use anyhow::Context;
use serde::Serialize;
use sqlx::{FromRow, SqlitePool};

#[derive(Clone)]
pub struct AppState {
    pub pool: SqlitePool,
}

#[derive(Debug, Clone, FromRow, Serialize)]
pub struct EmployeeRow {
    pub id: i64,
    pub name: String,
    pub salary: i64,
    pub manager_id: Option<i64>,
    pub manager_name: Option<String>,
}

#[derive(Debug, Clone, FromRow, Serialize)]
pub struct OutlierRow {
    pub employee: String,
    pub employee_salary: i64,
    pub manager: String,
    pub manager_salary: i64,
    pub delta: i64,
}

#[derive(Debug, Clone, Serialize)]
pub struct DashboardSnapshot {
    pub employees: Vec<EmployeeRow>,
    pub outliers: Vec<OutlierRow>,
    pub headcount: i64,
    pub managers: i64,
    pub outlier_count: i64,
}

const SCHEMA_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS employee (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    salary INTEGER NOT NULL,
    manager_id INTEGER REFERENCES employee(id)
);
"#;

/// Open a pool, apply schema, and seed LeetCode 181 sample rows if empty.
pub async fn connect_and_seed(database_url: &str) -> anyhow::Result<SqlitePool> {
    let pool = SqlitePool::connect(database_url)
        .await
        .with_context(|| format!("failed to connect to {database_url}"))?;

    sqlx::query(SCHEMA_SQL).execute(&pool).await?;

    let count: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM employee")
        .fetch_one(&pool)
        .await?;

    if count.0 == 0 {
        seed_example(&pool).await?;
        tracing::info!("seeded LeetCode 181 employee sample");
    }

    Ok(pool)
}

async fn seed_example(pool: &SqlitePool) -> anyhow::Result<()> {
    let mut tx = pool.begin().await?;
    // Managers first so the self-reference is valid.
    sqlx::query("INSERT INTO employee (id, name, salary, manager_id) VALUES (?, ?, ?, ?)")
        .bind(3_i64)
        .bind("Sam")
        .bind(60_000_i64)
        .bind(None::<i64>)
        .execute(&mut *tx)
        .await?;
    sqlx::query("INSERT INTO employee (id, name, salary, manager_id) VALUES (?, ?, ?, ?)")
        .bind(4_i64)
        .bind("Max")
        .bind(90_000_i64)
        .bind(None::<i64>)
        .execute(&mut *tx)
        .await?;
    sqlx::query("INSERT INTO employee (id, name, salary, manager_id) VALUES (?, ?, ?, ?)")
        .bind(1_i64)
        .bind("Joe")
        .bind(70_000_i64)
        .bind(Some(3_i64))
        .execute(&mut *tx)
        .await?;
    sqlx::query("INSERT INTO employee (id, name, salary, manager_id) VALUES (?, ?, ?, ?)")
        .bind(2_i64)
        .bind("Henry")
        .bind(80_000_i64)
        .bind(Some(4_i64))
        .execute(&mut *tx)
        .await?;
    tx.commit().await?;
    Ok(())
}

/// Run independent reads concurrently with `try_join!`.
pub async fn load_snapshot(pool: &SqlitePool) -> anyhow::Result<DashboardSnapshot> {
    let employees_fut = sqlx::query_as::<_, EmployeeRow>(
        r#"
        SELECT
            e.id,
            e.name,
            e.salary,
            e.manager_id,
            m.name AS manager_name
        FROM employee e
        LEFT JOIN employee m ON e.manager_id = m.id
        ORDER BY e.id
        "#,
    )
    .fetch_all(pool);

    let outliers_fut = sqlx::query_as::<_, OutlierRow>(
        r#"
        SELECT
            e.name AS employee,
            e.salary AS employee_salary,
            m.name AS manager,
            m.salary AS manager_salary,
            (e.salary - m.salary) AS delta
        FROM employee e
        INNER JOIN employee m ON e.manager_id = m.id
        WHERE e.salary > m.salary
        ORDER BY delta DESC
        "#,
    )
    .fetch_all(pool);

    let counts_fut = sqlx::query_as::<_, (i64, i64, i64)>(
        r#"
        SELECT
            COUNT(*) AS headcount,
            SUM(CASE WHEN manager_id IS NULL THEN 1 ELSE 0 END) AS managers,
            (
                SELECT COUNT(*)
                FROM employee e
                INNER JOIN employee m ON e.manager_id = m.id
                WHERE e.salary > m.salary
            ) AS outlier_count
        FROM employee
        "#,
    )
    .fetch_one(pool);

    let (employees, outliers, counts) =
        tokio::try_join!(employees_fut, outliers_fut, counts_fut)?;

    Ok(DashboardSnapshot {
        employees,
        outliers,
        headcount: counts.0,
        managers: counts.1,
        outlier_count: counts.2,
    })
}

pub async fn insert_employee(
    pool: &SqlitePool,
    name: &str,
    salary: i64,
    manager_id: Option<i64>,
) -> anyhow::Result<i64> {
    let result = sqlx::query("INSERT INTO employee (name, salary, manager_id) VALUES (?, ?, ?)")
        .bind(name)
        .bind(salary)
        .bind(manager_id)
        .execute(pool)
        .await?;
    Ok(result.last_insert_rowid())
}
