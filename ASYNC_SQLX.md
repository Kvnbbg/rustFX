# Async sqlx dashboard

LeetCode 181 is implemented as a live payroll control room inside rustFX.

## Run

```bash
cargo run -p rustfx-dashboard
```

Open http://127.0.0.1:8080/payroll.html

Optional environment:

- `DATABASE_URL` — defaults to `sqlite://rustfx-payroll.db?mode=rwc`
- `RUSTFX_DASHBOARD_BIND` — defaults to `127.0.0.1:8080`

## Async patterns in this crate

| Pattern | Where |
| --- | --- |
| Multi-thread Tokio runtime | `#[tokio::main]` in `main.rs` |
| Shared pool as cheap clone | `AppState { pool }` |
| Concurrent independent queries | `tokio::try_join!` in `db::load_snapshot` |
| Transactional seed | `pool.begin()` then `tx.commit()` |
| Runtime SQL (no offline `query!`) | `sqlx::query` / `query_as` so compile does not need a live database |
| Async HTTP | Axum extractors + `Json` |

## SQL (same join as LeetCode 181)

```sql
SELECT e.name AS employee, e.salary AS employee_salary,
       m.name AS manager, m.salary AS manager_salary,
       (e.salary - m.salary) AS delta
FROM employee e
INNER JOIN employee m ON e.manager_id = m.id
WHERE e.salary > m.salary;
```

## API

- `GET /api/health`
- `GET /api/dashboard` — roster, outliers, counts
- `POST /api/employees` — `{ "name", "salary", "manager_id" }`
