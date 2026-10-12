# rustFX as Laura's Rust Web3 engine

Laura (`github.com/Kvnbbg/Laura`) is the preferred Go CLI. It can check, suggest, and
build this crate from Ubuntu or any host with `rustup`. rustFX owns chain/coin engine
code in Rust. COPY remains the paper market loop. This crate does not sign.

```text
Ubuntu / other terminal
        |
        v
   laura  (Go)
        |-- /run rustfx check|status|coins|build
        |-- /install rustfx   (suggests rustup + cargo only)
        v
   rustfx-web3  (Rust binary)
        |
        +-- status JSON  (engine identity, no keys)
        +-- coins JSON   (catalog)
        v
   COPY paper desk (optional sibling)
```

## Build

```bash
rustup toolchain install stable
cargo build -p rustfx-web3 --release
./target/release/rustfx-web3 status
```

## Pilot from Laura

```bash
export LAURA_RUSTFX_REPO="$HOME/src/rustFX"
npm run chat   # then: /run rustfx status
```

Laura never pipes an installer through `curl | sh` for rustFX. The operator reviews
`rustup` / `cargo` commands first.
