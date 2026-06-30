# Contributing to zai-codex-proxy

## Setup

Install Rust 2024 tooling, then run:

```bash
cargo fmt
cargo test
cargo clippy --all-targets -- -D warnings
```

The optional zodex integration tests require Python/pytest plus real Z.AI credentials:

```bash
RUN_ZODEX_E2E=1 CODEX_PROXY_ZAI_API_KEY=... pytest -q tests/integration
```

## Development Workflow

1. Create a feature branch.
2. Keep changes focused on Codex/zodex and Z.AI.
3. Add or update tests for behavior changes.
4. Update docs for config, CLI, or runtime behavior changes.
5. Run formatting, tests, and clippy before opening a PR.

## Code Style

- Prefer the existing Rust module structure and helper APIs.
- Keep request normalization and Z.AI payload adaptation in `src/codex_wire`.
- Keep runtime provider execution in `src/providers`.
- Keep routing/account selection in `src/account_pool`.
- Do not add non-Z.AI provider support unless the project goal changes.

## Documentation

- Update `README.md` for user-facing behavior.
- Update files under `docs/` for runtime behavior details.
- Keep `AGENTS.md` in sync with implementation guidance.
