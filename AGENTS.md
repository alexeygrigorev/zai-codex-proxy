# Codex Proxy Context

## Overview

`zai-codex-proxy` is a Rust service for running Codex/zodex against Z.AI. It
accepts Codex's OpenAI Responses-shaped requests, normalizes them into a shared
typed internal form, then translates them to Z.AI chat completions.

## Current Architecture

### Request flow

The main request path lives in `src/server.rs` and follows this sequence:

1. Validate incoming Responses API-style request payloads.
2. Normalize `ResponsesRequest` payloads into the internal typed `ChatRequest` form.
3. Resolve routing in one place:
   - resolve the route via `routing.model_routes` (requested model key or `*`)
   - compute sticky-routing key from normalized message content
   - choose a healthy Z.AI account for the selected route
4. Dispatch to `src/providers/zai.rs`.
5. Apply success/failure health updates and sticky binding updates after execution.

### Providers

Current upstream provider module:

- `src/providers/zai.rs`

Runtime provider execution is a single Z.AI provider path. Do not add non-Z.AI
upstream support unless the project goal changes.

### Runtime routing and account state

Shared routing/account logic lives in:

- `src/account_pool/pool.rs`
- `src/account_pool/routing.rs`

Important runtime behaviors:

- multiple Z.AI accounts
- health-aware account selection
- sticky routing for KV-cache reuse
- cooldown-based unhealthy recovery
- per-account health stats surfaced through config/model endpoints

### Auth model

Z.AI uses account-scoped API-key auth via `accounts[]`.

## Configuration

Configuration uses a structured schema centered on:

- `server`
- `zai` (single upstream Z.AI config)
- `models`
- `routing`
- `accounts`
- `reasoning`
- `timeouts`
- `compaction`

Key behaviors:

- `accounts[]` is the source of truth for Z.AI credentials.
- `routing.model_routes` maps the requested model (or `*`) to ordered Z.AI physical targets and logical aliases.
- the config file has a single current format.

## UI and config endpoint

`GET /config` returns a typed snapshot of:

- structured config sections
- masked account auth data
- account health snapshot
- routing stats such as sticky binding count

Treat `POST /config` as read-only/stub unless the user explicitly asks to fully
implement config persistence for the schema.

## Development expectations

When changing routing/auth/provider code:

- keep Codex request normalization and Z.AI payload adaptation in `src/codex_wire/*`
- keep provider selection/account selection in shared routing
- do not reintroduce model-prefix routing inside provider registry
- do not read provider credentials directly from global flat config
- preserve Z.AI web search tool passthrough

## Verification

Preferred verification commands:

```bash
cargo fmt
cargo check
cargo test
```

When touching routing/auth/config logic, also sanity check:

- config boot path
- provider override routing
- multi-account isolation
- sticky-routing failover behavior
- zodex E2E with `RUN_ZODEX_E2E=1`
