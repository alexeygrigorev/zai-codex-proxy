# zai-codex-proxy

[![CI](https://github.com/alexeygrigorev/zai-codex-proxy/workflows/CI/badge.svg)](https://github.com/alexeygrigorev/zai-codex-proxy/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

An OpenAI Responses API-compatible proxy for running Codex/zodex against Z.AI.

This project is based on [`JakkuSakura/codex-proxy`](https://github.com/JakkuSakura/codex-proxy),
which served as the basis for this code.

This project is used by the `.agents` dotfiles project at
https://github.com/alexeygrigorev/.agents. In that setup, `zodex` downloads a
released `zai-codex-proxy` binary and uses it to route an isolated Codex profile
through Z.AI.

The proxy accepts Codex's Responses-shaped requests, normalizes them into a
shared typed internal form, resolves logical-model to Z.AI account/model routing,
and translates requests to the Z.AI chat completions API.

## Features

- OpenAI Responses-compatible `/responses` and `/v1/responses` endpoints for Codex/zodex
- Z.AI upstream
- Codex multi-agent namespace-tool flattening for Z.AI Chat Completions
- Z.AI web search tool passthrough
- Codex `agent_message` / encrypted-content normalization for subagent tasks
- Multi-account Z.AI routing
- Ordered preferred route targets per logical model
- Sticky routing for KV-cache reuse on the exact resolved model/account path
- Account health tracking with exponential backoff and recovery probes
- Route-scoped reasoning defaults mapped to Z.AI `thinking.type`
- Optional 429 retry support
- Z.AI summary auto-compaction after context-length errors

## Quick start

For `.agents`, no Rust toolchain is required. Configure `zodex` there and it
will download the latest release binary from:

https://github.com/alexeygrigorev/zai-codex-proxy/releases/latest

Manual Linux binary install:

```bash
mkdir -p ~/.zodex/bin
curl -fL \
  https://github.com/alexeygrigorev/zai-codex-proxy/releases/latest/download/zai-codex-proxy-linux-amd64 \
  -o ~/.zodex/bin/zai-codex-proxy
chmod +x ~/.zodex/bin/zai-codex-proxy
```

Source build requires [Rust](https://www.rust-lang.org/tools/install) (edition
2024).

```bash
git clone https://github.com/alexeygrigorev/zai-codex-proxy.git
cd zai-codex-proxy
CODEX_PROXY_ZAI_API_KEY=... cargo run --release
```

## Codex configuration

Example `~/.codex/config.toml`: see `config/codex/config.toml`.

```toml
model = "glm-5-turbo"
model_provider = "codex-proxy"
service_tier = "fast"
disable_response_storage = true
personality = "pragmatic"

[model_providers.codex-proxy]
name = "openai"
base_url = "http://127.0.0.1:8765/v1"
wire_api = "responses"
requires_openai_auth = false
env_key = "CODEX_PROXY_API_KEY"
```

The `name = "openai"` setting is Codex CLI terminology for an OpenAI-shaped
wire protocol; the upstream provider remains Z.AI.

## Running as a daemon (systemd)

A template user unit is provided at `deploy/zai-codex-proxy.service`. It uses
`%h` (your home directory), so it installs as-is without editing, and runs the
proxy under your own user so it can read `~/.zodex`.

```bash
mkdir -p ~/.config/systemd/user
cp deploy/zai-codex-proxy.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now zai-codex-proxy
loginctl enable-linger "$USER"   # start at boot, survive logout
```

Prerequisites (already present for an existing zodex install): the binary at
`~/.zodex/bin/zai-codex-proxy`, the config at
`~/.zodex/codex-proxy/config.json`, and `~/.zodex/zai.env` exporting
`ZAI_API_KEY`. The unit loads that env file; the proxy accepts `ZAI_API_KEY`
directly (or `CODEX_PROXY_ZAI_API_KEY`).

Manage it with the usual user-service commands; logs go to the user journal:

```bash
systemctl --user status|restart|stop zai-codex-proxy
journalctl --user -u zai-codex-proxy -f
```

## Configuration

Configuration search order is:

1. `config/config.json.local`
2. `~/.config/codex-proxy/config.json`
3. `config/config.json`

Top-level sections:

- `server`
- `zai`
- `models`
- `routing`
- `health`
- `accounts`
- `reasoning`
- `timeouts`
- `retry`
- `compaction`

### Example config

```json
{
  "server": {
    "host": "127.0.0.1",
    "port": 8765,
    "log_level": "INFO"
  },
  "zai": {
    "api_url": "https://api.z.ai/api/coding/paas/v4/chat/completions",
    "models": ["glm-5.2", "glm-5-turbo"]
  },
  "models": {
    "served": ["glm-5.2", "glm-5-turbo", "compact-default"]
  },
  "model_metadata": {
    "glm-5.2": {
      "context_window": 128000,
      "max_output_tokens": 16384
    }
  },
  "routing": {
    "model_routes": {
      "*": ["glm-5.2"],
      "glm-5.2": [
        {
          "type": "physical",
          "model": "glm-5.2",
          "reasoning": { "effort": "high" }
        }
      ],
      "glm-5-turbo": [
        {
          "type": "physical",
          "model": "glm-5-turbo",
          "reasoning": { "effort": "medium" }
        }
      ],
      "compact-default": [
        {
          "type": "physical",
          "model": "glm-5-turbo",
          "reasoning": { "effort": "none" }
        }
      ]
    }
  },
  "health": {
    "auth_failure_immediate_unhealthy": true,
    "failure_threshold": 3,
    "cooldown_seconds": 60
  },
  "accounts": [
    {
      "id": "zai-primary",
      "enabled": true,
      "weight": 1,
      "auth": {
        "type": "api_key",
        "api_key": "zai-..."
      }
    }
  ],
  "auto_compaction": {
    "enabled": true,
    "max_attempts_per_request": 1,
    "tail_items_to_keep": 8
  },
  "reasoning": {
    "default_effort": "medium",
    "effort_levels": {
      "none": { "budget": 0, "level": "LOW" },
      "medium": { "budget": 16384, "level": "MEDIUM" },
      "high": { "budget": 32768, "level": "HIGH" }
    }
  },
  "timeouts": {
    "connect_seconds": 10,
    "read_seconds": 600
  },
  "retry": {
    "enabled": true,
    "max_attempts": 5,
    "initial_delay_ms": 1000,
    "max_delay_ms": 60000,
    "backoff_multiplier": 2.0
  },
  "compaction": {
    "temperature": 0.1,
    "preferred_targets": ["compact-default"]
  }
}
```

### Routing model

- `routing.model_routes[logical_model]` is the canonical ordered list of route steps.
- A route step can be written as a shorthand string:
  - Physical Z.AI model: `"model"` (example: `"glm-5.2"`)
  - Logical: `"proxy:logical_model"` (example: `"proxy:glm-5-turbo"`)
- The `zai` section is the single upstream config.
- For per-step `reasoning`, use the structured object form.
- Routing resolves `routing.model_routes[requested_model]` first, then falls back to `routing.model_routes["*"]`.
- Sticky routing is always enabled and reuses the exact chosen `(model, account)` path when it is still healthy.
- If the sticky-bound path is unhealthy, routing falls through to the next compatible preferred target.
- `accounts[].models`, when omitted, means the account can use any model from `zai.models`.
- `accounts[].weight` is used as a stable tiebreaker between otherwise equivalent compatible accounts.
- Accounts are marked unhealthy on the first failure, back off exponentially, and only return after a recovery-probe path succeeds.

### Reasoning model

- Reusable reasoning presets live in `reasoning.effort_levels`.
- `reasoning.default_effort` is optional and acts as the fallback when a route target does not specify one.
- A route target can reference a preset with `reasoning.effort` or provide inline `budget` / `level` overrides.
- Z.AI route reasoning is forwarded as `thinking.type` (`enabled` / `disabled`).

### Notes

- `accounts[]` is the source of truth for upstream Z.AI credentials.
- `CODEX_PROXY_ZAI_API_KEY` can provide a default `zai-default` account when no config file is loaded.
- Auto-compaction uses the same ordered route planning path as normal responses, starting from `compaction.preferred_targets`.
- Auto-compaction prompt text is embedded in the binary; config controls whether it runs and how aggressively it trims.

## Credits

The basis for this code was [`JakkuSakura/codex-proxy`](https://github.com/JakkuSakura/codex-proxy).
This project builds on that work to route Codex/zodex through Z.AI.
