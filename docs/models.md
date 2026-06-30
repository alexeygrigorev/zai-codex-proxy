# Models endpoint

codex-proxy exposes the configured public model list through the OpenAI-compatible models API.

## `GET /v1/models`

codex-proxy serves an OpenAI-style models list on:

- `GET /v1/models`
- `GET /models`

The returned list is always `models.served`. Each served model is projected with metadata from the first resolved Z.AI
route target when `model_metadata` contains an entry for that upstream model.

## Metadata and pricing

Configure model metadata directly by Z.AI model id:

```json
{
  "model_metadata": {
    "glm-5.2": {
      "context_window": 128000,
      "max_output_tokens": 16384,
      "pricing": {
        "input_per_mtoken": 10.0,
        "output_per_mtoken": 30.0
      }
    }
  }
}
```
