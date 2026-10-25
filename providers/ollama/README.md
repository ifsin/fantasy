# Ollama

Provider for [Ollama](https://ollama.com) using its native `/api/chat` endpoint.

Ollama's OpenAI-compatible `/v1` endpoint cannot set per-request runtime
options such as `num_ctx`, so a long prompt is silently truncated to the
server's default context length. This provider talks to the native API, where
those options are available.

```go
import "charm.land/fantasy/providers/ollama"

provider, err := ollama.New(
	ollama.WithBaseURL("http://localhost:11434"),
	ollama.WithDefaultOptions(ollama.ProviderOptions{NumCtx: ptr(int64(8192))}),
)
model, err := provider.LanguageModel(ctx, "llama3.2")
```

A trailing `/v1` or `/api` on the base URL is ignored, so the same URL works for
the OpenAI-compatible providers.

Per-call options go through `ollama.NewProviderOptions`: `num_ctx`, `num_predict`,
`seed`, `min_p`, `repeat_penalty`, `stop`, `think`, `keep_alive` and an `Extra` map
for any other Ollama option. Options set on a call override the provider defaults.

Notes:

- Ollama has no tool choice setting. `none` drops the tools, a specific tool
  filters the list to that tool, and `required` is reported as a warning.
- `ObjectModeAuto` and `ObjectModeJSON` use Ollama's structured outputs.
- `WithAPIKey` sends a bearer token for hosted endpoints such as `https://ollama.com`.
