package ollama

import (
	"net/http"

	"charm.land/fantasy"
)

// HTTPClient is the interface used to send requests to the Ollama server.
type HTTPClient interface {
	Do(*http.Request) (*http.Response, error)
}

type options struct {
	name       string
	baseURL    string
	apiKey     string
	headers    map[string]string
	userAgent  string
	client     HTTPClient
	defaults   ProviderOptions
	objectMode fantasy.ObjectMode
}

// Option defines a function that configures Ollama provider options.
type Option = func(*options)

// WithName sets the name for the Ollama provider.
func WithName(name string) Option {
	return func(o *options) {
		o.name = name
	}
}

// WithBaseURL sets the base URL for the Ollama provider. A trailing "/v1" or
// "/api" is ignored, so an OpenAI-compatible base URL can be passed as is.
func WithBaseURL(url string) Option {
	return func(o *options) {
		o.baseURL = url
	}
}

// WithAPIKey sets the API key for the Ollama provider. It is sent as a bearer
// token, which hosted Ollama endpoints require and a local server ignores.
func WithAPIKey(apiKey string) Option {
	return func(o *options) {
		o.apiKey = apiKey
	}
}

// WithHeaders sets the headers for the Ollama provider.
func WithHeaders(headers map[string]string) Option {
	return func(o *options) {
		o.headers = headers
	}
}

// WithUserAgent sets an explicit User-Agent header, overriding the default and
// any value set via WithHeaders.
func WithUserAgent(ua string) Option {
	return func(o *options) {
		o.userAgent = ua
	}
}

// WithHTTPClient sets the HTTP client for the Ollama provider.
func WithHTTPClient(client HTTPClient) Option {
	return func(o *options) {
		o.client = client
	}
}

// WithDefaultOptions sets the provider options applied to every call. Options
// set on a call take precedence.
func WithDefaultOptions(opts ProviderOptions) Option {
	return func(o *options) {
		o.defaults = opts
	}
}

// WithObjectMode sets the object generation mode for the Ollama provider.
// ObjectModeAuto and ObjectModeJSON use Ollama's structured outputs.
func WithObjectMode(om fantasy.ObjectMode) Option {
	return func(o *options) {
		o.objectMode = om
	}
}
