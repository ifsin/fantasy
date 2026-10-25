// Package ollama provides an implementation of the fantasy AI SDK for Ollama's native chat API.
package ollama

import (
	"context"
	"net/http"
	"strings"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/internal/httpheaders"
)

const (
	// DefaultURL is the default URL for a local Ollama server.
	DefaultURL = "http://localhost:11434"
	// Name is the name of the Ollama provider.
	Name = "ollama"
)

type provider struct {
	options options
}

// New creates a new Ollama provider with the given options.
func New(opts ...Option) (fantasy.Provider, error) {
	providerOptions := options{
		name:       Name,
		baseURL:    DefaultURL,
		objectMode: fantasy.ObjectModeAuto,
	}
	for _, o := range opts {
		o(&providerOptions)
	}

	providerOptions.baseURL = normalizeBaseURL(providerOptions.baseURL)

	return &provider{options: providerOptions}, nil
}

// Name implements fantasy.Provider.
func (p *provider) Name() string {
	return p.options.name
}

// LanguageModel implements fantasy.Provider.
func (p *provider) LanguageModel(_ context.Context, modelID string) (fantasy.LanguageModel, error) {
	client := p.options.client
	if client == nil {
		client = http.DefaultClient
	}

	headers := httpheaders.ResolveHeaders(
		p.options.headers,
		p.options.userAgent,
		httpheaders.DefaultUserAgent(fantasy.Version),
	)
	if p.options.apiKey != "" {
		headers["Authorization"] = "Bearer " + p.options.apiKey
	}

	return &languageModel{
		modelID:    modelID,
		provider:   p.options.name,
		baseURL:    p.options.baseURL,
		headers:    headers,
		client:     client,
		defaults:   p.options.defaults,
		objectMode: p.options.objectMode,
	}, nil
}

func normalizeBaseURL(url string) string {
	url = strings.TrimRight(url, "/")
	url = strings.TrimSuffix(url, "/v1")
	url = strings.TrimSuffix(url, "/api")
	return strings.TrimRight(url, "/")
}
