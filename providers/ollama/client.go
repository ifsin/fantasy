package ollama

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/internal/httpheaders"
)

const maxErrorBody = 1 << 20

func (l *languageModel) post(ctx context.Context, req chatRequest, userAgent string, headers map[string]string) (*http.Response, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, &fantasy.Error{Title: "invalid request", Message: err.Error(), Cause: err}
	}

	url := l.baseURL + "/api/chat"
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, &fantasy.Error{Title: "invalid request", Message: err.Error(), Cause: err}
	}

	httpReq.Header.Set("Content-Type", "application/json")
	for key, value := range l.headers {
		httpReq.Header.Set(key, value)
	}
	if ua, ok := httpheaders.CallUserAgent(userAgent); ok {
		httpReq.Header.Set("User-Agent", ua)
	}
	if callHeaders, ok := httpheaders.CallHeaders(headers); ok {
		for key, value := range callHeaders {
			httpReq.Header.Set(key, value)
		}
	}

	resp, err := l.client.Do(httpReq)
	if err != nil {
		return nil, toProviderErr(err)
	}

	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		defer func() { _ = resp.Body.Close() }()
		responseBody, _ := io.ReadAll(io.LimitReader(resp.Body, maxErrorBody))
		return nil, newStatusErr(url, resp.Header, resp.StatusCode, body, responseBody)
	}

	return resp, nil
}

func (l *languageModel) chat(ctx context.Context, req chatRequest, userAgent string, headers map[string]string) (*chatResponse, error) {
	req.Stream = false

	resp, err := l.post(ctx, req, userAgent, headers)
	if err != nil {
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()

	var out chatResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, toProviderErr(err)
	}
	if out.Error != "" {
		return nil, newStreamErr(out.Error)
	}

	return &out, nil
}

func (l *languageModel) chatStream(ctx context.Context, req chatRequest, userAgent string, headers map[string]string) (*json.Decoder, io.Closer, error) {
	req.Stream = true

	resp, err := l.post(ctx, req, userAgent, headers)
	if err != nil {
		return nil, nil, err
	}

	return json.NewDecoder(resp.Body), resp.Body, nil
}
