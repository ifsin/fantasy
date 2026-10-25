package ollama

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"regexp"
	"strings"

	"charm.land/fantasy"
)

var contextPattern = regexp.MustCompile(`exceeded max context length by (\d+) tokens`)

func toProviderErr(err error) error {
	if err == nil || errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return err
	}
	if _, ok := errors.AsType[*fantasy.ProviderError](err); ok {
		return err
	}
	return fantasy.WrapTransportError(err)
}

func newStatusErr(url string, header http.Header, statusCode int, requestBody, responseBody []byte) *fantasy.ProviderError {
	message := strings.TrimSpace(string(responseBody))

	var parsed errorResponse
	if json.Unmarshal(responseBody, &parsed) == nil && parsed.Error != "" {
		message = parsed.Error
	}

	providerErr := &fantasy.ProviderError{
		Title:           cmp.Or(fantasy.ErrorTitleForStatusCode(statusCode), "provider request failed"),
		Message:         cmp.Or(message, http.StatusText(statusCode)),
		URL:             url,
		StatusCode:      statusCode,
		RequestBody:     requestBody,
		ResponseHeaders: toHeaderMap(header),
		ResponseBody:    responseBody,
	}
	parseContextTooLargeError(providerErr.Message, providerErr)

	return providerErr
}

func newStreamErr(message string) *fantasy.ProviderError {
	providerErr := &fantasy.ProviderError{
		Title:   "ollama stream error",
		Message: message,
	}
	parseContextTooLargeError(message, providerErr)

	return providerErr
}

func parseContextTooLargeError(message string, providerErr *fantasy.ProviderError) {
	if contextPattern.MatchString(message) || strings.Contains(message, "exceeds the available context size") {
		providerErr.ContextTooLargeErr = true
	}
}

func toHeaderMap(header http.Header) map[string]string {
	out := make(map[string]string, len(header))
	for k, v := range header {
		out[k] = strings.Join(v, ", ")
	}
	return out
}
