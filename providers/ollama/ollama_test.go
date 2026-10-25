package ollama

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"charm.land/fantasy"
	"charm.land/fantasy/schema"
	"github.com/stretchr/testify/require"
)

type capture struct {
	path    string
	header  http.Header
	request map[string]any
}

func newServer(t *testing.T, status int, body string) (*httptest.Server, *capture) {
	t.Helper()

	got := &capture{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got.path = r.URL.Path
		got.header = r.Header.Clone()
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &got.request)

		w.WriteHeader(status)
		_, _ = io.WriteString(w, body)
	}))
	t.Cleanup(server.Close)

	return server, got
}

func newModel(t *testing.T, opts ...Option) fantasy.LanguageModel {
	t.Helper()

	provider, err := New(opts...)
	require.NoError(t, err)

	model, err := provider.LanguageModel(context.Background(), "muse")
	require.NoError(t, err)

	return model
}

func userCall(text string) fantasy.Call {
	return fantasy.Call{
		Prompt: fantasy.Prompt{
			{Role: fantasy.MessageRoleUser, Content: []fantasy.MessagePart{fantasy.TextPart{Text: text}}},
		},
	}
}

func lines(chunks ...string) string {
	return strings.Join(chunks, "\n") + "\n"
}

func collect(t *testing.T, stream fantasy.StreamResponse) []fantasy.StreamPart {
	t.Helper()

	var parts []fantasy.StreamPart
	for part := range stream {
		parts = append(parts, part)
	}
	return parts
}

func types(parts []fantasy.StreamPart) []fantasy.StreamPartType {
	out := make([]fantasy.StreamPartType, 0, len(parts))
	for _, part := range parts {
		out = append(out, part.Type)
	}
	return out
}

func TestGenerate(t *testing.T) {
	t.Parallel()

	server, got := newServer(t, http.StatusOK, `{"model":"muse","message":{"role":"assistant","content":"Hello","thinking":"hmm"},"done":true,"done_reason":"stop","prompt_eval_count":12,"eval_count":3,"total_duration":5}`)

	model := newModel(t,
		WithBaseURL(server.URL+"/v1/"),
		WithAPIKey("secret"),
		WithHeaders(map[string]string{"X-Test": "yes"}),
		WithDefaultOptions(ProviderOptions{NumCtx: new(int64(8192)), Seed: new(int64(7))}),
	)

	temperature := 0.5
	maxTokens := int64(64)
	call := userCall("hi")
	call.Temperature = &temperature
	call.MaxOutputTokens = &maxTokens
	call.ProviderOptions = NewProviderOptions(&ProviderOptions{Seed: new(int64(9)), Think: new(true)})

	resp, err := model.Generate(context.Background(), call)
	require.NoError(t, err)

	require.Equal(t, "/api/chat", got.path)
	require.Equal(t, "Bearer secret", got.header.Get("Authorization"))
	require.Equal(t, "yes", got.header.Get("X-Test"))
	require.Contains(t, got.header.Get("User-Agent"), "Fantasy")
	require.Equal(t, false, got.request["stream"])
	require.Equal(t, true, got.request["think"])
	require.Equal(t, map[string]any{
		"num_ctx":     float64(8192),
		"seed":        float64(9),
		"temperature": 0.5,
		"num_predict": float64(64),
	}, got.request["options"])

	require.Equal(t, "Hello", resp.Content.Text())
	require.Equal(t, "hmm", resp.Content.ReasoningText())
	require.Equal(t, fantasy.FinishReasonStop, resp.FinishReason)
	require.Equal(t, fantasy.Usage{InputTokens: 12, OutputTokens: 3, TotalTokens: 15}, resp.Usage)

	metadata, ok := resp.ProviderMetadata[Name].(*ProviderMetadata)
	require.True(t, ok)
	require.Equal(t, "muse", metadata.Model)
	require.Equal(t, int64(5), metadata.TotalDurationNS)
}

func TestGenerateToolCalls(t *testing.T) {
	t.Parallel()

	server, got := newServer(t, http.StatusOK, `{"model":"muse","message":{"role":"assistant","content":"","tool_calls":[{"function":{"name":"weather","arguments":{"city":"Oslo"}}}]},"done":true,"done_reason":"stop","prompt_eval_count":1,"eval_count":1}`)
	model := newModel(t, WithBaseURL(server.URL))

	call := userCall("weather?")
	call.Tools = []fantasy.Tool{fantasy.FunctionTool{
		Name:        "weather",
		Description: "get weather",
		InputSchema: map[string]any{"type": "object"},
	}}

	resp, err := model.Generate(context.Background(), call)
	require.NoError(t, err)

	tools, ok := got.request["tools"].([]any)
	require.True(t, ok)
	require.Len(t, tools, 1)

	calls := resp.Content.ToolCalls()
	require.Len(t, calls, 1)
	require.NotEmpty(t, calls[0].ToolCallID)
	require.Equal(t, "weather", calls[0].ToolName)
	require.JSONEq(t, `{"city":"Oslo"}`, calls[0].Input)
	require.Equal(t, fantasy.FinishReasonToolCalls, resp.FinishReason)
}

func TestStream(t *testing.T) {
	t.Parallel()

	server, got := newServer(t, http.StatusOK, lines(
		`{"model":"muse","message":{"role":"assistant","content":"","thinking":"a"},"done":false}`,
		`{"model":"muse","message":{"role":"assistant","content":"","thinking":"b"},"done":false}`,
		`{"model":"muse","message":{"role":"assistant","content":"Hel"},"done":false}`,
		`{"model":"muse","message":{"role":"assistant","content":"lo"},"done":false}`,
		`{"model":"muse","message":{"role":"assistant","content":""},"done":true,"done_reason":"length","prompt_eval_count":10,"eval_count":2}`,
	))
	model := newModel(t, WithBaseURL(server.URL))

	stream, err := model.Stream(context.Background(), userCall("hi"))
	require.NoError(t, err)
	parts := collect(t, stream)

	require.Equal(t, true, got.request["stream"])
	require.Equal(t, []fantasy.StreamPartType{
		fantasy.StreamPartTypeReasoningStart,
		fantasy.StreamPartTypeReasoningDelta,
		fantasy.StreamPartTypeReasoningDelta,
		fantasy.StreamPartTypeReasoningEnd,
		fantasy.StreamPartTypeTextStart,
		fantasy.StreamPartTypeTextDelta,
		fantasy.StreamPartTypeTextDelta,
		fantasy.StreamPartTypeTextEnd,
		fantasy.StreamPartTypeFinish,
	}, types(parts))

	finish := parts[len(parts)-1]
	require.Equal(t, fantasy.FinishReasonLength, finish.FinishReason)
	require.Equal(t, fantasy.Usage{InputTokens: 10, OutputTokens: 2, TotalTokens: 12}, finish.Usage)
}

func TestStreamToolCall(t *testing.T) {
	t.Parallel()

	server, _ := newServer(t, http.StatusOK, lines(
		`{"model":"muse","message":{"role":"assistant","content":"ok"},"done":false}`,
		`{"model":"muse","message":{"role":"assistant","content":"","tool_calls":[{"id":"call_1","function":{"name":"weather","arguments":{"city":"Oslo"}}}]},"done":false}`,
		`{"model":"muse","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop"}`,
	))
	model := newModel(t, WithBaseURL(server.URL))

	stream, err := model.Stream(context.Background(), userCall("hi"))
	require.NoError(t, err)
	parts := collect(t, stream)

	require.Equal(t, []fantasy.StreamPartType{
		fantasy.StreamPartTypeTextStart,
		fantasy.StreamPartTypeTextDelta,
		fantasy.StreamPartTypeTextEnd,
		fantasy.StreamPartTypeToolInputStart,
		fantasy.StreamPartTypeToolInputDelta,
		fantasy.StreamPartTypeToolInputEnd,
		fantasy.StreamPartTypeToolCall,
		fantasy.StreamPartTypeFinish,
	}, types(parts))

	call := parts[6]
	require.Equal(t, "call_1", call.ID)
	require.Equal(t, "weather", call.ToolCallName)
	require.JSONEq(t, `{"city":"Oslo"}`, call.ToolCallInput)
	require.Equal(t, fantasy.FinishReasonToolCalls, parts[7].FinishReason)
}

func TestStreamIncomplete(t *testing.T) {
	t.Parallel()

	server, _ := newServer(t, http.StatusOK, lines(
		`{"model":"muse","message":{"role":"assistant","content":"partial"},"done":false}`,
	))
	model := newModel(t, WithBaseURL(server.URL))

	stream, err := model.Stream(context.Background(), userCall("hi"))
	require.NoError(t, err)
	parts := collect(t, stream)

	last := parts[len(parts)-1]
	require.Equal(t, fantasy.StreamPartTypeError, last.Type)

	var providerErr *fantasy.ProviderError
	require.True(t, errors.As(last.Error, &providerErr))
	require.True(t, providerErr.IsRetryable())
}

func TestStreamInBandError(t *testing.T) {
	t.Parallel()

	server, _ := newServer(t, http.StatusOK, lines(
		`{"model":"muse","message":{"role":"assistant","content":"a"},"done":false}`,
		`{"error":"model runner has unexpectedly stopped"}`,
	))
	model := newModel(t, WithBaseURL(server.URL))

	stream, err := model.Stream(context.Background(), userCall("hi"))
	require.NoError(t, err)
	parts := collect(t, stream)

	last := parts[len(parts)-1]
	require.Equal(t, fantasy.StreamPartTypeError, last.Type)
	require.ErrorContains(t, last.Error, "model runner has unexpectedly stopped")
}

func TestStatusError(t *testing.T) {
	t.Parallel()

	server, _ := newServer(t, http.StatusNotFound, `{"error":"model 'muse' not found"}`)
	model := newModel(t, WithBaseURL(server.URL))

	_, err := model.Generate(context.Background(), userCall("hi"))

	var providerErr *fantasy.ProviderError
	require.True(t, errors.As(err, &providerErr))
	require.Equal(t, http.StatusNotFound, providerErr.StatusCode)
	require.Equal(t, "model 'muse' not found", providerErr.Message)
	require.False(t, providerErr.IsRetryable())
}

func TestContextTooLargeError(t *testing.T) {
	t.Parallel()

	server, _ := newServer(t, http.StatusBadRequest, `{"error":"prompt too long; exceeded max context length by 120 tokens"}`)
	model := newModel(t, WithBaseURL(server.URL))

	_, err := model.Generate(context.Background(), userCall("hi"))

	var providerErr *fantasy.ProviderError
	require.True(t, errors.As(err, &providerErr))
	require.True(t, providerErr.IsContextTooLarge())
}

func TestGenerateObject(t *testing.T) {
	t.Parallel()

	server, got := newServer(t, http.StatusOK, `{"model":"muse","message":{"role":"assistant","content":"{\"name\":\"Ada\"}"},"done":true,"done_reason":"stop","prompt_eval_count":4,"eval_count":2}`)
	model := newModel(t, WithBaseURL(server.URL))

	resp, err := model.GenerateObject(context.Background(), fantasy.ObjectCall{
		Prompt: userCall("who?").Prompt,
		Schema: schema.Schema{
			Type:       "object",
			Properties: map[string]*schema.Schema{"name": {Type: "string"}},
			Required:   []string{"name"},
		},
	})
	require.NoError(t, err)

	format, ok := got.request["format"].(map[string]any)
	require.True(t, ok)
	require.Equal(t, "object", format["type"])
	require.Equal(t, map[string]any{"name": "Ada"}, resp.Object)
	require.Equal(t, int64(6), resp.Usage.TotalTokens)
}

func TestStreamObject(t *testing.T) {
	t.Parallel()

	server, _ := newServer(t, http.StatusOK, lines(
		`{"model":"muse","message":{"role":"assistant","content":"{\"name\":"},"done":false}`,
		`{"model":"muse","message":{"role":"assistant","content":"\"Ada\"}"},"done":false}`,
		`{"model":"muse","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop","prompt_eval_count":4,"eval_count":2}`,
	))
	model := newModel(t, WithBaseURL(server.URL))

	stream, err := model.StreamObject(context.Background(), fantasy.ObjectCall{
		Prompt: userCall("who?").Prompt,
		Schema: schema.Schema{
			Type:       "object",
			Properties: map[string]*schema.Schema{"name": {Type: "string"}},
			Required:   []string{"name"},
		},
	})
	require.NoError(t, err)

	var objects []any
	var finish *fantasy.ObjectStreamPart
	for part := range stream {
		switch part.Type {
		case fantasy.ObjectStreamPartTypeObject:
			objects = append(objects, part.Object)
		case fantasy.ObjectStreamPartTypeFinish:
			finish = &part
		case fantasy.ObjectStreamPartTypeError:
			t.Fatalf("unexpected error: %v", part.Error)
		}
	}

	require.NotEmpty(t, objects)
	require.Equal(t, map[string]any{"name": "Ada"}, objects[len(objects)-1])
	require.NotNil(t, finish)
	require.Equal(t, fantasy.FinishReasonStop, finish.FinishReason)
}

func TestToPrompt(t *testing.T) {
	t.Parallel()

	prompt := fantasy.Prompt{
		{Role: fantasy.MessageRoleSystem, Content: []fantasy.MessagePart{fantasy.TextPart{Text: "be brief"}}},
		{Role: fantasy.MessageRoleUser, Content: []fantasy.MessagePart{
			fantasy.TextPart{Text: "look"},
			fantasy.TextPart{Text: "here"},
			fantasy.FilePart{MediaType: "image/png", Data: []byte{1, 2, 3}},
			fantasy.FilePart{MediaType: "application/pdf", Data: []byte{4}},
		}},
		{Role: fantasy.MessageRoleAssistant, Content: []fantasy.MessagePart{
			fantasy.ReasoningPart{Text: "a"},
			fantasy.ReasoningPart{Text: "b"},
			fantasy.TextPart{Text: "x"},
			fantasy.TextPart{Text: "y"},
			fantasy.ToolCallPart{ToolCallID: "c1", ToolName: "weather", Input: `{"city":"Oslo"}`},
			fantasy.ToolCallPart{ToolCallID: "c2", ToolName: "time", Input: ``},
		}},
		{Role: fantasy.MessageRoleTool, Content: []fantasy.MessagePart{
			fantasy.ToolResultPart{ToolCallID: "c1", Output: fantasy.ToolResultOutputContentText{Text: "rain"}},
			fantasy.ToolResultPart{ToolCallID: "c2", Output: fantasy.ToolResultOutputContentError{Error: errors.New("boom")}},
		}},
		{Role: fantasy.MessageRoleAssistant, Content: []fantasy.MessagePart{fantasy.ReasoningPart{Text: "only thoughts"}}},
	}

	messages, warnings := toPrompt(prompt)

	require.Len(t, warnings, 1)
	require.Contains(t, warnings[0].Message, "application/pdf")
	require.Len(t, messages, 5)

	require.Equal(t, message{Role: "system", Content: "be brief"}, messages[0])
	require.Equal(t, message{Role: "user", Content: "look\nhere", Images: [][]byte{{1, 2, 3}}}, messages[1])

	assistant := messages[2]
	require.Equal(t, "x\ny", assistant.Content)
	require.Equal(t, "a\nb", assistant.Thinking)
	require.Len(t, assistant.ToolCalls, 2)
	require.Equal(t, 1, assistant.ToolCalls[1].Function.Index)
	require.JSONEq(t, `{"city":"Oslo"}`, string(assistant.ToolCalls[0].Function.Arguments))
	require.JSONEq(t, `{}`, string(assistant.ToolCalls[1].Function.Arguments))

	require.Equal(t, message{Role: "tool", Content: "rain", ToolName: "weather", ToolCallID: "c1"}, messages[3])
	require.Equal(t, message{Role: "tool", Content: "boom", ToolName: "time", ToolCallID: "c2"}, messages[4])
}

func TestToolChoice(t *testing.T) {
	t.Parallel()

	tools := []fantasy.Tool{
		fantasy.FunctionTool{Name: "a", InputSchema: map[string]any{}},
		fantasy.FunctionTool{Name: "b", InputSchema: map[string]any{}},
	}

	choice := func(c fantasy.ToolChoice) *fantasy.ToolChoice { return &c }

	out, warnings := toTools(tools, nil)
	require.Len(t, out, 2)
	require.Empty(t, warnings)

	out, _ = toTools(tools, choice(fantasy.ToolChoiceNone))
	require.Empty(t, out)

	out, _ = toTools(tools, choice(fantasy.SpecificToolChoice("b")))
	require.Len(t, out, 1)
	require.Equal(t, "b", out[0].Function.Name)

	out, warnings = toTools(tools, choice(fantasy.ToolChoiceRequired))
	require.Len(t, out, 2)
	require.Len(t, warnings, 1)
	require.Equal(t, fantasy.CallWarningTypeUnsupportedSetting, warnings[0].Type)
}

func TestNormalizeBaseURL(t *testing.T) {
	t.Parallel()

	for in, want := range map[string]string{
		"http://localhost:11434":      "http://localhost:11434",
		"http://localhost:11434/":     "http://localhost:11434",
		"http://localhost:11434/v1":   "http://localhost:11434",
		"http://localhost:11434/v1/":  "http://localhost:11434",
		"https://ollama.com/api":      "https://ollama.com",
		"http://host/prefix/v1":       "http://host/prefix",
		"http://host:11434/api/v1///": "http://host:11434",
	} {
		require.Equal(t, want, normalizeBaseURL(in), in)
	}
}

func TestProviderOptionsRoundTrip(t *testing.T) {
	t.Parallel()

	call := userCall("hi")
	call.ProviderOptions = NewProviderOptions(&ProviderOptions{
		NumCtx:      new(int64(4096)),
		ThinkEffort: new("low"),
		Stop:        []string{"END"},
		Extra:       map[string]any{"mirostat": 2},
	})

	raw, err := json.Marshal(call)
	require.NoError(t, err)

	var decoded fantasy.Call
	require.NoError(t, json.Unmarshal(raw, &decoded))

	opts, ok := decoded.ProviderOptions[Name].(*ProviderOptions)
	require.True(t, ok)
	require.Equal(t, int64(4096), *opts.NumCtx)
	require.Equal(t, "low", *opts.ThinkEffort)

	req := chatRequest{}
	opts.apply(&req)
	require.Equal(t, "low", req.Think)
	require.Equal(t, int64(4096), req.Options["num_ctx"])
	require.Equal(t, []string{"END"}, req.Options["stop"])
	require.EqualValues(t, 2, req.Options["mirostat"])
}

func TestCallHeadersAndUserAgent(t *testing.T) {
	t.Parallel()

	server, got := newServer(t, http.StatusOK, `{"model":"muse","message":{"role":"assistant","content":"ok"},"done":true}`)
	model := newModel(t, WithBaseURL(server.URL), WithUserAgent("base-agent"))

	call := userCall("hi")
	call.UserAgent = "call-agent"
	call.Headers = map[string]string{"X-Call": "1"}

	_, err := model.Generate(context.Background(), call)
	require.NoError(t, err)
	require.Equal(t, "call-agent", got.header.Get("User-Agent"))
	require.Equal(t, "1", got.header.Get("X-Call"))
}

func TestContextCanceled(t *testing.T) {
	t.Parallel()

	server, _ := newServer(t, http.StatusOK, `{}`)
	model := newModel(t, WithBaseURL(server.URL))

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := model.Generate(ctx, userCall("hi"))
	require.ErrorIs(t, err, context.Canceled, fmt.Sprint(err))
}
