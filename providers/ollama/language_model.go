package ollama

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"reflect"

	"charm.land/fantasy"
	"charm.land/fantasy/object"
	"charm.land/fantasy/schema"
	"github.com/google/uuid"
)

const (
	textID      = "0"
	reasoningID = "reasoning-0"
)

type languageModel struct {
	modelID    string
	provider   string
	baseURL    string
	headers    map[string]string
	client     HTTPClient
	defaults   ProviderOptions
	objectMode fantasy.ObjectMode
}

// Model implements fantasy.LanguageModel.
func (l *languageModel) Model() string {
	return l.modelID
}

// Provider implements fantasy.LanguageModel.
func (l *languageModel) Provider() string {
	return l.provider
}

// Generate implements fantasy.LanguageModel.
func (l *languageModel) Generate(ctx context.Context, call fantasy.Call) (*fantasy.Response, error) {
	req, warnings, err := l.prepareRequest(call)
	if err != nil {
		return nil, err
	}

	resp, err := l.chat(ctx, req, call.UserAgent, call.Headers)
	if err != nil {
		return nil, err
	}

	content := make([]fantasy.Content, 0, 2+len(resp.Message.ToolCalls))
	if resp.Message.Thinking != "" {
		content = append(content, fantasy.ReasoningContent{Text: resp.Message.Thinking})
	}
	if resp.Message.Content != "" {
		content = append(content, fantasy.TextContent{Text: resp.Message.Content})
	}
	for _, tc := range resp.Message.ToolCalls {
		content = append(content, fantasy.ToolCallContent{
			ToolCallID: cmp.Or(tc.ID, uuid.NewString()),
			ToolName:   tc.Function.Name,
			Input:      toolArguments(tc.Function.Arguments),
		})
	}

	finishReason := mapFinishReason(resp.DoneReason)
	if len(resp.Message.ToolCalls) > 0 {
		finishReason = fantasy.FinishReasonToolCalls
	}

	return &fantasy.Response{
		Content:          content,
		Usage:            toUsage(*resp),
		FinishReason:     finishReason,
		ProviderMetadata: toMetadata(*resp),
		Warnings:         warnings,
	}, nil
}

// Stream implements fantasy.LanguageModel.
func (l *languageModel) Stream(ctx context.Context, call fantasy.Call) (fantasy.StreamResponse, error) {
	req, warnings, err := l.prepareRequest(call)
	if err != nil {
		return nil, err
	}

	dec, body, err := l.chatStream(ctx, req, call.UserAgent, call.Headers)
	if err != nil {
		return nil, err
	}

	return func(yield func(fantasy.StreamPart) bool) {
		if len(warnings) > 0 {
			if !yield(fantasy.StreamPart{Type: fantasy.StreamPartTypeWarnings, Warnings: warnings}) {
				return
			}
		}

		var (
			activeText      bool
			activeReasoning bool
			sawToolCall     bool
			final           *chatResponse
		)

		endReasoning := func() bool {
			if !activeReasoning {
				return true
			}
			activeReasoning = false
			return yield(fantasy.StreamPart{Type: fantasy.StreamPartTypeReasoningEnd, ID: reasoningID})
		}

		endText := func() bool {
			if !activeText {
				return true
			}
			activeText = false
			return yield(fantasy.StreamPart{Type: fantasy.StreamPartTypeTextEnd, ID: textID})
		}

		for chunk, err := range readChunks(dec, body) {
			if err != nil {
				yield(fantasy.StreamPart{Type: fantasy.StreamPartTypeError, Error: err})
				return
			}

			if chunk.Message.Thinking != "" {
				if !activeReasoning {
					activeReasoning = true
					if !yield(fantasy.StreamPart{Type: fantasy.StreamPartTypeReasoningStart, ID: reasoningID}) {
						return
					}
				}
				if !yield(fantasy.StreamPart{Type: fantasy.StreamPartTypeReasoningDelta, ID: reasoningID, Delta: chunk.Message.Thinking}) {
					return
				}
			}

			if chunk.Message.Content != "" {
				if !endReasoning() {
					return
				}
				if !activeText {
					activeText = true
					if !yield(fantasy.StreamPart{Type: fantasy.StreamPartTypeTextStart, ID: textID}) {
						return
					}
				}
				if !yield(fantasy.StreamPart{Type: fantasy.StreamPartTypeTextDelta, ID: textID, Delta: chunk.Message.Content}) {
					return
				}
			}

			for _, tc := range chunk.Message.ToolCalls {
				if !endReasoning() || !endText() {
					return
				}
				sawToolCall = true

				id := cmp.Or(tc.ID, uuid.NewString())
				input := toolArguments(tc.Function.Arguments)

				parts := []fantasy.StreamPart{
					{Type: fantasy.StreamPartTypeToolInputStart, ID: id, ToolCallName: tc.Function.Name},
					{Type: fantasy.StreamPartTypeToolInputDelta, ID: id, Delta: input},
					{Type: fantasy.StreamPartTypeToolInputEnd, ID: id},
					{Type: fantasy.StreamPartTypeToolCall, ID: id, ToolCallName: tc.Function.Name, ToolCallInput: input},
				}
				for _, part := range parts {
					if !yield(part) {
						return
					}
				}
			}

			if chunk.Done {
				final = &chunk
			}
		}

		if !endReasoning() || !endText() {
			return
		}

		if final == nil {
			err := ctx.Err()
			if err == nil {
				err = fantasy.NewIncompleteStreamError()
			}
			yield(fantasy.StreamPart{Type: fantasy.StreamPartTypeError, Error: err})
			return
		}

		finishReason := mapFinishReason(final.DoneReason)
		if sawToolCall {
			finishReason = fantasy.FinishReasonToolCalls
		}

		yield(fantasy.StreamPart{
			Type:             fantasy.StreamPartTypeFinish,
			Usage:            toUsage(*final),
			FinishReason:     finishReason,
			ProviderMetadata: toMetadata(*final),
		})
	}, nil
}

// GenerateObject implements fantasy.LanguageModel.
func (l *languageModel) GenerateObject(ctx context.Context, call fantasy.ObjectCall) (*fantasy.ObjectResponse, error) {
	switch l.objectMode {
	case fantasy.ObjectModeText:
		return object.GenerateWithText(ctx, l, call)
	case fantasy.ObjectModeTool:
		return object.GenerateWithTool(ctx, l, call)
	default:
		return l.generateObjectWithJSONMode(ctx, call)
	}
}

// StreamObject implements fantasy.LanguageModel.
func (l *languageModel) StreamObject(ctx context.Context, call fantasy.ObjectCall) (fantasy.ObjectStreamResponse, error) {
	switch l.objectMode {
	case fantasy.ObjectModeText:
		return object.StreamWithText(ctx, l, call)
	case fantasy.ObjectModeTool:
		return object.StreamWithTool(ctx, l, call)
	default:
		return l.streamObjectWithJSONMode(ctx, call)
	}
}

func (l *languageModel) prepareObjectRequest(call fantasy.ObjectCall) (chatRequest, []fantasy.CallWarning, error) {
	req, warnings, err := l.prepareRequest(fantasy.Call{
		Prompt:           call.Prompt,
		MaxOutputTokens:  call.MaxOutputTokens,
		Temperature:      call.Temperature,
		TopP:             call.TopP,
		TopK:             call.TopK,
		PresencePenalty:  call.PresencePenalty,
		FrequencyPenalty: call.FrequencyPenalty,
		ProviderOptions:  call.ProviderOptions,
	})
	if err != nil {
		return chatRequest{}, nil, err
	}

	req.Format = schema.ToMap(call.Schema)

	return req, warnings, nil
}

func (l *languageModel) generateObjectWithJSONMode(ctx context.Context, call fantasy.ObjectCall) (*fantasy.ObjectResponse, error) {
	req, warnings, err := l.prepareObjectRequest(call)
	if err != nil {
		return nil, err
	}

	resp, err := l.chat(ctx, req, call.UserAgent, call.Headers)
	if err != nil {
		return nil, err
	}

	jsonText := resp.Message.Content

	var obj any
	if call.RepairText != nil {
		obj, err = schema.ParseAndValidateWithRepair(ctx, jsonText, call.Schema, call.RepairText)
	} else {
		obj, err = schema.ParseAndValidate(jsonText, call.Schema)
	}

	usage := toUsage(*resp)
	finishReason := mapFinishReason(resp.DoneReason)

	if err != nil {
		if nogErr, ok := err.(*fantasy.NoObjectGeneratedError); ok {
			nogErr.Usage = usage
			nogErr.FinishReason = finishReason
		}
		return nil, err
	}

	return &fantasy.ObjectResponse{
		Object:           obj,
		RawText:          jsonText,
		Usage:            usage,
		FinishReason:     finishReason,
		Warnings:         warnings,
		ProviderMetadata: toMetadata(*resp),
	}, nil
}

func (l *languageModel) streamObjectWithJSONMode(ctx context.Context, call fantasy.ObjectCall) (fantasy.ObjectStreamResponse, error) {
	req, warnings, err := l.prepareObjectRequest(call)
	if err != nil {
		return nil, err
	}

	dec, body, err := l.chatStream(ctx, req, call.UserAgent, call.Headers)
	if err != nil {
		return nil, err
	}

	return func(yield func(fantasy.ObjectStreamPart) bool) {
		if len(warnings) > 0 {
			if !yield(fantasy.ObjectStreamPart{Type: fantasy.ObjectStreamPartTypeObject, Warnings: warnings}) {
				return
			}
		}

		var accumulated string
		var last any
		var final *chatResponse

		emit := func(obj any) bool {
			if reflect.DeepEqual(obj, last) {
				return true
			}
			last = obj
			return yield(fantasy.ObjectStreamPart{Type: fantasy.ObjectStreamPartTypeObject, Object: obj})
		}

		for chunk, err := range readChunks(dec, body) {
			if err != nil {
				yield(fantasy.ObjectStreamPart{Type: fantasy.ObjectStreamPartTypeError, Error: err})
				return
			}

			if chunk.Done {
				final = &chunk
			}

			if chunk.Message.Content == "" {
				continue
			}
			accumulated += chunk.Message.Content

			obj, state, parseErr := schema.ParsePartialJSON(accumulated)

			if (state == schema.ParseStateSuccessful || state == schema.ParseStateRepaired) && schema.ValidateAgainstSchema(obj, call.Schema) == nil {
				if !emit(obj) {
					return
				}
			}

			if state == schema.ParseStateFailed && call.RepairText != nil {
				repaired, repairErr := call.RepairText(ctx, accumulated, parseErr)
				if repairErr != nil {
					continue
				}
				obj, state, _ := schema.ParsePartialJSON(repaired)
				if (state == schema.ParseStateSuccessful || state == schema.ParseStateRepaired) && schema.ValidateAgainstSchema(obj, call.Schema) == nil {
					if !emit(obj) {
						return
					}
				}
			}
		}

		if final == nil {
			err := ctx.Err()
			if err == nil {
				err = fantasy.NewIncompleteStreamError()
			}
			yield(fantasy.ObjectStreamPart{Type: fantasy.ObjectStreamPartTypeError, Error: err})
			return
		}

		usage := toUsage(*final)
		finishReason := mapFinishReason(final.DoneReason)

		if last == nil {
			yield(fantasy.ObjectStreamPart{
				Type: fantasy.ObjectStreamPartTypeError,
				Error: &fantasy.NoObjectGeneratedError{
					RawText:      accumulated,
					ParseError:   fmt.Errorf("no valid object generated in stream"),
					Usage:        usage,
					FinishReason: finishReason,
				},
			})
			return
		}

		yield(fantasy.ObjectStreamPart{
			Type:             fantasy.ObjectStreamPartTypeFinish,
			Usage:            usage,
			FinishReason:     finishReason,
			ProviderMetadata: toMetadata(*final),
		})
	}, nil
}

func readChunks(dec *json.Decoder, body io.Closer) iter.Seq2[chatResponse, error] {
	return func(yield func(chatResponse, error) bool) {
		defer func() { _ = body.Close() }()

		for {
			var chunk chatResponse
			if err := dec.Decode(&chunk); err != nil {
				if !errors.Is(err, io.EOF) {
					yield(chatResponse{}, toProviderErr(err))
				}
				return
			}
			if chunk.Error != "" {
				yield(chatResponse{}, newStreamErr(chunk.Error))
				return
			}
			if !yield(chunk, nil) {
				return
			}
		}
	}
}
