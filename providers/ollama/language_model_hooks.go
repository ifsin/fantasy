package ollama

import (
	"encoding/json"
	"fmt"
	"strings"

	"charm.land/fantasy"
)

func (l *languageModel) prepareRequest(call fantasy.Call) (chatRequest, []fantasy.CallWarning, error) {
	messages, warnings := toPrompt(call.Prompt)

	req := chatRequest{
		Model:    l.modelID,
		Messages: messages,
		Options:  make(map[string]any),
	}

	if call.MaxOutputTokens != nil {
		req.Options["num_predict"] = *call.MaxOutputTokens
	}
	if call.Temperature != nil {
		req.Options["temperature"] = *call.Temperature
	}
	if call.TopP != nil {
		req.Options["top_p"] = *call.TopP
	}
	if call.TopK != nil {
		req.Options["top_k"] = *call.TopK
	}
	if call.PresencePenalty != nil {
		req.Options["presence_penalty"] = *call.PresencePenalty
	}
	if call.FrequencyPenalty != nil {
		req.Options["frequency_penalty"] = *call.FrequencyPenalty
	}

	l.defaults.apply(&req)

	callOptions, err := l.callOptions(call.ProviderOptions)
	if err != nil {
		return chatRequest{}, nil, err
	}
	if callOptions != nil {
		callOptions.apply(&req)
	}

	tools, toolWarnings := toTools(call.Tools, call.ToolChoice)
	req.Tools = tools
	warnings = append(warnings, toolWarnings...)

	return req, warnings, nil
}

func (l *languageModel) callOptions(options fantasy.ProviderOptions) (*ProviderOptions, error) {
	if options == nil {
		return nil, nil
	}

	value, ok := options[l.provider]
	if !ok {
		value, ok = options[Name]
	}
	if !ok {
		return nil, nil
	}

	parsed, ok := value.(*ProviderOptions)
	if !ok {
		return nil, &fantasy.Error{Title: "invalid argument", Message: "ollama provider options should be *ollama.ProviderOptions"}
	}

	return parsed, nil
}

func mapFinishReason(doneReason string) fantasy.FinishReason {
	switch doneReason {
	case "", "stop":
		return fantasy.FinishReasonStop
	case "length":
		return fantasy.FinishReasonLength
	default:
		return fantasy.FinishReasonOther
	}
}

func toUsage(resp chatResponse) fantasy.Usage {
	return fantasy.Usage{
		InputTokens:  int64(resp.PromptEvalCount),
		OutputTokens: int64(resp.EvalCount),
		TotalTokens:  int64(resp.PromptEvalCount + resp.EvalCount),
	}
}

func toMetadata(resp chatResponse) fantasy.ProviderMetadata {
	return fantasy.ProviderMetadata{
		Name: &ProviderMetadata{
			Model:              resp.Model,
			DoneReason:         resp.DoneReason,
			TotalDurationNS:    int64(resp.TotalDuration),
			LoadDurationNS:     int64(resp.LoadDuration),
			PromptEvalDuration: int64(resp.PromptEvalDuration),
			EvalDurationNS:     int64(resp.EvalDuration),
		},
	}
}

func toolArguments(raw json.RawMessage) string {
	trimmed := strings.TrimSpace(string(raw))
	if trimmed == "" || trimmed == "null" || !json.Valid([]byte(trimmed)) {
		return "{}"
	}
	return trimmed
}

func toTools(tools []fantasy.Tool, choice *fantasy.ToolChoice) ([]tool, []fantasy.CallWarning) {
	var warnings []fantasy.CallWarning

	if choice != nil && *choice == fantasy.ToolChoiceNone {
		return nil, nil
	}

	var out []tool
	for _, t := range tools {
		ft, ok := t.(fantasy.FunctionTool)
		if !ok {
			warnings = append(warnings, fantasy.CallWarning{
				Type:    fantasy.CallWarningTypeUnsupportedTool,
				Tool:    t,
				Message: "tool is not supported",
			})
			continue
		}

		if choice != nil && *choice != fantasy.ToolChoiceAuto && *choice != fantasy.ToolChoiceRequired && string(*choice) != ft.Name {
			continue
		}

		out = append(out, tool{
			Type: "function",
			Function: toolFunction{
				Name:        ft.Name,
				Description: ft.Description,
				Parameters:  ft.InputSchema,
			},
		})
	}

	if choice != nil && *choice == fantasy.ToolChoiceRequired {
		warnings = append(warnings, fantasy.CallWarning{
			Type:    fantasy.CallWarningTypeUnsupportedSetting,
			Setting: "tool_choice",
			Details: "ollama cannot require a tool call",
		})
	}

	return out, warnings
}

func toPrompt(prompt fantasy.Prompt) ([]message, []fantasy.CallWarning) {
	var messages []message
	var warnings []fantasy.CallWarning
	toolNames := make(map[string]string)

	warn := func(text string) {
		warnings = append(warnings, fantasy.CallWarning{
			Type:    fantasy.CallWarningTypeOther,
			Message: text,
		})
	}

	for _, msg := range prompt {
		switch msg.Role {
		case fantasy.MessageRoleSystem:
			for _, c := range msg.Content {
				if c.GetType() != fantasy.ContentTypeText {
					continue
				}
				textPart, ok := fantasy.AsMessagePart[fantasy.TextPart](c)
				if !ok {
					warn("system message text part does not have the right type")
					continue
				}
				messages = append(messages, message{Role: "system", Content: textPart.Text})
			}

		case fantasy.MessageRoleUser:
			var texts []string
			var images [][]byte

			for _, c := range msg.Content {
				switch c.GetType() {
				case fantasy.ContentTypeText:
					textPart, ok := fantasy.AsMessagePart[fantasy.TextPart](c)
					if !ok {
						warn("user message text part does not have the right type")
						continue
					}
					texts = append(texts, textPart.Text)

				case fantasy.ContentTypeFile:
					filePart, ok := fantasy.AsMessagePart[fantasy.FilePart](c)
					if !ok {
						warn("user message file part does not have the right type")
						continue
					}
					if !strings.HasPrefix(filePart.MediaType, "image/") {
						warn(fmt.Sprintf("file part media type %s not supported", filePart.MediaType))
						continue
					}
					images = append(images, filePart.Data)
				}
			}

			if len(texts) > 0 || len(images) > 0 {
				messages = append(messages, message{
					Role:    "user",
					Content: strings.Join(texts, "\n"),
					Images:  images,
				})
			}

		case fantasy.MessageRoleAssistant:
			var texts, thoughts []string
			var calls []toolCall

			for _, c := range msg.Content {
				switch c.GetType() {
				case fantasy.ContentTypeText:
					textPart, ok := fantasy.AsMessagePart[fantasy.TextPart](c)
					if !ok {
						warn("assistant message text part does not have the right type")
						continue
					}
					texts = append(texts, textPart.Text)

				case fantasy.ContentTypeReasoning:
					reasoningPart, ok := fantasy.AsMessagePart[fantasy.ReasoningPart](c)
					if !ok {
						warn("assistant message reasoning part does not have the right type")
						continue
					}
					thoughts = append(thoughts, reasoningPart.Text)

				case fantasy.ContentTypeToolCall:
					toolCallPart, ok := fantasy.AsMessagePart[fantasy.ToolCallPart](c)
					if !ok {
						warn("assistant message tool part does not have the right type")
						continue
					}
					toolNames[toolCallPart.ToolCallID] = toolCallPart.ToolName
					calls = append(calls, toolCall{
						ID: toolCallPart.ToolCallID,
						Function: toolCallFunction{
							Index:     len(calls),
							Name:      toolCallPart.ToolName,
							Arguments: json.RawMessage(toolArguments(json.RawMessage(toolCallPart.Input))),
						},
					})
				}
			}

			content := strings.Join(texts, "\n")
			if content == "" && len(calls) == 0 {
				continue
			}

			messages = append(messages, message{
				Role:      "assistant",
				Content:   content,
				Thinking:  strings.Join(thoughts, "\n"),
				ToolCalls: calls,
			})

		case fantasy.MessageRoleTool:
			for _, c := range msg.Content {
				if c.GetType() != fantasy.ContentTypeToolResult {
					warn("tool message can only have tool result content")
					continue
				}

				toolResultPart, ok := fantasy.AsMessagePart[fantasy.ToolResultPart](c)
				if !ok {
					warn("tool message result part does not have the right type")
					continue
				}

				var content string

				switch toolResultPart.Output.GetType() {
				case fantasy.ToolResultContentTypeText:
					if output, ok := fantasy.AsToolResultOutputType[fantasy.ToolResultOutputContentText](toolResultPart.Output); ok {
						content = output.Text
					}

				case fantasy.ToolResultContentTypeError:
					if output, ok := fantasy.AsToolResultOutputType[fantasy.ToolResultOutputContentError](toolResultPart.Output); ok && output.Error != nil {
						content = output.Error.Error()
					}

				case fantasy.ToolResultContentTypeMedia:
					warn("tool result media is not supported")
					if output, ok := fantasy.AsToolResultOutputType[fantasy.ToolResultOutputContentMedia](toolResultPart.Output); ok {
						content = output.Text
					}
				}

				messages = append(messages, message{
					Role:       "tool",
					Content:    content,
					ToolName:   toolNames[toolResultPart.ToolCallID],
					ToolCallID: toolResultPart.ToolCallID,
				})
			}
		}
	}

	return messages, warnings
}
