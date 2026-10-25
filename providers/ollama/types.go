package ollama

import (
	"encoding/json"
	"time"
)

type chatRequest struct {
	Model     string         `json:"model"`
	Messages  []message      `json:"messages"`
	Tools     []tool         `json:"tools,omitempty"`
	Format    any            `json:"format,omitempty"`
	Options   map[string]any `json:"options,omitempty"`
	Stream    bool           `json:"stream"`
	Think     any            `json:"think,omitempty"`
	KeepAlive any            `json:"keep_alive,omitempty"`
}

type message struct {
	Role       string     `json:"role"`
	Content    string     `json:"content"`
	Thinking   string     `json:"thinking,omitempty"`
	Images     [][]byte   `json:"images,omitempty"`
	ToolCalls  []toolCall `json:"tool_calls,omitempty"`
	ToolName   string     `json:"tool_name,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
}

type toolCall struct {
	ID       string           `json:"id,omitempty"`
	Function toolCallFunction `json:"function"`
}

type toolCallFunction struct {
	Index     int             `json:"index"`
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}

type tool struct {
	Type     string       `json:"type"`
	Function toolFunction `json:"function"`
}

type toolFunction struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Parameters  any    `json:"parameters"`
}

type chatResponse struct {
	Model              string        `json:"model"`
	Message            message       `json:"message"`
	Done               bool          `json:"done"`
	DoneReason         string        `json:"done_reason"`
	TotalDuration      time.Duration `json:"total_duration"`
	LoadDuration       time.Duration `json:"load_duration"`
	PromptEvalCount    int           `json:"prompt_eval_count"`
	PromptEvalDuration time.Duration `json:"prompt_eval_duration"`
	EvalCount          int           `json:"eval_count"`
	EvalDuration       time.Duration `json:"eval_duration"`
	Error              string        `json:"error"`
}

type errorResponse struct {
	Error string `json:"error"`
}
