package ollama

import (
	"encoding/json"
	"maps"

	"charm.land/fantasy"
)

// Global type identifiers for Ollama-specific provider data.
const (
	TypeProviderOptions  = Name + ".options"
	TypeProviderMetadata = Name + ".metadata"
)

// Register Ollama provider-specific types with the global registry.
func init() {
	fantasy.RegisterProviderType(TypeProviderOptions, func(data []byte) (fantasy.ProviderOptionsData, error) {
		var v ProviderOptions
		if err := json.Unmarshal(data, &v); err != nil {
			return nil, err
		}
		return &v, nil
	})

	fantasy.RegisterProviderType(TypeProviderMetadata, func(data []byte) (fantasy.ProviderOptionsData, error) {
		var v ProviderMetadata
		if err := json.Unmarshal(data, &v); err != nil {
			return nil, err
		}
		return &v, nil
	})
}

// ProviderMetadata represents additional metadata from the Ollama provider.
type ProviderMetadata struct {
	Model              string `json:"model"`
	DoneReason         string `json:"done_reason,omitempty"`
	TotalDurationNS    int64  `json:"total_duration_ns,omitempty"`
	LoadDurationNS     int64  `json:"load_duration_ns,omitempty"`
	PromptEvalDuration int64  `json:"prompt_eval_duration_ns,omitempty"`
	EvalDurationNS     int64  `json:"eval_duration_ns,omitempty"`
}

// Options implements the ProviderOptionsData interface.
func (*ProviderMetadata) Options() {}

// MarshalJSON implements custom JSON marshaling with type info for ProviderMetadata.
func (m ProviderMetadata) MarshalJSON() ([]byte, error) {
	type plain ProviderMetadata
	return fantasy.MarshalProviderType(TypeProviderMetadata, plain(m))
}

// UnmarshalJSON implements custom JSON unmarshaling with type info for ProviderMetadata.
func (m *ProviderMetadata) UnmarshalJSON(data []byte) error {
	type plain ProviderMetadata
	var p plain
	if err := fantasy.UnmarshalProviderType(data, &p); err != nil {
		return err
	}
	*m = ProviderMetadata(p)
	return nil
}

// ProviderOptions represents additional options for the Ollama provider.
type ProviderOptions struct {
	NumCtx        *int64         `json:"num_ctx"`
	NumPredict    *int64         `json:"num_predict"`
	NumKeep       *int64         `json:"num_keep"`
	NumBatch      *int64         `json:"num_batch"`
	NumGPU        *int64         `json:"num_gpu"`
	NumThread     *int64         `json:"num_thread"`
	Seed          *int64         `json:"seed"`
	MinP          *float64       `json:"min_p"`
	TypicalP      *float64       `json:"typical_p"`
	RepeatPenalty *float64       `json:"repeat_penalty"`
	RepeatLastN   *int64         `json:"repeat_last_n"`
	Stop          []string       `json:"stop"`
	Think         *bool          `json:"think"`
	ThinkEffort   *string        `json:"think_effort"`
	KeepAlive     *string        `json:"keep_alive"`
	Extra         map[string]any `json:"extra"`
}

// Options implements the ProviderOptionsData interface.
func (*ProviderOptions) Options() {}

// MarshalJSON implements custom JSON marshaling with type info for ProviderOptions.
func (o ProviderOptions) MarshalJSON() ([]byte, error) {
	type plain ProviderOptions
	return fantasy.MarshalProviderType(TypeProviderOptions, plain(o))
}

// UnmarshalJSON implements custom JSON unmarshaling with type info for ProviderOptions.
func (o *ProviderOptions) UnmarshalJSON(data []byte) error {
	type plain ProviderOptions
	var p plain
	if err := fantasy.UnmarshalProviderType(data, &p); err != nil {
		return err
	}
	*o = ProviderOptions(p)
	return nil
}

// NewProviderOptions creates new provider options for Ollama.
func NewProviderOptions(opts *ProviderOptions) fantasy.ProviderOptions {
	return fantasy.ProviderOptions{
		Name: opts,
	}
}

// ParseOptions parses provider options from a map.
func ParseOptions(data map[string]any) (*ProviderOptions, error) {
	var options ProviderOptions
	if err := fantasy.ParseOptions(data, &options); err != nil {
		return nil, err
	}
	return &options, nil
}

func (o *ProviderOptions) apply(req *chatRequest) {
	if req.Options == nil {
		req.Options = make(map[string]any)
	}

	set := func(key string, value any) {
		req.Options[key] = value
	}

	if o.NumCtx != nil {
		set("num_ctx", *o.NumCtx)
	}
	if o.NumPredict != nil {
		set("num_predict", *o.NumPredict)
	}
	if o.NumKeep != nil {
		set("num_keep", *o.NumKeep)
	}
	if o.NumBatch != nil {
		set("num_batch", *o.NumBatch)
	}
	if o.NumGPU != nil {
		set("num_gpu", *o.NumGPU)
	}
	if o.NumThread != nil {
		set("num_thread", *o.NumThread)
	}
	if o.Seed != nil {
		set("seed", *o.Seed)
	}
	if o.MinP != nil {
		set("min_p", *o.MinP)
	}
	if o.TypicalP != nil {
		set("typical_p", *o.TypicalP)
	}
	if o.RepeatPenalty != nil {
		set("repeat_penalty", *o.RepeatPenalty)
	}
	if o.RepeatLastN != nil {
		set("repeat_last_n", *o.RepeatLastN)
	}
	if o.Stop != nil {
		set("stop", o.Stop)
	}
	maps.Copy(req.Options, o.Extra)

	switch {
	case o.ThinkEffort != nil:
		req.Think = *o.ThinkEffort
	case o.Think != nil:
		req.Think = *o.Think
	}
	if o.KeepAlive != nil {
		req.KeepAlive = *o.KeepAlive
	}
}
