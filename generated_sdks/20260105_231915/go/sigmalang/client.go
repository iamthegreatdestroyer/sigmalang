// Package sigmalang provides a Go SDK for SigmaLang semantic compression
package sigmalang

import (
    "context"
    "encoding/json"
    "fmt"
    "github.com/go-resty/resty/v2"
)

// Client represents a SigmaLang API client
type Client struct {
    client *resty.Client
}

// CompressOptions represents compression options
type CompressOptions struct {
    Level             string `json:"level,omitempty"`
    PreserveFormatting bool   `json:"preserveFormatting,omitempty"`
}

// AnalysisResult represents text analysis results
type AnalysisResult struct {
    CompressionRatio  float64 `json:"compressionRatio"`
    SemanticDensity   float64 `json:"semanticDensity"`
    Complexity        float64 `json:"complexity"`
}

// NewClient creates a new SigmaLang client
func NewClient(apiKey, baseURL string) *Client {
    if baseURL == "" {
        baseURL = "https://api.sigmalang.com"
    }

    client := resty.New().
        SetBaseURL(baseURL).
        SetAuthToken(apiKey).
        SetHeader("Content-Type", "application/json")

    return &Client{client: client}
}

// Compress compresses text using SigmaLang
func (c *Client) Compress(ctx context.Context, text string, options *CompressOptions) (string, error) {
    payload := map[string]interface{}{"text": text}
    if options != nil {
        if options.Level != "" {
            payload["level"] = options.Level
        }
        payload["preserveFormatting"] = options.PreserveFormatting
    }

    resp, err := c.client.R().
        SetContext(ctx).
        SetBody(payload).
        Post("/compress")

    if err != nil {
        return "", fmt.Errorf("compression request failed: %w", err)
    }

    if resp.StatusCode() != 200 {
        return "", fmt.Errorf("API error: %s", resp.String())
    }

    var result struct {
        Compressed string `json:"compressed"`
    }

    if err := json.Unmarshal(resp.Body(), &result); err != nil {
        return "", fmt.Errorf("failed to parse response: %w", err)
    }

    return result.Compressed, nil
}

// Decompress decompresses SigmaLang compressed data
func (c *Client) Decompress(ctx context.Context, compressed string) (string, error) {
    payload := map[string]string{"compressed": compressed}

    resp, err := c.client.R().
        SetContext(ctx).
        SetBody(payload).
        Post("/decompress")

    if err != nil {
        return "", fmt.Errorf("decompression request failed: %w", err)
    }

    if resp.StatusCode() != 200 {
        return "", fmt.Errorf("API error: %s", resp.String())
    }

    var result struct {
        Text string `json:"text"`
    }

    if err := json.Unmarshal(resp.Body(), &result); err != nil {
        return "", fmt.Errorf("failed to parse response: %w", err)
    }

    return result.Text, nil
}

// Analyze analyzes text semantic structure
func (c *Client) Analyze(ctx context.Context, text string) (*AnalysisResult, error) {
    payload := map[string]string{"text": text}

    resp, err := c.client.R().
        SetContext(ctx).
        SetBody(payload).
        Post("/analyze")

    if err != nil {
        return nil, fmt.Errorf("analysis request failed: %w", err)
    }

    if resp.StatusCode() != 200 {
        return nil, fmt.Errorf("API error: %s", resp.String())
    }

    var result AnalysisResult
    if err := json.Unmarshal(resp.Body(), &result); err != nil {
        return nil, fmt.Errorf("failed to parse response: %w", err)
    }

    return &result, nil
}
