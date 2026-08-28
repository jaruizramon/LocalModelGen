package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"image"
	"image/png"
	"io"
	"log"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

const defaultOrch = "http://127.0.0.1:8080"

func orchURL() string {
	if v := os.Getenv("LMG_ORCH"); v != "" {
		return v
	}
	return defaultOrch
}
// callOrch performs an orchestrator HTTP call with a per-call timeout, and logs
// the method, URL, status, duration and a human-readable classification of any
// failure. The returned bytes are the response body.
func callOrch(method, rawURL, contentType string, body io.Reader, timeout time.Duration) ([]byte, error) {
	start := time.Now()
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, method, rawURL, body)
	if err != nil {
		log.Printf("[orch] %s %s -> request build failed: %v", method, rawURL, err)
		return nil, err
	}
	if contentType != "" {
		req.Header.Set("Content-Type", contentType)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		log.Printf("[orch] %s %s -> FAIL in %.2fs: %v (%s)", method, rawURL, time.Since(start).Seconds(), err, classifyNet(err))
		return nil, err
	}
	defer resp.Body.Close()
	data, rerr := io.ReadAll(resp.Body)
	if rerr != nil {
		log.Printf("[orch] %s %s -> %d, BODY READ FAIL in %.2fs: %v (%s)", method, rawURL, resp.StatusCode, time.Since(start).Seconds(), rerr, classifyNet(rerr))
		return nil, fmt.Errorf("read response: %w", rerr)
	}
	log.Printf("[orch] %s %s -> %d (%s) in %.2fs", method, rawURL, resp.StatusCode, summarize(data), time.Since(start).Seconds())
	if resp.StatusCode < 200 || resp.StatusCode > 299 {
		return nil, fmt.Errorf("orchestrator returned HTTP %d: %s", resp.StatusCode, summarize(data))
	}
	return data, nil
}

// classifyNet turns raw net/http errors into an actionable one-liner so the log
// tells you WHY (orchestrator down, worker hung, connection dropped) rather than
// dumping the raw string.
func classifyNet(err error) string {
	if err == nil {
		return ""
	}
	s := err.Error()
	switch {
	case strings.Contains(s, "dial tcp"), strings.Contains(s, "connection refused"):
		return "orchestrator UNREACHABLE on " + orchURL() + " (is 3dgen running?)"
	case strings.Contains(s, "EOF"):
		return "connection dropped mid-response (worker hung or 3dgen restarted)"
	case strings.Contains(s, "reset by peer"):
		return "connection reset by peer"
	case strings.Contains(s, "deadline exceeded"):
		return "no response within timeout (3dgen not answering)"
	}
	return s
}

// summarize truncates a response body for log lines so a giant JSON blob never
// floods the log.
func summarize(b []byte) string {
	s := strings.TrimSpace(string(b))
	if len(s) > 160 {
		s = s[:160] + "…"
	}
	return s
}

type genParams struct {
	Model                   string
	Seed, SSSteps, SlatSteps int
	SSCfg, SlatCfg           float64
	Tris, TexSize            int
	SubsampleRes             int // 0 = auto (HW tier); 128..256 explicit
	SmoothIters              int // surface-smoothing passes (0 = off)
	MeshCleanup              bool // keep only the largest component + fill holes
	Smooth                   bool
	Offload                  float64
}

func orchGenerate(img *image.RGBA, p genParams) (map[string]any, error) {
	var buf bytes.Buffer
	mw := multipart.NewWriter(&buf)
	var imgBuf bytes.Buffer
	if err := png.Encode(&imgBuf, img); err != nil {
		return nil, err
	}
	pw, err := mw.CreateFormFile("image", "image.png")
	if err != nil {
		return nil, err
	}
	if _, err := pw.Write(imgBuf.Bytes()); err != nil {
		return nil, err
	}
	fields := map[string]string{
		"model":        p.Model,
		"seed":         fmt.Sprint(p.Seed),
		"ss_steps":     fmt.Sprint(p.SSSteps),
		"slat_steps":   fmt.Sprint(p.SlatSteps),
		"ss_cfg":       fmt.Sprint(p.SSCfg),
		"slat_cfg":     fmt.Sprint(p.SlatCfg),
		"target_tris":   fmt.Sprint(p.Tris),
		"texture_size":  fmt.Sprint(p.TexSize),
		"subsample_res": fmt.Sprint(p.SubsampleRes),
		"smooth_iters":  fmt.Sprint(p.SmoothIters),
		"mesh_cleanup":  fmt.Sprint(p.MeshCleanup),
		"smooth_mesh":   fmt.Sprint(p.Smooth),
		"offload_ratio": fmt.Sprint(p.Offload),
	}
	for k, v := range fields {
		if err := mw.WriteField(k, v); err != nil {
			return nil, err
		}
	}
	if err := mw.Close(); err != nil {
		return nil, err
	}
	body, err := callOrch("POST", orchURL()+"/api/generate", mw.FormDataContentType(), &buf, 15*time.Minute)
	if err != nil {
		return nil, err
	}
	var j map[string]any
	if err := json.Unmarshal(body, &j); err != nil {
		return nil, fmt.Errorf("bad json (%d bytes): %s", len(body), summarize(body))
	}
	if errText, ok := j["error"].(string); ok && errText != "" {
		return nil, fmt.Errorf("%s", errText)
	}
	return j, nil
}

func orchClear() (map[string]any, error) {
	body, err := callOrch("POST", orchURL()+"/api/clear", "application/json", bytes.NewReader(nil), 60*time.Second)
	if err != nil {
		return nil, err
	}
	var j map[string]any
	if err := json.Unmarshal(body, &j); err != nil {
		return nil, fmt.Errorf("bad json")
	}
	return j, nil
}

func orchConvert(glbPath string) (string, error) {
	var buf bytes.Buffer
	mw := multipart.NewWriter(&buf)
	f, err := os.Open(glbPath)
	if err != nil {
		return "", err
	}
	defer f.Close()
	pw, err := mw.CreateFormFile("glb_file", baseName(glbPath))
	if err != nil {
		return "", err
	}
	if _, err := io.Copy(pw, f); err != nil {
		return "", err
	}
	mw.Close()
	body, err := callOrch("POST", orchURL()+"/api/convert", mw.FormDataContentType(), &buf, 2*time.Minute)
	if err != nil {
		return "", err
	}
	var j map[string]any
	if err := json.Unmarshal(body, &j); err != nil {
		return "", fmt.Errorf("bad json")
	}
	if errText, ok := j["error"].(string); ok && errText != "" {
		return "", fmt.Errorf("%s", errText)
	}
	z, _ := j["zip"].(string)
	if z == "" {
		return "", fmt.Errorf("no zip returned")
	}
	return z, nil
}

func fetchPhase() string {
	// The sync generate blocks the worker's event loop, so /status can't serve
	// the live phase mid-job; the worker writes it to ../tmp/phase.txt instead,
	// which we read directly (independent of the busy HTTP server).
	b, err := os.ReadFile(filepath.Join("..", "tmp", "phase.txt"))
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(b))
}

func baseName(p string) string {
	for i := len(p) - 1; i >= 0; i-- {
		if p[i] == '/' || p[i] == '\\' {
			return p[i+1:]
		}
	}
	return p
}
