// 3DModelGen orchestrator.
//
// Owns the TRELLIS inference worker (a Python/FastAPI subprocess), serializes
// GPU jobs, watches VRAM, and serves a web UI + REST API. The heavy math stays
// in the torch worker; Go handles scheduling, lifecycle, and serving.
//
// Process-lifecycle contract: this program must never leave the Python worker
// (or anything the worker spawned, e.g. headless Blender) running after it
// exits, and must never leave a zombie behind while it runs. That is enforced
// by the explicit boolean state machine in workerProc below plus a single
// shutdown path that every exit route funnels through.
package main

import (
	"bytes"
	"context"
	_ "embed"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
)

//go:embed index.html
var indexHTML []byte

var (
	workerURL    = envOr("WORKER_URL", "http://127.0.0.1:8001")
	workerPython = envOr("WORKER_PYTHON", "/home/pipo/miniconda3/envs/trellis/bin/python")
	workerDir    = envOr("WORKER_DIR", "../worker")
	preload      = envOr("LD_PRELOAD", "/home/pipo/miniconda3/envs/trellis/lib/libittnotify.so.0")
	listen       = envOr("LISTEN", ":8080")
	resultsDir   = envOr("RESULTS_DIR", "../tmp")
	staticDir    = envOr("STATIC_DIR", "../static")

	mu  sync.Mutex
	job int64 // 0 = idle, else a running job id

	// Dedicated clients: the default client has NO timeout, so a wedged worker
	// used to pin the job flag forever (permanent 429 until restart) and leak a
	// goroutine per poll. Every outbound call now has a finite deadline.
	probeClient = &http.Client{Timeout: 3 * time.Second}  // /health, /status
	ctlClient   = &http.Client{Timeout: 2 * time.Minute}  // /clear
	jobClient   = &http.Client{Timeout: 30 * time.Minute} // /generate proxy
)

func envOr(k, d string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return d
}

// ---------------------------------------------------------------------------
// Worker process lifecycle
//
// Every transition is an explicit boolean so the shutdown path can distinguish
// "never started" from "running" from "already reaped", and so cleanup is
// idempotent no matter how many routes call it (signal, ListenAndServe error,
// readiness timeout, normal exit):
//
//	started -> Start() returned successfully; the PID exists
//	ready   -> /health answered 200 at least once
//	signaled-> WE sent it a signal (as opposed to it dying on its own)
//	exited  -> Wait() returned; the PID is reaped, no zombie. Terminal.
//
// The child is put in its OWN process group (Setpgid) so that killing -pgid
// takes down the worker AND everything it spawned. Without that, a headless
// Blender started by the worker survives, gets re-parented to init, and keeps
// its GPU context.
type workerProc struct {
	mu           sync.Mutex
	cmd          *exec.Cmd
	logFile      *os.File
	pgid         int
	started      bool
	ready        bool
	signaled     bool
	exited       bool
	groupCleaned bool // the final group sweep has run; makes stopWorker idempotent
	waitErr      error
	done         chan struct{} // closed by the reaper once exited is set
}

var wp = workerProc{done: make(chan struct{})}

// shutdownOnce guards the whole teardown sequence; several routes can race to
// trigger it (SIGTERM while ListenAndServe is also failing).
var shutdownOnce sync.Once

func (w *workerProc) snapshot() map[string]any {
	w.mu.Lock()
	defer w.mu.Unlock()
	m := map[string]any{
		"started": w.started, "ready": w.ready,
		"signaled": w.signaled, "exited": w.exited,
		"group_cleaned": w.groupCleaned,
	}
	if w.cmd != nil && w.cmd.Process != nil {
		m["pid"] = w.cmd.Process.Pid
	}
	if w.waitErr != nil {
		m["wait_err"] = w.waitErr.Error()
	}
	return m
}

func startWorker(dir string) error {
	logPath := filepath.Join(resultsDir, "worker.log")
	if !filepath.IsAbs(logPath) {
		if abs, err := filepath.Abs(logPath); err == nil {
			logPath = abs
		}
	}
	if err := os.MkdirAll(filepath.Dir(logPath), 0o755); err != nil {
		return fmt.Errorf("worker log dir: %w", err)
	}
	// Error was previously dropped, which handed the child a nil *os.File and
	// silently discarded every worker log line.
	lf, err := os.OpenFile(logPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return fmt.Errorf("worker log %s: %w", logPath, err)
	}

	cmd := exec.Command(workerPython, "worker.py")
	cmd.Dir = dir
	cmd.Env = append(os.Environ(),
		"PYTHONUNBUFFERED=1",
		"LD_PRELOAD="+preload,
		// Arms the worker's parent-death watchdog. Covers the one case the
		// process-group kill below cannot: this orchestrator being SIGKILLed
		// or OOM-killed, which would otherwise leave the worker holding the
		// model, ~1.5GB of VRAM and port 8001 after being reparented to init.
		"LMG_SUPERVISED=1",
	)
	cmd.Stdout = lf
	cmd.Stderr = lf
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}

	if err := cmd.Start(); err != nil {
		lf.Close()
		return fmt.Errorf("failed to start worker: %w", err)
	}

	w := &wp
	w.mu.Lock()
	w.cmd = cmd
	w.logFile = lf
	w.started = true
	w.pgid = cmd.Process.Pid // Setpgid makes the child its own group leader
	pid := cmd.Process.Pid
	w.mu.Unlock()

	// The reaper: without this Wait() the exited worker sits as a zombie for as
	// long as the orchestrator runs, and its death is never noticed.
	go func() {
		err := cmd.Wait()
		w.mu.Lock()
		w.exited = true
		w.ready = false
		w.waitErr = err
		signaled := w.signaled
		if w.logFile != nil {
			w.logFile.Close()
			w.logFile = nil
		}
		w.mu.Unlock()
		close(w.done)
		if signaled {
			log.Printf("worker (pid %d) exited after signal: %v", pid, err)
		} else {
			log.Printf("WARNING: worker (pid %d) exited on its own: %v", pid, err)
		}
	}()

	log.Printf("worker started (pid %d, pgid %d); waiting for readiness...", pid, pid)
	for range 120 { // up to ~120s
		w.mu.Lock()
		gone := w.exited
		w.mu.Unlock()
		if gone {
			return fmt.Errorf("worker died during startup; see %s", logPath)
		}
		if workerHealthy() {
			w.mu.Lock()
			w.ready = true
			w.mu.Unlock()
			log.Printf("worker ready: %s", workerURL)
			return nil
		}
		time.Sleep(time.Second)
	}
	return fmt.Errorf("worker did not become ready within 120s")
}

// stopWorker kills the worker's whole process group and waits for the reaper to
// confirm it is gone. Idempotent via groupCleaned.
//
// The final group sweep runs even when the worker already exited on its own:
// its children are still in that process group, and the group outlives the
// leader. Returning early on `exited` (as an earlier version did) left an
// orphaned Blender running -- caught by the lifecycle test.
func stopWorker(grace time.Duration) {
	w := &wp
	w.mu.Lock()
	if !w.started || w.groupCleaned {
		w.mu.Unlock()
		return
	}
	w.groupCleaned = true
	alive := !w.exited
	if alive {
		w.signaled = true
	}
	pgid := w.pgid
	pid := w.cmd.Process.Pid
	w.mu.Unlock()

	if alive {
		// Negative pid = the whole process group, so anything the worker
		// spawned (headless Blender) goes down with it.
		log.Printf("stopping worker (pid %d, pgid %d): SIGTERM", pid, pgid)
		if err := syscall.Kill(-pgid, syscall.SIGTERM); err != nil {
			// Group already gone; fall back to the pid.
			_ = w.cmd.Process.Signal(syscall.SIGTERM)
		}
		select {
		case <-w.done:
			log.Printf("worker stopped cleanly")
		case <-time.After(grace):
			log.Printf("worker did not exit within %s: SIGKILL", grace)
			if err := syscall.Kill(-pgid, syscall.SIGKILL); err != nil {
				_ = w.cmd.Process.Kill()
			}
			select {
			case <-w.done:
				log.Printf("worker killed")
			case <-time.After(10 * time.Second):
				log.Printf("ERROR: worker pid %d unreaped after SIGKILL", pid)
			}
		}
	} else {
		log.Printf("worker (pid %d) already reaped; sweeping its process group", pid)
	}

	// Final sweep: whether the worker exited on its own or on our signal, any
	// straggler it spawned is still a member of pgid. ESRCH means the group is
	// empty, which is the outcome we want.
	if err := syscall.Kill(-pgid, syscall.SIGKILL); err == nil {
		log.Printf("process group %d: swept leftover children", pgid)
	} else if err != syscall.ESRCH {
		log.Printf("process group %d sweep: %v", pgid, err)
	}
}

// die is the only exit route once the worker exists: log.Fatal would skip
// teardown and orphan the worker holding VRAM and port 8001.
func die(format string, a ...any) {
	log.Printf("fatal: "+format, a...)
	shutdownOnce.Do(func() { stopWorker(10 * time.Second) })
	os.Exit(1)
}

// cors lets the Gradio frontend (a different origin) use the WebGL viewer:
// it fetches /static/* modules and /download/* GLBs cross-origin.
func cors(h http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "*")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		h.ServeHTTP(w, r)
	})
}

// requirePOST rejects other methods before any shared state is touched. A GET
// (browser prefetch, crawler, curl) used to take the GPU job lock.
func requirePOST(h http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			w.Header().Set("Allow", "POST")
			http.Error(w, `{"error":"method not allowed"}`, http.StatusMethodNotAllowed)
			return
		}
		h(w, r)
	}
}

func main() {
	absWorker, err := filepath.Abs(workerDir)
	if err != nil {
		log.Fatalf("bad WORKER_DIR %q: %v", workerDir, err)
	}
	if err := startWorker(absWorker); err != nil {
		die("%v", err)
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/", uiHandler)
	mux.HandleFunc("/api/status", statusHandler)
	mux.HandleFunc("/api/generate", requirePOST(generateHandler))
	mux.HandleFunc("/api/clear", requirePOST(clearHandler))
	mux.HandleFunc("/api/convert", requirePOST(convertHandler))
	mux.HandleFunc("/api/models", modelsHandler)
	mux.HandleFunc("/api/results", resultsHandler)
	mux.Handle("/download/", cors(http.HandlerFunc(downloadHandler)))
	mux.Handle("/static/", cors(http.StripPrefix("/static/",
		http.FileServer(http.Dir(staticDir)))))

	srv := &http.Server{
		Addr:    listen,
		Handler: mux,
		// WriteTimeout stays 0: a generation legitimately holds the response
		// open for ~30s and the job proxy has its own deadline.
		ReadHeaderTimeout: 10 * time.Second,
		IdleTimeout:       60 * time.Second,
	}

	watchdogCtx, stopWatchdog := context.WithCancel(context.Background())
	go vramWatchdog(watchdogCtx)

	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		s := <-sig
		log.Printf("received %v: shutting down", s)
		shutdownOnce.Do(func() {
			stopWatchdog()
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()
			if err := srv.Shutdown(ctx); err != nil {
				log.Printf("http shutdown: %v", err)
			}
			stopWorker(10 * time.Second)
		})
		os.Exit(0)
	}()

	log.Printf("3DModelGen orchestrator listening on %s", listen)
	if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		stopWatchdog()
		die("listen %s: %v", listen, err)
	}
	stopWatchdog()
	shutdownOnce.Do(func() { stopWorker(10 * time.Second) })
}

func workerHealthy() bool {
	resp, err := probeClient.Get(workerURL + "/health")
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	io.Copy(io.Discard, resp.Body) // drain so the connection can be reused
	return resp.StatusCode == 200
}

func httpPostJSON(url string, body io.Reader, contentType string) (*http.Response, error) {
	req, err := http.NewRequest("POST", url, body)
	if err != nil {
		return nil, err
	}
	if contentType != "" {
		req.Header.Set("Content-Type", contentType)
	}
	return ctlClient.Do(req)
}

// uiHandler serves the embedded page.
func uiHandler(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Write(indexHTML)
}

// gpuVRAM forks nvidia-smi under a deadline: an unbounded exec.Command would
// leave a stuck nvidia-smi process behind on every call.
func gpuVRAM() int {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, "nvidia-smi",
		"--query-gpu=memory.used", "--format=csv,noheader,nounits").Output()
	if err != nil {
		return 0
	}
	s := strings.TrimSpace(string(out))
	if i := strings.Index(s, "\n"); i >= 0 {
		s = strings.TrimSpace(s[:i])
	}
	n, _ := strconv.Atoi(s)
	return n
}

// workerRSS reports the worker process's DRAM RSS (MB) via /status.
func workerRSS() int {
	resp, err := probeClient.Get(workerURL + "/status")
	if err != nil {
		return 0
	}
	defer resp.Body.Close()
	var s struct {
		DRAMRSS int `json:"dram_rss_mb"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&s); err != nil {
		return 0
	}
	return s.DRAMRSS
}

// vramWatchdog logs GPU VRAM + worker DRAM RSS every 30s with the delta vs the
// previous sample. A persistent positive delta across samples / jobs is the
// accumulation signature (something not being freed / offloaded).
func vramWatchdog(ctx context.Context) {
	tick := time.NewTicker(30 * time.Second)
	defer tick.Stop()
	last := -1 // -1 so the first sample reports no bogus delta
	for {
		select {
		case <-ctx.Done():
			return
		case <-tick.C:
		}
		mb := gpuVRAM()
		delta := ""
		if last >= 0 {
			delta = fmt.Sprintf(" (%+d vs last)", mb-last)
		}
		last = mb
		line := fmt.Sprintf("watchdog: gpu=%dMB%s worker_rss=%dMB",
			mb, delta, workerRSS())
		if mb > 7000 {
			log.Printf("WARNING: %s (near limit)", line)
		} else {
			log.Printf("%s", line)
		}
	}
}

func modelsHandler(w http.ResponseWriter, r *http.Request) {
	// Registry of selectable models. Add backends here as they become available.
	models := []string{"trellis-image"}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(models)
}

func statusHandler(w http.ResponseWriter, r *http.Request) {
	ws := map[string]any{"orchestrator": "ok", "gpu_mb": gpuVRAM()}
	mu.Lock()
	ws["job_running"] = job != 0
	mu.Unlock()
	ws["worker_proc"] = wp.snapshot()
	resp, err := probeClient.Get(workerURL + "/status")
	if err != nil {
		ws["worker"] = "down"
	} else {
		defer resp.Body.Close()
		var s map[string]any
		if err := json.NewDecoder(resp.Body).Decode(&s); err != nil {
			log.Printf("worker /status decode: %v", err)
			ws["worker"] = "down"
		} else {
			ws["worker"] = s
		}
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(ws)
}

// generateHandler proxies a multipart generation to the worker. GPU jobs are
// serialized here so only one runs at a time.
func generateHandler(w http.ResponseWriter, r *http.Request) {
	mu.Lock()
	if job != 0 {
		mu.Unlock()
		http.Error(w, `{"error":"a job is already running"}`, http.StatusTooManyRequests)
		return
	}
	job = time.Now().UnixNano()
	mu.Unlock()
	gpuBefore := gpuVRAM()
	log.Printf("job start: gpu=%dMB worker_rss=%dMB", gpuBefore, workerRSS())
	defer func() {
		mu.Lock()
		job = 0
		mu.Unlock()
		gpuAfter := gpuVRAM()
		log.Printf("job end: gpu=%dMB (%+d vs start) worker_rss=%dMB",
			gpuAfter, gpuAfter-gpuBefore, workerRSS())
	}()

	// Bound to the client request: a disconnect cancels the proxy instead of
	// pinning the job flag, and jobClient's deadline caps a wedged worker.
	req, err := http.NewRequestWithContext(r.Context(), http.MethodPost,
		workerURL+"/generate", r.Body)
	if err != nil {
		http.Error(w, fmt.Sprintf(`{"error":%q}`, err.Error()), http.StatusInternalServerError)
		return
	}
	req.Header.Set("Content-Type", r.Header.Get("Content-Type"))
	req.ContentLength = r.ContentLength

	resp, err := jobClient.Do(req)
	if err != nil {
		http.Error(w, fmt.Sprintf(`{"error":%q}`, err.Error()), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()
	w.Header().Set("Content-Type", resp.Header.Get("Content-Type"))
	w.WriteHeader(resp.StatusCode)
	io.Copy(w, resp.Body)
}

func clearHandler(w http.ResponseWriter, r *http.Request) {
	gpuBefore := gpuVRAM()
	log.Printf("clear: gpu=%dMB before (worker_rss=%dMB)", gpuBefore, workerRSS())
	resp, err := httpPostJSON(workerURL+"/clear", bytes.NewReader(nil), "application/json")
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()
	w.Header().Set("Content-Type", resp.Header.Get("Content-Type"))
	w.WriteHeader(resp.StatusCode)
	io.Copy(w, resp.Body)
	gpuAfter := gpuVRAM()
	log.Printf("clear: gpu=%dMB after (%+d)", gpuAfter, gpuAfter-gpuBefore)
}

// convertHandler proxies a GLB upload to the worker /convert (glb -> obj zip).
func convertHandler(w http.ResponseWriter, r *http.Request) {
	req, err := http.NewRequestWithContext(r.Context(), http.MethodPost,
		workerURL+"/convert", r.Body)
	if err != nil {
		http.Error(w, fmt.Sprintf(`{"error":%q}`, err.Error()), http.StatusInternalServerError)
		return
	}
	req.Header.Set("Content-Type", r.Header.Get("Content-Type"))
	req.ContentLength = r.ContentLength
	resp, err := jobClient.Do(req)
	if err != nil {
		http.Error(w, fmt.Sprintf(`{"error":%q}`, err.Error()), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()
	w.Header().Set("Content-Type", resp.Header.Get("Content-Type"))
	w.WriteHeader(resp.StatusCode)
	io.Copy(w, resp.Body)
}

// resultExts is the single whitelist shared by the listing and the download
// route. worker.log lives in resultsDir too, and /download/ is served with a
// wildcard CORS origin, so an unfiltered download route exposed it to any page.
var resultExts = []string{".glb", ".ply", ".obj", ".zip", ".mp4", ".blend"}

func isResult(name string) bool {
	for _, ext := range resultExts {
		if strings.HasSuffix(name, ext) {
			return true
		}
	}
	return false
}

func resultsHandler(w http.ResponseWriter, r *http.Request) {
	entries, err := os.ReadDir(resultsDir)
	if err != nil {
		log.Printf("results dir %s: %v", resultsDir, err)
	}
	files := []string{}
	for _, e := range entries {
		if !e.IsDir() && isResult(e.Name()) {
			files = append(files, e.Name())
		}
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(files)
}

func downloadHandler(w http.ResponseWriter, r *http.Request) {
	name := filepath.Base(r.URL.Path)
	if !isResult(name) {
		http.NotFound(w, r)
		return
	}
	path := filepath.Join(resultsDir, name)
	if st, err := os.Stat(path); err != nil || st.IsDir() {
		http.NotFound(w, r)
		return
	}
	w.Header().Set("Content-Disposition", "attachment; filename="+name)
	http.ServeFile(w, r, path)
}
