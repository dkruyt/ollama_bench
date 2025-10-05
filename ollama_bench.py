#!/usr/bin/env python3
"""
ollama_bench.py — simple, parallel benchmark tool for Ollama models

Features
- Launches many requests in parallel to one or more models
- Measures per‑request total wall‑clock time (start → completion)
- Optional time‑to‑first‑token (TTFT) when streaming
- Works with `generate` or `chat` endpoints
- Aggregates per‑model and overall stats (p50/p95/p99, avg, throughput)
- Real‑time TUI display with live metrics (--tui flag)
- Exports detailed results to JSON/CSV if you want

Examples
---------
# 100 requests to llama3, 20 in parallel, using a prompt string
python ollama_bench.py --models llama3 --requests 100 --concurrency 20 \
  --prompt "Explain quantum tunneling in one sentence." --stream

# Compare two models side‑by‑side with the same workload
python ollama_bench.py --models llama3 llama3:8b --requests 50 --concurrency 10 \
  --prompt-file prompt.txt --stream --options '{"temperature":0.2,"num_predict":64}'

# Chat mode
python ollama_bench.py --models llama3 --requests 30 --concurrency 10 \
  --chat --system "You are concise." --prompt "Summarize: {text}" --variables text=input.txt

# Real-time TUI display with live metrics
python ollama_bench.py --models llama3 llama3:8b --requests 100 --concurrency 20 \
  --prompt "Explain AI in one sentence." --stream --tui

Prereqs
-------
pip install ollama==0.3.*  # https://github.com/ollama/ollama-python
pip install rich           # For TUI display (optional, only needed for --tui)

Make sure the Ollama server is running and the models are pulled.
You can point to a remote host with --host or OLLAMA_HOST.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import datetime as dt
import json
import random
import statistics as stats
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from ollama import Client
except Exception as e:
    print("[fatal] Could not import ollama. Install with `pip install ollama`.\n", e, file=sys.stderr)
    sys.exit(1)

try:
    from rich.live import Live
    from rich.console import Console
    from rich.table import Table
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.text import Text
    from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
    from rich.align import Align
    RICH_AVAILABLE = True
except Exception:
    RICH_AVAILABLE = False

ISO = "%Y-%m-%dT%H:%M:%S.%fZ"

def iso_now() -> str:
    return dt.datetime.utcnow().strftime(ISO)


def parse_iso(ts: str) -> dt.datetime:
    """Parse ISO timestamp produced by iso_now()"""
    return dt.datetime.strptime(ts, ISO)

# ------------------------------ CLI ---------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parallel benchmark tool for Ollama models")
    p.add_argument("--models", nargs="+", required=True,
                   help="One or more model names (e.g., llama3, llama3:8b)")
    p.add_argument("--requests", type=int, default=50,
                   help="Total requests per model (default: 50)")
    p.add_argument("--concurrency", type=int, default=10,
                   help="Max concurrent in‑flight requests across all models (default: 10)")
    p.add_argument("--prompt", type=str, default=None,
                   help="Prompt string for generate/chat user message. Use --prompt-file to read from file.")
    p.add_argument("--prompt-file", type=str, default=None,
                   help="Path to a file with the prompt text. If provided, overrides --prompt.")
    p.add_argument("--prompts-jsonl", type=str, default=None,
                   help="Optional JSONL file with per‑request prompts: each line {\"prompt\":\"...\"}. Cycled if shorter than requests.")
    p.add_argument("--chat", action="store_true", help="Use chat endpoint instead of generate.")
    p.add_argument("--system", type=str, default=None, help="System message for chat mode.")
    p.add_argument("--variables", type=str, default=None,
                   help="Optional variable replacements in the prompt. Formats supported: \n"
                        " - key=value[,key=value...] \n"
                        " - path/to/vars.json (object of {key:value}) \n"
                        " - key=@path/to/file to inject file contents")
    p.add_argument("--stream", action="store_true",
                   help="Stream tokens. Enables TTFT measurement (time‑to‑first‑token).")
    p.add_argument("--options", type=str, default=None,
                   help="JSON dict of generation options passed to Ollama (e.g., '{\"temperature\":0.2,\"num_predict\":64}').")
    p.add_argument("--host", type=str, default=None,
                   help="Ollama host like http://localhost:11434. Defaults to env OLLAMA_HOST or library default.")
    p.add_argument("--timeout", type=float, default=0,
                   help="Per‑request timeout in seconds (0 = no timeout). Applies to the client socket.")
    p.add_argument("--warmup", type=int, default=0, help="Number of warmup requests per model (not measured).")
    p.add_argument("--shuffle", action="store_true", help="Shuffle work across models to mix load.")
    p.add_argument("--seed", type=int, default=None, help="Random seed for shuffling / prompt cycling.")
    p.add_argument("--out-json", type=str, default=None, help="Write detailed results JSON to this path.")
    p.add_argument("--out-csv", type=str, default=None, help="Write detailed results CSV to this path.")
    p.add_argument("--silent", action="store_true", help="Reduce console output (only summary).")
    p.add_argument("--tui", action="store_true", help="Enable real-time TUI (terminal UI) display. Requires 'rich' library.")
    return p.parse_args(argv)

# ------------------------------ Helpers ------------------------------

def load_prompt(args: argparse.Namespace) -> str:
    if args.prompts_jsonl:
        return ""  # handled dynamically per task
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8")
    if args.prompt is not None:
        return args.prompt
    # Fallback tiny prompt
    return "Say 'ok'."


def parse_variables(spec: Optional[str]) -> Dict[str, str]:
    if not spec:
        return {}
    # If it's a path to json
    p = Path(spec)
    if p.exists() and p.is_file():
        return json.loads(p.read_text(encoding="utf-8"))
    # Otherwise parse key=value[,key=value] with @file support
    vars: Dict[str, str] = {}
    for pair in spec.split(","):
        if not pair:
            continue
        if "=" not in pair:
            raise ValueError(f"Invalid variable pair: {pair}")
        k, v = pair.split("=", 1)
        if v.startswith("@") and Path(v[1:]).exists():
            vars[k] = Path(v[1:]).read_text(encoding="utf-8")
        else:
            vars[k] = v
    return vars


def apply_vars(text: str, variables: Dict[str, str]) -> str:
    out = text
    for k, v in variables.items():
        out = out.replace("{" + k + "}", v)
    return out


def maybe_json(s: Optional[str]) -> Optional[Dict[str, Any]]:
    if not s:
        return None
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        raise SystemExit(f"--options must be JSON: {e}")

# ------------------------------ Real-time metrics --------------------

class LiveMetrics:
    """Thread-safe live metrics tracker for TUI display"""
    def __init__(self):
        self.lock = threading.Lock()
        self.total_tasks = 0
        self.completed = 0
        self.errors = 0
        self.in_flight = 0
        self.per_model: Dict[str, Dict[str, Any]] = {}
        self.recent_latencies: List[float] = []
        self.recent_ttfts: List[float] = []
        self.start_time = time.time()
        self.last_completed_times: List[float] = []
        # Track active requests: req_id -> (model, start_time, req_idx, current_tokens, streaming_content)
        self.active_requests: Dict[int, Tuple[str, float, int, int, str]] = {}
        self._next_req_id = 0
        # Token tracking
        self.total_tokens = 0
        self.recent_tokens: List[Tuple[float, int]] = []  # (timestamp, token_count)
        # Error tracking
        self.error_log: List[Dict[str, Any]] = []  # Recent errors with details
        # Token preview
        self.show_preview = False  # Toggle for token preview display
        # UI state
        self.show_help = False  # Help panel display
        self.show_info = False  # Info panel display
        self.show_graph = False  # Graph view display
        self.should_quit = False  # Signal to quit benchmark
        # Time-series data for graphs
        self.rps_history: List[Tuple[float, float]] = []  # (timestamp, req/s)
        self.latency_history: List[Tuple[float, float]] = []  # (timestamp, avg latency ms)
        self.tokens_per_sec_history: List[Tuple[float, float]] = []  # (timestamp, tok/s)

    def init_model(self, model: str, count: int):
        with self.lock:
            if model not in self.per_model:
                self.per_model[model] = {
                    "total": count,
                    "completed": 0,
                    "errors": 0,
                    "in_flight": 0,
                    "latencies": [],
                    "ttfts": [],
                    "tokens": [],
                    "tokens_per_sec": [],  # Token throughput per request
                }

    def start_request(self, model: str, req_idx: int) -> int:
        with self.lock:
            req_id = self._next_req_id
            self._next_req_id += 1
            self.in_flight += 1
            self.active_requests[req_id] = (model, time.time(), req_idx, 0, "")  # Add token counter and content
            if model in self.per_model:
                self.per_model[model]["in_flight"] += 1
            return req_id

    def update_request_tokens(self, req_id: int, tokens: int, content: str = ""):
        """Update token count and streaming content for an active request"""
        with self.lock:
            if req_id in self.active_requests:
                model, start_time, req_idx, _, _ = self.active_requests[req_id]
                # Keep last 500 chars to avoid memory issues
                truncated_content = content[-500:] if len(content) > 500 else content
                self.active_requests[req_id] = (model, start_time, req_idx, tokens, truncated_content)

    def finish_request(self, req_id: int, duration_ms: float, ttft_ms: Optional[float], error: Optional[str], tokens: int = 0):
        with self.lock:
            if req_id not in self.active_requests:
                return

            model, _, req_idx, _, _ = self.active_requests.pop(req_id)
            self.in_flight -= 1
            self.completed += 1
            now = time.time()
            self.last_completed_times.append(now)
            # Keep only last 10 completion times for RPS calculation
            self.last_completed_times = self.last_completed_times[-10:]

            if error:
                self.errors += 1
                # Log error details
                self.error_log.append({
                    "timestamp": now,
                    "model": model,
                    "req_idx": req_idx,
                    "error": error,
                })
                # Keep only last 20 errors
                self.error_log = self.error_log[-20:]

                if model in self.per_model:
                    self.per_model[model]["errors"] += 1
                    self.per_model[model]["in_flight"] -= 1
            else:
                self.recent_latencies.append(duration_ms)
                self.recent_latencies = self.recent_latencies[-100:]  # keep last 100
                if ttft_ms is not None:
                    self.recent_ttfts.append(ttft_ms)
                    self.recent_ttfts = self.recent_ttfts[-100:]

                # Track tokens
                if tokens > 0:
                    self.total_tokens += tokens
                    self.recent_tokens.append((now, tokens))
                    # Keep only last 60 seconds of token data
                    cutoff = now - 60
                    self.recent_tokens = [(t, c) for t, c in self.recent_tokens if t > cutoff]

                if model in self.per_model:
                    self.per_model[model]["completed"] += 1
                    self.per_model[model]["in_flight"] -= 1
                    self.per_model[model]["latencies"].append(duration_ms)
                    self.per_model[model]["latencies"] = self.per_model[model]["latencies"][-200:]  # Keep last 200
                    if ttft_ms is not None:
                        self.per_model[model]["ttfts"].append(ttft_ms)
                        self.per_model[model]["ttfts"] = self.per_model[model]["ttfts"][-200:]  # Keep last 200
                    if tokens > 0:
                        self.per_model[model]["tokens"].append(tokens)
                        self.per_model[model]["tokens"] = self.per_model[model]["tokens"][-200:]  # Keep last 200
                        # Calculate tokens/sec for this request
                        duration_sec = duration_ms / 1000
                        tok_per_sec = tokens / duration_sec if duration_sec > 0 else 0
                        self.per_model[model]["tokens_per_sec"].append(tok_per_sec)
                        self.per_model[model]["tokens_per_sec"] = self.per_model[model]["tokens_per_sec"][-200:]  # Keep last 200

    def reset_metrics(self):
        """Reset all metrics for restart (keeps config)"""
        with self.lock:
            self.completed = 0
            self.errors = 0
            self.in_flight = 0
            self.recent_latencies = []
            self.recent_ttfts = []
            self.start_time = time.time()
            self.last_completed_times = []
            self.active_requests = {}
            self.total_tokens = 0
            self.recent_tokens = []
            self.error_log = []
            for model in self.per_model:
                self.per_model[model].update({
                    "completed": 0,
                    "errors": 0,
                    "in_flight": 0,
                    "latencies": [],
                    "ttfts": [],
                    "tokens": [],
                    "tokens_per_sec": [],
                })

    def get_snapshot(self) -> Dict[str, Any]:
        with self.lock:
            elapsed = time.time() - self.start_time
            now = time.time()
            # Calculate recent throughput from last completed times
            recent_rps = 0.0
            if len(self.last_completed_times) >= 2:
                time_span = self.last_completed_times[-1] - self.last_completed_times[0]
                if time_span > 0:
                    recent_rps = (len(self.last_completed_times) - 1) / time_span

            # Calculate ETA
            eta_seconds = None
            if self.completed > 0 and recent_rps > 0:
                remaining = self.total_tasks - self.completed
                eta_seconds = remaining / recent_rps

            # Calculate tokens/sec
            tokens_per_sec = 0.0
            if self.recent_tokens:
                if len(self.recent_tokens) >= 2:
                    time_span = self.recent_tokens[-1][0] - self.recent_tokens[0][0]
                    if time_span > 0:
                        total_recent_tokens = sum(c for _, c in self.recent_tokens)
                        tokens_per_sec = total_recent_tokens / time_span

            # Build active requests snapshot with elapsed time and current token/sec
            active = []
            for req_id, (model, start_time, req_idx, current_tokens, content) in self.active_requests.items():
                elapsed_sec = now - start_time
                elapsed_ms = elapsed_sec * 1000
                # Calculate current tokens/sec for this request
                cur_tok_per_sec = current_tokens / elapsed_sec if elapsed_sec > 0 and current_tokens > 0 else 0
                active.append({
                    "req_id": req_id,
                    "model": model,
                    "req_idx": req_idx,
                    "elapsed_ms": elapsed_ms,
                    "current_tokens": current_tokens,
                    "tokens_per_sec": cur_tok_per_sec,
                    "content": content,  # Add streaming content
                })
            # Sort by elapsed time (longest running first)
            active.sort(key=lambda x: x["elapsed_ms"], reverse=True)

            # Record history for graphs (sample every ~1 second)
            if not self.rps_history or (now - self.rps_history[-1][0]) >= 1.0:
                self.rps_history.append((now, recent_rps))
                # Keep last 60 samples (1 minute of data)
                self.rps_history = self.rps_history[-60:]

                # Record average latency
                avg_latency = sum(self.recent_latencies) / len(self.recent_latencies) if self.recent_latencies else 0
                self.latency_history.append((now, avg_latency))
                self.latency_history = self.latency_history[-60:]

                # Record tokens/sec
                self.tokens_per_sec_history.append((now, tokens_per_sec))
                self.tokens_per_sec_history = self.tokens_per_sec_history[-60:]

            return {
                "total_tasks": self.total_tasks,
                "completed": self.completed,
                "errors": self.errors,
                "in_flight": self.in_flight,
                "elapsed": elapsed,
                "eta_seconds": eta_seconds,
                "recent_rps": recent_rps,
                "overall_rps": self.completed / elapsed if elapsed > 0 else 0,
                "recent_latencies": list(self.recent_latencies),
                "recent_ttfts": list(self.recent_ttfts),
                "active_requests": active,
                "total_tokens": self.total_tokens,
                "tokens_per_sec": tokens_per_sec,
                "error_log": list(self.error_log[-10:]),  # Last 10 errors
                "per_model": {
                    m: {
                        "total": d["total"],
                        "completed": d["completed"],
                        "errors": d["errors"],
                        "in_flight": d["in_flight"],
                        "latencies": list(d["latencies"]),
                        "ttfts": list(d["ttfts"]),
                        "tokens": list(d["tokens"]),
                        "tokens_per_sec": list(d["tokens_per_sec"]),
                    }
                    for m, d in self.per_model.items()
                },
                "rps_history": list(self.rps_history),
                "latency_history": list(self.latency_history),
                "tokens_per_sec_history": list(self.tokens_per_sec_history),
            }

# ------------------------------ TUI Display --------------------------

def create_ascii_graph(data: List[Tuple[float, float]], width: int = 60, height: int = 15, title: str = "", unit: str = "") -> str:
    """Create an ASCII line graph from time-series data"""
    if not data or len(data) < 2:
        return f"Not enough data for {title} (need at least 2 data points)"

    # Extract values
    values = [v for _, v in data]
    max_val = max(values) if values else 1
    min_val = min(values) if values else 0

    # Avoid division by zero
    range_val = max_val - min_val
    if range_val == 0:
        range_val = 1

    # Create graph grid
    graph = []
    for _ in range(height):
        graph.append([" "] * width)

    # Plot data points
    for i, (_, value) in enumerate(data):
        x = int((i / (len(data) - 1)) * (width - 1))
        # Normalize value to graph height
        normalized = (value - min_val) / range_val
        y = height - 1 - int(normalized * (height - 1))

        if 0 <= x < width and 0 <= y < height:
            graph[y][x] = "█"

    # Connect points with lines
    for i in range(len(data) - 1):
        x1 = int((i / (len(data) - 1)) * (width - 1))
        x2 = int(((i + 1) / (len(data) - 1)) * (width - 1))

        val1 = (data[i][1] - min_val) / range_val
        val2 = (data[i + 1][1] - min_val) / range_val

        y1 = height - 1 - int(val1 * (height - 1))
        y2 = height - 1 - int(val2 * (height - 1))

        # Simple line drawing
        if x1 != x2:
            steps = abs(x2 - x1)
            for step in range(steps + 1):
                x = x1 + (step if x2 > x1 else -step)
                progress = step / steps if steps > 0 else 0
                y = int(y1 + (y2 - y1) * progress)
                if 0 <= x < width and 0 <= y < height:
                    graph[y][x] = "█"

    # Build output string
    result = []
    result.append(f"┌{'─' * (width + 2)}┐")
    result.append(f"│ {title:<{width}} │")
    result.append(f"├{'─' * (width + 2)}┤")

    # Y-axis labels and graph
    for i, row in enumerate(graph):
        # Calculate value for this row
        row_value = max_val - (i / (height - 1)) * range_val
        label = f"{row_value:6.1f}{unit}"
        result.append(f"│{''.join(row)} │{label}")

    # X-axis
    result.append(f"└{'─' * (width + 2)}┘")
    result.append(f"  {'Time (last 60s)':^{width}}")
    result.append(f"  Min: {min_val:.1f}{unit}  Max: {max_val:.1f}{unit}  Avg: {sum(values)/len(values):.1f}{unit}")

    return "\n".join(result)


def create_graph_panel(snapshot: Dict[str, Any]) -> Panel:
    """Create panel with ASCII graphs for throughput and latency"""
    graph_text = Text()

    # Throughput graph
    if snapshot.get("rps_history"):
        rps_graph = create_ascii_graph(
            snapshot["rps_history"],
            width=55,
            height=12,
            title="Requests per Second",
            unit=" req/s"
        )
        graph_text.append(rps_graph + "\n\n", style="cyan")

    # Latency graph
    if snapshot.get("latency_history"):
        latency_graph = create_ascii_graph(
            snapshot["latency_history"],
            width=55,
            height=12,
            title="Average Latency",
            unit=" ms"
        )
        graph_text.append(latency_graph + "\n\n", style="yellow")

    # Tokens/sec graph
    if snapshot.get("tokens_per_sec_history"):
        tokens_graph = create_ascii_graph(
            snapshot["tokens_per_sec_history"],
            width=55,
            height=12,
            title="Tokens per Second",
            unit=" tok/s"
        )
        graph_text.append(tokens_graph, style="green")

    if not graph_text.plain:
        graph_text.append("Collecting data for graphs...\n", style="dim italic")
        graph_text.append("(Graphs will appear after a few seconds)", style="dim italic")

    return Panel(graph_text, title="[bold]Performance Graphs (press 'g' to toggle)[/bold]", border_style="blue", expand=True)


def create_help_panel() -> Panel:
    """Create help panel with keyboard shortcuts and metrics explanation"""
    help_text = Text()
    help_text.append("Keyboard Shortcuts\n\n", style="bold cyan")

    shortcuts = [
        ("[p]", "Toggle token preview display", "cyan"),
        ("[g]", "Toggle ASCII performance graphs", "blue"),
        ("[r]", "Restart benchmark (resets metrics)", "magenta"),
        ("[i]", "Show/Hide benchmark info", "blue"),
        ("[h] or [?]", "Show/Hide this help", "green"),
        ("[q] or [Esc]", "Quit benchmark", "red"),
    ]

    for key, desc, color in shortcuts:
        help_text.append(f"  {key:<15}", style=f"bold {color}")
        help_text.append(f" {desc}\n", style="white")

    help_text.append("\n━━━ Metrics Explained ━━━\n\n", style="bold yellow")

    help_text.append("Overall Stats:\n", style="bold green")
    help_text.append("  • ", style="dim")
    help_text.append("Completed/Total", style="cyan")
    help_text.append(" - Finished requests vs total requests\n", style="dim")
    help_text.append("  • ", style="dim")
    help_text.append("In-Flight", style="cyan")
    help_text.append(" - Requests currently being processed\n", style="dim")
    help_text.append("  • ", style="dim")
    help_text.append("Errors", style="cyan")
    help_text.append(" - Failed requests (timeouts, connection errors, etc.)\n", style="dim")
    help_text.append("  • ", style="dim")
    help_text.append("Elapsed", style="cyan")
    help_text.append(" - Total time since benchmark started\n", style="dim")
    help_text.append("  • ", style="dim")
    help_text.append("Req/s", style="cyan")
    help_text.append(" - Requests completed per second (recent avg)\n", style="dim")
    help_text.append("  • ", style="dim")
    help_text.append("Total Tokens", style="cyan")
    help_text.append(" - All tokens generated across all requests\n", style="dim")
    help_text.append("  • ", style="dim")
    help_text.append("Tok/s", style="cyan")
    help_text.append(" - Tokens generated per second (recent avg)\n", style="dim")

    help_text.append("\nLatency Metrics:\n", style="bold green")
    help_text.append("  • ", style="dim")
    help_text.append("TTFT", style="cyan")
    help_text.append(" - Time To First Token (streaming responsiveness)\n", style="dim")
    help_text.append("  • ", style="dim")
    help_text.append("Duration", style="cyan")
    help_text.append(" - Total request time from start to completion\n", style="dim")
    help_text.append("  • ", style="dim")
    help_text.append("Percentiles (p50/p95/p99)", style="cyan")
    help_text.append(" - Distribution of latencies\n", style="dim")
    help_text.append("    (p50 = median, p99 = 99% of requests faster than this)\n", style="dim")

    help_text.append("\nPer-Model Stats:\n", style="bold green")
    help_text.append("  Shows individual performance for each model being tested\n", style="dim")
    help_text.append("  including completion counts, latencies, and token throughput\n", style="dim")

    help_text.append("\nError Log:\n", style="bold green")
    help_text.append("  Displays recent errors with timestamps and details\n", style="dim")

    help_text.append("\nToken Preview:\n", style="bold green")
    help_text.append("  Press [p] to see live token generation from active requests\n", style="dim")
    help_text.append("  Shows model, request#, and streamed content in real-time\n", style="dim")

    help_text.append("\nPerformance Graphs:\n", style="bold green")
    help_text.append("  Press [g] to view ASCII graphs showing trends over time\n", style="dim")
    help_text.append("  • Requests/sec - Throughput over the last 60 seconds\n", style="dim")
    help_text.append("  • Avg Latency - Request latency trend over time\n", style="dim")
    help_text.append("  • Tokens/sec - Token generation rate over time\n", style="dim")
    help_text.append("  Graphs update every second and show min/max/avg values\n", style="dim")

    return Panel(help_text, title="[bold]Help & Metrics Guide[/bold]", border_style="green", expand=True)


def create_info_panel(args: argparse.Namespace, metrics: LiveMetrics) -> Panel:
    """Create info panel with benchmark configuration"""
    info_text = Text()
    info_text.append("Benchmark Configuration\n\n", style="bold cyan")

    snapshot = metrics.get_snapshot()

    # Basic config
    info_text.append("Models:\n", style="bold yellow")
    for model in args.models:
        info_text.append(f"  • {model}\n", style="white")

    info_text.append(f"\nRequests: ", style="bold yellow")
    info_text.append(f"{args.requests} per model ({snapshot['total_tasks']} total)\n", style="white")

    info_text.append(f"Concurrency: ", style="bold yellow")
    info_text.append(f"{args.concurrency}\n", style="white")

    info_text.append(f"Mode: ", style="bold yellow")
    mode = "chat" if args.chat else "generate"
    mode += " (streaming)" if args.stream else " (non-streaming)"
    info_text.append(f"{mode}\n", style="white")

    if args.host:
        info_text.append(f"Host: ", style="bold yellow")
        info_text.append(f"{args.host}\n", style="white")

    if args.warmup > 0:
        info_text.append(f"Warmup: ", style="bold yellow")
        info_text.append(f"{args.warmup} requests per model\n", style="white")

    if args.options:
        info_text.append(f"\nOptions: ", style="bold yellow")
        info_text.append(f"{args.options}\n", style="dim white")

    return Panel(info_text, title="[bold]Info[/bold]", border_style="blue", expand=True)


def create_tui_layout(metrics: LiveMetrics, benchmark_done: bool = False, menu_text: str = "", model_info: Optional[Dict[str, Dict[str, Any]]] = None, args: Optional[argparse.Namespace] = None) -> Layout:
    """Build rich Layout with live metrics"""
    snapshot = metrics.get_snapshot()

    # Overall stats panel - 2 column layout
    elapsed = snapshot["elapsed"]
    progress_pct = (snapshot["completed"] / snapshot["total_tasks"] * 100) if snapshot["total_tasks"] > 0 else 0

    # Left column
    left_col = Text()
    bar_width = 40
    filled = int(bar_width * progress_pct / 100)
    bar = "█" * filled + "░" * (bar_width - filled)
    left_col.append(f"Progress: {bar} ", style="bold cyan")
    left_col.append(f"{progress_pct:.1f}%\n", style="bold cyan")
    left_col.append(f"          {snapshot['completed']}/{snapshot['total_tasks']} requests", style="dim cyan")

    # ETA
    if snapshot["eta_seconds"] is not None and snapshot["eta_seconds"] > 0:
        eta_min = int(snapshot["eta_seconds"] // 60)
        eta_sec = int(snapshot["eta_seconds"] % 60)
        left_col.append(f" | ETA: {eta_min}m {eta_sec}s\n", style="dim yellow")
    else:
        left_col.append("\n")

    left_col.append(f"In-flight: {snapshot['in_flight']}\n", style="yellow")
    left_col.append(f"Errors: {snapshot['errors']}\n", style="red" if snapshot['errors'] > 0 else "green")
    left_col.append(f"Elapsed: {elapsed:.1f}s\n", style="white")

    # Right column
    right_col = Text()
    right_col.append(f"Overall RPS: {snapshot['overall_rps']:.2f}\n", style="bold green")
    right_col.append(f"Recent RPS: {snapshot['recent_rps']:.2f}\n", style="green")

    # Tokens/sec if available
    if snapshot["tokens_per_sec"] > 0:
        right_col.append(f"Total Tokens: {snapshot['total_tokens']:,}\n", style="bold blue")
        right_col.append(f"Tokens/sec: {snapshot['tokens_per_sec']:.1f}\n", style="blue")
    else:
        right_col.append("\n\n")

    # Latency stats
    if snapshot["recent_latencies"]:
        lats = snapshot["recent_latencies"]
        avg_lat = stats.fmean(lats)
        p50_lat = percentile(lats, 50) or 0
        p95_lat = percentile(lats, 95) or 0
        right_col.append(f"Latency: avg={avg_lat:.0f}ms\n", style="magenta")
        right_col.append(f"         p50={p50_lat:.0f}ms p95={p95_lat:.0f}ms\n", style="magenta")

    if snapshot["recent_ttfts"]:
        ttfts = snapshot["recent_ttfts"]
        avg_ttft = stats.fmean(ttfts)
        p50_ttft = percentile(ttfts, 50) or 0
        right_col.append(f"TTFT: avg={avg_ttft:.0f}ms p50={p50_ttft:.0f}ms\n", style="magenta")

    # Combine columns side-by-side
    from rich.columns import Columns
    overall_columns = Columns([left_col, right_col], equal=True, expand=True)
    overall_panel = Panel(overall_columns, title="[bold]Overall Status[/bold]", border_style="blue")

    # Per-model table
    table = Table(show_header=True, header_style="bold magenta", expand=True)
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Progress", justify="right")
    table.add_column("In-flight", justify="right")
    table.add_column("Errors", justify="right")
    table.add_column("Avg Latency", justify="right")
    table.add_column("Avg TTFT", justify="right")
    table.add_column("Tok/s Avg", justify="right")
    table.add_column("Tok/s Max", justify="right")

    for model, data in sorted(snapshot["per_model"].items()):
        progress_str = f"{data['completed']}/{data['total']}"
        in_flight_str = str(data['in_flight'])
        error_str = str(data['errors'])

        # Format model name with info
        model_display = model
        if model_info and model in model_info:
            info = model_info[model]
            parts = []
            if info.get('parameter_size'):
                parts.append(f"{info['parameter_size']}")
            if info.get('quantization_level'):
                parts.append(f"{info['quantization_level']}")
            if parts:
                model_display = f"{model} [dim]({', '.join(parts)})[/dim]"

        avg_lat = "—"
        avg_ttft = "—"
        avg_tok_sec = "—"
        max_tok_sec = "—"

        if data["latencies"]:
            avg_lat = f"{stats.fmean(data['latencies']):.1f}ms"

        if data["ttfts"]:
            avg_ttft = f"{stats.fmean(data['ttfts']):.1f}ms"

        if data["tokens_per_sec"]:
            avg_tok_sec = f"{stats.fmean(data['tokens_per_sec']):.1f}"
            max_tok_sec = f"{max(data['tokens_per_sec']):.1f}"

        table.add_row(
            model_display,
            progress_str,
            in_flight_str,
            error_str,
            avg_lat,
            avg_ttft,
            avg_tok_sec,
            max_tok_sec
        )

    model_panel = Panel(table, title="[bold]Per-Model Stats[/bold]", border_style="green")

    # Active requests table (in-flight)
    active_table = Table(show_header=True, header_style="bold yellow", expand=True, show_lines=False)
    active_table.add_column("Req ID", style="dim", width=8)
    active_table.add_column("Model", style="cyan", no_wrap=True)
    active_table.add_column("Task #", justify="right", width=8)
    active_table.add_column("Running", justify="right", width=10)
    active_table.add_column("Tokens", justify="right", width=8)
    active_table.add_column("Tok/s", justify="right", width=10)

    # Show up to 15 active requests
    active_requests = snapshot["active_requests"][:15]
    if active_requests:
        for req in active_requests:
            elapsed_sec = req["elapsed_ms"] / 1000
            # Color code by elapsed time
            if elapsed_sec < 2:
                time_style = "green"
            elif elapsed_sec < 5:
                time_style = "yellow"
            else:
                time_style = "red"

            # Format tokens and tok/sec
            tokens_str = str(req["current_tokens"]) if req["current_tokens"] > 0 else "—"
            tok_sec_str = f"{req['tokens_per_sec']:.1f}" if req['tokens_per_sec'] > 0 else "—"

            active_table.add_row(
                str(req["req_id"]),
                req["model"],
                str(req["req_idx"]),
                Text(f"{elapsed_sec:.1f}s", style=time_style),
                tokens_str,
                tok_sec_str
            )
    else:
        active_table.add_row("—", "—", "—", "—", "—", "—")

    active_panel = Panel(
        active_table,
        title=f"[bold]Active Requests ({len(snapshot['active_requests'])} in-flight)[/bold]",
        border_style="yellow"
    )

    # Error log panel
    error_text = Text()
    if snapshot["error_log"]:
        for err in snapshot["error_log"][-5:]:  # Show last 5 errors
            time_ago = snapshot["elapsed"] - (err["timestamp"] - metrics.start_time)
            error_text.append(f"[{time_ago:.1f}s ago] ", style="dim")
            error_text.append(f"{err['model']} req#{err['req_idx']}: ", style="red")
            # Truncate error message to 60 chars
            err_msg = err["error"][:60] + "..." if len(err["error"]) > 60 else err["error"]
            error_text.append(f"{err_msg}\n", style="dim red")
    else:
        error_text.append("No errors ✓", style="green")

    error_panel = Panel(
        error_text,
        title=f"[bold]Recent Errors ({snapshot['errors']} total)[/bold]",
        border_style="red" if snapshot["errors"] > 0 else "green",
        height=8
    )

    # Status bar
    if benchmark_done:
        status_text = Text("✓ BENCHMARK COMPLETE", style="bold green")
    else:
        status_text = Text("⚡ RUNNING...", style="bold green")
    status_panel = Panel(Align.center(status_text), style="bold", height=3)

    # Combine layout
    # Preview panel (shown when toggle is enabled)
    preview_panel = None
    if metrics.show_preview:
        if snapshot["active_requests"]:
            from rich.columns import Columns
            from rich.console import Group

            preview_count = min(4, len(snapshot["active_requests"]))  # Show up to 3 requests
            stream_panels = []
            has_content = False

            for i, req in enumerate(snapshot["active_requests"][:preview_count]):
                content = req.get("content", "")

                stream_text = Text()
                # Compact header - fixed format
                model_name = req['model'][:20]  # Truncate long model names
                stream_text.append(f"{model_name:<20}\n", style="cyan bold")
                stream_text.append(f"Req #{req['req_idx']:<3} │ ", style="dim")
                stream_text.append(f"{req['current_tokens']:>4} tok │ ", style="yellow")
                stream_text.append(f"{req['elapsed_ms']/1000:>5.1f}s\n", style="green")
                stream_text.append("─" * 40 + "\n", style="dim")

                if content:
                    has_content = True
                    # Pad/truncate content to fixed height (8 lines of ~40 chars each)
                    lines = []
                    # Split content into lines that fit the column width
                    words = content.split()
                    current_line = ""
                    for word in words:
                        if len(current_line) + len(word) + 1 <= 38:
                            current_line += (" " if current_line else "") + word
                        else:
                            if current_line:
                                lines.append(current_line)
                            current_line = word
                    if current_line:
                        lines.append(current_line)

                    # Show last 8 lines to keep height consistent
                    display_lines = lines[-8:] if len(lines) > 8 else lines
                    # Pad to exactly 8 lines
                    while len(display_lines) < 8:
                        display_lines.append("")

                    for line in display_lines:
                        stream_text.append(f"{line:<38}\n", style="white")
                else:
                    stream_text.append("[waiting for first token...]\n", style="dim italic")
                    # Pad empty space to maintain height
                    for _ in range(7):
                        stream_text.append("\n")

                stream_panels.append(Panel(stream_text, border_style="blue", expand=True))

            # Arrange panels side-by-side
            columns = Columns(stream_panels, equal=True, expand=True)

            if not has_content:
                tip_text = Text("\n💡 Tip: Content will appear once streaming starts", style="yellow", justify="center")
                preview_content = Group(columns, tip_text)
            else:
                preview_content = columns

            preview_panel = Panel(preview_content, title="[bold]Live Token Preview (press 'p' to toggle)[/bold]", border_style="magenta")
        else:
            preview_text = Text("No active requests at the moment", style="yellow italic")
            preview_panel = Panel(preview_text, title="[bold]Live Token Preview (press 'p' to toggle)[/bold]", border_style="magenta")

    # Menu bar at bottom
    menu_text = Text()
    menu_text.append("[p]", style="bold cyan")
    menu_text.append(" Preview  ", style="white")
    menu_text.append("[g]", style="bold blue")
    menu_text.append(" Graphs  ", style="white")
    menu_text.append("[r]", style="bold magenta")
    menu_text.append(" Restart  ", style="white")
    menu_text.append("[i]", style="bold blue")
    menu_text.append(" Info  ", style="white")
    menu_text.append("[h]", style="bold green")
    menu_text.append(" Help  ", style="white")
    menu_text.append("[q]", style="bold red")
    menu_text.append(" Quit", style="white")

    # Status indicators
    menu_text.append("  │  ", style="dim")
    if metrics.show_preview:
        menu_text.append("Preview:", style="white")
        menu_text.append(" ON ", style="bold green")
    if metrics.show_graph:
        menu_text.append("Graphs:", style="white")
        menu_text.append(" ON ", style="bold green")
    if metrics.show_help:
        menu_text.append("Help:", style="white")
        menu_text.append(" ON ", style="bold green")
    if metrics.show_info:
        menu_text.append("Info:", style="white")
        menu_text.append(" ON ", style="bold green")

    menu_panel = Panel(Align.center(menu_text), style="dim", height=3)

    # Help/Info/Graph overlays
    help_panel_display = create_help_panel() if metrics.show_help else None
    info_panel_display = create_info_panel(args, metrics) if metrics.show_info and args else None
    graph_panel_display = create_graph_panel(snapshot) if metrics.show_graph else None

    layout = Layout()

    # If help, info, or graph is shown, create a simpler layout with overlay
    if help_panel_display or info_panel_display or graph_panel_display:
        # Show help/info/graph in a 2-column layout with main stats
        main_layout = Layout()
        main_layout.split_column(
            Layout(status_panel, size=3),
            Layout(overall_panel, size=10),
            Layout(model_panel, size=8),
            Layout(menu_panel, size=3)
        )

        # Determine which panels to show
        panels_to_show = []
        if help_panel_display:
            panels_to_show.append(help_panel_display)
        if info_panel_display:
            panels_to_show.append(info_panel_display)
        if graph_panel_display:
            panels_to_show.append(graph_panel_display)

        if len(panels_to_show) == 1:
            # Single panel on right
            layout.split_row(
                Layout(main_layout, ratio=2),
                Layout(panels_to_show[0], ratio=1)
            )
        elif len(panels_to_show) == 2:
            # Two panels on right, stacked
            overlay_layout = Layout()
            overlay_layout.split_column(
                Layout(panels_to_show[0]),
                Layout(panels_to_show[1])
            )
            layout.split_row(
                Layout(main_layout, ratio=1),
                Layout(overlay_layout, ratio=1)
            )
        else:
            # Three panels - show in vertical stack on right
            overlay_layout = Layout()
            overlay_layout.split_column(
                Layout(panels_to_show[0]),
                Layout(panels_to_show[1]),
                Layout(panels_to_show[2])
            )
            layout.split_row(
                Layout(main_layout, ratio=1),
                Layout(overlay_layout, ratio=1)
            )
        return layout

    # Normal layout - adjust sizes based on whether we have errors and preview
    if metrics.show_preview and preview_panel:
        if snapshot["errors"] > 0:
            layout.split_column(
                Layout(status_panel, size=3),
                Layout(overall_panel, size=10),
                Layout(model_panel, size=8),
                Layout(preview_panel, size=16),  # Fixed reasonable size for preview
                Layout(active_panel, ratio=1),  # Active panel expands to fill remaining
                Layout(error_panel, size=5),
                Layout(menu_panel, size=3)
            )
        else:
            layout.split_column(
                Layout(status_panel, size=3),
                Layout(overall_panel, size=10),
                Layout(model_panel, size=8),
                Layout(preview_panel, size=16),  # Fixed reasonable size for preview
                Layout(active_panel, ratio=1),  # Active panel expands to fill remaining
                Layout(menu_panel, size=3)
            )
    else:
        if snapshot["errors"] > 0:
            layout.split_column(
                Layout(status_panel, size=3),
                Layout(overall_panel, size=15),
                Layout(model_panel, size=10),
                Layout(active_panel, size=10),
                Layout(error_panel, size=7),
                Layout(menu_panel, size=3)
            )
        else:
            layout.split_column(
                Layout(status_panel, size=3),
                Layout(overall_panel, size=15),
                Layout(model_panel, size=10),
                Layout(active_panel, ratio=1),  # Expand to fill remaining space
                Layout(menu_panel, size=3)
            )

    return layout

# ------------------------------ Benchmark core -----------------------

class Bench:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.base_prompt = load_prompt(args)
        self.variables = parse_variables(args.variables)
        self.options = maybe_json(args.options)
        if args.seed is not None:
            random.seed(args.seed)
        self.client = Client(host=args.host, timeout=args.timeout if args.timeout > 0 else None)
        self.prompts_pool = self._load_prompts_jsonl(args.prompts_jsonl) if args.prompts_jsonl else None
        self.live_metrics: Optional[LiveMetrics] = None
        self.model_info: Dict[str, Dict[str, Any]] = {}  # Cache model information
        if args.tui:
            if not RICH_AVAILABLE:
                raise SystemExit("--tui requires 'rich' library. Install with: pip install rich")
            self.live_metrics = LiveMetrics()

    def _fetch_model_info(self, model: str) -> Dict[str, Any]:
        """Fetch model information from Ollama API"""
        if model in self.model_info:
            return self.model_info[model]

        try:
            # Use ollama show to get model details
            info = self.client.show(model)
            model_data = info.get('modelinfo', {}) if hasattr(info, 'get') else getattr(info, 'modelinfo', {})
            details = info.get('details', {}) if hasattr(info, 'get') else getattr(info, 'details', {})

            # Extract relevant information
            result = {
                'parameter_size': details.get('parameter_size', '') if hasattr(details, 'get') else getattr(details, 'parameter_size', ''),
                'quantization_level': details.get('quantization_level', '') if hasattr(details, 'get') else getattr(details, 'quantization_level', ''),
                'family': details.get('family', '') if hasattr(details, 'get') else getattr(details, 'family', ''),
            }
            self.model_info[model] = result
            return result
        except Exception:
            # Fallback if API call fails
            self.model_info[model] = {'parameter_size': '', 'quantization_level': '', 'family': ''}
            return self.model_info[model]

    def _load_prompts_jsonl(self, path: str) -> List[str]:
        pool: List[str] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if "prompt" in obj:
                        pool.append(str(obj["prompt"]))
                except json.JSONDecodeError:
                    # allow raw text line
                    pool.append(line)
        if not pool:
            raise SystemExit("--prompts-jsonl file had no prompts")
        return pool

    def _pick_prompt(self, idx: int) -> str:
        if self.prompts_pool:
            return self.prompts_pool[idx % len(self.prompts_pool)]
        return self.base_prompt

    def _build_messages(self, user_text: str) -> List[Dict[str, str]]:
        msgs: List[Dict[str, str]] = []
        if self.args.system:
            msgs.append({"role": "system", "content": self.args.system})
        msgs.append({"role": "user", "content": user_text})
        return msgs

    def warmup(self) -> None:
        if self.args.warmup <= 0:
            return
        total_warmup = len(self.args.models) * self.args.warmup

        # If TUI mode, show warmup with rich Progress
        if self.args.tui and RICH_AVAILABLE:
            from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]Warming up..."),
                BarColumn(),
                TaskProgressColumn(),
                TextColumn("[cyan]{task.description}"),
            ) as progress:
                task = progress.add_task(f"{len(self.args.models)} model(s)", total=total_warmup)
                completed = 0
                for m in self.args.models:
                    for i in range(self.args.warmup):
                        prompt = apply_vars(self._pick_prompt(i), self.variables)
                        try:
                            if self.args.chat:
                                self.client.chat(model=m, messages=self._build_messages(prompt), options=self.options, stream=False)
                            else:
                                self.client.generate(model=m, prompt=prompt, options=self.options, stream=False)
                            completed += 1
                            progress.update(task, completed=completed, description=f"({m})")
                        except Exception:
                            completed += 1
                            progress.update(task, completed=completed, description=f"({m}, error)")
        else:
            # Non-TUI mode
            if not self.args.silent:
                print(f"Warming up {len(self.args.models)} model(s) with {self.args.warmup} requests each ({total_warmup} total)...")
            completed = 0
            for m in self.args.models:
                for i in range(self.args.warmup):
                    prompt = apply_vars(self._pick_prompt(i), self.variables)
                    try:
                        if self.args.chat:
                            self.client.chat(model=m, messages=self._build_messages(prompt), options=self.options, stream=False)
                        else:
                            self.client.generate(model=m, prompt=prompt, options=self.options, stream=False)
                        completed += 1
                        if not self.args.silent:
                            print(f"  Warmup progress: {completed}/{total_warmup} ({m})", end="\r")
                    except Exception:
                        completed += 1
                        if not self.args.silent:
                            print(f"  Warmup progress: {completed}/{total_warmup} ({m}, error ignored)", end="\r")
            if not self.args.silent:
                print(f"  Warmup complete: {completed}/{total_warmup}          ")

    async def run(self) -> Dict[str, Any]:
        # Fetch model information before starting
        for model in self.args.models:
            self._fetch_model_info(model)

        self.warmup()
        tasks: List[Tuple[str, int]] = []  # (model, idx)

        # Build tasks list - interleave models to ensure parallel distribution
        if len(self.args.models) > 1 and not self.args.shuffle:
            # Interleave requests across models for better parallelism
            for i in range(self.args.requests):
                for m in self.args.models:
                    tasks.append((m, i))
        else:
            # Original behavior for single model or when shuffling
            for m in self.args.models:
                for i in range(self.args.requests):
                    tasks.append((m, i))

        if self.args.shuffle:
            random.shuffle(tasks)

        # Initialize live metrics if TUI enabled
        if self.live_metrics:
            self.live_metrics.total_tasks = len(tasks)
            for m in self.args.models:
                self.live_metrics.init_model(m, self.args.requests)

        # Use per-model semaphores for fair concurrency distribution
        num_models = len(self.args.models)
        if num_models > 1:
            # Split concurrency evenly across models
            per_model_concurrency = max(1, self.args.concurrency // num_models)
            model_semaphores = {m: asyncio.Semaphore(per_model_concurrency) for m in self.args.models}
        else:
            # Single model - use global semaphore
            model_semaphores = {self.args.models[0]: asyncio.Semaphore(self.args.concurrency)}

        results: List[Dict[str, Any]] = []

        async def worker(model: str, idx: int):
            nonlocal results
            async with model_semaphores[model]:
                # Check for quit signal
                if self.live_metrics and self.live_metrics.should_quit:
                    return

                result = await asyncio.to_thread(self._one_request, model, idx)
                results.append(result)
                if not self.args.silent and not self.args.tui:
                    status = "OK" if result.get("error") is None else "ERR"
                    print(f"[{status}] {result['model']} req#{result['req_id']} duration={result['duration_ms']:.1f} ms" +
                          (f", ttft={result['ttft_ms']:.1f} ms" if result.get('ttft_ms') is not None else ""))

        if self.args.tui and self.live_metrics:
            # Run with TUI - keyboard input handling
            import select
            import sys
            import termios
            import tty

            def check_keyboard():
                """Check for keyboard input non-blocking"""
                try:
                    # Check if stdin is ready
                    if select.select([sys.stdin], [], [], 0)[0]:
                        ch = sys.stdin.read(1)
                        if ch == 'p':
                            self.live_metrics.show_preview = not self.live_metrics.show_preview
                        elif ch == 'g':  # Graph toggle
                            self.live_metrics.show_graph = not self.live_metrics.show_graph
                        elif ch == 'r':  # Restart
                            self.live_metrics.reset_metrics()
                        elif ch == 'i':  # Info toggle
                            self.live_metrics.show_info = not self.live_metrics.show_info
                        elif ch == 'h' or ch == '?':  # Help toggle
                            self.live_metrics.show_help = not self.live_metrics.show_help
                        elif ch == 'q' or ch == '\x1b':  # q or Escape
                            # Signal to stop
                            self.live_metrics.should_quit = True
                            return True
                except Exception:
                    pass
                return False

            # Run with TUI
            with Live(create_tui_layout(self.live_metrics, model_info=self.model_info, args=self.args), refresh_per_second=4, screen=True) as live:
                # Save terminal settings
                old_settings = None
                try:
                    old_settings = termios.tcgetattr(sys.stdin)
                    tty.setcbreak(sys.stdin.fileno())
                except Exception:
                    pass  # Not a TTY or Windows

                async def update_display():
                    while self.live_metrics and self.live_metrics.completed < self.live_metrics.total_tasks:
                        await asyncio.sleep(0.25)
                        if check_keyboard():  # Check for quit
                            break
                        live.update(create_tui_layout(self.live_metrics, model_info=self.model_info, args=self.args))
                    # Final update
                    if self.live_metrics:
                        live.update(create_tui_layout(self.live_metrics, benchmark_done=True, model_info=self.model_info, args=self.args))

                try:
                    await asyncio.gather(
                        asyncio.gather(*(worker(m, i) for (m, i) in tasks)),
                        update_display()
                    )
                finally:
                    # Restore terminal settings
                    if old_settings:
                        try:
                            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                        except Exception:
                            pass
        else:
            # Run without TUI
            await asyncio.gather(*(worker(m, i) for (m, i) in tasks))

        return self._summarize(results)

    def _one_request(self, model: str, idx: int) -> Dict[str, Any]:
        prompt = apply_vars(self._pick_prompt(idx), self.variables)

        # Notify live metrics that request is starting
        req_id = 0
        if self.live_metrics:
            req_id = self.live_metrics.start_request(model, idx)

        started = time.perf_counter()
        started_iso = iso_now()
        ttft_ms: Optional[float] = None
        out_text_len = 0
        error = None
        meta: Dict[str, Any] = {}
        eval_count = 0  # Token count from eval_count in response
        prompt_eval_count = 0  # Prompt tokens
        chunk_count = 0  # Count chunks as proxy for tokens during streaming

        accumulated_text = ""  # Track full response for preview
        try:
            if self.args.chat:
                if self.args.stream:
                    first = True
                    for chunk in self.client.chat(model=model, messages=self._build_messages(prompt), options=self.options, stream=True):
                        if first:
                            ttft_ms = (time.perf_counter() - started) * 1000
                            first = False

                        # Handle ChatResponse objects (ollama returns these, not dicts)
                        if hasattr(chunk, 'message'):
                            content = chunk.message.content if hasattr(chunk.message, 'content') else ""
                            if content:
                                out_text_len += len(content)
                                chunk_count += 1
                                accumulated_text += content
                                # Update live metrics with streaming content
                                if self.live_metrics and self.args.tui:
                                    self.live_metrics.update_request_tokens(req_id, chunk_count, accumulated_text)
                            # Extract token counts if present (only in final chunk where done=True)
                            if hasattr(chunk, 'prompt_eval_count') and chunk.prompt_eval_count:
                                prompt_eval_count = chunk.prompt_eval_count
                            if hasattr(chunk, 'eval_count') and chunk.eval_count:
                                eval_count = chunk.eval_count
                            # Store metadata from final chunk
                            if hasattr(chunk, 'done') and chunk.done and hasattr(chunk, '__dict__'):
                                meta = chunk.__dict__
                        elif isinstance(chunk, dict):
                            # Fallback for dict format (just in case)
                            c = chunk.get("message", {}).get("content", "")
                            if c:
                                out_text_len += len(c)
                                chunk_count += 1
                            meta.update({k: v for k, v in chunk.items() if k not in {"message"}})
                            if "prompt_eval_count" in chunk and chunk["prompt_eval_count"]:
                                prompt_eval_count = chunk["prompt_eval_count"]
                            if "eval_count" in chunk and chunk["eval_count"]:
                                eval_count = chunk["eval_count"]

                        # Update live metrics with current token count (use eval_count or chunk_count as estimate)
                        if self.live_metrics and self.args.tui:
                            current_tokens = eval_count if eval_count > 0 else chunk_count
                            self.live_metrics.update_request_tokens(req_id, current_tokens, accumulated_text)
                else:
                    resp = self.client.chat(model=model, messages=self._build_messages(prompt), options=self.options, stream=False)
                    # Handle ChatResponse object
                    if hasattr(resp, 'message'):
                        content = resp.message.content if hasattr(resp.message, 'content') else ""
                        out_text_len = len(content)
                        if hasattr(resp, 'eval_count') and resp.eval_count:
                            eval_count = resp.eval_count
                        if hasattr(resp, '__dict__'):
                            meta = resp.__dict__
                    elif isinstance(resp, dict):
                        out_text_len = len(str(resp.get("message", {}).get("content", "")))
                        meta = resp
                        eval_count = resp.get("eval_count", 0)
            else:
                if self.args.stream:
                    first = True
                    for chunk in self.client.generate(model=model, prompt=prompt, options=self.options, stream=True):
                        if first:
                            ttft_ms = (time.perf_counter() - started) * 1000
                            first = False

                        # Handle GenerateResponse objects (ollama returns these, not dicts)
                        if hasattr(chunk, 'response'):
                            resp_text = chunk.response or ""
                            if resp_text:
                                out_text_len += len(resp_text)
                                chunk_count += 1
                                accumulated_text += resp_text
                                # Update live metrics with streaming content
                                if self.live_metrics and self.args.tui:
                                    self.live_metrics.update_request_tokens(req_id, chunk_count, accumulated_text)
                            # Extract token counts if present (only in final chunk where done=True)
                            if hasattr(chunk, 'prompt_eval_count') and chunk.prompt_eval_count:
                                prompt_eval_count = chunk.prompt_eval_count
                            if hasattr(chunk, 'eval_count') and chunk.eval_count:
                                eval_count = chunk.eval_count
                            # Store metadata from final chunk
                            if hasattr(chunk, 'done') and chunk.done and hasattr(chunk, '__dict__'):
                                meta = chunk.__dict__
                        elif isinstance(chunk, dict):
                            # Fallback for dict format (just in case)
                            resp_text = str(chunk.get("response", ""))
                            if resp_text:
                                out_text_len += len(resp_text)
                                chunk_count += 1
                            meta.update({k: v for k, v in chunk.items() if k not in {"response"}})
                            if "prompt_eval_count" in chunk and chunk["prompt_eval_count"]:
                                prompt_eval_count = chunk["prompt_eval_count"]
                            if "eval_count" in chunk and chunk["eval_count"]:
                                eval_count = chunk["eval_count"]

                        # Update live metrics with current token count (use eval_count or chunk_count as estimate)
                        if self.live_metrics and self.args.tui:
                            current_tokens = eval_count if eval_count > 0 else chunk_count
                            self.live_metrics.update_request_tokens(req_id, current_tokens, accumulated_text)
                else:
                    resp = self.client.generate(model=model, prompt=prompt, options=self.options, stream=False)
                    # Handle GenerateResponse object
                    if hasattr(resp, 'response'):
                        out_text_len = len(resp.response or "")
                        if hasattr(resp, 'eval_count') and resp.eval_count:
                            eval_count = resp.eval_count
                        if hasattr(resp, '__dict__'):
                            meta = resp.__dict__
                    elif isinstance(resp, dict):
                        out_text_len = len(str(resp.get("response", "")))
                        meta = resp
                        eval_count = resp.get("eval_count", 0)
        except Exception as e:
            error = str(e)
        ended = time.perf_counter()
        ended_iso = iso_now()
        duration_ms = (ended - started) * 1000

        # Use eval_count if available, otherwise fall back to chunk_count for streaming
        final_token_count = eval_count if eval_count > 0 else (chunk_count if self.args.stream else 0)

        # Notify live metrics that request finished
        if self.live_metrics:
            self.live_metrics.finish_request(req_id, duration_ms, ttft_ms, error, tokens=final_token_count)

        return {
            "req_id": idx,
            "model": model,
            "chat": self.args.chat,
            "stream": self.args.stream,
            "prompt_chars": len(prompt),
            "output_chars": out_text_len,
            "prompt_eval_count": prompt_eval_count,
            "eval_count": final_token_count,
            "started": started_iso,
            "ended": ended_iso,
            "duration_ms": duration_ms,
            "ttft_ms": ttft_ms,
            "error": error,
            "meta": safe_trim_meta(meta),
        }

    def _summarize(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        by_model: Dict[str, List[Dict[str, Any]]] = {}
        for r in results:
            by_model.setdefault(r["model"], []).append(r)

        def summarize_group(rs: List[Dict[str, Any]]) -> Dict[str, Any]:
            ok_results = [r for r in rs if r.get("error") is None]
            durs = [r["duration_ms"] for r in ok_results]
            ttfts = [r["ttft_ms"] for r in ok_results if r.get("ttft_ms") is not None]
            errs = [r for r in rs if r.get("error") is not None]
            prompt_tokens = [r["prompt_eval_count"] for r in ok_results if r.get("prompt_eval_count", 0) > 0]
            output_tokens = [r["eval_count"] for r in ok_results if r.get("eval_count", 0) > 0]
            count_ok = len(ok_results)

            # Measure throughput using wall-clock elapsed time between first start and last finish
            wall_seconds: Optional[float] = None
            if ok_results:
                starts = []
                ends = []
                for r in ok_results:
                    try:
                        starts.append(parse_iso(r["started"]))
                        ends.append(parse_iso(r["ended"]))
                    except Exception:
                        continue
                if starts and ends:
                    wall_seconds = (max(ends) - min(starts)).total_seconds()

            throughput = None
            if wall_seconds and wall_seconds > 0:
                throughput = count_ok / wall_seconds
            elif durs:
                total_s = sum(durs) / 1000.0
                if total_s > 0:
                    throughput = count_ok / total_s
            return {
                "count": len(rs),
                "ok": count_ok,
                "errors": len(errs),
                "throughput_rps": throughput,
                "latency_ms": percentile_summary(durs),
                "latency_raw": durs,  # Keep raw data for histogram
                "ttft_ms": percentile_summary(ttfts) if ttfts else None,
                "ttft_raw": ttfts if ttfts else None,  # Keep raw data for histogram
                "tokens": {
                    "prompt_tokens": {"total": sum(prompt_tokens), "avg": stats.fmean(prompt_tokens)} if prompt_tokens else None,
                    "output_tokens": {"total": sum(output_tokens), "avg": stats.fmean(output_tokens)} if output_tokens else None,
                    "total_tokens": sum(prompt_tokens) + sum(output_tokens) if (prompt_tokens or output_tokens) else 0,
                } if (prompt_tokens or output_tokens) else None,
            }

        summary = {m: summarize_group(rs) for m, rs in by_model.items()}
        overall = summarize_group(results)
        out = {
            "args": vars(self.args),
            "overall": overall,
            "per_model": summary,
            "results": results,
        }
        if self.args.out_json:
            Path(self.args.out_json).write_text(json.dumps(out, indent=2), encoding="utf-8")
        if self.args.out_csv:
            write_csv(self.args.out_csv, results)
        return out

# ------------------------------ Utilities ---------------------------

def percentile(data: List[float], p: float) -> Optional[float]:
    if not data:
        return None
    data = sorted(data)
    k = (len(data) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(data) - 1)
    if f == c:
        return data[int(k)]
    return data[f] + (data[c] - data[f]) * (k - f)


def percentile_summary(durs: List[float]) -> Optional[Dict[str, float]]:
    if not durs:
        return None
    return {
        "avg": stats.fmean(durs),
        "p50": percentile(durs, 50) or 0.0,
        "p95": percentile(durs, 95) or 0.0,
        "p99": percentile(durs, 99) or 0.0,
        "min": min(durs),
        "max": max(durs),
    }


def ascii_histogram(data: List[float], bins: int = 20, width: int = 60, title: str = "Distribution") -> str:
    """Generate ASCII histogram from data"""
    if not data or len(data) < 2:
        return f"{title}: insufficient data"

    min_val = min(data)
    max_val = max(data)
    range_val = max_val - min_val

    if range_val == 0:
        return f"{title}: all values are {min_val:.1f}"

    # Create bins
    bin_width = range_val / bins
    counts = [0] * bins

    for val in data:
        bin_idx = int((val - min_val) / bin_width)
        if bin_idx >= bins:
            bin_idx = bins - 1
        counts[bin_idx] += 1

    max_count = max(counts)
    if max_count == 0:
        return f"{title}: no data"

    # Build histogram
    lines = [f"\n{title}:"]
    for i, count in enumerate(counts):
        bin_start = min_val + i * bin_width
        bin_end = bin_start + bin_width
        bar_len = int((count / max_count) * width) if max_count > 0 else 0
        bar = "█" * bar_len
        lines.append(f"  {bin_start:7.1f}-{bin_end:7.1f}ms │{bar} {count}")

    return "\n".join(lines)


def safe_trim_meta(meta: Dict[str, Any], max_len: int = 2048) -> Dict[str, Any]:
    try:
        s = json.dumps(meta)
        if len(s) <= max_len:
            return meta
        return {"_trimmed": True}
    except Exception:
        return {}


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    keys = [
        "req_id","model","chat","stream","prompt_chars","output_chars","prompt_eval_count","eval_count","started","ended","duration_ms","ttft_ms","error"
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})

# ------------------------------ main --------------------------------

async def amain(argv: Optional[List[str]] = None):
    args = parse_args(argv)
    bench = Bench(args)
    report = await bench.run()

    # Pretty print summary
    def fmt_lat(s: Optional[Dict[str, float]]) -> str:
        if not s:
            return "—"
        return (
            f"avg={s['avg']:.1f}ms p50={s['p50']:.1f}ms p95={s['p95']:.1f}ms p99={s['p99']:.1f}ms "
            f"min={s['min']:.1f}ms max={s['max']:.1f}ms"
        )

    print("\n=== Summary ===")
    ov = report["overall"]
    print(f"Overall: {ov['ok']}/{ov['count']} ok, errors={ov['errors']}, throughput={ov['throughput_rps']:.2f} rps" if ov['throughput_rps'] else f"Overall: {ov['ok']}/{ov['count']} ok, errors={ov['errors']}")
    print("latency:", fmt_lat(ov.get("latency_ms")))
    if ov.get("ttft_ms"):
        print("ttft:", fmt_lat(ov.get("ttft_ms")))
    if ov.get("tokens"):
        tok = ov["tokens"]
        if tok.get("prompt_tokens"):
            print(f"prompt tokens: total={tok['prompt_tokens']['total']:,}, avg={tok['prompt_tokens']['avg']:.1f}")
        if tok.get("output_tokens"):
            print(f"output tokens: total={tok['output_tokens']['total']:,}, avg={tok['output_tokens']['avg']:.1f}")
        if tok.get("total_tokens"):
            print(f"total tokens: {tok['total_tokens']:,}")

    # Show latency histogram for overall
    if ov.get("latency_raw") and len(ov["latency_raw"]) >= 10:
        print(ascii_histogram(ov["latency_raw"], bins=15, width=50, title="Overall Latency Distribution"))

    for m, s in report["per_model"].items():
        tp = f", throughput={s['throughput_rps']:.2f} rps" if s.get('throughput_rps') else ""

        # Show model info if available
        model_info_str = ""
        if m in bench.model_info:
            info = bench.model_info[m]
            parts = []
            if info.get('parameter_size'):
                parts.append(f"{info['parameter_size']}")
            if info.get('quantization_level'):
                parts.append(f"{info['quantization_level']}")
            if parts:
                model_info_str = f" ({', '.join(parts)})"

        print(f"\n- {m}{model_info_str}: {s['ok']}/{s['count']} ok, errors={s['errors']}{tp}")
        print("  latency:", fmt_lat(s.get("latency_ms")))
        if s.get("ttft_ms"):
            print("  ttft:", fmt_lat(s.get("ttft_ms")))
        if s.get("tokens"):
            tok = s["tokens"]
            if tok.get("prompt_tokens"):
                print(f"  prompt tokens: total={tok['prompt_tokens']['total']:,}, avg={tok['prompt_tokens']['avg']:.1f}")
            if tok.get("output_tokens"):
                print(f"  output tokens: total={tok['output_tokens']['total']:,}, avg={tok['output_tokens']['avg']:.1f}")

        # Show latency histogram per model
        if s.get("latency_raw") and len(s["latency_raw"]) >= 10:
            hist = ascii_histogram(s["latency_raw"], bins=12, width=40, title=f"{m} Latency Distribution")
            # Indent histogram
            print("  " + hist.replace("\n", "\n  "))

if __name__ == "__main__":
    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        print("\nInterrupted.")
