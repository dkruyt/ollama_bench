# Ollama Bench

A powerful, parallel benchmark tool for [Ollama](https://ollama.ai) models with real-time TUI monitoring and comprehensive performance metrics.

## Features

- 🚀 **Parallel request execution** - Test models under realistic concurrent load
- 📊 **Real-time TUI dashboard** - Live metrics with progress tracking and performance stats
- 🎮 **Interactive controls** - Toggle previews, graphs, help, and restart on-the-fly
- 🎯 **Multiple model comparison** - Benchmark and compare different models side-by-side
- 📈 **Comprehensive metrics** - Latency (p50/p95/p99), TTFT, throughput, token counts
- 📉 **ASCII histograms & graphs** - Visualize latency distribution and time-series data
- 🔴 **Live token preview** - Watch streaming content from active requests in real-time
- 📈 **Performance graphs** - 60-second time-series graphs for RPS, latency, and token throughput
- 💾 **Export results** - Save detailed results to JSON or CSV
- 🔄 **Streaming support** - Test with or without streaming responses
- 🎨 **Flexible prompts** - Use inline prompts, files, JSONL, or templates with variables
- 🔥 **Warmup support** - Run warmup requests to load models before benchmarking
- 🎛️ **Model options** - Configure temperature, num_predict, and other parameters
- ⚖️ **Fair load balancing** - Automatic concurrency distribution across multiple models
- 🏷️ **Model metadata** - Display parameter size and quantization level for each model

## Installation

### Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai) installed and running locally

### Setup

```bash
# Clone the repository
git clone https://github.com/dkruyt/ollama_bench.git
cd ollama_bench

# Install dependencies
pip install -r requirements.txt

# Make executable (optional)
chmod +x ollama_bench.py
```

## Quick Start

```bash
# Basic benchmark - 100 requests to llama3
python ollama_bench.py --models llama3 --requests 100 --concurrency 20 \
  --prompt "Explain quantum computing in one sentence." --stream

# Real-time TUI dashboard
python ollama_bench.py --models llama3 --requests 100 --concurrency 20 \
  --prompt "Explain AI briefly." --stream --tui

# Compare multiple models
python ollama_bench.py --models llama3 qwen2.5:7b --requests 50 --concurrency 10 \
  --prompt-file prompt.txt --stream --out-json results.json
```

## Usage Examples

### Basic Benchmarking

```bash
# Single model, streaming responses
python ollama_bench.py \
  --models llama3 \
  --requests 100 \
  --concurrency 20 \
  --prompt "What is machine learning?" \
  --stream
```

### Chat Mode

```bash
# Chat endpoint with system prompt
python ollama_bench.py \
  --models llama3 \
  --requests 50 \
  --concurrency 10 \
  --chat \
  --system "You are a helpful assistant." \
  --prompt "Explain neural networks."
```

### Using Prompt Files with Variables

```bash
# Use a prompt file with variable substitution
python ollama_bench.py \
  --models llama3 \
  --requests 30 \
  --concurrency 5 \
  --prompt-file template.txt \
  --variables topic=quantum_physics name=Alice

# template.txt content:
# Hello {name}, please explain {topic} in simple terms.
```

### Model Comparison with Export

```bash
# Compare models and export results
python ollama_bench.py \
  --models llama3 qwen2.5:7b mistral \
  --requests 100 \
  --concurrency 15 \
  --prompt-file prompts.txt \
  --stream \
  --out-json comparison.json \
  --out-csv comparison.csv
```

### Advanced Configuration

```bash
# Custom model options, warmup, and multiple prompts
python ollama_bench.py \
  --models llama3 \
  --requests 200 \
  --concurrency 30 \
  --prompt-file prompts.txt \
  --stream \
  --warmup 5 \
  --options '{"temperature":0.7,"num_predict":256,"top_p":0.9}' \
  --tui
```

### Silent Mode (No TUI)

```bash
# Run without TUI, just final summary
python ollama_bench.py \
  --models llama3 \
  --requests 50 \
  --prompt "Explain AI" \
  --silent
```

## Command Line Options

### Required Options

- `--models MODEL [MODEL ...]` - One or more Ollama model names to benchmark

### Request Configuration

- `--requests N` - Number of requests per model (default: 10)
- `--concurrency N` - Maximum concurrent requests (default: 5)
- `--warmup N` - Number of warmup requests per model before benchmark (default: 0)

### Prompt Options

- `--prompt TEXT` - Inline prompt text
- `--prompt-file FILE` - Read prompt from file
- `--prompts-jsonl FILE` - JSONL file with per-request prompts (cycled if needed)
- `--variables KEY=VALUE[,KEY=VALUE...]` - Variables for prompt template substitution
  - Supports comma-separated key=value pairs
  - Supports JSON file: `--variables path/to/vars.json`
  - Supports file injection: `--variables text=@file.txt`

### Mode Options

- `--chat` - Use chat endpoint instead of generate
- `--stream` - Enable streaming responses (enables TTFT measurement)
- `--system TEXT` - System message for chat mode
- `--shuffle` - Shuffle request order to mix load across models

### Model Options

- `--options JSON` - JSON string of model options (temperature, num_predict, etc.)

### Output Options

- `--tui` - Enable real-time TUI dashboard with interactive controls
- `--silent` - Suppress all output except final summary (no TUI)
- `--out-json FILE` - Export detailed results to JSON file
- `--out-csv FILE` - Export per-request results to CSV file

### Connection Options

- `--host URL` - Ollama server URL (default: from OLLAMA_HOST env or library default)
- `--timeout SECONDS` - Per-request timeout in seconds (0 = no timeout)
- `--seed N` - Random seed for reproducible shuffling/prompt cycling

## Interactive TUI Controls

When using `--tui`, the following keyboard shortcuts are available:

- **`[p]`** - Toggle live token preview (shows streaming content from active requests)
- **`[g]`** - Toggle ASCII performance graphs (60-second time-series data)
- **`[r]`** - Restart benchmark (resets metrics and starts over)
- **`[i]`** - Show/hide benchmark configuration info panel
- **`[h]` or `[?]`** - Show/hide help panel with metrics explanations
- **`[q]` or `[Esc]`** - Quit benchmark gracefully

### TUI Panels

- **Overall Status** - Progress bar, completion stats, ETA, throughput (req/s), token/s
- **Per-Model Stats** - Table showing each model's progress, latencies, TTFT, and token throughput
- **Active Requests** - Live view of in-flight requests with elapsed time and token counts
- **Error Log** - Recent errors with timestamps and details (shown when errors occur)
- **Live Token Preview** - Real-time streaming content from up to 4 active requests (toggle with `[p]`)
- **Performance Graphs** - ASCII graphs showing RPS, latency, and token/s trends (toggle with `[g]`)
- **Help Panel** - Keyboard shortcuts and metrics explanations (toggle with `[h]`)
- **Info Panel** - Benchmark configuration details (toggle with `[i]`)

## Output Metrics

### Summary Statistics

- **Throughput** - Requests per second (RPS)
- **Latency** - avg, p50, p95, p99, min, max (in milliseconds)
- **TTFT** - Time to first token (streaming only)
- **Token counts** - Prompt tokens, output tokens, total tokens
- **Success rate** - Completed vs failed requests
- **Latency histogram** - ASCII visualization of distribution

### Real-time TUI Metrics

- **Progress bar** - Visual progress with percentage and ETA
- **Active requests** - Count and detailed list of in-flight requests
- **Recent metrics** - Rolling window of last 100 latencies and TTFTs
- **Tokens/second** - Real-time token generation throughput
- **Per-model statistics** - Individual performance breakdown with avg/max token/s
- **Error tracking** - Recent error log with timestamps and details
- **Live streaming** - Watch token generation from active requests in real-time
- **Performance graphs** - 60-second time-series graphs for RPS, latency, and token throughput
- **Model metadata** - Parameter size and quantization level for each model

### Export Formats

**JSON** - Complete results with all metadata:
```json
{
  "per_model": {
    "llama3": {
      "count": 100,
      "ok": 98,
      "errors": 2,
      "throughput_rps": 8.5,
      "latency_ms": { "avg": 235.4, "p50": 220.1, ... },
      "tokens": { ... }
    }
  },
  "overall": { ... },
  "results": [ ... ]
}
```

**CSV** - Per-request details for analysis:
```csv
req_id,model,duration_ms,ttft_ms,prompt_eval_count,eval_count,error,...
0,llama3,234.5,45.2,32,128,
1,llama3,245.1,48.3,32,135,
```

## Performance Tips

1. **Warmup requests** - Use `--warmup 3-5` to load models into memory before benchmarking
2. **Concurrency tuning** - Start with `--concurrency 10-20`, adjust based on your hardware
3. **Multiple models** - Concurrency is automatically distributed fairly across models
4. **Prompt variety** - Use `--prompts-jsonl` for varied workload testing
5. **Monitor resources** - Watch CPU/GPU/memory usage during benchmarks (use `[i]` in TUI)
6. **Streaming** - Use `--stream` to measure TTFT and simulate real-world usage
7. **Interactive monitoring** - Use `[p]` to watch live token generation, `[g]` for performance trends
8. **Restart benchmarks** - Press `[r]` in TUI to reset metrics and restart without restarting the process

## Troubleshooting

### Ollama not responding
```bash
# Check if Ollama is running
ollama list

# Start Ollama if needed
ollama serve
```

### Model not found
```bash
# Pull the model first
ollama pull llama3
```

### Connection refused
```bash
# Verify Ollama is running on default port
curl http://localhost:11434/api/version

# Or specify custom host
python ollama_bench.py --host http://your-host:11434 ...
```

### High error rates
- Reduce `--concurrency` to lower system load
- Increase per-request timeout: `--timeout 120`
- Check system resources (RAM, GPU memory)
- Check error log in TUI for specific error messages

### TUI not displaying correctly
- Ensure terminal size is at least 120x40 for optimal display
- Some terminals may not support all Rich library features
- Try a different terminal emulator (iTerm2, Windows Terminal, etc.)

## Examples Gallery

### Compare Latency Distribution with Live Graphs
```bash
python ollama_bench.py \
  --models llama3 qwen2.5:7b \
  --requests 200 \
  --concurrency 20 \
  --prompt "Explain quantum entanglement" \
  --stream --tui
```

Use `[g]` in TUI to view real-time performance graphs. Final output includes ASCII histograms showing latency distribution per model.

### Load Testing
```bash
python ollama_bench.py \
  --models llama3 \
  --requests 1000 \
  --concurrency 50 \
  --prompt-file load_test_prompts.txt \
  --stream \
  --out-json load_test_results.json
```

### Token Efficiency Analysis
```bash
python ollama_bench.py \
  --models llama3 qwen2.5:7b mistral \
  --requests 100 \
  --prompt "Write a haiku about programming" \
  --options '{"num_predict":50}' \
  --stream --tui \
  --out-csv tokens.csv
```

Watch live token generation with `[p]` in TUI. Analyze `prompt_eval_count` and `eval_count` columns in CSV output.

### Interactive Debugging
```bash
python ollama_bench.py \
  --models llama3 \
  --requests 50 \
  --concurrency 10 \
  --prompt-file prompts.txt \
  --stream --tui
```

During the run:
- Press `[p]` to watch streaming content from active requests
- Press `[g]` to see throughput and latency trends over time
- Press `[i]` to review benchmark configuration
- Press `[r]` to restart if you notice issues
- Press `[h]` for help with metrics interpretation

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built for [Ollama](https://ollama.ai)
- Uses [Rich](https://github.com/Textualize/rich) for TUI display

## Author

Created by Dennis Kruyt

## Links

- [Ollama Documentation](https://github.com/ollama/ollama)
- [Report Issues](https://github.com/dkruyt/ollama_bench/issues)
