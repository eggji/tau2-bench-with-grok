# τ²-Bench: Grok Benchmarking Framework

A streamlined version of τ²-bench focused on benchmarking conversational agents with Grok and other LLMs.

## Overview

This framework evaluates customer service agents across multiple domains:
- **airline** - Flight reservation customer service
- **retail** - E-commerce customer service  
- **telecom** - Technical support scenarios
- **mock** - Simple testing domain

## Quick Start

### Installation

```bash
# Install dependencies
pip install -e .

# Or using UV
uv sync
```

### Setup API Keys

Copy `.env.example` to `.env` and add your API keys:

```bash
cp .env.example .env
# Edit .env with your API keys
```

### Run Benchmark

**Using the standalone script (recommended for Grok integration):**

```bash
python run_benchmark.py \
  --domain airline \
  --agent-llm gpt-4.1 \
  --user-llm grok-4-fast-reasoning \
  --num-trials 1 \
  --num-tasks 5
```

**Using the CLI:**

```bash
tau2 run \
  --domain airline \
  --agent-llm gpt-4.1 \
  --user-llm grok-4-fast-reasoning \
  --num-trials 1 \
  --num-tasks 5
```

## Available Commands

### Run Benchmark
```bash
tau2 run \
  --domain <domain> \
  --agent-llm <llm_name> \
  --user-llm <llm_name> \
  --num-trials <trial_count> \
  --num-tasks <task_count>
```

### View Results
```bash
tau2 view
```

### Check Data Configuration
```bash
tau2 check-data
```

### View Domain Documentation
```bash
tau2 domain <domain>
```

## Domains

- **airline**: Flight booking, cancellations, seat changes
- **retail**: Product orders, returns, customer support
- **telecom**: Technical support, billing, service issues
- **mock**: Simple arithmetic and basic operations

## Results

Results are saved in `data/tau2/simulations/` as JSON files containing:
- Task execution trajectories
- Success/failure metrics
- Agent performance data

## LLM Support

Uses [LiteLLM](https://github.com/BerriAI/litellm) for LLM integration. Supports:
- OpenAI (GPT-4, GPT-3.5)
- Grok (grok-4-fast-reasoning, etc.)
- Anthropic (Claude)
- And many others

## Configuration

Key configuration options in `src/tau2/config.py`:
- Default LLM models
- Temperature settings
- Max steps and errors
- Concurrency limits

## Citation

```bibtex
@misc{barres2025tau2,
      title={$\tau^2$-Bench: Evaluating Conversational Agents in a Dual-Control Environment}, 
      author={Victor Barres and Honghua Dong and Soham Ray and Xujie Si and Karthik Narasimhan},
      year={2025},
      eprint={2506.07982},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2506.07982}, 
}
```