# Tau2 Grok Runner

`tau2-bench-with-grok` is a slimmed version of the Tau2 dual-control benchmark that ships with sane defaults for **xAI Grok** models. It lets you measure dialog agents on the airline, retail, and telecom task suites without hand-tuning the full Tau2 stack.

## Setup

```bash
pip install -e .   # or `uv sync`
cp .env.example .env
```

Populate `.env` with `XAI_API_KEY` (and optionally `OPENAI_API_KEY` if you compare against GPT baselines).

## Running with Grok

The helper script wires Grok into both the agent and user simulators. Run it once per domain or task subset:

```bash
python run_benchmark.py \
  --domain airline \
  --grok-model grok-4-fast-reasoning \
  --num-trials 1 \
  --num-tasks 5 \
  --evaluation-type process  # swap to `semantic_action` for relaxed matching
```

Behind the scenes this sets `--agent-llm` and `--user-llm` to the chosen Grok model unless you override them manually. Results land in `data/tau2/simulations/<timestamp>.json` and can be previewed via `tau2 view --file <path>`.

## Key Improvements

### Process-Aware Scoring

- Tracks how well the agent followed the *procedure* rather than only the final DB equality.
- Components: `StepReward`, `OrderReward`, `RedundancyPenalty`, `CausalPenalty`, normalized into `ProcessScoreNorm ∈ [0,1]`.
- Enable it with `--evaluation-type process`. Each simulation prints a "Process-Aware Score" under the reward and we aggregate the average score in the metrics footer.

### Semantic Argument Matching

- Replaces brittle string equality with field-aware fuzzy matching (numeric tolerance, date parsing, ID similarity ≥ 0.9, synonym tables).
- New evaluation mode `--evaluation-type semantic_action` annotates every action check with `(Semantic: ✅/❌, Score)` and drastically reduces false negatives such as `credit_card_01` vs `visa_card_01`.

### Robustness Mode (User Variants)

- Turn on rule-based user perturbations with `--robustness-k <N>` (plus optional `--robustness-*` probability knobs). The runner paraphrases, adds ambiguity, injects small talk, and proposes mild goal changes before launching extra simulations.
- Each base task reports `robustness = {k, success_vector, success_rate, stability@k}` where `stability@k = 1 − Var(success)`; use it to spot language-sensitive agents or tasks. Per-task YAML overrides (`robustness: {k_variants:5, paraphrase_prob:0.6, ...}`) keep experiments reproducible.

Together these upgrades make Grok benchmarking fairer *and* stress-tested: you can spot whether a failure came from poor planning (low ProcessScoreNorm), lexical mismatch (semantic scores), or brittle behavior under paraphrased users (low stability@k).

## Handy CLI commands

```bash
tau2 run ...                 # full Tau2 runner (same flags as script)
tau2 view --file <json>      # inspect a simulation file
tau2 check-data              # verify domain assets exist
tau2 domain <airline|...>    # print domain policy + tools
```

Happy benchmarking! Grok-specific issues? Set `TAU2_GROK_MODEL` or edit `run_benchmark.py` defaults and rerun.
