# Financial Agent Evaluation Suite

1. Put the `evaluation` folder in your project root.
2. Open `run_evaluation.py`.
3. Edit only `call_financial_assistant()` to call your actual backend entry point.
4. Run:

    python -m evaluation.run_evaluation --repetitions 5

Outputs are written to `evaluation/results/`.

Measured metrics:
- success rate
- mean latency
- P50 latency
- P95 latency
- min/max latency
- latency standard deviation
- response length
- required-term coverage
- forbidden-phrase violations
- deterministic instruction-adherence proxy
- extracted final rating
- per-agent/category summaries

Note: deterministic adherence is a proxy, not a complete hallucination or semantic-quality metric.
