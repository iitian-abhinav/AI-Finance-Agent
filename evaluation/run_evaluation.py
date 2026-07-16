from __future__ import annotations
import argparse, csv, json, os, time
from datetime import datetime, timezone

try:
    from .benchmark_cases import BENCHMARK_CASES
    from .evaluator import evaluate_response, result_to_dict, summarize_results
except ImportError:
    from benchmark_cases import BENCHMARK_CASES
    from evaluator import evaluate_response, result_to_dict, summarize_results

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results")

AGENT_MAP = {
    "sec_filing": "sec_filing",
    "financial_statement": "financial_statement",
    "market_data": "market_data",
    "news": "news_intelligence",
    "comparative": "comparative_analysis",
    "investment_decision": "investment_decision",
}

def call_financial_assistant(
    prompt: str,
    force_agent: str | None = None,
) -> str:
    import backend as bk

    conversation_id = bk.create_conversation(
        title="Evaluation Run",
        mode="chat",
    )

    stream = bk.stream_chat(
        conversation_id=conversation_id,
        user_message=prompt,
        force_agent=force_agent,
    )

    response_parts = []

    for chunk in stream:
        if chunk is not None:
            response_parts.append(str(chunk))

    return "".join(response_parts)

def run_single_case(case, repetition):
    started = time.perf_counter()
    response, error = "", None
    try:
        agent_key = AGENT_MAP.get(case["category"])

        if not agent_key:
            raise ValueError(
                f"No agent mapping configured for category: {case['category']}")

        response = call_financial_assistant(
            prompt=case["prompt"],
            force_agent=agent_key,
            )
        if not isinstance(response, str):
            response = str(response)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    latency_ms = (time.perf_counter()-started)*1000
    result = evaluate_response(case, response, latency_ms, repetition, error)
    print(
    f"{case['case_id']} "
    f"rep={repetition} "
    f"success={result.success} "
    f"latency={result.latency_ms}ms "
    f"adherence={result.instruction_adherence_score}"
    )

    if result.error:
        print(f"  ERROR: {result.error}")
    return result

def save_results(results, summary):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    json_path = os.path.join(OUTPUT_DIR, "evaluation.json")
    csv_path = os.path.join(OUTPUT_DIR, "evaluation.csv")
    summary_path = os.path.join(OUTPUT_DIR, "summary.json")

    # Save detailed JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            [result_to_dict(r) for r in results],
            f,
            indent=2,
            ensure_ascii=False,
        )

    # Save CSV
    fields = [
        "case_id",
        "category",
        "ticker",
        "repetition",
        "success",
        "status",
        "error",
        "latency_ms",
        "response_chars",
        "response_words",
        "response_paragraphs",
        "required_term_coverage_pct",
        "structure_score",
        "citation_score",
        "evidence_score",
        "missing_data_score",
        "hallucination_penalty",
        "instruction_adherence_score",
        "forbidden_violations",
        "rating",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for r in results:
            d = result_to_dict(r)
            writer.writerow({k: d.get(k) for k in fields})

    # Save summary
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nResults overwritten successfully.")
    print(json_path)
    print(csv_path)
    print(summary_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()

    if args.repetitions < 1:
        raise ValueError("--repetitions must be >= 1")

    results = [
        run_single_case(case, rep)
        for case in BENCHMARK_CASES
        for rep in range(1, args.repetitions + 1)
    ]

    summary = summarize_results(results)

    # Print summary
    print("\n================ Evaluation Summary ================\n")
    print(json.dumps(summary, indent=2))

    # Save files
    save_results(results, summary)

    print("\nResults successfully saved.")
    
if __name__ == "__main__":
    main()
