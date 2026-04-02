"""
run_evals.py
------------
Evaluates two Fireworks AI models on synthetic patient intake notes for
ClearPath Health. For each note, both models are asked to:
  1. Summarize the note in 2-3 sentences.
  2. List the key clinical risk factors present.

Metrics captured per request:
  - latency_ms      : end-to-end API round-trip time in milliseconds
  - output_tokens   : number of tokens in the completion (from usage metadata)
  - summary_length  : character length of the returned summary
  - risk_flags      : number of bullet-point risk factors identified
  - finished        : whether the model returned a complete (non-truncated) response

Results are written to results/eval_results.csv for downstream analysis.

Usage:
    python evals/run_evals.py

Requirements:
    FIREWORKS_API_KEY must be set in the .env file at the repo root.
"""

import json
import os
import time
import csv
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI  # Fireworks exposes an OpenAI-compatible endpoint

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Load API key from .env at the repo root (one level up from evals/)
load_dotenv(Path(__file__).parent.parent / ".env")

FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
if not FIREWORKS_API_KEY:
    raise EnvironmentError(
        "FIREWORKS_API_KEY not found. "
        "Add it to the .env file: FIREWORKS_API_KEY=fw_xxxx"
    )

# Fireworks OpenAI-compatible base URL
BASE_URL = "https://api.fireworks.ai/inference/v1"

# Models to compare. These are strong general-purpose instruction models
# available on the Fireworks serverless endpoint.
MODELS = [
    "accounts/fireworks/models/llama-v3p1-8b-instruct",
    "accounts/fireworks/models/llama-v3p1-70b-instruct",
]

# Prompt template — intentionally detailed to elicit structured clinical output.
SYSTEM_PROMPT = (
    "You are a clinical documentation assistant. Your task is to help "
    "healthcare providers quickly understand patient intake notes. "
    "Always be concise, accurate, and use plain clinical language."
)

USER_PROMPT_TEMPLATE = """Please review the following patient intake note and provide:

1. SUMMARY: A 2-3 sentence summary of the patient's presentation.
2. RISK FLAGS: A bulleted list of the key clinical risk factors or urgent concerns identified in the note.

Patient Note:
{note}
"""

# Paths
DATA_PATH = Path(__file__).parent.parent / "data" / "intake_notes.json"
RESULTS_DIR = Path(__file__).parent.parent / "results"
OUTPUT_CSV = RESULTS_DIR / "eval_results.csv"

# CSV columns
FIELDNAMES = [
    "patient_id",
    "age",
    "sex",
    "model",
    "latency_ms",
    "output_tokens",
    "prompt_tokens",
    "summary_length",
    "risk_flags",
    "finish_reason",
    "response_text",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def count_risk_flags(text: str) -> int:
    """Count bullet-point lines in the response as a proxy for risk flags."""
    return sum(1 for line in text.splitlines() if line.strip().startswith(("-", "•", "*")))


def call_model(client: OpenAI, model: str, note: str) -> dict:
    """
    Send a single note to the specified model and return a dict of metrics.
    Catches exceptions so one failed call doesn't abort the entire run.
    """
    prompt = USER_PROMPT_TEMPLATE.format(note=note)

    start_time = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,   # Low temperature for clinical consistency
            max_tokens=512,
        )
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 1)

        content = response.choices[0].message.content or ""
        finish_reason = response.choices[0].finish_reason
        output_tokens = response.usage.completion_tokens if response.usage else None
        prompt_tokens = response.usage.prompt_tokens if response.usage else None

        return {
            "latency_ms": elapsed_ms,
            "output_tokens": output_tokens,
            "prompt_tokens": prompt_tokens,
            "summary_length": len(content),
            "risk_flags": count_risk_flags(content),
            "finish_reason": finish_reason,
            "response_text": content.replace("\n", " | "),  # flatten for CSV
        }

    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 1)
        print(f"  [ERROR] {model} failed: {exc}")
        return {
            "latency_ms": elapsed_ms,
            "output_tokens": None,
            "prompt_tokens": None,
            "summary_length": 0,
            "risk_flags": 0,
            "finish_reason": "error",
            "response_text": f"ERROR: {exc}",
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Load patient notes
    with open(DATA_PATH, "r") as f:
        notes = json.load(f)

    print(f"Loaded {len(notes)} patient notes from {DATA_PATH}")
    print(f"Evaluating {len(MODELS)} models: {', '.join(MODELS)}\n")

    # Initialise Fireworks client (OpenAI-compatible)
    client = OpenAI(api_key=FIREWORKS_API_KEY, base_url=BASE_URL)

    # Ensure results directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    for note_obj in notes:
        patient_id = note_obj["patient_id"]
        note_text = note_obj["note"]

        for model in MODELS:
            short_model = model.split("/")[-1]  # human-readable label
            print(f"  [{patient_id}] {short_model} ...", end=" ", flush=True)

            metrics = call_model(client, model, note_text)

            row = {
                "patient_id": patient_id,
                "age": note_obj["age"],
                "sex": note_obj["sex"],
                "model": short_model,
                **metrics,
            }
            rows.append(row)
            print(
                f"done  latency={metrics['latency_ms']}ms  "
                f"tokens={metrics['output_tokens']}  "
                f"risk_flags={metrics['risk_flags']}"
            )

    # Write CSV
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults saved to {OUTPUT_CSV}")
    _print_summary(rows)


def _print_summary(rows: list[dict]):
    """Print a quick aggregate summary to stdout after the run."""
    print("\n--- Aggregate Summary ---")
    from collections import defaultdict
    import statistics

    by_model = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)

    for model, model_rows in by_model.items():
        latencies = [r["latency_ms"] for r in model_rows if isinstance(r["latency_ms"], (int, float))]
        tokens = [r["output_tokens"] for r in model_rows if r["output_tokens"] is not None]
        flags = [r["risk_flags"] for r in model_rows if isinstance(r["risk_flags"], int)]

        print(f"\nModel: {model}")
        print(f"  Avg latency  : {statistics.mean(latencies):.1f} ms")
        print(f"  Avg out tokens: {statistics.mean(tokens):.1f}" if tokens else "  Avg out tokens: N/A")
        print(f"  Avg risk flags: {statistics.mean(flags):.1f}" if flags else "  Avg risk flags: N/A")


if __name__ == "__main__":
    main()
