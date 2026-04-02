"""
param_sweep.py
--------------
Runs a parameter sweep over temperature and max_tokens settings for the
best-performing model identified during the eval phase (run_evals.py).

The goal is to find the configuration that optimises for ClearPath Health's
clinical use case: responses that are concise, complete (not truncated), and
consistently structured for downstream risk-flagging workflows.

Grid searched:
  - temperature  : [0.0, 0.2, 0.5, 0.8]  (0.0 = deterministic; 0.8 = creative)
  - max_tokens   : [128, 256, 512]

For each (temperature, max_tokens) combination, all 8 patient notes are
evaluated and the following metrics are averaged:
  - latency_ms        : round-trip API time in milliseconds
  - output_tokens     : tokens in the completion
  - summary_length    : character count of response
  - risk_flags        : bullet-point risk items identified
  - truncation_rate   : fraction of responses where finish_reason == "length"

Results are written to results/sweep_results.csv.

Usage:
    python sweeps/param_sweep.py

Requirements:
    FIREWORKS_API_KEY must be set in .env at the repo root.
    Run run_evals.py first. Set SWEEP_MODEL to the same endpoint you plan to deploy;
    the sweep optimises inference knobs for that model only — do not copy settings
    from a different model without re-running the sweep on the deployed one.
"""

import json
import os
import time
import csv
import itertools
import statistics
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv(Path(__file__).parent.parent / ".env")

FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
if not FIREWORKS_API_KEY:
    raise EnvironmentError(
        "FIREWORKS_API_KEY not found. "
        "Add it to the .env file: FIREWORKS_API_KEY=fw_xxxx"
    )

BASE_URL = "https://api.fireworks.ai/inference/v1"

# Must match the production model: sweep results apply only to this endpoint.
# After run_evals.py, set this to the model you are deploying (see eval_results.csv).
SWEEP_MODEL = "accounts/fireworks/models/mixtral-8x22b-instruct"

# Parameter grid
TEMPERATURES = [0.0, 0.2, 0.5, 0.8]
MAX_TOKENS_LIST = [128, 256, 512]

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

DATA_PATH = Path(__file__).parent.parent / "data" / "intake_notes.json"
RESULTS_DIR = Path(__file__).parent.parent / "results"
OUTPUT_CSV = RESULTS_DIR / "sweep_results.csv"

# Per-call row: one row per (patient, temperature, max_tokens) combination
FIELDNAMES_RAW = [
    "patient_id",
    "temperature",
    "max_tokens",
    "latency_ms",
    "output_tokens",
    "summary_length",
    "risk_flags",
    "finish_reason",
]

# Aggregated row: one row per (temperature, max_tokens) combination
FIELDNAMES_AGG = [
    "temperature",
    "max_tokens",
    "avg_latency_ms",
    "avg_output_tokens",
    "avg_summary_length",
    "avg_risk_flags",
    "truncation_rate",
    "n_samples",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def count_risk_flags(text: str) -> int:
    """Count bullet-point lines as a proxy for distinct risk factors named."""
    return sum(1 for line in text.splitlines() if line.strip().startswith(("-", "•", "*")))


def call_model(client: OpenAI, note: str, temperature: float, max_tokens: int) -> dict:
    """
    Single API call with specified hyperparameters.
    Returns a metrics dict; catches exceptions gracefully.
    """
    prompt = USER_PROMPT_TEMPLATE.format(note=note)
    start = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=SWEEP_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
        content = response.choices[0].message.content or ""
        return {
            "latency_ms": elapsed_ms,
            "output_tokens": response.usage.completion_tokens if response.usage else None,
            "summary_length": len(content),
            "risk_flags": count_risk_flags(content),
            "finish_reason": response.choices[0].finish_reason,
        }
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
        print(f"    [ERROR] {exc}")
        return {
            "latency_ms": elapsed_ms,
            "output_tokens": None,
            "summary_length": 0,
            "risk_flags": 0,
            "finish_reason": "error",
        }


def aggregate(raw_rows: list[dict]) -> list[dict]:
    """
    Collapse per-patient rows into per-(temperature, max_tokens) aggregate rows.
    Truncation rate = fraction of calls where finish_reason was 'length'.
    """
    from collections import defaultdict

    groups = defaultdict(list)
    for row in raw_rows:
        key = (row["temperature"], row["max_tokens"])
        groups[key].append(row)

    agg_rows = []
    for (temp, mt), rows in sorted(groups.items()):
        latencies = [r["latency_ms"] for r in rows if isinstance(r["latency_ms"], (int, float))]
        tokens = [r["output_tokens"] for r in rows if r["output_tokens"] is not None]
        lengths = [r["summary_length"] for r in rows]
        flags = [r["risk_flags"] for r in rows]
        truncated = sum(1 for r in rows if r["finish_reason"] == "length")

        agg_rows.append({
            "temperature": temp,
            "max_tokens": mt,
            "avg_latency_ms": round(statistics.mean(latencies), 1) if latencies else None,
            "avg_output_tokens": round(statistics.mean(tokens), 1) if tokens else None,
            "avg_summary_length": round(statistics.mean(lengths), 1),
            "avg_risk_flags": round(statistics.mean(flags), 2),
            "truncation_rate": round(truncated / len(rows), 3),
            "n_samples": len(rows),
        })

    return agg_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    with open(DATA_PATH, "r") as f:
        notes = json.load(f)

    client = OpenAI(api_key=FIREWORKS_API_KEY, base_url=BASE_URL)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    short_model = SWEEP_MODEL.split("/")[-1]
    total_calls = len(TEMPERATURES) * len(MAX_TOKENS_LIST) * len(notes)
    print(f"Sweep model   : {short_model}")
    print(f"Grid          : {len(TEMPERATURES)} temperatures x {len(MAX_TOKENS_LIST)} max_tokens settings")
    print(f"Notes         : {len(notes)}")
    print(f"Total API calls: {total_calls}\n")

    raw_rows = []

    for temperature, max_tokens in itertools.product(TEMPERATURES, MAX_TOKENS_LIST):
        print(f"[temp={temperature}, max_tokens={max_tokens}]")

        for note_obj in notes:
            patient_id = note_obj["patient_id"]
            print(f"  {patient_id} ...", end=" ", flush=True)

            metrics = call_model(client, note_obj["note"], temperature, max_tokens)

            row = {
                "patient_id": patient_id,
                "temperature": temperature,
                "max_tokens": max_tokens,
                **metrics,
            }
            raw_rows.append(row)
            print(
                f"done  latency={metrics['latency_ms']}ms  "
                f"tokens={metrics['output_tokens']}  "
                f"flags={metrics['risk_flags']}  "
                f"finish={metrics['finish_reason']}"
            )

    # Write raw per-call CSV
    raw_csv = RESULTS_DIR / "sweep_results_raw.csv"
    with open(raw_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES_RAW)
        writer.writeheader()
        writer.writerows(raw_rows)
    print(f"\nRaw results saved to {raw_csv}")

    # Write aggregated summary CSV
    agg_rows = aggregate(raw_rows)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES_AGG)
        writer.writeheader()
        writer.writerows(agg_rows)
    print(f"Aggregated results saved to {OUTPUT_CSV}")

    _print_summary(agg_rows)


def _print_summary(agg_rows: list[dict]):
    """Print a ranked summary table to help identify the best configuration."""
    print("\n--- Sweep Summary (ranked by avg_risk_flags desc, truncation_rate asc) ---")
    print(
        f"{'temp':>6}  {'max_tok':>7}  {'avg_lat_ms':>11}  "
        f"{'avg_tokens':>10}  {'avg_flags':>9}  {'trunc_rate':>10}"
    )
    print("-" * 70)

    # Rank: more risk flags identified is better; lower truncation rate is better
    ranked = sorted(agg_rows, key=lambda r: (-r["avg_risk_flags"], r["truncation_rate"]))
    for r in ranked:
        print(
            f"{r['temperature']:>6.1f}  {r['max_tokens']:>7}  "
            f"{r['avg_latency_ms']:>11.1f}  "
            f"{r['avg_output_tokens']:>10.1f}  "
            f"{r['avg_risk_flags']:>9.2f}  "
            f"{r['truncation_rate']:>10.3f}"
        )

    best = ranked[0]
    print(
        f"\nRecommended config: temperature={best['temperature']}, "
        f"max_tokens={best['max_tokens']}  "
        f"(avg_risk_flags={best['avg_risk_flags']}, "
        f"truncation_rate={best['truncation_rate']})"
    )


if __name__ == "__main__":
    main()
