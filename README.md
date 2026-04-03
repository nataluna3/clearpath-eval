# ClearPath Health – LLM Evaluation (Fireworks AI)

This repository is a **take-home–style evaluation** for a healthcare scenario: **ClearPath Health** wants to use large language models to **summarize patient intake notes** and **surface clinical risk factors**. It compares two Fireworks-hosted chat models on synthetic notes, runs a **hyperparameter sweep** on the recommended production model, and documents a **customer-facing memo** with findings.

---

## Quick setup (under ~5 minutes)

**Prerequisites:** Python **3.10+** and a [Fireworks AI](https://fireworks.ai) API key.

1. **Clone and enter the repo**

   ```bash
   git clone <repository-url>
   cd clearpath-eval
   ```

2. **Create a virtual environment (recommended)**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure your API key** (never commit this file)

   Create a file named `.env` in the **project root** with:

   ```bash
   FIREWORKS_API_KEY=your_key_here
   ```

   Replace `your_key_here` with your real key from the Fireworks dashboard. The repo’s `.gitignore` excludes `.env` so it stays local.

---

## Project layout

| Path | Purpose |
|------|---------|
| **`data/`** | Input data. **`intake_notes.json`** – eight synthetic patient intake notes used as eval/sweep inputs. |
| **`evals/`** | **`run_evals.py`** – sends each note to two models, records latency, tokens, risk-flag count, and truncation; writes **`results/eval_results.csv`**. |
| **`sweeps/`** | **`param_sweep.py`** – grid over `temperature` × `max_tokens` for the **deployed** model (Mixtral); writes **`results/sweep_results.csv`** (aggregated) and **`results/sweep_results_raw.csv`** (per-call). |
| **`results/`** | CSV outputs from eval and sweep runs (safe to regenerate; may be committed for reproducibility). |
| **`memo.md`** | One-page recommendation memo for ClearPath Health’s technical lead, aligned with the CSV results. |
| **`requirements.txt`** | Python packages (`openai` client for Fireworks’ OpenAI-compatible API, `python-dotenv`, etc.). |
| **`.gitignore`** | Keeps secrets and common junk out of git (e.g. `.env`). |

---

## Run the evaluation

From the project root:

```bash
python3 evals/run_evals.py
```

**What it does:** Loads **`data/intake_notes.json`**, calls the Fireworks chat API for **Mixtral 8×22B Instruct** and **DeepSeek V3.2** with the same system/user prompt (summary + bulleted risk flags), and prints progress per patient/model.

**Expected output:**

- Console lines per `(patient_id, model)` with **latency (ms)**, **output tokens**, and **risk flag count**.
- A short **aggregate summary** (mean latency, tokens, risk flags) per model.
- **`results/eval_results.csv`** – one row per call with `latency_ms`, `output_tokens`, `risk_flags`, `finish_reason`, and a flattened `response_text` column.

---

## Run the parameter sweep

From the project root:

```bash
python3 sweeps/param_sweep.py
```

**What it does:** For the model set in `SWEEP_MODEL` inside **`sweeps/param_sweep.py`** (intended to match **production**, currently **Mixtral**), it runs a grid: temperatures `{0.0, 0.2, 0.5, 0.8}` × `max_tokens` `{128, 256, 512}` across all eight notes (96 API calls).

**Expected output:**

- Console log of each `(temperature, max_tokens, patient_id)` with latency, tokens, flags, and `finish` reason.
- **`results/sweep_results_raw.csv`** – one row per API call.
- **`results/sweep_results.csv`** – **aggregated** metrics per grid cell (mean latency, mean risk flags, **truncation rate**, etc.) plus a ranked summary printed to the terminal.

---

## Key finding (one line)

**On this synthetic set, Mixtral 8×22B was ~2.5× faster than DeepSeek V3.2 with no length truncations in the eval (vs five of eight for DeepSeek at the same cap), and a Mixtral-only sweep supported `temperature = 0.5` and `max_tokens = 512` for strong risk-flag coverage with zero truncation in the 512-token band.** *(See `memo.md` and `results/*.csv` for exact numbers.)*

---

*Disclaimer: Notes are synthetic; any production use requires validation on real, appropriately governed data and clinical workflows.*
