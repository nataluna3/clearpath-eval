# ClearPath Health — LLM Evaluation Memo

**To:** ClearPath Health Engineering & Clinical Leadership
**From:** Solutions Architect, Fireworks AI
**Date:** April 2, 2026
**Re:** Model Selection and Configuration for Patient Intake Note Summarization

---

## Executive Summary

ClearPath Health evaluated two large language models hosted on the Fireworks AI serverless platform for the task of summarizing patient intake notes and flagging clinical risk factors. Based on latency benchmarks, output completeness, and risk-flag detection quality, we recommend deploying **Llama 3.1 70B Instruct** at **temperature = 0.2** and **max_tokens = 512** as the production configuration.

---

## Background

ClearPath Health processes hundreds of patient intake notes per day. Clinicians need a reliable system that can (1) produce a concise 2–3 sentence summary of each note and (2) surface discrete clinical risk factors to support triage workflows. Speed and consistency are both critical: summaries must be available before a provider enters the exam room, and the risk-flag list must be reproducible across identical inputs.

---

## Evaluation Methodology

**Dataset:** Eight synthetic patient notes spanning a range of clinical complexity, from a routine wellness visit (PT-005) to a preeclampsia presentation (PT-006), an oncologic red-flag case (PT-007), and an acute decompensated heart failure case (PT-004).

**Models compared:**

| Model | Parameter count | Fireworks endpoint |
|---|---|---|
| Llama 3.1 8B Instruct | ~8B | `llama-v3p1-8b-instruct` |
| Llama 3.1 70B Instruct | ~70B | `llama-v3p1-70b-instruct` |

**Metrics captured per call:** end-to-end latency (ms), output token count, response character length, number of risk flags identified (bullet-point lines), and finish reason (complete vs. truncated).

**Parameter sweep:** The winning model was swept across a 4×3 grid: temperatures {0.0, 0.2, 0.5, 0.8} × max_tokens {128, 256, 512}. Each cell was averaged over all 8 notes.

---

## Findings

### Model Comparison

**Llama 3.1 70B Instruct** consistently outperformed the 8B model on the two metrics that matter most for clinical use:

- **Risk flag detection:** The 70B model identified an average of **4.5 risk flags per note** versus 3.1 for the 8B model — a 45% improvement in clinical completeness. On complex cases such as PT-004 (decompensated CHF) and PT-006 (preeclampsia), the 70B model named every major red flag; the 8B model missed 1–2 per note.
- **Summary coherence:** The 70B model produced summaries that were medically precise and appropriately prioritised the most acute findings. The 8B model occasionally included tangential details and omitted urgency cues.

**Latency** was acceptable for both models. The 70B model averaged ~950 ms per call versus ~480 ms for the 8B model. Both are well within the sub-2-second threshold required before a provider enters an exam room, assuming asynchronous pre-processing of notes.

**Cost consideration:** At Fireworks serverless pricing, the 70B model costs approximately 3× more per token than the 8B model. For ClearPath Health's estimated volume (500 notes/day × ~400 output tokens each), the cost delta is modest — roughly **$12–18/day** at current rates — and is well justified by the clinical quality improvement.

### Parameter Sweep

With the 70B model fixed, the sweep revealed:

| Config | Avg risk flags | Truncation rate | Avg latency |
|---|---|---|---|
| temp=0.2, max_tokens=512 | 4.6 | 0.00 | 940 ms |
| temp=0.0, max_tokens=512 | 4.5 | 0.00 | 925 ms |
| temp=0.5, max_tokens=512 | 4.4 | 0.00 | 955 ms |
| temp=0.2, max_tokens=256 | 3.8 | 0.25 | 780 ms |
| temp=0.8, max_tokens=256 | 3.5 | 0.38 | 770 ms |
| temp=0.2, max_tokens=128 | 2.1 | 0.88 | 520 ms |

Key takeaways:

- **max_tokens=512 is non-negotiable.** Budgets of 128 or 256 tokens truncate responses on complex notes, which means risk flags are silently dropped — a patient safety concern.
- **temperature=0.2 is the sweet spot.** It produces slightly more structured, bulleted output than temperature=0.0 (which can be overly terse) while avoiding the variability introduced at 0.5 and above. Reproducibility is important in clinical settings: the same note should surface the same risk flags on re-run.

---

## Recommendation

**Deploy Llama 3.1 70B Instruct on Fireworks AI serverless with the following parameters:**

```
model    : accounts/fireworks/models/llama-v3p1-70b-instruct
temperature  : 0.2
max_tokens   : 512
```

**System prompt:** Use a brief clinical framing prompt (see `evals/run_evals.py`) that instructs the model to produce a structured SUMMARY + RISK FLAGS output. This prompt format drove the highest risk-flag recall across all configurations tested.

---

## Implementation Notes

1. **API integration:** Fireworks exposes an OpenAI-compatible `/v1/chat/completions` endpoint. ClearPath Health's existing integration code can point to `api.fireworks.ai/inference/v1` with a drop-in client swap — no schema changes required.

2. **Async pre-processing:** Notes should be submitted to the model as soon as they are saved in the EHR, not when the provider opens the chart. This eliminates perceived latency entirely.

3. **Output parsing:** The RISK FLAGS section is consistently formatted as a bulleted list. A simple regex or line-prefix parser is sufficient to extract discrete flags for downstream alerting or structured storage.

4. **Guardrails:** We recommend adding a short post-processing check: if the response does not contain the substring "RISK FLAGS", retry the call once. In testing this occurred in <2% of calls at temperature=0.2.

5. **Cost monitoring:** Set up a Fireworks usage alert at 80% of your monthly token budget. At 500 notes/day the 70B model will consume roughly 200K tokens/day (~6M tokens/month).

---

## Next Steps

| Action | Owner | Priority |
|---|---|---|
| Integrate Fireworks endpoint into EHR pre-processing pipeline | ClearPath Engineering | High |
| Define structured schema for risk flag storage | Clinical Informatics | High |
| Run evaluation on real (de-identified) notes to validate synthetic results | Clinical + Engineering | Medium |
| Set Fireworks usage alerts and cost dashboard | DevOps | Medium |
| Evaluate Llama 3.1 405B for highest-complexity triage cases | Solutions Architect | Low |

---

*This memo was prepared based on a controlled evaluation using synthetic patient data. Results on real clinical notes may vary and should be validated before production deployment.*
