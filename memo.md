# Recommendation: Intake Note Summarization Models (Fireworks AI)

**To:** ClearPath Health, Technical Lead  
**From:** Solutions Architect, Fireworks AI  
**Date:** April 2, 2026  
**Re:** Model choice and inference settings for patient intake summarization and risk-flag extraction

---

## Executive summary

We evaluated **Mixtral 8×22B Instruct** and **DeepSeek V3.2** on eight synthetic intake notes using the same prompt: a short clinical summary plus a bulleted list of risk factors. **Mixtral is the better fit for production** on the metrics that matter operationally: it was **~2.5× faster** (mean latency ~2.3s vs ~5.8s per note) and returned **complete responses on every note** (zero API responses stopped for length), while DeepSeek hit the output cap **5 of 8** times at the same `max_tokens=512` used in the eval.

We then ran a **parameter sweep on the same model we recommend for production — Mixtral 8×22B Instruct** (temperature × `max_tokens` over the same eight notes). **Every `max_tokens=128` configuration had a 100% truncation rate** (outputs cut before the list was usable). **`max_tokens=256` still showed meaningful truncation** (12.5–37.5% of notes per cell). **`max_tokens=512` produced zero truncation across all four temperatures tested.** At 512 tokens, **temperature = 0.5** achieved the **highest mean risk-flag count (5.88)** among that row.

**Recommendation:** Deploy **`accounts/fireworks/models/mixtral-8x22b-instruct`** with **`temperature = 0.5`** and **`max_tokens = 512`**, matching both the eval winner and the Mixtral-specific sweep optimum.

---

## Model comparison (eval)

Eight notes, two models, identical prompts and `max_tokens=512` / `temperature=0.2` (eval script defaults). “Risk flags” = count of bulleted lines in the model output. “Truncated” = API `finish_reason` = `length`.

| Model (Fireworks ID) | Mean latency | Mean output tokens | Mean risk flags | Truncated (notes) |
|----------------------|-------------:|-------------------:|----------------:|------------------:|
| `mixtral-8x22b-instruct` | **2,293 ms** | 240 | 6.0 | **0 / 8** |
| `deepseek-v3p2` | 5,781 ms | 483 | 6.1 | **5 / 8** |

**Plain language:** DeepSeek listed slightly more bullets on average, but often because generations ran long and were cut off—unsafe for a workflow that depends on a complete “RISK FLAGS” section. Mixtral stayed inside the cap every time, with much lower latency—better for **latency-sensitive or synchronous** use, and for **predictable parsing** downstream.

---

## Parameter sweep results (inference grid)

Sweep used **Mixtral 8×22B Instruct** (`mixtral-8x22b-instruct`) and the same eight notes; temperatures **{0.0, 0.2, 0.5, 0.8}** × **`max_tokens` {128, 256, 512}**. Below: mean risk flags, truncation rate (fraction of notes truncated), and mean latency.

| Temp | `max_tokens` | Mean risk flags | Truncation rate | Mean latency (ms) |
|-----:|---------------:|----------------:|----------------:|------------------:|
| 0.0 | 128 | 1.00 | 100% | 2,149 |
| 0.2 | 128 | 0.62 | 100% | 2,046 |
| 0.5 | 128 | 1.00 | 100% | 1,790 |
| 0.8 | 128 | 1.00 | 100% | 1,893 |
| 0.0 | 256 | 5.25 | 25% | 3,125 |
| 0.2 | 256 | 4.88 | 12.5% | 3,176 |
| 0.5 | 256 | 5.25 | 37.5% | 3,286 |
| 0.8 | 256 | 4.88 | 25% | 3,080 |
| 0.0 | 512 | 5.75 | **0%** | 3,571 |
| 0.2 | 512 | 5.62 | **0%** | 3,676 |
| **0.5** | **512** | **5.88** | **0%** | **3,658** |
| 0.8 | 512 | 5.25 | **0%** | 3,397 |

**Best 512-token configuration:** **temperature = 0.5**, **max_tokens = 512** (highest mean risk flags in the 512-token band, with **no truncations** in this run). **128-token budgets are unusable here** (100% truncation every time). **256 is risky—still truncates** on some notes depending on temperature—so **512 is the setting we standardise on** for reliable risk-flag sections.

---

## Final recommendation (with rationale)

| Setting | Value |
|--------|-------|
| **Model** | `accounts/fireworks/models/mixtral-8x22b-instruct` |
| **Temperature** | **0.5** (best mean risk-flag yield among Mixtral 512-token configs in the sweep) |
| **max_tokens** | **512** (only band with zero truncation across all temperatures tested; 256 remained partially truncated) |

**Why Mixtral over DeepSeek here:** Faster responses and **no length truncation in the head-to-head eval** outweigh a marginal difference in average bullet count, given the importance of **complete, parseable risk-flag sections** and **operational latency**.

**Why 0.5 and 512:** Chosen from a sweep **on Mixtral itself**, so inference tuning matches the deployed model.

---

## Follow-on question before architecture sign-off

**Can we validate this configuration on a holdout set of real, de-identified intake notes** (with optional blinded clinical review of summaries and flags) **to confirm the synthetic-note sweep transfers to production language and complexity?**

---

*Synthetic notes only; validate on representative de-identified production notes and clinical review before go-live.*
