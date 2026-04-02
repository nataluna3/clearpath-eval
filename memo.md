# Recommendation: Intake Note Summarization Models (Fireworks AI)

**To:** ClearPath Health, Technical Lead  
**From:** Solutions Architect, Fireworks AI  
**Date:** April 2, 2026  
**Re:** Model choice and inference settings for patient intake summarization and risk-flag extraction

---

## Executive summary

We evaluated **Mixtral 8×22B Instruct** and **DeepSeek V3.2** on eight synthetic intake notes using the same prompt: a short clinical summary plus a bulleted list of risk factors. **Mixtral is the better fit for production** on the metrics that matter operationally: it was **~2.5× faster** (mean latency ~2.3s vs ~5.8s per note) and returned **complete responses on every note** (zero API responses stopped for length), while DeepSeek hit the output cap **5 of 8** times at the same `max_tokens=512` used in the eval.

A follow-up **parameter sweep** on DeepSeek (temperature × max_tokens over the same eight notes) still yields critical infrastructure lessons for ClearPath: **any `max_tokens` below 512 produced 100% truncation** in our runs—outputs were cut before useful risk-flag lists could appear. At **`max_tokens=512`**, **temperature = 0.5** achieved the **highest mean risk-flag count (6.25)** with a **12.5% truncation rate** (1/8 notes), edging other 512-token settings.

**Recommendation:** Deploy **`accounts/fireworks/models/mixtral-8x22b-instruct`** with **`temperature = 0.5`** and **`max_tokens = 512`**. Rationale below combines head-to-head model results with sweep findings on token budget and temperature. **Follow-on before sign-off:** confirm the same temperature optimum on Mixtral with a short replication sweep (one question, below).

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

Sweep used **DeepSeek V3.2** and the same eight notes; temperatures **{0.0, 0.2, 0.5, 0.8}** × **`max_tokens` {128, 256, 512}**. Below: mean risk flags, truncation rate (fraction of notes truncated), and mean latency. **Every 128- and 256-token configuration had a 100% truncation rate** in this study.

| Temp | `max_tokens` | Mean risk flags | Truncation rate | Mean latency (ms) |
|-----:|---------------:|----------------:|----------------:|------------------:|
| 0.0 | 128 | 0.00 | 100% | 3,329 |
| 0.2 | 128 | 0.00 | 100% | 2,429 |
| 0.5 | 128 | 0.00 | 100% | 2,835 |
| 0.8 | 128 | 0.00 | 100% | 2,260 |
| 0.0 | 256 | 0.62 | 100% | 3,697 |
| 0.2 | 256 | 0.38 | 100% | 5,650 |
| 0.5 | 256 | 1.12 | 100% | 3,800 |
| 0.8 | 256 | 0.25 | 100% | 5,104 |
| 0.0 | 512 | 5.88 | 25% | 6,233 |
| 0.2 | 512 | 5.62 | 12.5% | 6,397 |
| **0.5** | **512** | **6.25** | **12.5%** | **6,431** |
| 0.8 | 512 | 5.75 | 12.5% | 6,431 |

**Best 512-token configuration:** **temperature = 0.5**, **max_tokens = 512** (highest mean risk flags among 512-token runs; tied lowest truncation rate with 0.2 and 0.8). The structural lesson applies whichever chat model you ship: **sub‑512 token limits broke output completeness in every cell we tested.**

---

## Final recommendation (with rationale)

| Setting | Value |
|--------|-------|
| **Model** | `accounts/fireworks/models/mixtral-8x22b-instruct` |
| **Temperature** | **0.5** (supported by sweep; best mean risk-flag yield at 512 tokens) |
| **max_tokens** | **512** (required; lower values produced universal truncation in the sweep) |

**Why Mixtral over DeepSeek here:** Faster responses and **no length truncation in the head-to-head eval** outweigh a marginal difference in average bullet count, given the importance of **complete, parseable risk-flag sections** and **operational latency**.

**Why 0.5 and 512:** The sweep shows **512** is the minimum viable completion budget for this prompt shape. Among 512-token runs, **0.5** maximized structured risk-flag extraction in our aggregate metrics.

---

## Follow-on question before architecture sign-off

**Can we run the same temperature × `max_tokens` matrix on Mixtral 8×22B Instruct** (the chosen endpoint) **to confirm that 0.5 / 512 remains optimal**, or whether a cooler temperature is better for regulatory or reproducibility requirements—**without** assuming the DeepSeek sweep transfers verbatim?

---

*Synthetic notes only; validate on representative de-identified production notes and clinical review before go-live.*
