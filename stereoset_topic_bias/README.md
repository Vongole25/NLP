# Implicit Bias × Hallucination in LLMs under Ambiguous Contexts

> **NLP / AI Safety / LLM Evaluation**
> Investigating whether **implicit bias** in large language models is
> correlated with **hallucination** when the input context is ambiguous.

## TL;DR

- **Research Question:** Under ambiguous contexts, does an LLM's tendency
  to produce **implicit-biased** outputs correlate with its tendency to
  **hallucinate**?
- **Approach:**
  - Build a **hallucination-trigger dataset** designed to elicit
    bias-laden responses under ambiguity.
  - Run controlled generation with **Llama-3-8B-Instruct** (and Gemini-class
    models for comparison).
  - Evaluate using a **StereoSet-style** evaluation pipeline (`llm-ss`).
  - Pilot study: **Gender × Major** attribute pair.
  - Augment with **RAG** to see if retrieval grounding changes the
    bias-hallucination relationship.

## Repository Structure

```
stereoset_topic_bias/
├── README.md / README_KR.md
├── .gitignore
├── notebooks/
│   ├── 01_correlation_bias_hallucination.ipynb   Main experiment design
│   ├── 02_llama3_generation.ipynb                Llama-3-8B-Instruct generation
│   ├── 03_rag_experiment.ipynb                   RAG-augmented generation
│   ├── 04_big_dataset_processing.ipynb           Large-scale data pipeline
│   └── 05_llm_ss_evaluation.ipynb                StereoSet-style evaluation (`llm-ss`)
└── docs/
    ├── README.md
    ├── pilot_experiment_gender_major_en.pdf
    └── pilot_experiment_gender_major_kr.pdf
```

## Methods

### Pilot Experiment — Gender × Major
A small-scale design crossing **gender** with **academic major** to probe
how the model responds to ambiguous queries where stereotyped associations
might surface. Used as a foundation for the larger study.

### Hallucination Trigger Set
Curated prompts designed to leave key facts under-specified, so the model
must either (a) refuse, (b) produce a calibrated answer, or (c) confabulate.
We measure the rate of (c) and check whether that rate correlates with
stereotype-aligned outputs.

### LLM-SS (LLM StereoSet) Evaluation
A StereoSet-style evaluation harness running the model on both stereotypical
and anti-stereotypical continuations and measuring preference. Notebook 05
runs `llm-ss` with seed-controlled, BF16 inference.

### Models Used
- **`meta-llama/Meta-Llama-3-8B-Instruct`** (primary)
- Additional generative models for comparison (see notebook 02)

## Stack

`Python` · `PyTorch` · `transformers` (HuggingFace) · `accelerate` ·
`sentencepiece` · `scipy` · `huggingface_hub`

## Reproduction

```bash
# 1. Install dependencies
pip install transformers torch pandas numpy sentencepiece accelerate \
            huggingface_hub scipy

# 2. Set HuggingFace token via environment variable (NEVER hardcode)
export HF_TOKEN=your_hf_token_here

# 3. Run notebooks in order
jupyter notebook notebooks/01_correlation_bias_hallucination.ipynb
```

### Security note

All notebooks have been sanitized — the HF token is read from
`os.environ.get("HF_TOKEN")`. **Do not commit `.env` or hardcoded tokens.**
See `.gitignore`.

## Why this matters (research framing)

- **AI Safety / Responsible AI:** Two well-known LLM failure modes —
  *bias* and *hallucination* — are typically studied in isolation.
  This project asks whether they share a common substrate: model
  uncertainty under ambiguity.
- **Practical implication:** If correlated, **calibration-based**
  mitigations (e.g., abstention, retrieval grounding) could reduce both
  simultaneously.
- **Methodological:** Combines standard LLM-eval harnesses (StereoSet-style)
  with custom hallucination triggers, enabling correlated measurement.

## Related Work / Tools

- StereoSet (Nadeem et al., 2020) — bias measurement benchmark.
- Llama 3 (Meta AI, 2024) — primary model under study.
- `llm-ss` — StereoSet-style evaluation runner used here.

---

*Author: Sunjae Lee · B.S. Big Data Science @ Korea University Sejong*
*Part of [NLP](https://github.com/Vongole25/NLP) repo. Preparing for NLP graduate school admission (2027 Spring).*
