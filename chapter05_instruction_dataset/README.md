# Chapter 05: The Instruction Dataset Pipeline

This chapter implements the **Instruction Dataset Pipeline** under the Feature Pipeline umbrella. It extracts cleaned documents from the logical feature store (Qdrant), transforms unstructured text into structured instruction-response pairs for Supervised Fine-Tuning (SFT), curates data quality, and exports versioned dataset artifacts.

---

## 📑 Table of Contents

- [1. Role of Instruction Datasets in Fine-Tuning](#1-role-of-instruction-datasets-in-fine-tuning)
  - [From Plain Text to Supervised Pairs](#from-plain-text-to-supervised-pairs)
  - [Persona & Style Transfer Alignment](#persona--style-transfer-alignment)
- [2. Pipeline Architecture](#2-pipeline-architecture)
  - [Feature Store Integration (Snapshot 1)](#feature-store-integration-snapshot-1)
  - [Synthetic Generation & Self-Instruct](#synthetic-generation--self-instruct)
  - [Data Curation & Filtering](#data-curation--filtering)
- [3. Target Dataset Formats](#3-target-dataset-formats)
  - [Alpaca Format](#alpaca-format)
  - [ChatML / Conversational Format](#chatml--conversational-format)
- [4. Implementation & Execution](#4-implementation--execution)
- [5. Directory Layout](#5-directory-layout)

---

## 1. Role of Instruction Datasets in Fine-Tuning

Raw internet text is effective for pre-training, but adapting a Large Language Model to act as a responsive persona requires **instruction-tuning**.

### From Plain Text to Supervised Pairs
Supervised Fine-Tuning (SFT) optimizes models to map queries to outputs directly:
$$\mathcal{L}_{\text{SFT}}(\theta) = -\sum_{t=1}^{T} \log P_\theta(y_t \mid x, y_{<t})$$
The loss is computed strictly over target tokens $y$, while prompt tokens $x$ are masked out during backpropagation.

### Persona & Style Transfer Alignment
To make the LLM Twin replicate your specific tone:
* **Input (Instruction & Context):** The task prompt and surrounding context framing the request.
* **Target Output:** Your actual writing (retrieved from articles, posts, and repositories) representing the ground-truth voice.

---

## 2. Pipeline Architecture

```text
       ┌────────────────────────┐
       │ Qdrant: Cleaned Docs   │ (Snapshot 1: Articles, Posts, Code)
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │  Instruction Generator │ (Self-Instruct / Prompt Synthesis)
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │   Quality Curator      │ (Length filters, toxicity, deduplication)
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │   Artifact Registry    │ (data/artifacts/datasets/instruct_sft.jsonl)
       └────────────────────────┘
```

### Feature Store Integration (Snapshot 1)
Instead of querying raw MongoDB databases directly, this pipeline consumes the **Cleaned Documents** stored in the Qdrant metadata index during Chapter 04. This preserves architectural boundaries and avoids pipeline coupling.

### Synthetic Generation & Self-Instruct
When raw text lacks explicit user queries, an LLM (or deterministic prompt synthesizer) formulates realistic instructions that naturally prompt the target excerpt.

### Data Curation & Filtering
High data quality outperforms sheer volume in modern fine-tuning:
1. **Length Validation:** Discards entries that are too short to capture stylistic nuance or exceed token thresholds.
2. **Boilerplate Scrubbing:** Strips residual platform markers, headers, and social noise.
3. **Deduplication:** Hash-based checks ensure no duplicate samples distort loss gradients.

---

## 3. Target Dataset Formats

### Alpaca Format
```json
{
  "instruction": "Explain why decoupling pipelines prevents training-serving skew.",
  "input": "",
  "output": "Decoupling feature stores from model inference avoids skew and enables modularity.",
  "system": "You are Oscar's LLM Twin, an AI system architect."
}
```

### ChatML / Conversational Format
```json
{
  "messages": [
    {"role": "system", "content": "You are Oscar's LLM Twin."},
    {"role": "user", "content": "Explain why decoupling pipelines prevents training-serving skew."},
    {"role": "assistant", "content": "Decoupling feature stores from model inference avoids skew..."}
  ]
}
```

---

## 4. Implementation & Execution

```bash
python -m chapter05_instruction_dataset.src.run
```

---

## 5. Directory Layout

```text
chapter05_instruction_dataset/
├── configs/
│   └── dataset.yaml           # Generation and filtering thresholds
├── src/
│   ├── __init__.py
│   ├── models.py              # Pydantic schemas (InstructionRecord, Formats)
│   ├── prompts.py             # Prompt synthesis templates
│   ├── generator.py           # Instruction-pair synthesis engine
│   ├── curator.py             # Deduplication and quality filtering
│   └── run.py                 # Pipeline orchestrator
├── requirements.txt
└── README.md
```
