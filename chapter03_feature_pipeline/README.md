# Chapter 03: Feature Pipeline, Vector Database & Logical Feature Store

This chapter covers the implementation of the **Feature Pipeline** (the **F** in the FTI pattern), transforming raw unstructured documents into ML-ready artifacts for training and dense vector indices for real-time RAG[cite: 1, 2].

---

## 📑 Table of Contents

- [1. Role of the Feature Pipeline](#1-role-of-the-feature-pipeline)
  - [Dual Snapshot Architecture](#dual-snapshot-architecture)
  - [Logical Feature Store vs. Dedicated Platforms](#logical-feature-store-vs-dedicated-platforms)
- [2. Pipeline Transformations](#2-pipeline-transformations)
  - [A. Cleaning & Normalization](#a-cleaning--normalization)
  - [B. Domain-Specific Chunking](#b-domain-specific-chunking)
  - [C. Vector Embeddings](#c-vector-embeddings)
- [3. Storage & Artifact Generation](#3-storage--artifact-generation)
  - [Online Store: Vector DB (Qdrant)](#online-store-vector-db-qdrant)
  - [Offline Store: Instruction Dataset Artifacts](#offline-store-instruction-dataset-artifacts)
- [4. Implementation & Execution](#4-implementation--execution)
- [5. Directory Layout](#5-directory-layout)

---

## 1. Role of the Feature Pipeline

The Feature Pipeline bridges raw crawled text and model consumption[cite: 1, 2]. It prevents **training-serving skew** by standardizing feature calculations across training and inference workloads[cite: 1, 2].

### Dual Snapshot Architecture
The pipeline produces two distinct outputs from the same ground-truth text[cite: 1, 2]:
1. **Offline Training Snapshot:** Cleaned, structured instruction-response pairs stored as versioned JSONL artifacts for fine-tuning (LoRA/QLoRA)[cite: 1, 2].
2. **Online Inference Snapshot:** Segmented semantic chunks converted into dense vector embeddings and loaded into a Vector Database for real-time RAG[cite: 1, 2].

### Logical Feature Store vs. Dedicated Platforms
Rather than introducing expensive enterprise feature store infrastructure (e.g., Feast, Hopsworks) for unstructured text, this system operates a **Logical Feature Store**[cite: 1, 2]:
- **Point Retrieval & Online Features:** Handled via collection queries in a Vector Database (Qdrant)[cite: 1, 2].
- **Batch Datasets & Lineage:** Managed via immutable dataset artifacts versioned with metadata hashes[cite: 1, 2].

---

## 2. Pipeline Transformations

```text
       ┌────────────────────────┐
       │   Raw Data Warehouse   │ (MongoDB / JSON)
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │  Cleaning & Filtering  │ (Remove boilerplate, Markdown cleanup)
       └───────────┬────────────┘
                   │
         ┌─────────┴─────────┐
         │                   │
         ▼                   ▼
┌──────────────────┐ ┌───────────────────┐
│ Domain Chunking  │ │ Dataset Formatter │
└────────┬─────────┘ └─────────┬─────────┘
         │                     │
         ▼                     ▼
┌──────────────────┐ ┌───────────────────┐
│ Dense Embeddings │ │ Versioned JSONL   │
└────────┬─────────┘ └─────────┬─────────┘
         │                     │
         ▼                     ▼
  Vector Database       Model Artifacts
  (Online RAG Store)    (Offline Training)
```

### A. Cleaning & Normalization
- Removes noise, HTML residuals, trailing spaces, and excessive linebreaks[cite: 1].
- Strips personal signatures or tracking links while retaining source provenance URLs[cite: 1, 2].

### B. Domain-Specific Chunking
Chunking strategies vary by category[cite: 1]:
* **Articles:** Recursive character/sentence splitting with overlap (e.g., 500 chars with 50 chars overlap) to preserve context continuity[cite: 1].
* **Posts:** Kept atomic or partitioned with small boundaries to preserve single-thought coherence[cite: 1].
* **Code:** Split by function, class boundary, or logical blocks rather than arbitrary character cuts[cite: 1].

### C. Vector Embeddings
Chunks are mapped into high-dimensional latent space using dense embedding models (e.g., `BAAI/bge-small-en-v1.5` or `sentence-transformers`), outputting normalized floating-point arrays for cosine similarity search[cite: 1].

---

## 3. Storage & Artifact Generation

### Online Store: Vector DB (Qdrant)
Points are inserted with dense vector payloads including:
- `chunk_id`: Unique deterministic hash.
- `document_id`: Parent document reference.
- `category`: `posts`, `articles`, or `code`.
- `text`: Chunk content used to populate LLM prompt context during RAG.

### Offline Store: Instruction Dataset Artifacts
The training snapshot compiles cleaned records into instruction-response pairs formatted for Supervised Fine-Tuning (SFT):
```json
{
  "instruction": "Write a LinkedIn post discussing MLOps pipelines.",
  "input": "",
  "output": "Decoupling feature extraction from training avoids skew...",
  "category": "posts"
}
```

---

## 4. Implementation & Execution

### 1. Install Dependencies
```bash
pip install -r chapter03_feature_pipeline/requirements.txt
```

### 2. Run the Feature Pipeline
```bash
python -m chapter03_feature_pipeline.src.run
```

---

## 5. Directory Layout

```text
chapter03_feature_pipeline/
├── configs/
│   └── features.yaml           # Chunking and embedding configurations
├── src/
│   ├── __init__.py
│   ├── cleaners.py             # Category-specific cleaners
│   ├── chunkers.py             # Sentence and block chunking algorithms
│   ├── embeddings.py           # Dense vector embedding generator
│   ├── feature_store.py        # Logical feature store & Qdrant adapter
│   └── run.py                  # Pipeline execution runner
├── requirements.txt
└── README.md
```