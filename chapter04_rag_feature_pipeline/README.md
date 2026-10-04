# Chapter 04: RAG Feature Pipeline & Vector Databases

This chapter implements the **RAG Feature Pipeline**, the ingestion backbone responsible for transforming raw unstructured content into vectorized representations indexed in a logical feature store (Qdrant & artifacts).

---

## 📑 Table of Contents

- [1. Fundamentals of RAG](#1-fundamentals-of-rag)
  - [Core Concept: Retrieval-Augmented Generation](#core-concept-retrieval-augmented-generation)
  - [The Two Core Problems Solved](#the-two-core-problems-solved)
- [2. The Vanilla RAG Framework](#2-the-vanilla-rag-framework)
  - [The Three Decoupled Pipelines](#the-three-decoupled-pipelines)
  - [Vector Embeddings & Similarity](#vector-embeddings--similarity)
  - [Vector Databases & ANN Algorithms](#vector-databases--ann-algorithms)
- [3. Advanced RAG Overview](#3-advanced-rag-overview)
  - [Pre-Retrieval Optimizations](#pre-retrieval-optimizations)
  - [Retrieval Optimizations](#retrieval-optimizations)
  - [Post-Retrieval Optimizations](#post-retrieval-optimizations)
- [4. LLM Twin RAG Pipeline Architecture](#4-llm-twin-rag-pipeline-architecture)
  - [Batch vs. Streaming Pipeline Design](#batch-vs-streaming-pipeline-design)
  - [Change Data Capture (CDC)](#change-data-capture-cdc)
  - [Dual Snapshot Storage Pattern](#dual-snapshot-storage-pattern)
- [5. Software Design Patterns Implemented](#5-software-design-patterns-implemented)
  - [Domain-Driven Design (DDD) & OVM](#domain-driven-design-ddd--ovm)
  - [Abstract Factory & Strategy Patterns](#abstract-factory--strategy-patterns)
  - [Singleton Pattern](#singleton-pattern)
- [6. Directory Layout & Verification](#6-directory-layout--verification)

---

## 1. Fundamentals of RAG

Retrieval-Augmented Generation (RAG) injects dynamic, domain-specific, or proprietary external knowledge into Large Language Model prompts at inference time, turning the LLM into a pure reasoning engine conditioned on retrieved facts.

### Core Concept: Retrieval-Augmented Generation
* **Retrieval:** Fetch semantically relevant context fragments from persistent external storage.
* **Augmented:** Inject retrieved context fragments into the model prompt alongside user instructions.
* **Generation:** Condition LLM response generation strictly on the augmented prompt.

### The Two Core Problems Solved
1. **Hallucinations:** Enforces factual grounding against verified context chunks, mitigating probabilistic fabulation.
2. **Knowledge Cutoffs & Private Data:** Bypasses costly, continuous pre-training cycles by updating external vector stores incrementally.

---

## 2. The Vanilla RAG Framework

### The Three Decoupled Pipelines
```text
┌────────────────────────┐      ┌────────────────────────┐      ┌────────────────────────┐
│   Ingestion Pipeline   │ ──>  │   Retrieval Pipeline   │ ──>  │  Generation Pipeline   │
│ (Extract, Chunk, Embed)│      │  (Vector / ANN Search) │      │  (Augment & Complete)  │
└────────────────────────┘      └────────────────────────┘      └────────────────────────┘
```

### Vector Embeddings & Similarity
Dense embeddings represent arbitrary digital objects (words, sentences, code, images) as dense numerical vectors in a continuous geometric space $\mathbb{R}^d$. 

Semantic similarity is evaluated via metric distances, predominantly **Cosine Similarity**:
$$\text{Cosine Similarity}(A, B) = \frac{A \cdot B}{\Vert{}A\Vert{} \Vert{}B\Vert{}} = \frac{\sum_{i=1}^n A_i B_i}{\sqrt{\sum_{i=1}^n A_i^2} \sqrt{\sum_{i=1}^n B_i^2}}$$

### Vector Databases & ANN Algorithms
Because exhaustive brute-force $k$-Nearest Neighbors ($k$-NN) incurs $\mathcal{O}(N \cdot d)$ complexity, vector databases leverage Approximate Nearest Neighbor (ANN) index structures:
* **HNSW (Hierarchical Navigable Small World):** Multi-layer graph traversal with logarithmic search complexity.
* **Product Quantization (PQ):** Sub-vector decomposition and centroid quantization for high-density memory compression.
* **Locality-Sensitive Hashing (LSH):** Probabilistic bucket mapping ensuring close points hash into identical partitions.

---

## 3. Advanced RAG Overview

```text
┌─────────────────────────┐     ┌─────────────────────────┐     ┌─────────────────────────┐
│      Pre-Retrieval      │ ──> │        Retrieval        │ ──> │     Post-Retrieval      │
│  - Sliding Window       │     │  - Fine-Tuned Encoders  │     │  - Re-Ranking (Cross-Enc│
│  - Small-to-Big Chunking│     │  - Instructor Models    │     │  - Prompt Compression   │
│  - Query Rewriting/HyDE │     │  - Hybrid (Dense+BM25)  │     │  - Context Pruning      │
│  - Semantic Routing     │     │  - Filtered Metadata    │     │                         │
└─────────────────────────┘     └─────────────────────────┘     └─────────────────────────┘
```

---

## 4. LLM Twin RAG Pipeline Architecture

### Batch vs. Streaming Pipeline Design
The LLM Twin implements a **scheduled batch pipeline** rather than a real-time event streaming pipeline:
* **Throughput & Efficiency:** Processes updates in bulk with GPU vectorization.
* **Low Complexity:** Avoids distributed state management (Kafka, Flink) when update latencies of minutes are fully acceptable.

### Change Data Capture (CDC)
To synchronize the raw NoSQL Data Warehouse (MongoDB) and the Vector Store (Qdrant), three CDC paradigms apply:
1. **Timestamp-Based:** Querying records updated since last execution timestamp.
2. **Trigger-Based:** Database hooks logging mutations into dedicated audit tables.
3. **Log-Based (Production Benchmark):** Reading database write-ahead transaction logs directly, achieving zero I/O overhead on primary tables.

### Dual Snapshot Storage Pattern
The Feature Store persists data across two distinct lifecycle checkpoints:
1. **Cleaned Documents Snapshot:** Stored without embeddings (leveraging Qdrant metadata index as a NoSQL document layer) to generate instruction-tuning datasets for offline fine-tuning.
2. **Embedded Chunks Snapshot:** Stored with dense embedding vectors for real-time online RAG semantic search.

---

## 5. Software Design Patterns Implemented

1. **Object-Vector Mapping (OVM):** `VectorBaseDocument` abstracts Qdrant operations (`PointStruct`, `upsert`, `scroll`, `search`) using generic type constraints `Generic[T]`.
2. **Abstract Factory & Strategy:** `CleaningDispatcher`, `ChunkingDispatcher`, and `EmbeddingDispatcher` dynamically instantiate dedicated handlers per category (`articles`, `posts`, `repositories`) without conditional branching.
3. **Singleton Pattern:** `EmbeddingModelSingleton` guarantees transformer weights are loaded into memory and GPU VRAM exactly once.

---

## 6. Directory Layout & Verification

```text
chapter04_rag_feature_pipeline/
├── configs/
│   └── feature_engineering.yaml
├── src/
│   ├── domain/
│   │   ├── base.py
│   │   └── documents.py
│   ├── networks/
│   │   └── embedding.py
│   ├── preprocessing/
│   │   ├── dispatchers.py
│   │   └── handlers/
│   │       ├── chunking.py
│   │       ├── cleaning.py
│   │       └── embedding.py
│   ├── steps/
│   │   └── feature_steps.py
│   ├── settings.py
│   └── run.py
├── requirements.txt
└── README.md
```

To execute the pipeline:
```bash
python -m chapter04_rag_feature_pipeline.src.run
```
