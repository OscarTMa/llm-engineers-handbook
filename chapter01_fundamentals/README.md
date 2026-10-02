# LLM Engineer's Handbook — Hands-On Exercises & LLM Twin

Implementation of production-grade Large Language Model systems based on the book **"LLM Engineer's Handbook"** by Paul Iusztin and Maxime Labonne.

This repository contains the end-to-end engineering implementation of an **LLM Twin**: an AI agent fine-tuned to capture personal writing style, tone, and technical perspective across digital platforms[cite: 2].

---

## System Architecture: FTI Pattern

The system implements the decoupled **Feature / Training / Inference (FTI)** architectural design to eliminate monolithic pipelines and prevent training-serving skew[cite: 2]:

```text
[ Data Sources: LinkedIn / Medium / Substack / GitHub ]
                           │
                           ▼
  ┌─────────────────────────────────────────────────┐
  │   1. Data Collection Pipeline (ETL)             │ ──> NoSQL Data Warehouse (MongoDB)
  └────────────────────────┬────────────────────────┘
                           │
                           ▼
  ┌─────────────────────────────────────────────────┐
  │   2. Feature Pipeline                           │ ──> Cleaning, Chunking & Embeddings
  └────────────┬────────────────────────┬───────────┘
               │                        │
               ▼                        ▼
     Instruction Datasets        Vector Database
        (Artifacts)                 (Qdrant)
               │                        │
               ▼                        │
  ┌─────────────────────────┐           │
  │   3. Training Pipeline  │           │
  │      (PEFT / QLoRA)     │           │
  └────────────┬────────────┘           │
               ▼                        │
         Model Registry                 │
               │                        │
               └───────────┬────────────┘
                           ▼
  ┌─────────────────────────────────────────────────┐
  │   4. Inference Pipeline                         │ ──> Fast REST API + RAG Retrieval
  └─────────────────────────────────────────────────┘
```

* **Data Collection Pipeline (ETL):** Ingests raw data across articles, posts, and code into a NoSQL document database[cite: 2].
* **Feature Pipeline:** Cleans, chunks, and computes dense embeddings to populate both instruction dataset artifacts and the vector store[cite: 2].
* **Training Pipeline:** Executes parameter-efficient fine-tuning (PEFT / QLoRA) and validates candidates before registering them[cite: 2].
* **Inference Pipeline:** Exposes a low-latency REST API, performing vector search for RAG and generating completions with prompt observability[cite: 2].

---

## Roadmap & Chapters

| Chapter | Title | Status | Directory |
| :---: | :--- | :---: | :--- |
| **01** | **LLM Twin Concept & FTI Architecture**[cite: 2] | Completed | `chapter01_fundamentals/`[cite: 2] |
| **02** | **Data Engineering & Crawling Pipeline** | Planned | `chapter02_data_collection/` |
| **03** | **Feature Store, Chunking & Vector DB** | Planned | `chapter03_feature_pipeline/` |
| **04** | **Instruction Fine-Tuning (SFT & QLoRA)** | Planned | `chapter04_training/` |
| **05** | **Preference Alignment (DPO / RLHF)** | Planned | `chapter05_alignment/` |
| **06** | **RAG & Advanced Semantic Retrieval** | Planned | `chapter06_rag/` |
| **07** | **LLM Evaluation & Observability** | Planned | `chapter07_evaluation/` |
| **08** | **High-Throughput Inference & Serving** | Planned | `chapter08_serving/` |

---

## Repository Structure

```text
llm-engineers-handbook/
├── assets/
├── chapter01_fundamentals/
│   ├── architecture/
│   │   └── fti_design.md
│   ├── configs/
│   │   └── system_specs.yaml
│   ├── src/
│   │   ├── __init__.py
│   │   ├── pipeline_interfaces.py
│   │   └── run.py
│   └── README.md
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Quickstart & Setup

### 1. Clone the Repository

```bash
git clone [https://github.com/OscarTMa/llm-engineers-handbook.git](https://github.com/OscarTMa/llm-engineers-handbook.git)
cd llm-engineers-handbook
```
*(Clone URL configured for GitHub account `OscarTMa`[cite: 4]).*

### 2. Configure Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run Architecture Simulation

```bash
python -m chapter01_fundamentals.src.run
```

---

## Tech Stack & Tooling

* **Data & Feature Engineering:** MongoDB, Qdrant, Pydantic, FastEmbed / Hugging Face[cite: 2].
* **Model Training & Alignment:** PyTorch, PEFT, TRL (LoRA, QLoRA, DPO)[cite: 2].
* **Inference & Serving:** FastAPI, vLLM, Docker[cite: 2].
* **Code Quality & Tooling:** Ruff, Pytest, Python typing.

---

## License

Distributed under the MIT License.