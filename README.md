# LLM Engineer's Handbook — Hands-On Exercises & LLM Twin

Implementation of production-grade Large Language Model systems based on the book **"LLM Engineer's Handbook"** by Paul Iusztin and Maxime Labonne. 

This repository contains the end-to-end engineering implementation of an **LLM Twin**: an AI agent fine-tuned to capture personal writing style, tone, and technical perspective across digital platforms.

---

## System Architecture: FTI Pattern

The system implements the decoupled **Feature / Training / Inference (FTI)** architectural design to eliminate monolithic pipelines and prevent training-serving skew:

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

---

## Roadmap & Chapters

| Chapter | Title | Status | Directory |
| :---: | :--- | :---: | :--- |
| **01** | **LLM Twin Concept & FTI Architecture** | Completed | `chapter01_fundamentals/` |
| **02** | **Data Engineering & Crawling Pipeline** | Planned | `chapter02_data_collection/` |
| **03** | **Feature Store, Chunking & Vector DB** | Planned | `chapter03_feature_pipeline/` |
| **04** | **RAG Feature Pipeline & Vector Databases** | ✅ Completed | `chapter04_rag_feature_pipeline/` |
| **05** | **The Instruction Dataset Pipeline** | ✅ Completed | `chapter05_instruction_dataset/` |
| **05A** | **Supervised Fine-Tuning (SFT) & Unsloth** | ✅ Completed | `chapter05A_supervised_fine_tuning/` |
| **06** | **Preference Alignment (DPO)** | ✅ Completed | `chapter06_preference_alignment/` |
| **07** | **LLM & RAG Evaluation (Ragas & Judge)** | ✅ Completed | `chapter07_evaluation/` |
| **08** | **High-Throughput Inference & Serving** | ⚪ Planned | `chapter08_serving/` |
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
git clone [https://github.com/](https://github.com/)<your-username>/llm-engineers-handbook.git
cd llm-engineers-handbook
```

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

* **Data & Feature Engineering:** MongoDB, Qdrant, Pydantic, FastEmbed / Hugging Face.
* **Model Training & Alignment:** PyTorch, PEFT, TRL (LoRA, QLoRA, DPO).
* **Inference & Serving:** FastAPI, vLLM, Docker.
* **Code Quality & Tooling:** Ruff, Pytest, Python typing.

---

## License

Distributed under the MIT License.
