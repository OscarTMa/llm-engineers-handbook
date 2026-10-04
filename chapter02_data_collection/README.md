# Chapter 02: Data Engineering & Crawling Pipeline (ETL)

This chapter focuses on the **Data Collection Pipeline**, implementing the Extract, Load, Transform (ETL) pattern to gather, normalize, and store unstructured personal data into a NoSQL Data Warehouse[cite: 1, 2].

---

## 📑 Table of Contents

- [1. Data Engineering for LLMs](#1-data-engineering-for-llms)
  - [The Role of Raw Data in LLM Twins](#the-role-of-raw-data-in-llm-twins)
  - [Category-Agnostic Abstraction](#category-agnostic-abstraction)
- [2. The ETL Pipeline Architecture](#2-the-etl-pipeline-architecture)
  - [Extract: Crawlers & Scrapers](#extract-crawlers--scrapers)
  - [Transform: Schema Standardization](#transform-schema-standardization)
  - [Load: NoSQL Data Warehouse (MongoDB)](#load-nosql-data-warehouse-mongodb)
- [3. Data Models & Schemas](#3-data-models--schemas)
- [4. Implementation & Execution](#4-implementation--execution)
- [5. Directory Layout](#5-directory-layout)

---

## 1. Data Engineering for LLMs

An LLM Twin directly reflects the distribution, tone, and domain coverage of its training data[cite: 1]. Therefore, establishing robust, automated data collection is critical before starting feature engineering or fine-tuning[cite: 1].

### The Role of Raw Data in LLM Twins
* **Identity Grounding:** Data crawled from professional platforms establishes your specific tone and technical depth[cite: 1].
* **Continuous Ingestion:** New articles, code commits, and social updates must be ingested incrementally to keep the digital twin updated[cite: 1].

### Category-Agnostic Abstraction
Rather than coupling pipelines to specific platforms (e.g., handling Substack vs. Medium separately), incoming data is normalized into three platform-agnostic categories[cite: 1, 2]:
1. **Posts:** Short-form, conversational content (e.g., LinkedIn posts, X threads)[cite: 1, 2].
2. **Articles:** Long-form, structured prose and tutorials (e.g., Medium, Substack)[cite: 1, 2].
3. **Code:** Syntax trees, scripts, and documentation (e.g., GitHub repositories)[cite: 1, 2].

Original URLs and source platform names are preserved as metadata for lineage and citation[cite: 1, 2].

---

## 2. The ETL Pipeline Architecture

```text
[ LinkedIn / Medium / Substack / GitHub ]
                    │
                    ▼
     ┌─────────────────────────────┐
     │      Extract (Crawlers)     │ ──> Raw HTML, Markdown, REST API responses
     └──────────────┬──────────────┘
                    ▼
     ┌─────────────────────────────┐
     │   Transform (Normalization) │ ──> Pydantic validation (Clean text & metadata)
     └──────────────┬──────────────┘
                    ▼
     ┌─────────────────────────────┐
     │      Load (Warehouse)       │ ──> NoSQL Document Store (MongoDB)
     └─────────────────────────────┘
```

### Extract: Crawlers & Scrapers
Custom scrapers query public endpoints and web pages, parsing HTML structure and extracting main text while discarding navigation and UI boilerplate[cite: 1].

### Transform: Schema Standardization
Every ingested record is parsed through a unified Pydantic schema to validate required fields: content, category, author, and timestamp.

### Load: NoSQL Data Warehouse (MongoDB)
Unstructured text fits document-oriented storage cleanly[cite: 1]. MongoDB acts as our raw data warehouse, storing documents indexed by unique ID and platform source[cite: 1].

---

## 3. Data Models & Schemas

The central unit of transmission across the data pipeline is the `RawDocument`:

```python
class RawDocument(BaseModel):
    id: str
    category: Literal["posts", "articles", "code"]
    content: str
    author: str
    source_url: str
    platform: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
```

---

## 4. Implementation & Execution

### 1. Install dependencies
```bash
pip install -r chapter02_data_collection/requirements.txt
```

### 2. Run Data Collection Pipeline
```bash
python -m chapter02_data_collection.src.run
```

---

## 5. Directory Layout

```text
chapter02_data_collection/
├── configs/
│   └── crawlers.yaml            # Crawling targets and intervals
├── src/
│   ├── crawlers/
│   │   ├── __init__.py
│   │   ├── base.py              # Abstract crawler definition
│   │   ├── github.py            # GitHub repository extractor
│   │   └── medium.py            # Article scraper
│   ├── models.py                # Pydantic data schemas
│   ├── warehouse.py             # MongoDB / local store adapter
│   └── run.py                   # Main pipeline entrypoint
├── requirements.txt             # Chapter dependencies
└── README.md                    # Chapter overview
```