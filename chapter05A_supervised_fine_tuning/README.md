# Chapter 05A: Supervised Fine-Tuning (SFT) & Instruction Engineering

This chapter implements the **Supervised Fine-Tuning (SFT)** stage for the LLM Twin, specializing a base foundation model (Llama 3.1 8B) into an instruction-following conversational assistant aligned with the author's voice, tone, and technical perspective.

---

## 📑 Table of Contents

- [1. Role of Supervised Fine-Tuning](#1-role-of-supervised-fine-tuning)
  - [Next-Token Prediction to Assistant](#next-token-prediction-to-assistant)
  - [When to Fine-Tune vs. RAG vs. Prompting](#when-to-fine-tune-vs-rag-vs-prompting)
  - [Risks: Hallucinations and Catastrophic Forgetting](#risks-hallucinations-and-catastrophic-forgetting)
- [2. Instruction Dataset Engineering](#2-instruction-dataset-engineering)
  - [Quality Dimensions: Accuracy, Diversity, Complexity](#quality-dimensions-accuracy-diversity-complexity)
  - [Rule-Based Filtering](#rule-based-filtering)
  - [Deduplication: Exact, MinHash & Semantic](#deduplication-exact-minhash--semantic)
  - [Data Decontamination](#data-decontamination)
  - [LLM-as-a-Judge & Reward Models](#llm-as-a-judge--reward-models)
  - [Synthetic Generation & Evol-Instruct](#synthetic-generation--evol-instruct)
- [3. Instruction Storage Formats & Chat Templates](#3-instruction-storage-formats--chat-templates)
  - [Storage Schemas: Alpaca vs. ShareGPT vs. ChatML](#storage-schemas-alpaca-vs-sharegpt-vs-chatml)
  - [Jinja Chat Templates](#jinja-chat-templates)
- [4. PEFT: Full Fine-Tuning vs. LoRA vs. QLoRA](#4-peft-full-fine-tuning-vs-lora-vs-qlora)
  - [VRAM Consumption Mathematics](#vram-consumption-mathematics)
  - [Low-Rank Adaptation (LoRA) Formulation](#low-rank-adaptation-lora-formulation)
  - [Quantized LoRA (QLoRA) Optimizations](#quantized-lora-qlora-optimizations)
- [5. Hyperparameters & Training Stability](#5-hyperparameters--training-stability)
  - [Learning Rate, Schedulers & Warmup](#learning-rate-schedulers--warmup)
  - [Batch Size & Gradient Accumulation](#batch-size--gradient-accumulation)
  - [Sequence Length & Sample Packing](#sequence-length--sample-packing)
  - [Monitoring Metrics: Loss Curves & Gradient Norm](#monitoring-metrics-loss-curves--gradient-norm)
- [6. Directory Layout & Verification](#6-directory-layout--verification)

---

## 1. Role of Supervised Fine-Tuning

Pre-trained base LLMs are optimized solely to perform self-supervised next-token prediction across raw web-scale text. Supervised Fine-Tuning conditions model parameters on prompt-response pairs, steering the model into an instruction-following conversational persona.

### Next-Token Prediction to Assistant
The SFT loss is calculated autoregressively, masking prompt tokens and evaluating cross-entropy strictly across target response tokens $y$:
$$\mathcal{L}_{\text{SFT}}(\theta) = -\sum_{t=1}^{T} \log P_\theta(y_t \mid x, y_{<t})$$

### When to Fine-Tune vs. RAG vs. Prompting
1. **Prompt Engineering / Few-Shot:** First-line baseline to validate task feasibility without infrastructure overhead.
2. **RAG:** Injects dynamic, volatile, or private knowledge at inference time to prevent knowledge-cutoff hallucinations.
3. **Fine-Tuning (SFT):** Modifies internal representations to internalize tone, structure, style, and specific output syntax.

### Risks: Hallucinations and Catastrophic Forgetting
* **Catastrophic Forgetting:** Overwriting base weights during unconstrained full training erases generalized reasoning skills.
* **Knowledge Distortion:** Forcing models to memorize novel factual information exclusively through weights induces hallucinations. Grounding is delegated to RAG, while style is enforced via SFT.

---

## 2. Instruction Dataset Engineering

### Quality Dimensions: Accuracy, Diversity, Complexity
* **Accuracy:** Factual alignment and direct relevance between the instruction and response.
* **Diversity:** Broad sampling across topics, vocabulary, syntax lengths, and technical domains.
* **Complexity:** Challenging, multi-step reasoning problems that prevent trivial overfitting.

### Rule-Based Filtering
* **Length Thresholds:** Discarding responses below 30 characters (insufficient style signal) or beyond maximum token limits.
* **Keyword Exclusion:** Stripping boilerplate metadata, URLs, HTML artifacts, and platform signatures.
* **Syntax Validation:** Verifying JSON or Python code blocks compile cleanly.

### Deduplication: Exact, MinHash & Semantic
1. **Exact Deduplication:** Hashing standardized strings with MD5/SHA-256 to drop duplicate entries.
2. **MinHash Fuzzy Deduplication:** Shingling texts into $k$-grams, hashing min-values into signature vectors, and clustering via Jaccard similarity:
$$J(A, B) = \frac{\vert{}A \cap B\vert{}}{\vert{}A \cup B\vert{}}$$
3. **Semantic Deduplication:** Generating sentence embeddings and computing cosine similarity; clusters exceeding thresholds are reduced to a single representative sample.

### Data Decontamination
Removes samples from the training set that appear in validation or benchmark evaluation splits (via exact matching, $n$-gram overlaps, or embedding thresholds).

### LLM-as-a-Judge & Reward Models
* **LLM-as-a-Judge:** Prompting evaluator models (GPT-4o, Llama 3 70B) to score candidate answers (1 to 4 scale). Mitigating biases (position bias, verbosity bias, and intra-model favoritism) using juries and score normalization.
* **Reward Models:** Scoring pairs with dedicated regression architectures (e.g., ArmoRM-Llama3-8B-v0.1) or classifier encoders (e.g., `fineweb-edu-classifier`).

### Synthetic Generation & Evol-Instruct
When natural instruction-answer pairs are scarce in raw articles, synthetic pipelines apply:
* **Backtranslation:** Providing raw article chunks as the expected output and prompting an LLM to generate the prompting query.
* **Evol-Instruct:**
  * *In-Depth Evolving:* Adding constraints, deepening questions, concretizing abstract concepts, and mandating reasoning steps.
  * *In-Breadth Evolving:* Synthesizing rare, long-tail variations inspired by the seed prompt.

---

## 3. Instruction Storage Formats & Chat Templates

### Storage Schemas: Alpaca vs. ShareGPT vs. ChatML
* **Alpaca (Single-Turn):** `{"instruction": "...", "input": "...", "output": "..."}`
* **ShareGPT (Multi-Turn):** `{"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}`
* **ChatML (Standardized):** `{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}`

### Jinja Chat Templates
Base models do not possess conversational delimiters. SFT trains the model to respect delimiter tokens (such as `<|im_start|>` and `<|im_end|>`).

```text
<|im_start|>system
You are Oscar's LLM Twin.<|im_end|>
<|im_start|>user
Explain the FTI architecture.<|im_end|>
<|im_start|>assistant
The FTI architecture decouples Feature, Training, and Inference pipelines...<|im_end|>
```

---

## 4. PEFT: Full Fine-Tuning vs. LoRA vs. QLoRA

### VRAM Consumption Mathematics
Total memory in a baseline single-GPU 32-bit floating point (FP32) training setup:
$$\text{Memory} = \text{Parameters} + \text{Gradients} + \text{Optimizer States} + \text{Activations}$$

| Component | FP32 Bytes/Param | FP16/BF16 Bytes/Param | 7B Model (FP32) |
|---|---|---|---|
| **Parameters** | 4 bytes | 2 bytes | 28 GB |
| **Gradients** | 4 bytes | 2 bytes | 28 GB |
| **Optimizer States (AdamW)** | 8 bytes | 8 bytes | 56 GB |
| **Total Baseline** | **16 bytes/param** | **12 bytes/param** | **112 GB VRAM** |

### Low-Rank Adaptation (LoRA) Formulation
Freezes pre-trained weights $W_0 \in \mathbb{R}^{d \times k}$ and injects trainable decomposition matrices $A \in \mathbb{R}^{r \times k}$ and $B \in \mathbb{R}^{d \times r}$ with rank $r \ll \min(d, k)$:
$$W' = W_0 + \Delta W = W_0 + \frac{\alpha}{r} (B \cdot A)$$

* Trainable parameters drop to $< 1\%$ of the total architecture.
* Common heuristic: $\alpha = 2r$.
* Target modules: attention (`q_proj`, `k_proj`, `v_proj`, `o_proj`) and MLP blocks (`gate_proj`, `up_proj`, `down_proj`).

### Quantized LoRA (QLoRA) Optimizations
1. **NF4 (4-bit NormalFloat):** Quantizes frozen base weights into theoretical information-dense intervals.
2. **Double Quantization:** Quantizes quantization constants, saving 0.37 bits per parameter.
3. **Paged Optimizers:** Offloads memory spikes to CPU RAM during gradient backward passes.
* **Footprint:** Reduces 7B/8B model fine-tuning footprint down to ~9.3 GB VRAM.

---

## 5. Hyperparameters & Training Stability

* **Effective Batch Size:**
  $$\text{Effective Batch Size} = \text{Batch Size per Device} \times \text{GPUs} \times \text{Gradient Accumulation Steps}$$
* **Learning Rate & Schedulers:** $1\times 10^{-4}$ to $3\times 10^{-4}$ for LoRA, employing a linear/cosine decay with a 5% warmup period.
* **Sample Packing:** Concatenating short training sequences into a single context window length (e.g., 2,048 tokens) with distinct cross-attention masking to maximize GPU saturation.
* **Loss Curve Diagnostics:**
  * *Healthy:* Sharp initial descent followed by gradual asymptotic stabilization.
  * *Overfitting:* Training loss continues declining while validation loss diverges upward.
  * *Exploding Gradients:* Spikes in gradient norm, mitigated using gradient clipping ($\text{max\_norm} \le 1.0$).

---

## 6. Directory Layout & Verification

```text
chapter05A_supervised_fine_tuning/
├── configs/
│   └── sft.yaml
├── src/
│   ├── __init__.py
│   ├── dataset_generator.py   # Backtranslation & Evol synthetic generation
│   ├── chat_templates.py      # Alpaca & ChatML formatters with EOS tokens
│   ├── trainer.py             # SFTTrainer engine (Unsloth / TRL / PyTorch)
│   ├── inference.py           # TextStreamer inference runner
│   └── run.py                 # Chapter pipeline orchestrator
├── requirements.txt
└── README.md
```

Run the pipeline:
```bash
python -m chapter05A_supervised_fine_tuning.src.run
```
