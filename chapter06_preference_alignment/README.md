# Chapter 06: Fine-Tuning with Preference Alignment (DPO)

This chapter implements the **Preference Alignment** stage for the LLM Twin. While Supervised Fine-Tuning (SFT) teaches base models conversational formatting and task execution, preference alignment captures human judgment, calibrates voice authenticity, and eliminates artificial verbosity using **Direct Preference Optimization (DPO)**.

---

## 📑 Table of Contents

- [1. Motivation: Beyond Supervised Fine-Tuning](#1-motivation-beyond-supervised-fine-tuning)
  - [Limitations of SFT & The Synthetic Voice Problem](#limitations-of-sft--the-synthetic-voice-problem)
  - [Use Cases for Preference Optimization](#use-cases-for-preference-optimization)
- [2. Anatomy of Preference Datasets](#2-anatomy-of-preference-datasets)
  - [Triples Structure: Prompt, Chosen, Rejected](#triples-structure-prompt-chosen-rejected)
  - [Sample Efficiency & Dataset Volume](#sample-efficiency--dataset-volume)
  - [Synthetic Generation & Ground-Truth Contrast](#synthetic-generation--ground-truth-contrast)
  - [Quality Filtering & Heuristics](#quality-filtering--heuristics)
- [3. Evaluation Biases & Mitigation Strategies](#3-evaluation-biases--mitigation-strategies)
- [4. Direct Preference Optimization (DPO) vs. RLHF](#4-direct-preference-optimization-dpo-vs-rlhf)
  - [RLHF & PPO Architecture Complexity](#rlhf--ppo-architecture-complexity)
  - [Mathematical Derivation of DPO](#mathematical-derivation-of-dpo)
  - [The Role of the $\beta$ Parameter](#the-role-of-the-beta-parameter)
- [5. Training Dynamics & Telemetry](#5-training-dynamics--telemetry)
  - [Hyperparameters & Conservative Fine-Tuning](#hyperparameters--conservative-fine-tuning)
  - [Key Metrics: Implicit Rewards, Margins, and Accuracy](#key-metrics-implicit-rewards-margins-and-accuracy)
- [6. Directory Layout & Verification](#6-directory-layout--verification)

---

## 1. Motivation: Beyond Supervised Fine-Tuning

### Limitations of SFT & The Synthetic Voice Problem
Supervised Fine-Tuning (SFT) optimizes models via token cross-entropy loss against static targets. However:
* **Nuance Blindness:** SFT treats all unselected tokens as equally incorrect, failing to discern subtle stylistic preferences.
* **Robotic Verbosity:** Models trained with typical instruction sets tend to default to synthetic, overly polite phrases (e.g., *"delve into"*, *"it is crucial to remember"*).
* **Persona Drift:** Capturing an author's natural cadence requires explicitly penalizing generic outputs while rewarding authentic expressions.

### Use Cases for Preference Optimization
* **Style Calibration:** Penalizing formal, robotic phrasing in favor of casual, direct technical prose.
* **Content Moderation & Refusals:** Training safe boundaries without blunt model degradation.
* **Source Attribution:** Discouraging false claims about model origin (e.g., claiming to be trained by OpenAI or Meta).
* **Summarization & Code Review:** Rewarding concise, idiomatic implementations over long, redundant code.

---

## 2. Anatomy of Preference Datasets

### Triples Structure: Prompt, Chosen, Rejected
Unlike instruction datasets which consist of prompt-response pairs, DPO requires triples:
$$\mathcal{D}_{\text{DPO}} = \left\{ (x^{(i)}, y_w^{(i)}, y_l^{(i)}) \right\}_{i=1}^N$$
* **$x$ (Prompt):** The incoming instruction or query.
* **$y_w$ (Chosen / Preferred):** High-quality, authentic response (ground-truth article excerpt).
* **$y_l$ (Rejected / Dispreferred):** Overly generic, verbose, or synthetic AI response.

### Sample Efficiency & Dataset Volume
DPO is significantly more sample-efficient than SFT:
* **Task-Specific Alignment (Style / Voice):** Requires between **200 and 2,500 pairs** to recalibrate tone.
* **General-Purpose Foundation Alignment:** Requires **10,000 to 100,000+ pairs**.

### Synthetic Generation & Ground-Truth Contrast
Rather than relying on human annotators, the LLM Twin leverages an automated contrastive setup:
1. Extract natural paragraphs from crawled blog posts and repositories as the **Chosen** answer ($y_w$).
2. Prompt an LLM to generate a synthetic response to the same prompt to serve as the **Rejected** answer ($y_l$).
3. This creates a natural gradient toward authentic human style without complex reward modeling.

### Quality Filtering & Heuristics
1. **Length Validation:** Discards chosen samples under 100 characters to prevent low-signal updates.
2. **Grammar & Punctuation Filtering:** Ensures chosen responses begin with an uppercase letter and terminate with valid closing punctuation (`.`, `!`, `?`).

---

## 3. Evaluation Biases & Mitigation Strategies

When evaluating candidate responses via an LLM judge, three systematic biases occur:
1. **Position Bias:** The judge favors the first option ($A$ over $B$).
   * *Mitigation:* Randomize input order across evaluation calls.
2. **Length Bias:** The judge equates verbosity with quality.
   * *Mitigation:* Apply length-normalized scoring and strict conciseness constraints.
3. **Family Bias:** LLMs prefer generations from their own model lineage.
   * *Mitigation:* Employ a multi-model jury combining diverse open and closed model families.

---

## 4. Direct Preference Optimization (DPO) vs. RLHF

### RLHF & PPO Architecture Complexity
Traditional RLHF via Proximal Policy Optimization (PPO) requires managing **four concurrent models** during training:
1. Active Policy (Actor being optimized).
2. Reference Policy (Frozen SFT baseline).
3. Reward Model (Pre-trained classifier scoring completions).
4. Value Model (Critic estimating cumulative rewards).

This architecture causes significant VRAM overhead, distributed orchestration challenges, and training instability.

### Mathematical Derivation of DPO
Rafailov et al. (2023) proved that the optimal policy under the KL-constrained RLHF objective can be expressed analytically, showing that an LLM's policy implicitly represents its own reward model:
$$r^*(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$$

Substituting this relation directly into the Bradley-Terry preference model eliminates the reward model entirely, yielding the **DPO Loss Function**:
$$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]$$

Where:
* $\pi_\theta$: The active parameterized policy.
* $\pi_{\text{ref}}$: The frozen SFT reference model.
* $\sigma$: The sigmoid function.
* $\beta$: Regularization coefficient scaling KL divergence penalty.

### The Role of the $\beta$ Parameter
* $\beta \to 0$: The reference model is ignored, maximizing preference separation at the risk of language degradation.
* **Standard default:** $\beta = 0.1$.
* **LLM Twin customization ($\beta = 0.5$):** Because standard DPO tends to steer models toward formal language, increasing $\beta$ anchors the policy closer to the reference model while making selective stylistic adjustments.

---

## 5. Training Dynamics & Telemetry

### Hyperparameters & Conservative Fine-Tuning
* **Learning Rate:** $2 \times 10^{-6}$ (roughly two orders of magnitude lower than SFT's $3 \times 10^{-4}$).
* **Epochs:** 1 epoch (preventing over-optimization and mode collapse).
* **LoRA Rank ($r$) & Alpha ($\alpha$):** Scaled to $r=64$, $\alpha=64$ for greater stylistic expressiveness.

### Key Metrics: Implicit Rewards, Margins, and Accuracy
1. **Implicit Rewards ($R_{\text{chosen}}, R_{\text{rejected}}$):**
   $$R(x, y) = \beta \left( \log \pi_\theta(y \mid x) - \log \pi_{\text{ref}}(y \mid x) \right)$$
2. **Margin:**
   $$\text{Margin} = R_{\text{chosen}} - R_{\text{rejected}}$$
   *Healthy dynamic:* Increases steadily and plateaus without exploding.
3. **Accuracy:** The percentage of training pairs where $R_{\text{chosen}} > R_{\text{rejected}}$. Steady increases toward 80–90% indicate balanced convergence.

---

## 6. Directory Layout & Verification

```text
chapter06_preference_alignment/
├── configs/
│   └── dpo.yaml
├── src/
│   ├── __init__.py
│   ├── dataset_generator.py   # Preference triples builder and format filters
│   ├── dpo_trainer.py         # DPOTrainer execution & metrics logger
│   ├── inference.py           # Side-by-side generation test (SFT vs DPO)
│   └── run.py                 # Chapter pipeline orchestrator
├── requirements.txt
└── README.md
```

Run the pipeline:
```bash
python -m chapter06_preference_alignment.src.run
```
