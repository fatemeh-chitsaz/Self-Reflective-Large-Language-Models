# Self-Reflective Large Language Models for Financial Reasoning

An experimental LLM pipeline for **financial sentiment / stock-movement reasoning** using iterative reflection and parameter-efficient fine-tuning.

The project explores whether an LLM can improve a failed prediction by diagnosing its previous reasoning and using that reflection on a subsequent attempt.

## Core idea

```text
financial text / facts
        ↓
initial prediction
        ↓
correct? ── yes → keep sample
   │
   no
   ↓
reflection on failure
        ↓
revised prediction
        ↓
training / preference data
```

The notebook contains agents that:

- summarize financial/tweet information;
- predict positive vs. negative stock movement;
- compare predictions with the target;
- generate a textual reflection after an incorrect attempt;
- incorporate previous reflections into later prompts.

## Fine-tuning

The repository also contains a PEFT adapter built from:

**`lmsys/vicuna-7b-v1.5-16k`**

The training code uses LoRA-style parameter-efficient fine-tuning with:

- 4-bit model loading;
- PEFT;
- Transformers;
- Datasets;
- TRL-related experimentation.

The current notebook configuration includes LoRA adapters on attention projection modules such as `q_proj` and `v_proj`.

## Repository contents

```text
.
├── code.ipynb
├── database/
│   └── comparison_data.json
├── reward_adapter/
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── README.md
├── requirements.txt
└── supporting data / reference material
```

## Setup

```bash
pip install -r requirements.txt
```

GPU execution is strongly recommended for the fine-tuning sections.

## Research status

This repository is an **experimental research prototype**, not a production trading system. It should not be interpreted as financial advice or as evidence that the model can predict markets reliably.

The strongest next research improvements would be:

- strict temporal train/validation/test splits;
- leakage checks;
- comparison against non-reflective LLM baselines;
- comparison against simple statistical/ML baselines;
- ablation of reflection vs. fine-tuning;
- multiple random seeds;
- clearly reported accuracy/F1 and calibration;
- reproducible data-generation scripts.

## Attribution

The project uses open-source Hugging Face/PEFT tooling and a Vicuna base model. Consult the respective upstream licenses before redistribution or deployment.
