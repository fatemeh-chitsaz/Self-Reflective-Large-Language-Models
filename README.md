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


