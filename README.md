# phi3-mini-os-finetuned 🧠

Fine-tuning [Phi-3 Mini](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) (3.8B) on a custom Operating Systems dataset using **QLoRA** — training only 0.26% of parameters on a free Colab T4 GPU.

🤗 **Model on HuggingFace:** [Prithvi-Aithal/phi3-mini-os-finetuned](https://huggingface.co/Prithvi-Aithal/phi3-mini-os-finetuned)

---

## What This Project Does

1. Curates 200 OS Q&A pairs from the PES University Operating Systems syllabus
2. Loads Phi-3 Mini in **4-bit quantization** to fit in ~6GB VRAM
3. Attaches **LoRA adapters** and trains only the attention layers
4. Merges the adapters and pushes the final model to HuggingFace Hub

The result is a model that answers OS exam questions — scheduling, deadlocks, memory management, file systems — in a textbook-accurate style.

---

## Project Structure

```
phi3-os-finetune/
├── README.md
├── requirements.txt
├── data/
│   └── full_data.jsonl         # Q&A training set (one JSON object per line)
└── notebooks/
    └── finetune.ipynb          # Full fine-tuning notebook (run on Colab)
```

---

## Quickstart

### Run Inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "Prithvi-Aithal/phi3-mini-os-finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

def ask_os(question):
    prompt = f"<|user|>\n{question}<|end|>\n<|assistant|>\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).split("<|assistant|>")[-1].strip()

print(ask_os("Explain the Banker's Algorithm."))
```

### Reproduce the Fine-Tuning

1. Open `notebooks/finetune.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Set runtime to **T4 GPU** (free tier works)
3. Install dependencies — first cell handles this
4. Run all cells — training takes ~15–20 minutes on T4

---

## Key Techniques

| Technique | Why |
|-----------|-----|
| **4-bit quantization (NF4)** | Reduces Phi-3's memory from ~8GB → ~3GB |
| **LoRA (r=16)** | Trains only 0.26% of parameters, prevents catastrophic forgetting |
| **SFTTrainer (TRL)** | Handles chat template formatting automatically |
| **Gradient checkpointing** | Lets us use larger batch sizes without OOM |

---

## Topics Covered in Training Data

- Process lifecycle and PCB
- CPU scheduling (FCFS, SJF, SRTF, Round Robin, Priority)
- Deadlock — Coffman conditions, Banker's Algorithm, detection & recovery
- Memory management — paging, segmentation, TLB
- Virtual memory and page replacement (LRU, FIFO, Optimal)
- File systems and directory structures
- Synchronization — semaphores, mutexes, monitors, the dining philosophers problem

---

## Requirements

See `requirements.txt`. Main dependencies:

- `transformers >= 4.40`
- `peft >= 0.10`
- `trl >= 0.8`
- `bitsandbytes >= 0.43`
- `accelerate >= 0.29`
- `datasets >= 2.18`

---

## Results

The model produces coherent, exam-appropriate answers for standard OS questions. Given the small training set (200 examples), it performs best on topics well-represented in the syllabus.

**Example:**

> **Q:** What are the four conditions for deadlock?
>
> **A:** The four necessary conditions (Coffman conditions) are: (1) Mutual Exclusion, (2) Hold and Wait, (3) No Preemption, and (4) Circular Wait. All four must hold simultaneously for a deadlock to occur.

---

## Built By

**Prithvi Aithal** — [HuggingFace](https://huggingface.co/Prithvi-Aithal)

Fine-tuned as a portfolio project demonstrating practical LLM training on domain-specific data with consumer-grade hardware.
