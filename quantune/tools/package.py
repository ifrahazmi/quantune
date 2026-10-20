"""
Package Generator Tool
======================
Generates a complete, ready-to-run training package using Unsloth for
fast LoRA fine-tuning of chat models.

Unsloth provides:
- 2-5x faster training
- 70% less memory usage
- Native 4-bit quantization
- Easy model export (GGUF, GGML, etc.)
"""

import json
import os
import shutil
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quantune.state import QuantuneState


# ============================================================================
# TEMPLATE: train.py using Unsloth
# ============================================================================

TRAIN_UNSLOTH = '''"""
Quantune Training Script - Unsloth LoRA Fine-Tuning
===================================================
Run: python train.py

This script uses Unsloth for 2-5x faster training with 70% less memory.
Your dataset should be in JSONL format with "messages" structure.
"""

import os
# Disable torch dynamo to avoid compilation errors with some models
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import json
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# ============================================================================
# LOAD CONFIG
# ============================================================================

with open("config.json") as f:
    config = json.load(f)

print("="*60)
print("QUANTUNE - UNSLOTH LORA TRAINING")
print("="*60)
print(f"Model: {config['base_model']}")
print(f"Data:  {config['data_path']}")
print(f"Output: {config['output_dir']}")
print("="*60)

# ============================================================================
# LOAD MODEL WITH UNSLOTH (4-BIT QUANTIZATION)
# ============================================================================

print("\\nLoading model with Unsloth...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=config["base_model"],
    max_seq_length=config.get("max_seq_length", 2048),
    dtype=None,  # Auto-detect
    load_in_4bit=True,  # 4-bit quantization for memory efficiency
)

# ============================================================================
# CONFIGURE LORA
# ============================================================================

lora_config = config.get("lora_config", {})

model = FastLanguageModel.get_peft_model(
    model,
    r=lora_config.get("r", 16),
    lora_alpha=lora_config.get("lora_alpha", 16),
    lora_dropout=lora_config.get("lora_dropout", 0),
    target_modules=lora_config.get("target_modules", [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]),
    bias="none",
    use_gradient_checkpointing="unsloth",  # Unsloth's optimized checkpointing
    random_state=42,
    use_rslora=False,  # Rank-stabilized LoRA
    loftq_config=None,
)

print("\\nLoRA Configuration:")
print(f"  Rank (r): {lora_config.get('r', 16)}")
print(f"  Alpha: {lora_config.get('lora_alpha', 16)}")
print(f"  Dropout: {lora_config.get('lora_dropout', 0)}")

# ============================================================================
# LOAD AND PREPARE DATASET
# ============================================================================

print("\\nLoading dataset...")

dataset = load_dataset("json", data_files=config["data_path"], split="train")
print(f"Loaded {len(dataset)} examples")

# Verify format
if len(dataset) == 0:
    raise ValueError("Dataset is empty!")
if "messages" not in dataset[0]:
    raise ValueError('Dataset must have "messages" field.')

# Format using chat template
def format_chat(example):
    """Apply chat template to messages."""
    messages = example["messages"]
    
    # Use tokenizer's chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    
    return {"text": text}

print("Formatting with chat template...")
dataset = dataset.map(format_chat, remove_columns=dataset.column_names)

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

training_args_config = config.get("training_args", {})

training_args = TrainingArguments(
    output_dir=config["output_dir"],
    
    # Training params
    num_train_epochs=training_args_config.get("num_train_epochs", 3),
    per_device_train_batch_size=training_args_config.get("per_device_train_batch_size", 2),
    gradient_accumulation_steps=training_args_config.get("gradient_accumulation_steps", 4),
    
    # Optimizer
    learning_rate=training_args_config.get("learning_rate", 2e-4),
    weight_decay=training_args_config.get("weight_decay", 0.01),
    warmup_steps=training_args_config.get("warmup_steps", 5),
    
    # Precision
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    
    # Logging
    logging_steps=training_args_config.get("logging_steps", 10),
    
    # Saving
    save_strategy="epoch",
    save_total_limit=2,
    
    # Optimization
    optim="adamw_8bit",  # 8-bit Adam for memory efficiency
    lr_scheduler_type="linear",
    seed=42,
    
    # Disable unused features
    push_to_hub=False,
    report_to="none",
)

# ============================================================================
# TRAIN
# ============================================================================

print("\\nInitializing trainer...")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=config.get("max_seq_length", 2048),
    packing=False,  # Can set to True for short sequences
    args=training_args,
)

# Show GPU stats before training
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"\\nGPU: {gpu_stats.name}")
print(f"Memory: {start_gpu_memory}GB / {max_memory}GB")

print("\\n" + "="*60)
print("STARTING TRAINING")
print("="*60 + "\\n")

trainer_stats = trainer.train()

# Show final GPU stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_pct = round(used_memory / max_memory * 100, 2)
print(f"\\nPeak GPU memory: {used_memory}GB ({used_memory_pct}% of {max_memory}GB)")

# ============================================================================
# SAVE MODEL
# ============================================================================

print("\\n" + "="*60)
print("SAVING MODEL")
print("="*60)

# Save LoRA adapter
lora_path = config["output_dir"]
model.save_pretrained(lora_path)
tokenizer.save_pretrained(lora_path)
print(f"\\n✓ LoRA adapter saved to: {lora_path}")

# Optional: Save merged model (full weights)
# print("\\nMerging LoRA with base model...")
# model.save_pretrained_merged(lora_path + "_merged", tokenizer, save_method="merged_16bit")
# print(f"✓ Merged model saved to: {lora_path}_merged")

# ============================================================================
# EXPORT OPTIONS (UNCOMMENT AS NEEDED)
# ============================================================================

# # Save as GGUF for llama.cpp / Ollama
# print("\\nExporting to GGUF format...")
# model.save_pretrained_gguf(lora_path + "_gguf", tokenizer, quantization_method="q4_k_m")
# print(f"✓ GGUF model saved to: {lora_path}_gguf")

# # Save as 4-bit quantized
# print("\\nExporting 4-bit quantized model...")
# model.save_pretrained_merged(lora_path + "_4bit", tokenizer, save_method="merged_4bit_forced")
# print(f"✓ 4-bit model saved to: {lora_path}_4bit")

print("\\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print(f"\\nNext steps:")
print(f"  1. Test: python eval.py")
print(f"  2. Use LoRA adapter: {lora_path}")
print("="*60)
'''


# ============================================================================
# TEMPLATE: eval.py
# ============================================================================

EVAL_SCRIPT = '''"""
Quantune Evaluation Script - Test Your Fine-Tuned Model
========================================================
Run: python eval.py
"""

import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import json
import torch
from unsloth import FastLanguageModel

# Load config
with open("config.json") as f:
    config = json.load(f)

print("="*60)
print("QUANTUNE - MODEL EVALUATION")
print("="*60)
print(f"Loading LoRA adapter from: {config['output_dir']}")
print("="*60)

# Load model with LoRA adapter
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=config["output_dir"],  # LoRA adapter path
    max_seq_length=config.get("max_seq_length", 2048),
    dtype=None,
    load_in_4bit=True,
)

# Enable faster inference
FastLanguageModel.for_inference(model)

print("\\nModel loaded successfully!\\n")

# ============================================================================
# LOAD TEST SAMPLES FROM TRAINING DATA
# ============================================================================

test_samples = []
data_path = config.get("data_path", "./data.jsonl")

if os.path.exists(data_path):
    with open(data_path, "r") as f:
        lines = f.readlines()
        # Take 5 samples from different parts of the dataset
        indices = [0, len(lines)//4, len(lines)//2, 3*len(lines)//4, -1]
        for idx in indices:
            try:
                example = json.loads(lines[idx])
                if "messages" in example:
                    test_samples.append(example["messages"])
            except:
                pass

# Fallback if no data found
if not test_samples:
    test_samples = [
        [{"role": "user", "content": "Hello!"}],
    ]

print("="*60)
print("TESTING MODEL WITH YOUR DATA")
print("="*60)

correct = 0
total = 0

for i, messages in enumerate(test_samples[:5], 1):
    # Get the user input (last user message before assistant)
    user_msg = None
    expected = None

    for j, msg in enumerate(messages):
        if msg["role"] == "user":
            user_msg = msg["content"]
        elif msg["role"] == "assistant" and user_msg:
            expected = msg["content"]
            break

    if not user_msg:
        continue

    total += 1
    print(f"\\n--- Test {i} ---")
    print(f"Input: {user_msg[:100]}{'...' if len(user_msg) > 100 else ''}")
    if expected:
        print(f"Expected: {expected[:50]}{'...' if len(expected) > 50 else ''}")

    # Build messages for generation (without the expected response)
    gen_messages = []
    for msg in messages:
        if msg["role"] == "assistant":
            break
        gen_messages.append(msg)

    # Apply chat template with generation prompt
    formatted = tokenizer.apply_chat_template(
        gen_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Tokenize
    inputs = tokenizer(formatted, return_tensors="pt").to("cuda")

    # Generate
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        use_cache=True,
        temperature=0.1,  # Low temperature for more deterministic output
    )

    # Decode only the new tokens
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    print(f"Got: {response[:50]}{'...' if len(response) > 50 else ''}")

    # Check if correct (simple match)
    if expected and expected.lower().strip() in response.lower():
        print("✓ MATCH")
        correct += 1
    elif expected:
        print("✗ MISMATCH")

    print("-"*60)

if total > 0:
    print(f"\\nAccuracy: {correct}/{total} ({100*correct/total:.1f}%)")

# ============================================================================
# INTERACTIVE MODE
# ============================================================================

print("\\n" + "="*60)
print("INTERACTIVE MODE")
print("="*60)
print("Enter messages to chat with your model (type 'quit' to exit)")
print("="*60)

while True:
    user_input = input("\\nYou: ").strip()
    if user_input.lower() in ['quit', 'exit', 'q']:
        print("Goodbye!")
        break
    
    if not user_input:
        continue
    
    messages = [{"role": "user", "content": user_input}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    inputs = tokenizer(formatted, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs, max_new_tokens=300, use_cache=True, temperature=0.7
    )
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    
    print(f"\\nAssistant: {response.strip()}")
'''


# ============================================================================
# TEMPLATE: export.py - Export to various formats
# ============================================================================

EXPORT_SCRIPT = '''"""
Quantune Export Script - Export Model to Various Formats
=========================================================
Run: python export.py

Export your fine-tuned model to:
- GGUF (for llama.cpp, Ollama, LM Studio)
- 16-bit merged (for HuggingFace)
- 4-bit quantized (for low-memory inference)
"""

import json
import os
from unsloth import FastLanguageModel

# Load config
with open("config.json") as f:
    config = json.load(f)

print("="*60)
print("QUANTUNE - MODEL EXPORT")
print("="*60)

# Load the trained LoRA model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=config["output_dir"],
    max_seq_length=config.get("max_seq_length", 2048),
    dtype=None,
    load_in_4bit=True,
)

base_output = config["output_dir"]

# ============================================================================
# EXPORT OPTIONS
# ============================================================================

print("\\nSelect export format:")
print("  1. GGUF (q4_k_m) - Best for llama.cpp, Ollama, LM Studio")
print("  2. GGUF (q8_0)   - Higher quality, larger file")
print("  3. 16-bit merged - Full precision, HuggingFace compatible")
print("  4. 4-bit merged  - Quantized, memory efficient")
print("  5. All formats")
print("  0. Exit")

choice = input("\\nEnter choice (0-5): ").strip()

if choice == "0":
    print("Exiting.")
    exit()

# GGUF q4_k_m
if choice in ["1", "5"]:
    print("\\nExporting GGUF (q4_k_m)...")
    output_path = base_output + "_gguf_q4"
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained_gguf(output_path, tokenizer, quantization_method="q4_k_m")
    print(f"✓ Saved to: {output_path}")

# GGUF q8_0
if choice in ["2", "5"]:
    print("\\nExporting GGUF (q8_0)...")
    output_path = base_output + "_gguf_q8"
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained_gguf(output_path, tokenizer, quantization_method="q8_0")
    print(f"✓ Saved to: {output_path}")

# 16-bit merged
if choice in ["3", "5"]:
    print("\\nExporting 16-bit merged model...")
    output_path = base_output + "_merged_16bit"
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained_merged(output_path, tokenizer, save_method="merged_16bit")
    print(f"✓ Saved to: {output_path}")

# 4-bit merged
if choice in ["4", "5"]:
    print("\\nExporting 4-bit merged model...")
    output_path = base_output + "_merged_4bit"
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained_merged(output_path, tokenizer, save_method="merged_4bit_forced")
    print(f"✓ Saved to: {output_path}")

print("\\n" + "="*60)
print("EXPORT COMPLETE!")
print("="*60)
print("\\nUsage instructions:")
print("  - GGUF: Use with llama.cpp, Ollama, or LM Studio")
print("  - Merged: Use with HuggingFace transformers")
print("="*60)
'''


# ============================================================================
# TEMPLATE: requirements.txt
# ============================================================================

def _generate_requirements() -> str:
    """Generate requirements.txt for Unsloth training."""
    return """# Quantune Training Requirements
# ================================
# Install with: pip install -r requirements.txt
#
# For best results, use a virtual environment:
#   python -m venv venv
#   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
#   pip install -r requirements.txt

# Unsloth - Fast LoRA training (2-5x faster, 70% less memory)
# This will install all required dependencies automatically
unsloth

# If you encounter issues, try:
#   pip uninstall unsloth unsloth_zoo -y
#   pip install --no-deps git+https://github.com/unslothai/unsloth_zoo.git
#   pip install --no-deps git+https://github.com/unslothai/unsloth.git
"""


# ============================================================================
# TEMPLATE: README.md
# ============================================================================

def _generate_readme(state: "QuantuneState") -> str:
    """Generate comprehensive README."""
    config = state.get("training_config", {})
    model = state.get("base_model", "unknown")
    reasoning = state.get("planning_reasoning", "")
    lora_config = config.get("lora_config", {})
    training_args = config.get("training_args", {})
    
    readme = f'''# Quantune Training Package

## Overview

This training package was generated by **[Quantune](https://github.com/riyanshibohra/Quantune)** using **Unsloth** for fast, memory-efficient LoRA fine-tuning.

| Property | Value |
|----------|-------|
| Base Model | `{model}` |
| Training Method | LoRA with Unsloth (2-5x faster) |
| Data Format | JSONL with "messages" structure |
| Memory Usage | ~70% less than standard training |
| Generated | {datetime.now().strftime("%Y-%m-%d %H:%M")} |

## Why Unsloth?

Unsloth provides significant advantages:
- ⚡ **2-5x faster training** with optimized kernels
- 💾 **70% less memory** usage
- 🔧 **Native 4-bit quantization** for efficiency
- 📦 **Easy export** to GGUF, 16-bit, 4-bit formats
- 🎯 **Optimized gradient checkpointing**

## Agent's Reasoning

> {reasoning}

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Your Dataset

Your JSONL file should have this format:
```json
{{"messages": [{{"role": "user", "content": "..."}}, {{"role": "assistant", "content": "..."}}]}}
{{"messages": [{{"role": "system", "content": "..."}}, {{"role": "user", "content": "..."}}, {{"role": "assistant", "content": "..."}}]}}
```

### 3. Train

```bash
python train.py
```

### 4. Evaluate

```bash
python eval.py
```

### 5. Export (Optional)

```bash
python export.py
```

Export to GGUF for llama.cpp/Ollama, or merged formats for HuggingFace.

## Files Included

| File | Description |
|------|-------------|
| `data.jsonl` | Your training dataset (included) |
| `config.json` | Training configuration |
| `train.py` | Unsloth LoRA training script |
| `eval.py` | Evaluation + interactive testing |
| `export.py` | Export to GGUF, 16-bit, 4-bit |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |

## LoRA Configuration

```json
{{
  "r": {lora_config.get("r", 16)},
  "lora_alpha": {lora_config.get("lora_alpha", 16)},
  "lora_dropout": {lora_config.get("lora_dropout", 0)},
  "target_modules": {json.dumps(lora_config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))}
}}
```

**What these mean:**
- **r (rank)**: Lower = smaller adapter, faster training. 16 is a good default.
- **lora_alpha**: Scaling factor. Usually equals r.
- **target_modules**: Which layers to train. More modules = better quality, more memory.

## Training Configuration

| Setting | Value |
|---------|-------|
| Learning Rate | {training_args.get("learning_rate", 2e-4)} |
| Batch Size | {training_args.get("per_device_train_batch_size", 2)} |
| Gradient Accumulation | {training_args.get("gradient_accumulation_steps", 4)} |
| Epochs | {training_args.get("num_train_epochs", 3)} |
| Max Sequence Length | {config.get("max_seq_length", 2048)} |
| Precision | Auto (bf16 if supported, else fp16) |
| Optimizer | AdamW 8-bit |

## Using Your Fine-Tuned Model

### Option 1: With Unsloth (Recommended)

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="{config.get("output_dir", "./output")}",
    max_seq_length=2048,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

messages = [{{"role": "user", "content": "Hello!"}}]
formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(formatted, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Option 2: With HuggingFace (after merging)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("{config.get("output_dir", "./output")}_merged_16bit")
model = AutoModelForCausalLM.from_pretrained("{config.get("output_dir", "./output")}_merged_16bit")
```

### Option 3: With Ollama (after GGUF export)

```bash
ollama create my-model -f Modelfile
ollama run my-model
```

## Export Formats

| Format | Use Case | Size |
|--------|----------|------|
| LoRA Adapter | Load with base model | ~50MB |
| GGUF q4_k_m | llama.cpp, Ollama, LM Studio | ~2-4GB |
| GGUF q8_0 | Higher quality GGUF | ~4-8GB |
| 16-bit Merged | HuggingFace, full precision | ~6-14GB |
| 4-bit Merged | Low-memory inference | ~2-4GB |

## Troubleshooting

### Out of Memory

- Reduce `per_device_train_batch_size` to 1
- Increase `gradient_accumulation_steps` to 8
- Reduce `max_seq_length` to 1024
- Use a smaller model

### Slow Training

- Ensure you're using a GPU
- Check that Unsloth is properly installed
- Use `load_in_4bit=True` (default)

### Chat Template Issues

Different models use different templates. If output looks wrong:
- Check the model's HuggingFace page for the correct template
- Some models need `add_generation_prompt=True`

## Resources

- [Quantune GitHub](https://github.com/riyanshibohra/Quantune)
- [Unsloth GitHub](https://github.com/unslothai/unsloth)
- [HuggingFace PEFT](https://huggingface.co/docs/peft)
- [TRL Library](https://huggingface.co/docs/trl)

---
*Generated by [Quantune](https://github.com/riyanshibohra/Quantune) - Automated Fine-Tuning Pipeline*
'''
    
    return readme


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def generate_package(state: "QuantuneState") -> dict:
    """
    Generate a complete training package using Unsloth for LoRA fine-tuning.
    
    Inputs (from state):
        - base_model: str
        - training_config: dict
        - planning_reasoning: str
        - file_path: str (path to JSONL file)
    
    Generated files:
        - config.json
        - train.py
        - eval.py
        - export.py
        - requirements.txt
        - README.md
    
    Returns:
        - package_path: str (path to generated package folder)
    """
    config = state["training_config"].copy()
    
    # Get the original data file path
    original_data_path = state["file_path"]
    
    # Set base model
    config["base_model"] = state.get("base_model", config.get("model_name", "unsloth/Phi-4"))
    
    # Ensure output_dir is set
    if "output_dir" not in config:
        config["output_dir"] = "./lora_adapter"
    
    # Set good defaults for Unsloth
    if "max_seq_length" not in config:
        config["max_seq_length"] = 2048
    
    # Ensure lora_config exists with good defaults
    if "lora_config" not in config:
        # Detect model architecture for correct target_modules
        model_id = config.get("base_model", "").lower()

        if "phi" in model_id:
            # Phi models use fc1/fc2 for MLP layers
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2"]
        else:
            # Llama, Mistral, Gemma, Qwen use gate_proj/up_proj/down_proj
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

        config["lora_config"] = {
            "r": 16,
            "lora_alpha": 16,
            "lora_dropout": 0,
            "target_modules": target_modules
        }
    
    # Ensure training_args exists with good defaults
    if "training_args" not in config:
        config["training_args"] = {
            "num_train_epochs": 3,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2e-4,
            "warmup_steps": 5,
            "weight_decay": 0.01,
            "logging_steps": 10,
        }
    
    # Create output folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    package_name = f"quantune_lora_{timestamp}"
    package_path = os.path.join("output", package_name)
    os.makedirs(package_path, exist_ok=True)
    
    # Copy data file into the package (makes it self-contained)
    data_filename = "data.jsonl"
    data_dest_path = os.path.join(package_path, data_filename)
    if os.path.exists(original_data_path):
        shutil.copy2(original_data_path, data_dest_path)
        config["data_path"] = f"./{data_filename}"  # Relative path for portability
    else:
        # Fallback to original path if file doesn't exist
        config["data_path"] = original_data_path
    
    # 1. Generate config.json
    config_path = os.path.join(package_path, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    
    # 2. Generate train.py
    train_path = os.path.join(package_path, "train.py")
    with open(train_path, "w", encoding="utf-8") as f:
        f.write(TRAIN_UNSLOTH)
    
    # 3. Generate eval.py
    eval_path = os.path.join(package_path, "eval.py")
    with open(eval_path, "w", encoding="utf-8") as f:
        f.write(EVAL_SCRIPT)
    
    # 4. Generate export.py
    export_path = os.path.join(package_path, "export.py")
    with open(export_path, "w", encoding="utf-8") as f:
        f.write(EXPORT_SCRIPT)
    
    # 5. Generate requirements.txt
    req_path = os.path.join(package_path, "requirements.txt")
    with open(req_path, "w", encoding="utf-8") as f:
        f.write(_generate_requirements())
    
    # 6. Generate README.md
    readme_path = os.path.join(package_path, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(_generate_readme(state))
    
    return {
        "package_path": package_path,
    }
