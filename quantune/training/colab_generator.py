"""
Colab Notebook Generator
========================
Generates Jupyter notebooks for training SLMs on Google Colab using Unsloth.
No external dependencies required - notebooks are just JSON!

Unsloth provides:
- 2-5x faster training
- 70% less memory usage
- Native 4-bit quantization
- Easy model export (GGUF, merged, etc.)
"""

import json
import base64
from typing import Dict, List


def new_markdown_cell(source: str) -> dict:
    """Create a markdown cell."""
    # Jupyter expects source as list of lines WITH newlines, or as a single string
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source  # Keep as single string - Jupyter handles it fine
    }


def new_code_cell(source: str, collapsed: bool = False) -> dict:
    """Create a code cell."""
    # Jupyter expects source as list of lines WITH newlines, or as a single string
    metadata = {}
    if collapsed:
        # Colab respects cellView: "form" to collapse cells
        metadata["cellView"] = "form"
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": metadata,
        "outputs": [],
        "source": source  # Keep as single string - Jupyter handles it fine
    }


def get_target_modules(model_id: str) -> list:
    """Get LoRA target modules based on model architecture."""
    model_id_lower = model_id.lower()

    if "phi" in model_id_lower:
        return ["q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2"]
    else:
        # Llama, Mistral, Gemma, Qwen - standard transformer
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def is_gated_model(model_id: str) -> bool:
    """Check if model is likely gated on HuggingFace."""
    model_id_lower = model_id.lower()

    # Known gated model families
    gated_patterns = [
        "meta-llama",
        "llama-2",
        "llama-3",
        "llama2",
        "llama3",
        "google/gemma",
        "gemma-2",
        "mistralai/mistral",
        "mistralai/mixtral",
        "bigscience/bloom",
        "tiiuae/falcon",
    ]

    for pattern in gated_patterns:
        if pattern in model_id_lower:
            return True

    return False


def estimate_training_time(num_examples: int, model_size: str) -> str:
    """Estimate training time based on dataset size and model."""
    # Rough estimates for T4 GPU on Colab with Unsloth (2x faster than standard)
    base_minutes = {
        "2B": 1,
        "3B": 1.5,
        "3.8B": 2,
        "7B": 4,
        "8B": 5,
    }

    base = base_minutes.get(model_size, 2)
    # Scale by dataset size (per 100 examples)
    minutes = base * (num_examples / 100) * 3  # 3 epochs

    if minutes < 5:
        return "~5 minutes"
    elif minutes < 30:
        return f"~{int(minutes)} minutes"
    else:
        hours = minutes / 60
        return f"~{hours:.1f} hours"


def generate_training_notebook(
    dataset_jsonl: str,
    model_id: str,
    model_name: str,
    analysis: dict
) -> str:
    """
    Generate a Jupyter notebook for training on Google Colab using Unsloth.

    Args:
        dataset_jsonl: User's JSONL data as string
        model_id: HuggingFace model ID (e.g., "microsoft/Phi-4-mini-instruct")
        model_name: Display name (e.g., "Phi-4 Mini")
        analysis: Dict from analyze.py with task_type, num_examples, etc.

    Returns:
        String with notebook content in JSON format
    """

    UNSLOTH_MAP = {
        "microsoft/Phi-4-mini-instruct": "unsloth/Phi-4-mini-instruct",
        "google/gemma-2-2b-it": "unsloth/gemma-2-2b-it-bnb-4bit",
        "meta-llama/Llama-3.2-3B-Instruct": "unsloth/Llama-3.2-3B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct": "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
        "mistralai/Mistral-7B-Instruct-v0.3": "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    }
    actual_model_id = UNSLOTH_MAP.get(model_id, model_id)

    # Extract analysis info
    num_examples = analysis.get("num_examples", 0)
    task_type = analysis.get("task_type", "chat")
    model_size = analysis.get("model_size", "3B")

    # Get target modules for this model
    target_modules = get_target_modules(actual_model_id)

    # Check if model is gated (unsloth mirrors are generally un-gated)
    is_gated = is_gated_model(model_id) and not actual_model_id.startswith("unsloth/")

    # Estimate training time
    est_time = estimate_training_time(num_examples, model_size)

    # Encode the JSONL data as base64 to avoid escaping issues
    dataset_b64 = base64.b64encode(dataset_jsonl.encode()).decode()

    # Extract system prompt and sample user prompt from dataset if present
    system_prompt = None
    sample_user_prompt = "Hello! Can you help me?"
    try:
        lines = [l for l in dataset_jsonl.strip().split('\n') if l.strip()]
        if lines:
            first_example = json.loads(lines[0])
            messages = first_example.get("messages", [])
            for m in messages:
                if m.get("role") == "system" and not system_prompt:
                    system_prompt = m.get("content", "")
                elif m.get("role") == "user" and sample_user_prompt == "Hello! Can you help me?":
                    sample_user_prompt = m.get("content", "").strip()
    except (json.JSONDecodeError, IndexError, KeyError):
        pass

    # Build notebook structure
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            },
            "accelerator": "GPU",
            "gpuClass": "standard",
            "colab": {
                "provenance": [],
                "gpuType": "T4"
            }
        },
        "cells": []
    }

    cells = []

    # =========================================================================
    # Cell 1: Title & Info (Markdown)
    # =========================================================================
    gated_warning = ""
    if is_gated:
        gated_warning = f"""
> **Note:** `{model_id}` is a gated model. You need to:
> 1. Accept the license at [huggingface.co/{model_id}](https://huggingface.co/{model_id})
> 2. Login with your HuggingFace token (Cell 3 below)
"""

    title_md = f"""# Quantune: Fine-Tune {model_name}

## Dataset Overview
| Metric | Value |
|--------|-------|
| **Model** | {model_name} (`{model_id}`) |
| **Examples** | {num_examples:,} conversations |
| **Task Type** | {task_type.title()} |
| **Estimated Time** | {est_time} |
{gated_warning}
## Before You Start
1. **GPU Required**: Go to `Runtime` > `Change runtime type` > Select **T4 GPU**
2. **Run All Cells**: Click `Runtime` > `Run all` or press `Ctrl+F9`
3. **Download Model**: After training, download your fine-tuned model from the Files panel

## Why Unsloth?
- **2-5x faster training** with optimized kernels
- **70% less memory** usage
- **Native 4-bit quantization** for efficiency
- **Easy export** to GGUF, 16-bit, 4-bit formats

---
*Generated by [Quantune](https://github.com/riyanshibohra/Quantune) - Fine-tuning made simple*
"""
    cells.append(new_markdown_cell(title_md))

    # =========================================================================
    # Cell 2: Install Dependencies (Unsloth)
    # =========================================================================
    install_code = """%%capture
# Install Unsloth - 2x faster training, 70% less memory
!pip install unsloth
"""
    cells.append(new_code_cell(install_code))

    # GPU verification (separate cell so output is visible)
    gpu_check_code = """# Verify GPU
import torch
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("⚠️ WARNING: No GPU! Go to Runtime > Change runtime type > T4 GPU")
"""
    cells.append(new_code_cell(gpu_check_code))

    # =========================================================================
    # Cell 3: HuggingFace Login (for gated models)
    # =========================================================================
    if is_gated:
        hf_login_md = f"""## HuggingFace Login Required

**`{model_id}`** is a gated model. You must:
1. **Accept the license** at [huggingface.co/{model_id}](https://huggingface.co/{model_id})
2. **Paste your token below** and run the cell

Get your token at: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
"""
        cells.append(new_markdown_cell(hf_login_md))

        hf_login_code = '''#@title HuggingFace Login (Required for this model)
#@markdown Paste your HuggingFace token below. Get one at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
HF_TOKEN = ""  #@param {type:"string"}

if not HF_TOKEN or not HF_TOKEN.startswith("hf_"):
    raise ValueError("⛔ STOP: Please paste your HuggingFace token above (starts with 'hf_') and re-run this cell.")

from huggingface_hub import login
login(token=HF_TOKEN)
print("✅ Successfully logged in to HuggingFace!")
'''
        cells.append(new_code_cell(hf_login_code, collapsed=True))
    else:
        # For non-gated models, add optional login cell
        hf_login_md = """## HuggingFace Login (Optional)

If you want to push your model to HuggingFace Hub later, login now.
Otherwise, skip this cell.
"""
        cells.append(new_markdown_cell(hf_login_md))

        hf_login_code = """# Optional: Login to HuggingFace (for pushing models to Hub)
# Uncomment the lines below if you want to upload your model later

# from huggingface_hub import login
# login()  # Will prompt for your token

# Get your token at: https://huggingface.co/settings/tokens
"""
        cells.append(new_code_cell(hf_login_code))

    # =========================================================================
    # Cell 4: Embed Dataset (collapsed by default - large base64 data)
    # =========================================================================
    dataset_code = f'''#@title Load Training Dataset (click to expand)
import base64
import json

# Your dataset (embedded from Quantune)
DATASET_B64 = """{dataset_b64}"""

# Decode and parse
dataset_jsonl = base64.b64decode(DATASET_B64).decode()
conversations = [json.loads(line) for line in dataset_jsonl.strip().split("\\n") if line.strip()]

print(f"Loaded {{len(conversations):,}} conversations")
print(f"\\nSample conversation:")
if conversations:
    sample = conversations[0]
    for msg in sample.get("messages", [])[:3]:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")[:100]
        print(f"  {{role}}: {{content}}...")
'''
    cells.append(new_code_cell(dataset_code, collapsed=True))

    # =========================================================================
    # Cell 4: Load Model with Unsloth
    # =========================================================================
    load_model_code = f'''# Load model with Unsloth (4-bit quantization for memory efficiency)
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"  # Disable dynamo to avoid compilation errors

from unsloth import FastLanguageModel
import torch

MODEL_ID = "{actual_model_id}"
MAX_SEQ_LENGTH = 2048

print(f"Loading {{MODEL_ID}} with Unsloth...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_ID,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,  # Auto-detect (float16 for T4)
    load_in_4bit=True,  # 4-bit quantization
)

# Configure LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # Rank - lower = smaller adapter, faster training
    lora_alpha=16,  # Scaling factor
    lora_dropout=0,  # No dropout for faster training
    target_modules={json.dumps(target_modules)},
    bias="none",
    use_gradient_checkpointing="unsloth",  # Optimized checkpointing (4x longer context)
    random_state=42,
)

# Print trainable parameters
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Model loaded!")
print(f"Trainable parameters: {{trainable:,}} / {{total:,}} ({{100*trainable/total:.2f}}%)")
'''
    cells.append(new_code_cell(load_model_code))

    # =========================================================================
    # Cell 5: Prepare Dataset
    # =========================================================================
    prepare_data_code = '''# Prepare dataset for training
from datasets import Dataset

def format_conversation(example):
    """Format conversation using chat template."""
    messages = example.get("messages", [])

    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

    return {"text": text}

# Create dataset
dataset = Dataset.from_list(conversations)
dataset = dataset.map(format_conversation, remove_columns=dataset.column_names)

print(f"Dataset prepared: {len(dataset)} examples")
print(f"\\nSample formatted text (first 500 chars):")
print(dataset[0]["text"][:500] + "...")
'''
    cells.append(new_code_cell(prepare_data_code))

    # =========================================================================
    # Cell 6: Train with SFTTrainer
    # =========================================================================
    train_code = '''# Train with SFTTrainer (optimized for Unsloth)
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

training_args = TrainingArguments(
    output_dir="./results",

    # Training params
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,

    # Optimizer
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_steps=5,

    # Precision (auto-detect best for GPU)
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),

    # Logging
    logging_steps=10,

    # Saving
    save_strategy="epoch",
    save_total_limit=2,

    # Optimization
    optim="adamw_8bit",  # 8-bit Adam for memory efficiency
    lr_scheduler_type="linear",
    seed=42,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    packing=False,  # Set True for short sequences
    args=training_args,
)

# Apply response-only loss masking so loss is only calculated on the assistant replies
try:
    from unsloth.chat_templates import train_on_responses_only
    if "llama" in MODEL_ID.lower():
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|start_header_id|>user<|end_header_id|>\\n\\n",
            response_part="<|start_header_id|>assistant<|end_header_id|>\\n\\n",
        )
    elif "phi" in MODEL_ID.lower() or "qwen" in MODEL_ID.lower():
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|im_start|>user\\n",
            response_part="<|im_start|>assistant\\n",
        )
    print("✅ Response loss masking enabled (prompt tokens masked)")
except Exception as e:
    print(f"Standard loss weighting applied: {e}")

# GPU stats before training
gpu_stats = torch.cuda.get_device_properties(0)
start_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 2)
max_memory = round(gpu_stats.total_memory / 1024**3, 2)
print(f"GPU: {gpu_stats.name}")
print(f"Memory before: {start_memory} GB / {max_memory} GB")

print("\\nStarting training...")
print("=" * 50)

trainer_stats = trainer.train()

# GPU stats after training
end_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 2)
print("=" * 50)
print(f"Training complete!")
print(f"Peak GPU memory: {end_memory} GB ({round(end_memory/max_memory*100, 1)}% of {max_memory} GB)")
'''
    cells.append(new_code_cell(train_code))

    # =========================================================================
    # Cell 7: Save Model
    # =========================================================================
    model_slug = model_name.lower().replace(" ", "_").replace("-", "_")
    save_code = f'''# Save the fine-tuned model
import os

OUTPUT_DIR = "./fine_tuned_{model_slug}"

# Save LoRA adapter
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"LoRA adapter saved to: {{OUTPUT_DIR}}")

# List saved files
files = os.listdir(OUTPUT_DIR)
print(f"\\nFiles saved:")
for f in files:
    size = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1024
    print(f"  {{f}} ({{size:.1f}} KB)")
'''
    cells.append(new_code_cell(save_code))

    # =========================================================================
    # Cell 8: Test Inference
    # =========================================================================
    # Build system prompt setup code if present
    if system_prompt:
        # Escape the system prompt for Python string
        escaped_system_prompt = system_prompt.replace('\\', '\\\\').replace("'''", "\\'\\'\\'").replace('"""', '\\"\\"\\"')
        system_prompt_code = f'''# System prompt from your training data
SYSTEM_PROMPT = """{escaped_system_prompt}"""

'''
        messages_code = '''messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]'''
    else:
        system_prompt_code = ""
        messages_code = '''messages = [{"role": "user", "content": prompt}]'''

    test_code = f'''# Test your fine-tuned model
FastLanguageModel.for_inference(model)  # Enable faster inference

{system_prompt_code}def generate_response(prompt, max_new_tokens=256):
    """Generate a response from the fine-tuned model."""
    {messages_code}

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        do_sample=True,
        use_cache=True,
    )

    # Decode only new tokens
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response

# Test prompts (using a real example from your dataset + standard greeting)
sample_prompt_clean = """{sample_user_prompt.replace('"', '\\"')}"""
test_prompts = [
    sample_prompt_clean,
    "Summarize your core capabilities in one concise sentence.",
]

print("Testing fine-tuned model:")
print("=" * 50)
for prompt in test_prompts:
    print(f"\\nUser: {{prompt}}")
    response = generate_response(prompt)
    print(f"Assistant: {{response}}")
    print("-" * 50)
'''
    cells.append(new_code_cell(test_code))

    # =========================================================================
    # Cell 9: Export Options (Markdown)
    # =========================================================================
    export_md = """## Export Options

Run the cells below to export your model to different formats.
"""
    cells.append(new_markdown_cell(export_md))

    # =========================================================================
    # Cell 10: Export to GGUF (optional, skipped by default)
    # =========================================================================
    export_gguf_code = f'''#@title Export to GGUF (for Ollama, LM Studio) - Optional
#@markdown Check the box to enable export. **Adds ~5-10 min.**
run_gguf_export = False  #@param {{type:"boolean"}}

if run_gguf_export:
    GGUF_OUTPUT = "./fine_tuned_{model_slug}_gguf"
    QUANT_METHOD = "q4_k_m"  # Good balance of size/quality

    print(f"Exporting to GGUF ({{QUANT_METHOD}})...")
    model.save_pretrained_gguf(GGUF_OUTPUT, tokenizer, quantization_method=QUANT_METHOD)
    print(f"GGUF model saved to: {{GGUF_OUTPUT}}")

    import shutil
    shutil.make_archive(GGUF_OUTPUT, 'zip', GGUF_OUTPUT)
    print(f"\\nDownload: {{GGUF_OUTPUT}}.zip")
else:
    print("Skipped GGUF export. Check the box above and re-run if needed.")
'''
    cells.append(new_code_cell(export_gguf_code, collapsed=True))

    # =========================================================================
    # Cell 11: Export Merged Model (optional, skipped by default)
    # =========================================================================
    export_merged_code = f'''#@title Export Merged Model (for HuggingFace) - Optional
#@markdown Check the box to enable export. **Adds ~3-5 min.**
run_merged_export = False  #@param {{type:"boolean"}}

if run_merged_export:
    MERGED_OUTPUT = "./fine_tuned_{model_slug}_merged"

    print("Exporting 16-bit merged model...")
    model.save_pretrained_merged(MERGED_OUTPUT + "_16bit", tokenizer, save_method="merged_16bit")
    print(f"Saved to: {{MERGED_OUTPUT}}_16bit")
else:
    print("Skipped merged export. Check the box above and re-run if needed.")
'''
    cells.append(new_code_cell(export_merged_code, collapsed=True))

    # =========================================================================
    # Cell 12: Download Instructions (Markdown)
    # =========================================================================
    download_md = f"""## Download Your Model

1. **Click the folder icon** in the left sidebar
2. **Navigate to the output folder** (e.g., `fine_tuned_{model_slug}`)
3. **Right-click** on the file/folder > **Download**

### Using Your Model

**With Unsloth (recommended):**
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./fine_tuned_{model_slug}",
    max_seq_length=2048,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)
```

**With HuggingFace (after merging):**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./fine_tuned_{model_slug}_merged_16bit")
tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_{model_slug}_merged_16bit")
```

**With Ollama (after GGUF export):**
```bash
# Create Modelfile
echo 'FROM ./fine_tuned_{model_slug}_gguf/model.gguf' > Modelfile
ollama create my-model -f Modelfile
ollama run my-model
```

---
*Happy fine-tuning!*
"""
    cells.append(new_markdown_cell(download_md))

    # Add all cells to notebook
    notebook["cells"] = cells

    # Return as JSON string
    return json.dumps(notebook, indent=2)


def save_notebook(notebook_content: str, output_path: str) -> str:
    """Save notebook content to a file."""
    with open(output_path, 'w') as f:
        f.write(notebook_content)
    return output_path
