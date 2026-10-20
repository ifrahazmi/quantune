<div align="center">

<img src="frontend/logo.svg" alt="Quantune Logo" width="110" height="110" style="margin-bottom: 12px; filter: drop-shadow(0 0 20px rgba(99, 102, 241, 0.4));" />

# Quantune

### Autonomous Small Language Model (SLM) Fine-Tuning Studio

Turn raw JSONL datasets into production-grade fine-tuned models in 15 minutes — completely free on Google Colab T4 with Unsloth response-loss masking.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB.svg?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Powered by Unsloth](https://img.shields.io/badge/Powered%20by-Unsloth-FF6F00.svg)](https://github.com/unslothai/unsloth)
[![Author](https://img.shields.io/badge/Author-Ifrahuddin%20Azmi-indigo.svg)](https://github.com/ifrahazmi)

[**Live Demo**](http://localhost:8000) • [**Launch Studio**](http://localhost:8000/dashboard) • [**Architecture & Philosophy**](http://localhost:8000/about)

---

</div>

## 💡 What is Quantune?

Fine-tuning monolithic 70B+ LLMs is slow, expensive, and complex. **Quantune** changes the game by democratizing **Small Language Models (1B–7B parameters)**. 

Using **Unsloth response-loss masking**, Quantune trains models exclusively on completion tokens while masking prompt tokens. This yields a **70% VRAM reduction** and **2x–5x training speedups**, enabling full domain fine-tuning on Google's free T4 GPU at zero cost.

```
Raw JSONL/CSV  ──►  12-Point Diagnostics  ──►  SLM Matcher  ──►  Masked Colab Run  ──►  Accuracy Studio
```

---

## ⚡ Key Features

| Stage | Feature | Description |
| :--- | :--- | :--- |
| **1. Ingest** | **Universal Normalizer** | Parses messy JSONL and CSV files into standard ChatML conversation turns with role auto-detection. |
| **2. Validate** | **12-Point Deep Diagnostics** | Scores datasets from **A+ to F**, detects token anomalies, leaks, duplicate pairs, and schema errors. |
| **3. Match** | **Hardware-Aware SLM Engine** | Probes your available GPU VRAM and recommends optimal models (Phi-4 Mini, Llama 3.2, Mistral 7B). |
| **4. Train** | **Masked Colab Generation** | 1-click generates reproducible Google Colab notebooks configured with Unsloth QLoRA ($0 cost). |
| **5. Evaluate** | **Side-by-Side Accuracy Studio** | Test prompt responses in real time with side-by-side comparison (Base Generic Model vs Fine-Tuned). |

---

## 🚀 Quick Start (Run Locally)

### Prerequisites
- Python 3.10 or higher
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/ifrahazmi/Quantune.git
cd Quantune
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Application
```bash
# On Windows (PowerShell or Command Prompt):
.\run.bat
# or
.\run.ps1

# Or run directly with Uvicorn:
uvicorn api.main:app --reload
```

Open your browser at **[http://localhost:8000](http://localhost:8000)**.

---

## 📊 Dataset Format

Quantune supports the standard conversation format in JSONL:

```jsonl
{"messages": [{"role": "system", "content": "You are a customer support agent for Apex Logistics."}, {"role": "user", "content": "Where is my package #9821?"}, {"role": "assistant", "content": "Let me look up tracking ID #9821 in our portal right now."}]}
{"messages": [{"role": "system", "content": "You are a Python programming assistant."}, {"role": "user", "content": "Write an async HTTP GET request in Python."}, {"role": "assistant", "content": "Use `httpx` or `aiohttp`:\n\n```python\nimport httpx\nasync with httpx.AsyncClient() as client:\n    res = await client.get('https://api.example.com')\n```"}]}
```

> **Tip:** Quantune also includes pre-built sample datasets for Customer Support, Python Coding, Clinical Triage, and Financial Analysis right in the UI.

---

## 🧠 Supported Foundation SLMs

| Model | Size | Context | Best Use Case |
| :--- | :---: | :---: | :--- |
| **Phi-4 Mini** | 3.8B | 128k | Structured JSON, technical reasoning, domain Q&A |
| **Llama 3.2** | 1B / 3B | 128k | Fast conversational chatbots, summarization |
| **Mistral 7B** | 7.2B | 32k | Complex logical tasks, enterprise reasoning |
| **Qwen 2.5** | 1.5B / 3B / 7B | 128k | Multilingual agents, code synthesis, mathematical logic |
| **Gemma 2** | 2B | 8k | Edge compute, low-latency mobile inference |

---

## 📁 Project Structure

```text
Quantune/
├── api/
│   ├── main.py                # FastAPI endpoints & static file router
│   ├── playground.py          # Testing studio & evaluation endpoints
│   └── app.py                 # Core application state
├── frontend/
│   ├── landing.html           # Luxury landing page with animated workflow
│   ├── landing.css            # Dark obsidian & light mode landing styles
│   ├── index.html             # 5-step Launch Studio dashboard
│   ├── style.css              # Studio UI tokens & high-specificity theme rules
│   ├── playground.css         # Testing studio & response pane styles
│   ├── about.html             # Architecture & SLM philosophy page
│   ├── logo.svg               # Transparent vector logo mark
│   └── favicon.svg            # Browser icon
├── quantune/
│   ├── tools/
│   │   ├── ingest.py          # Dataset ingestion & normalization
│   │   ├── deep_validator.py  # 12-point quality scoring engine
│   │   ├── hardware_probe.py  # VRAM & GPU detection
│   │   └── model_rec.py       # SLM recommendation matcher
│   └── training/
│       ├── colab_generator.py # Unsloth Colab notebook exporter
│       └── local_trainer.py   # In-app background training manager
├── datasets/                  # Sample domain datasets (Support, Code, Finance)
├── requirements.txt           # Python dependencies
├── run.bat / run.ps1          # Quick start scripts
└── README.md
```

---

## 👤 Author

Crafted with precision by **Ifrahuddin Azmi**:
- **GitHub:** [@ifrahazmi](https://github.com/ifrahazmi)
- **LinkedIn:** [Ifrahuddin Azmi](https://www.linkedin.com/in/ifrahuddin-azmi/)
- **Email:** `ifrahazmi@live.com`

---

## 📄 License

This project is open-source under the [MIT License](LICENSE) — free for personal and commercial use.
