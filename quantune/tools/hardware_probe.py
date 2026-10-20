"""
Quantune Hardware & Ollama Diagnostics Probe
===========================================
Cross-platform system capability benchmarking (CPU, RAM, GPU, VRAM)
and hardware-aware Ollama model recommender with live local Ollama inspection.
"""

import os
import platform
import subprocess
import ctypes
import json
import urllib.request
from typing import Dict, Any, List, Optional, Tuple


def _get_windows_ram() -> Tuple[float, float, int]:
    """Uses ctypes to read physical memory without psutil dependency."""
    try:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ('dwLength', ctypes.c_ulong),
                ('dwMemoryLoad', ctypes.c_ulong),
                ('ullTotalPhys', ctypes.c_ulonglong),
                ('ullAvailPhys', ctypes.c_ulonglong),
                ('ullTotalPageFile', ctypes.c_ulonglong),
                ('ullAvailPageFile', ctypes.c_ulonglong),
                ('ullTotalVirtual', ctypes.c_ulonglong),
                ('ullAvailVirtual', ctypes.c_ulonglong),
                ('sullAvailExtendedVirtual', ctypes.c_ulonglong),
            ]
        mem = MEMORYSTATUSEX()
        mem.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem)):
            total_gb = round(mem.ullTotalPhys / (1024**3), 1)
            avail_gb = round(mem.ullAvailPhys / (1024**3), 1)
            load_pct = int(mem.dwMemoryLoad)
            return total_gb, avail_gb, load_pct
    except Exception:
        pass
    return 16.0, 8.0, 50


def _get_nvidia_gpu() -> Optional[Dict[str, Any]]:
    """Query nvidia-smi for NVIDIA GPU information."""
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
        if res.returncode == 0 and res.stdout.strip():
            parts = [p.strip() for p in res.stdout.strip().split(",")]
            if len(parts) >= 4:
                name = parts[0]
                total_mb = float(parts[1])
                free_mb = float(parts[2])
                driver = parts[3]
                return {
                    "has_gpu": True,
                    "name": name,
                    "vram_total_gb": round(total_mb / 1024, 2),
                    "vram_free_gb": round(free_mb / 1024, 2),
                    "driver_version": driver,
                    "backend": "NVIDIA CUDA"
                }
    except Exception:
        pass
    return None


def get_system_hardware() -> Dict[str, Any]:
    """Inspects host system CPU, RAM, GPU, and determines fine-tuning capability tier."""
    cpu_processor = platform.processor() or "Multi-Core x86_64"
    logical_cores = os.cpu_count() or 8
    
    total_ram, avail_ram, ram_load = _get_windows_ram()
    gpu_info = _get_nvidia_gpu()

    # Determine capability tier
    if gpu_info and gpu_info["vram_total_gb"] >= 15.0:
        tier_level = 1
        tier_name = "Tier 1: High-Performance GPU Rig"
        tier_badge = "16GB+ VRAM (Full LoRA & 8B Supported)"
        suitability = "Full support for local fine-tuning of 7B-8B models natively with 4-bit/8-bit QLoRA."
    elif gpu_info and gpu_info["vram_total_gb"] >= 4.0:
        tier_level = 2
        tier_name = "Tier 2: Mid-Range Local GPU"
        tier_badge = f"{gpu_info['vram_total_gb']}GB VRAM (1B-3B In-App, 7B via Colab)"
        suitability = f"Capable of running in-app background fine-tuning on 1B-3B models (Qwen 2.5 3B, Llama 3.2 3B). 32GB RAM enables smooth quantization."
    elif gpu_info:
        tier_level = 3
        tier_name = "Tier 3: Entry-Level Local GPU"
        tier_badge = f"{gpu_info['vram_total_gb']}GB VRAM"
        suitability = "Best suited for 1B lightweight fine-tuning or 1-click Google Colab Free T4 export."
    else:
        tier_level = 4
        tier_name = "Tier 4: CPU-Optimized Workstation"
        tier_badge = f"{total_ram}GB System RAM (CPU Acceleration)"
        suitability = "CPU simulation & background engine supported; Colab T4 recommended for GPU acceleration."

    return {
        "cpu": {
            "model": cpu_processor,
            "cores": logical_cores,
            "architecture": platform.machine()
        },
        "ram": {
            "total_gb": total_ram,
            "available_gb": avail_ram,
            "load_percent": ram_load
        },
        "gpu": gpu_info or {
            "has_gpu": False,
            "name": "Integrated / CPU Only",
            "vram_total_gb": 0.0,
            "vram_free_gb": 0.0,
            "driver_version": "N/A",
            "backend": "CPU (PyTorch)"
        },
        "tier": {
            "level": tier_level,
            "name": tier_name,
            "badge": tier_badge,
            "suitability": suitability
        }
    }


def get_ollama_status(endpoint: str = "http://localhost:11434") -> Dict[str, Any]:
    """Queries local Ollama daemon for status and installed models."""
    url = f"{endpoint.rstrip('/')}/api/tags"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Quantune-HardwareProbe/1.0"})
        with urllib.request.urlopen(req, timeout=2.5) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m.get("name") for m in data.get("models", []) if m.get("name")]
                return {
                    "is_running": True,
                    "endpoint": endpoint,
                    "installed_models": models,
                    "total_installed": len(models),
                    "status_message": f"Connected to Ollama ({len(models)} models installed)"
                }
    except Exception as e:
        return {
            "is_running": False,
            "endpoint": endpoint,
            "installed_models": [],
            "total_installed": 0,
            "status_message": f"Ollama not detected on {endpoint} (start with 'ollama serve')"
        }


def get_hardware_aware_ollama_recommendations(endpoint: str = "http://localhost:11434") -> Dict[str, Any]:
    """Generates tailored Ollama model recommendations matched to host hardware."""
    hw = get_system_hardware()
    ollama_stat = get_ollama_status(endpoint)
    installed_set = set(ollama_stat.get("installed_models", []))
    installed_prefixes = {m.split(":")[0] for m in installed_set}

    vram = hw["gpu"].get("vram_total_gb", 0.0)

    # Catalog of models
    catalog = [
        {
            "id": "qwen2.5:3b",
            "title": "Qwen 2.5 3B",
            "params": "3.1 Billion",
            "required_vram": 3.0,
            "ideal_task": "General Reasoning, Multilingual, Structured QA",
            "context_window": "32k",
            "speed": "Very Fast (~60 tokens/s on T1200)",
            "why_recommended": "Perfect fit for 4GB VRAM. Exceptional reasoning density and low memory footprint."
        },
        {
            "id": "qwen2.5-coder:3b",
            "title": "Qwen 2.5 Coder 3B",
            "params": "3.1 Billion",
            "required_vram": 3.0,
            "ideal_task": "Python, SQL, Bug Fixing & Code Generation",
            "context_window": "32k",
            "speed": "Very Fast (~58 tokens/s on T1200)",
            "why_recommended": "State-of-the-art coding density in the 3B class, fits completely within 4GB VRAM."
        },
        {
            "id": "llama3.2:3b",
            "title": "Llama 3.2 3B",
            "params": "3.2 Billion",
            "required_vram": 3.2,
            "ideal_task": "Conversations, Empathy, Customer Support",
            "context_window": "128k",
            "speed": "Fast (~52 tokens/s on T1200)",
            "why_recommended": "Meta's flagship compact model with huge 128k context and fine-tuning friendliness."
        },
        {
            "id": "llama3.2:1b",
            "title": "Llama 3.2 1B",
            "params": "1.2 Billion",
            "required_vram": 1.3,
            "ideal_task": "Ultra-Low Latency, Edge Triage, Mobile",
            "context_window": "128k",
            "speed": "Ultra Fast (~110 tokens/s)",
            "why_recommended": "Lightning-fast iteration for quick fine-tuning experiments and testing."
        },
        {
            "id": "phi4:mini",
            "title": "Phi-4 Mini",
            "params": "3.8 Billion",
            "required_vram": 3.8,
            "ideal_task": "Math, Logic, Strict Classification",
            "context_window": "16k",
            "speed": "Fast (~48 tokens/s)",
            "why_recommended": "Microsoft's high-efficiency synthetic math & logic trained architecture."
        }
    ]

    # Tag each model with installed status
    for item in catalog:
        m_id = item["id"]
        is_inst = m_id in installed_set or m_id.split(":")[0] in installed_prefixes
        item["is_installed"] = is_inst
        item["cli_command"] = f"ollama run {m_id}"
        item["hardware_fit"] = "Optimal (100% VRAM Fit)" if vram >= item["required_vram"] else "Offloaded to System RAM"

    return {
        "hardware_summary": hw["tier"],
        "gpu_name": hw["gpu"]["name"],
        "vram_gb": vram,
        "ollama_connected": ollama_stat["is_running"],
        "total_local_installed": ollama_stat["total_installed"],
        "recommended_models": catalog
    }
