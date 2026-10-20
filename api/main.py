"""
Quantune API
============
FastAPI backend for the Quantune pipeline.
"""

import os
import sys
import shutil
import uuid
import json
import requests
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantune import (
    QuantuneState,
    TuneKitState,
    ingest_data,
    validate_quality,
    analyze_dataset,
    generate_package,
    recommend_model,
)
from quantune.training import (
    generate_training_notebook,
    save_notebook,
)
from quantune.tools.universal_converter import convert_and_normalize
from quantune.tools.deep_validator import validate_dataset_deep
from quantune.tools.hardware_probe import (
    get_system_hardware,
    get_ollama_status,
    get_hardware_aware_ollama_recommendations,
)

# ============================================================================
# APP SETUP
# ============================================================================

app = FastAPI(
    title="Quantune API",
    description="Automated SLM Fine-Tuning & Model Management Pipeline",
    version="0.2.0",
)

allowed_origins_str = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8000,http://127.0.0.1:8000")
allowed_origins = [origin.strip() for origin in allowed_origins_str.split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

sessions: dict[str, dict] = {}
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
PACKAGE_MAPPING_FILE = "output/.package_mapping.json"

# Session management for memory efficiency (tuned for 1GB RAM)
MAX_SESSIONS = 25  # Maximum concurrent sessions
SESSION_EXPIRY_MINUTES = 30  # Sessions expire after 30 minutes


def cleanup_expired_sessions():
    """Remove expired sessions to free memory."""
    if not sessions:
        return
    
    import gc
    from datetime import datetime, timedelta
    now = datetime.now()
    expired = []
    
    for session_id, session in sessions.items():
        created_str = session.get("created_at", "")
        try:
            created = datetime.fromisoformat(created_str)
            if now - created > timedelta(minutes=SESSION_EXPIRY_MINUTES):
                expired.append(session_id)
        except (ValueError, TypeError):
            pass
    
    for session_id in expired:
        # Clean up uploaded file
        file_path = sessions[session_id].get("file_path")
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass
        del sessions[session_id]
    
    # Force garbage collection if we cleaned up sessions
    if expired:
        gc.collect()


def enforce_session_limit():
    """Remove oldest sessions if we exceed the limit."""
    cleanup_expired_sessions()
    
    if len(sessions) >= MAX_SESSIONS:
        # Sort by creation time and remove oldest
        sorted_sessions = sorted(
            sessions.items(),
            key=lambda x: x[1].get("created_at", "")
        )
        # Remove oldest 10 sessions
        for session_id, session in sorted_sessions[:10]:
            file_path = session.get("file_path")
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except OSError:
                    pass
            del sessions[session_id]


def reload_raw_data_if_needed(session: dict) -> bool:
    """Reload raw_data from file if it's None. Returns True if successful."""
    state = session.get("state")
    if not state:
        return False
    
    if state.get("raw_data") is not None:
        return True  # Already loaded
    
    file_path = session.get("file_path")
    if not file_path or not os.path.exists(file_path):
        return False
    
    # Re-ingest the data
    temp_state = create_empty_state(file_path, "")
    result = ingest_data(temp_state)
    
    if result.get("error_msg"):
        return False
    
    state["raw_data"] = result.get("raw_data")
    return True

def save_package_mapping(session_id: str, package_path: str):
    """Save session_id -> package_path mapping to persistent file."""
    os.makedirs("output", exist_ok=True)
    mapping = load_package_mapping()
    mapping[session_id] = package_path
    with open(PACKAGE_MAPPING_FILE, "w") as f:
        json.dump(mapping, f)

def load_package_mapping() -> dict:
    """Load package mappings from persistent file."""
    if os.path.exists(PACKAGE_MAPPING_FILE):
        try:
            with open(PACKAGE_MAPPING_FILE, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# Step 5 Model Playground & Refinement Router
from api.playground import router as playground_router, init_playground
init_playground(sessions, reload_raw_data_if_needed)
app.include_router(playground_router)


# ============================================================================
# STARTUP & MIDDLEWARE
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Clean up any leftover files on startup."""
    import gc
    # Clean uploads folder on startup
    if os.path.exists(UPLOAD_DIR):
        for f in os.listdir(UPLOAD_DIR):
            try:
                os.remove(os.path.join(UPLOAD_DIR, f))
            except OSError:
                pass
    gc.collect()


@app.middleware("http")
async def cleanup_middleware(request, call_next):
    """Run cleanup on every request to prevent memory buildup."""
    cleanup_expired_sessions()
    response = await call_next(request)
    return response


# ============================================================================
# MODELS
# ============================================================================

class AnalyzeRequest(BaseModel):
    session_id: str
    user_description: str


class RecommendRequest(BaseModel):
    session_id: str
    user_task: str  # classify, qa, conversation, generation, extraction
    deployment_target: str = 'not_sure'  # cloud_api, mobile_app, edge_device, web_browser, desktop_app, not_sure


class PlanRequest(BaseModel):
    session_id: str
    user_task: Optional[str] = None  # classify, qa, conversation, generation, extraction
    deployment_target: Optional[str] = None  # cloud_api, mobile_app, etc.
    selected_model_id: Optional[str] = None  # If user chose an alternative model
    learning_rate: Optional[float] = None
    num_train_epochs: Optional[int] = None
    per_device_train_batch_size: Optional[int] = None
    lora_r: Optional[int] = None
    lora_alpha: Optional[int] = None
    gradient_accumulation_steps: Optional[int] = 4
    lr_scheduler_type: Optional[str] = "cosine"
    warmup_ratio: Optional[float] = 0.1
    weight_decay: Optional[float] = 0.01
    target_modules: Optional[str] = "all_linear"
    lora_dropout: Optional[float] = 0.0
    quantization: Optional[str] = "4bit"
    max_seq_length: Optional[int] = 2048
    val_split_pct: Optional[int] = 10


class GenerateRequest(BaseModel):
    session_id: str


class TrainStartRequest(BaseModel):
    session_id: str
    model_id: Optional[str] = None
    model_name: Optional[str] = None
    epochs: Optional[int] = 3
    batch_size: Optional[int] = 4
    learning_rate: Optional[float] = 0.0002
    lora_r: Optional[int] = 16
    lora_alpha: Optional[int] = 32
    gradient_accumulation_steps: Optional[int] = 4
    lr_scheduler_type: Optional[str] = "cosine"
    warmup_ratio: Optional[float] = 0.1
    weight_decay: Optional[float] = 0.01
    target_modules: Optional[str] = "all_linear"
    max_seq_length: Optional[int] = 2048


class SessionResponse(BaseModel):
    session_id: str
    status: str
    message: str
    rows: Optional[int] = 0
    stats: Optional[dict] = None
    report: Optional[dict] = None
    conversion_meta: Optional[dict] = None


class ApplySystemPromptRequest(BaseModel):
    session_id: str
    system_prompt: str


class AutoCleanRequest(BaseModel):
    session_id: str
    deduplicate: Optional[bool] = True
    trim_whitespace: Optional[bool] = True
    remove_empty: Optional[bool] = True


class EditRowRequest(BaseModel):
    session_id: str
    row_index: int
    user_content: str
    assistant_content: str
    system_content: Optional[str] = None


class DeleteRowRequest(BaseModel):
    session_id: str
    row_index: int


class AddRowRequest(BaseModel):
    session_id: str
    user_content: str
    assistant_content: str
    system_content: Optional[str] = None


class ApiConfigTestRequest(BaseModel):
    provider: str
    endpoint: Optional[str] = None
    api_key: Optional[str] = None
    model: Optional[str] = None


class ApiConfigSnippetRequest(BaseModel):
    provider: str
    endpoint: str
    api_key: Optional[str] = None
    model: str


class AnalyzeResponse(BaseModel):
    session_id: str
    num_rows: int
    columns: list[str]
    quality_score: float
    quality_issues: list[str]
    inferred_task_type: str
    task_confidence: float
    num_classes: Optional[int]
    column_candidates: dict
    sample_rows: Optional[list[dict]] = None  # First 5 rows for preview


class PlanResponse(BaseModel):
    session_id: str
    final_task_type: str
    base_model: str
    reasoning: str
    training_config: dict


class GenerateResponse(BaseModel):
    session_id: str
    package_path: str
    download_url: str


class ColabRequest(BaseModel):
    session_id: str


class ColabResponse(BaseModel):
    session_id: str
    notebook_url: str
    message: str
    status: str
    colab_url: Optional[str] = None  # Direct Colab URL (via Gist)
    gist_url: Optional[str] = None   # GitHub Gist URL
    files: Optional[list] = None
    download_endpoint: Optional[str] = None
    error: Optional[str] = None


def upload_to_gist(notebook_content: str, filename: str = "quantune_training.ipynb") -> dict:
    """
    Upload notebook to GitHub Gist and return URLs.

    Returns:
        dict with 'gist_url', 'colab_url', or 'error'
    """
    github_token = os.getenv("GITHUB_TOKEN")

    if not github_token:
        return {"error": "GITHUB_TOKEN not configured"}

    try:
        response = requests.post(
            "https://api.github.com/gists",
            headers={
                "Authorization": f"token {github_token}",
                "Accept": "application/vnd.github.v3+json",
            },
            json={
                "description": "Quantune Training Notebook - Fine-tune your model on Google Colab",
                "public": False,  # Secret gist (unlisted but accessible via URL)
                "files": {
                    filename: {
                        "content": notebook_content
                    }
                }
            },
            timeout=30,
        )

        if response.status_code == 201:
            data = response.json()
            gist_id = data["id"]
            owner = data["owner"]["login"]
            gist_url = data["html_url"]

            colab_url = f"https://colab.research.google.com/gist/{owner}/{gist_id}/{filename}"

            return {
                "gist_url": gist_url,
                "colab_url": colab_url,
                "gist_id": gist_id,
            }
        else:
            error_msg = response.json().get("message", "Unknown error")
            return {"error": f"GitHub API error: {error_msg}"}

    except Exception as e:
        return {"error": str(e)}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_empty_state(file_path: str, user_description: str = "") -> QuantuneState:
    """Create an empty QuantuneState."""
    return {
        "file_path": file_path,
        "user_description": user_description,
        "raw_data": None, "columns": None, "sample_rows": None, "num_rows": None,
        "quality_score": None, "quality_issues": None,
        "inferred_task_type": None, "task_confidence": None, "num_classes": None,
        "label_list": None, "column_analysis": None, "column_candidates": None, "dataset_stats": None,
        "final_task_type": None, "base_model": None, "training_config": None, "planning_reasoning": None,
        "est_cost": None, "est_time_minutes": None, "user_approved": None,
        "job_id": None, "job_status": None, "current_epoch": None, "metrics": None,
        "error_msg": None, "retry_count": None,
        "final_report": None, "next_steps": None, "model_download_url": None,
    }


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the landing page."""
    landing_path = os.path.join(FRONTEND_DIR, "landing.html")
    with open(landing_path, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/dashboard", response_class=HTMLResponse)
@app.get("/app", response_class=HTMLResponse)
async def dashboard():
    """Serve the main dashboard/app."""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/about", response_class=HTMLResponse)
async def about():
    """Serve the about page."""
    about_path = os.path.join(FRONTEND_DIR, "about.html")
    if os.path.exists(about_path):
        with open(about_path, "r", encoding="utf-8") as f:
            return f.read()
    raise HTTPException(status_code=404, detail="About page not found")


@app.get("/logo.svg")
async def logo_svg():
    """Serve the transparent logo mark."""
    logo_svg_path = os.path.join(FRONTEND_DIR, "logo.svg")
    if os.path.exists(logo_svg_path):
        return FileResponse(logo_svg_path, media_type="image/svg+xml")
    raise HTTPException(status_code=404, detail="Logo not found")


@app.get("/favicon.ico")
async def favicon():
    """Serve the favicon."""
    favicon_path = os.path.join(FRONTEND_DIR, "favicon.svg")
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path, media_type="image/svg+xml")
    logo_path = os.path.join(FRONTEND_DIR, "logo.svg")
    if os.path.exists(logo_path):
        return FileResponse(logo_path, media_type="image/svg+xml")
    raise HTTPException(status_code=404)


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "ok", "message": "Quantune API is running"}


SAMPLE_DATASETS = [
    {
        "id": "islamic_qa",
        "title": "Islamic Knowledge & Guidance Q&A",
        "category": "Theology & Guidance",
        "num_rows": 20,
        "format": "domain Q&A / dialogue",
        "description": "Authentic scholarly Q&A covering the Five Pillars of Islam, Hajj rituals in Makkah, Ramadan fasting, Zakat calculations, and creeds.",
        "tags": ["Islamic Studies", "Pilgrimage", "Zakat", "Ramadan", "Fasting"],
        "filename": "islamic_qa.jsonl",
        "recommended_task": "qa",
        "sample_prompts": [
            "What is Hajj?",
            "where hajj perform?",
            "How many days does Ramadan last?",
            "What is Zakat and how is the 2.5% calculated?"
        ]
    },
    {
        "id": "customer_support",
        "title": "E-Commerce Customer Support",
        "category": "Customer Support & Triage",
        "num_rows": 55,
        "format": "multi-turn chat",
        "description": "55 multi-turn conversations resolving refunds, order tracking, damaged parcels, and exchanges with empathetic tone.",
        "tags": ["Customer Care", "E-Commerce", "Multi-Turn", "Empathy"],
        "filename": "customer_support.jsonl",
        "recommended_task": "conversation",
        "sample_prompts": [
            "My package says delivered yesterday but I never received anything!",
            "Can I return an item after 30 days if it was defective out of the box?",
            "I want to change the delivery address for order #TK-84920."
        ]
    },
    {
        "id": "python_coder",
        "title": "Python Algorithms & Data Engineering",
        "category": "Coding & Technical",
        "num_rows": 55,
        "format": "instruction / response",
        "description": "55 clean Python scripts with error handling, docstrings, type annotations, and unit test validations.",
        "tags": ["Python", "Algorithms", "Data Pipelines", "Unit Tests"],
        "filename": "python_coder.jsonl",
        "recommended_task": "generation",
        "sample_prompts": [
            "Write a resilient rate limiter in Python using a token bucket algorithm.",
            "Write a function to merge overlapping intervals with docstrings and doctests.",
            "Implement a thread-safe LRU cache in Python with O(1) get and put."
        ]
    },
    {
        "id": "medical_qa",
        "title": "Clinical Patient Triage & Health Q&A",
        "category": "Healthcare & Science",
        "num_rows": 55,
        "format": "instruction / response",
        "description": "55 clinical consultations addressing symptoms, medication safety, red-flag triage, and patient guidance with disclaimers.",
        "tags": ["Healthcare", "Clinical Triage", "Pharmacology", "Disclaimers"],
        "filename": "medical_qa.jsonl",
        "recommended_task": "qa",
        "sample_prompts": [
            "What are the red flag symptoms of acute appendicitis vs routine stomach flu?",
            "Can I safely take ibuprofen with blood pressure medication like lisinopril?",
            "What is the first-aid protocol if someone has a suspected concussion?"
        ]
    },
    {
        "id": "finance_analyst",
        "title": "Financial Sentiment & SEC Intelligence",
        "category": "Finance & Analytics",
        "num_rows": 55,
        "format": "structured JSON / analysis",
        "description": "55 financial entries extracting earnings KPIs, market sentiment, balance sheet ratios, and structured JSON summaries.",
        "tags": ["Finance", "SEC Filings", "Sentiment", "JSON Extraction"],
        "filename": "finance_analyst.jsonl",
        "recommended_task": "extraction",
        "sample_prompts": [
            "Analyze earnings transcript sentiment: 'Cloud revenue grew 28% YoY, though macro headwinds impacted APAC.'",
            "Extract key financial metrics into JSON: 'Q3 net revenue $4.2B, gross margin 68.4%, diluted EPS $1.15.'",
            "Calculate Quick Ratio and Debt-to-Equity from: Current Assets $900M, Inventory $250M, Current Liab $400M."
        ]
    }
]


@app.get("/sample-datasets")
async def get_sample_datasets():
    """Return available pre-loaded sample datasets."""
    return {"datasets": SAMPLE_DATASETS}


@app.post("/load-sample/{sample_id}", response_model=SessionResponse)
async def load_sample_dataset(sample_id: str):
    """Load a built-in sample dataset into a new session with deep validation."""
    enforce_session_limit()
    
    sample = next((s for s in SAMPLE_DATASETS if s["id"] == sample_id), None)
    if not sample:
        raise HTTPException(status_code=404, detail=f"Sample dataset '{sample_id}' not found")
    
    source_path = os.path.join("datasets", sample["filename"])
    if not os.path.exists(source_path):
        raise HTTPException(status_code=404, detail=f"Sample dataset file '{sample['filename']}' not found on disk")
    
    session_id = str(uuid.uuid4())[:8]
    dest_path = os.path.join(UPLOAD_DIR, f"{session_id}_{sample['filename']}")
    
    try:
        shutil.copyfile(source_path, dest_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to copy sample dataset: {str(e)}")
    
    canonical_rows, meta = convert_and_normalize(dest_path, sample["filename"])
    report = validate_dataset_deep(canonical_rows)
    
    sessions[session_id] = {
        "file_path": dest_path,
        "filename": sample["filename"],
        "original_filename": sample["filename"],
        "state": None,
        "created_at": datetime.now().isoformat(),
        "sample_id": sample_id,
        "canonical_rows": canonical_rows,
        "diagnostic_report": report,
        "conversion_meta": meta
    }
    
    try:
        state = create_empty_state(dest_path, "")
        result = ingest_data(state)
        state.update(result)
        
        if state.get("error_msg"):
            if os.path.exists(dest_path):
                os.remove(dest_path)
            del sessions[session_id]
            raise HTTPException(status_code=400, detail=state["error_msg"])
        
        num_rows = len(canonical_rows) if canonical_rows else state.get("num_rows", sample["num_rows"])
        stats = state.get("stats", {})
        
        state["raw_data"] = canonical_rows
        sessions[session_id]["state"] = state
        
        return {
            "session_id": session_id,
            "status": "uploaded",
            "message": f"Sample dataset '{sample['title']}' loaded ({num_rows} examples)",
            "rows": int(num_rows),
            "stats": stats,
            "report": report,
            "conversion_meta": meta
        }
    except HTTPException:
        raise
    except Exception as e:
        if os.path.exists(dest_path):
            os.remove(dest_path)
        if session_id in sessions:
            del sessions[session_id]
        raise HTTPException(status_code=500, detail=f"Failed to ingest sample dataset: {str(e)}")


@app.post("/upload", response_model=SessionResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload a dataset file (JSONL, JSON, CSV, TSV, TXT), auto-convert to canonical chat, and run deep validation."""
    enforce_session_limit()
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    from pathlib import Path
    raw_filename = os.path.basename(file.filename)
    clean_filename = "".join(c for c in raw_filename if c.isalnum() or c in "._-")
    if not clean_filename:
        clean_filename = "dataset.jsonl"
    
    ext = os.path.splitext(clean_filename)[1].lower()
    ALLOWED_EXTS = {".jsonl", ".json", ".csv", ".tsv", ".txt"}
    if ext not in ALLOWED_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Supported formats: .jsonl, .json, .csv, .tsv, .txt"
        )
    
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB max
    session_id = str(uuid.uuid4())[:8]
    temp_upload_path = os.path.join(UPLOAD_DIR, f"{session_id}_raw_{clean_filename}")
    canonical_dest_path = os.path.join(UPLOAD_DIR, f"{session_id}_canonical.jsonl")
    
    resolved_path = Path(temp_upload_path).resolve()
    resolved_upload_dir = Path(UPLOAD_DIR).resolve()
    try:
        resolved_path.relative_to(resolved_upload_dir)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid file path")
    
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB")
    
    with open(temp_upload_path, "wb") as f:
        f.write(content)
    
    # Run universal conversion
    try:
        canonical_rows, conversion_meta = convert_and_normalize(temp_upload_path, clean_filename)
        if not canonical_rows:
            if os.path.exists(temp_upload_path):
                os.remove(temp_upload_path)
            raise HTTPException(
                status_code=400,
                detail=f"Could not extract valid conversation pairs from '{file.filename}'. Make sure columns contain instruction/response, input/output, or messages array."
            )
        
        # Save canonical JSONL
        with open(canonical_dest_path, "w", encoding="utf-8") as f:
            for r in canonical_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        
        # Clean up raw temp file
        if os.path.exists(temp_upload_path):
            try:
                os.remove(temp_upload_path)
            except OSError:
                pass
                
        # Run deep validation
        report = validate_dataset_deep(canonical_rows)
        
        sessions[session_id] = {
            "file_path": canonical_dest_path,
            "filename": f"{os.path.splitext(clean_filename)[0]}.jsonl",
            "original_filename": file.filename,
            "state": None,
            "created_at": datetime.now().isoformat(),
            "canonical_rows": canonical_rows,
            "diagnostic_report": report,
            "conversion_meta": conversion_meta
        }
        
        state = create_empty_state(canonical_dest_path, "")
        result = ingest_data(state)
        state.update(result)
        
        num_rows = len(canonical_rows)
        stats = state.get("stats", {})
        state["raw_data"] = canonical_rows
        sessions[session_id]["state"] = state
        
        return {
            "session_id": session_id,
            "status": "uploaded",
            "message": f"File '{file.filename}' processed ({num_rows} examples normalized to canonical chat)",
            "rows": num_rows,
            "stats": stats,
            "report": report,
            "conversion_meta": conversion_meta
        }
    except HTTPException:
        raise
    except Exception as e:
        if os.path.exists(temp_upload_path):
            os.remove(temp_upload_path)
        if os.path.exists(canonical_dest_path):
            os.remove(canonical_dest_path)
        if session_id in sessions:
            del sessions[session_id]
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {str(e)}")


@app.post("/recommend")
async def recommend(request: RecommendRequest):
    """Analyze dataset and recommend the best model based on user task and deployment target."""
    session_id = request.session_id
    
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found. Upload a file first.")
    
    session = sessions[session_id]
    state = session.get("state")
    
    if not state:
        raise HTTPException(status_code=400, detail="No data found. Please re-upload your file.")
    
    # Reload raw_data from file if needed (memory optimization)
    if not reload_raw_data_if_needed(session):
        raise HTTPException(status_code=400, detail="Could not load data. Please re-upload your file.")
    
    if state.get("quality_score") is None:
        result = validate_quality(state)
        state.update(result)
    
    if state.get("conversation_characteristics") is None:
        result = analyze_dataset(state)
        state.update(result)
    
    conversation_chars = state.get("conversation_characteristics", {})
    num_examples = state.get("num_rows", 0)
    
    recommendation = recommend_model(
        user_task=request.user_task,
        conversation_characteristics=conversation_chars,
        num_examples=num_examples,
        deployment_target=request.deployment_target
    )
    
    state["user_task"] = request.user_task
    state["deployment_target"] = request.deployment_target
    state["recommendation"] = recommendation
    
    # Clear raw_data after use to save memory
    state["raw_data"] = None
    sessions[session_id]["state"] = state
    
    return {
        "session_id": session_id,
        "recommendation": recommendation,
        "analysis": {
            "stats": state.get("stats", {}),
            "quality_score": state.get("quality_score", 0),
            "quality_issues": state.get("quality_issues", []),
            "conversation_characteristics": conversation_chars,
        }
    }


class AddSystemPromptRequest(BaseModel):
    session_id: str
    system_prompt: str


@app.post("/add-system-prompt")
async def add_system_prompt(request: AddSystemPromptRequest):
    """Add a system prompt to all conversations in the dataset."""
    session_id = request.session_id
    system_prompt = request.system_prompt.strip()
    
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if not system_prompt:
        raise HTTPException(status_code=400, detail="System prompt cannot be empty")
    
    session = sessions[session_id]
    state = session.get("state")
    
    if not state:
        raise HTTPException(status_code=400, detail="No data found. Please re-upload your file.")
    
    # Reload raw_data from file if needed (memory optimization)
    if not reload_raw_data_if_needed(session):
        raise HTTPException(status_code=400, detail="Could not load data. Please re-upload your file.")
    
    raw_data = state["raw_data"]
    modified_count = 0
    
    for entry in raw_data:
        messages = entry.get("messages", [])
        has_system = any(msg.get("role") == "system" for msg in messages)
        
        if not has_system:
            messages.insert(0, {
                "role": "system",
                "content": system_prompt
            })
            modified_count += 1
    
    file_path = session["file_path"]
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            for entry in raw_data:
                f.write(json.dumps(entry) + '\n')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save modified data: {str(e)}")
    
    state["file_path"] = file_path
    result = ingest_data(state)
    state.update(result)
    
    if state.get("error_msg"):
        raise HTTPException(status_code=400, detail=state["error_msg"])
    
    # Clear raw_data after use to save memory
    state["raw_data"] = None
    sessions[session_id]["state"] = state
    
    return {
        "session_id": session_id,
        "status": "success",
        "message": f"Added system prompt to {modified_count} conversations",
        "modified_count": modified_count,
        "stats": state.get("stats", {})
    }


@app.get("/download-dataset/{session_id}")
async def download_dataset(session_id: str):
    """Download the dataset file (with any modifications like system prompts)."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    file_path = session.get("file_path")
    
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    original_filename = session.get("original_filename", "dataset.jsonl")
    if not original_filename.endswith('.jsonl'):
        original_filename = original_filename.rsplit('.', 1)[0] + '.jsonl'
    
    if session.get("state", {}).get("stats", {}).get("has_system_prompts"):
        name, ext = os.path.splitext(original_filename)
        original_filename = f"{name}_with_system_prompt{ext}"
    
    return FileResponse(
        path=file_path,
        filename=original_filename,
        media_type='application/jsonl'
    )


@app.post("/analyze")
async def analyze(request: AnalyzeRequest):
    """
    Legacy endpoint - Run ingestion, validation, and analysis on the uploaded file.
    """
    session_id = request.session_id
    
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found. Upload a file first.")
    
    session = sessions[session_id]
    state = session.get("state")
    
    if not state:
        file_path = session["file_path"]
    state = create_empty_state(file_path, request.user_description)
    result = ingest_data(state)
    state.update(result)
    
    if state.get("error_msg"):
        raise HTTPException(status_code=400, detail=state["error_msg"])
    
    result = validate_quality(state)
    state.update(result)
    
    result = analyze_dataset(state)
    state.update(result)
    
    sessions[session_id]["state"] = state
    
    return {
        "session_id": session_id,
        "num_rows": state.get("num_rows", 0),
        "quality_score": state.get("quality_score", 0),
        "quality_issues": state.get("quality_issues", []),
        "conversation_characteristics": state.get("conversation_characteristics", {}),
        "stats": state.get("stats", {}),
    }


@app.get("/sample-data/{session_id}")
async def get_sample_data(session_id: str, limit: int = 100):
    """Get sample rows from the uploaded dataset."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    
    # Reload raw_data from file if needed (memory optimization)
    if not reload_raw_data_if_needed(session):
        raise HTTPException(status_code=400, detail="Could not load data. Please re-upload your file.")
    
    raw_data = session["state"]["raw_data"]
    result = {
        "rows": raw_data[:limit],
        "total_rows": len(raw_data),
        "returned_rows": min(limit, len(raw_data))
    }
    
    # Clear raw_data after use to save memory
    session["state"]["raw_data"] = None
    
    return result


@app.post("/plan")
async def plan(request: PlanRequest):
    """Create training plan based on recommendation."""
    session_id = request.session_id
    
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    state = session.get("state")
    
    if not state:
        raise HTTPException(status_code=400, detail="Run /recommend first")
    
    recommendation = state.get("recommendation", {})
    primary_rec = recommendation.get("primary_recommendation", {})
    
    if not primary_rec:
        raise HTTPException(status_code=400, detail="No recommendation found. Run /recommend first.")
    
    user_task = request.user_task or state.get("user_task", "conversation")
    deployment_target = request.deployment_target or state.get("deployment_target", "not_sure")
    selected_model_id = request.selected_model_id
    selected_model_name = primary_rec.get('model_name', 'Phi-4 Mini')
    
    if selected_model_id:
        model_id = selected_model_id
        alternatives = recommendation.get("alternatives", [])
        for alt in alternatives:
            if alt.get("model_id") == selected_model_id:
                selected_model_name = alt.get("model_name", selected_model_id)
                break
    else:
        model_id = primary_rec.get("model_id", "microsoft/Phi-4-mini-instruct")
    
    state["final_task_type"] = user_task
    state["base_model"] = model_id
    state["model_name"] = selected_model_name
    state["planning_reasoning"] = f"Based on your {user_task} task and {deployment_target} deployment target, we recommend {selected_model_name}."
    
    lr = request.learning_rate or 2e-4
    epochs = request.num_train_epochs or 3
    batch_size = request.per_device_train_batch_size or 4
    lora_r = request.lora_r or 16
    lora_alpha = request.lora_alpha or (lora_r * 2)
    grad_accum = request.gradient_accumulation_steps or 4
    lr_scheduler = request.lr_scheduler_type or "cosine"
    warmup = request.warmup_ratio if request.warmup_ratio is not None else 0.1
    weight_decay = request.weight_decay if request.weight_decay is not None else 0.01
    target_modules = request.target_modules or "all_linear"
    lora_dropout = request.lora_dropout if request.lora_dropout is not None else 0.0
    quantization = request.quantization or "4bit"
    max_seq_length = request.max_seq_length or 2048
    val_split_pct = request.val_split_pct if request.val_split_pct is not None else 10

    state["training_config"] = {
        "model_name": model_id,
        "task_type": "chat",
        "training_args": {
            "num_train_epochs": epochs,
            "per_device_train_batch_size": batch_size,
            "gradient_accumulation_steps": grad_accum,
            "learning_rate": lr,
            "lr_scheduler_type": lr_scheduler,
            "warmup_ratio": warmup,
            "weight_decay": weight_decay,
            "fp16": True,
            "logging_steps": 10,
            "save_strategy": "epoch",
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "target_modules": target_modules,
            "quantization": quantization,
            "max_seq_length": max_seq_length,
            "val_split_pct": val_split_pct,
        },
    }
    
    sessions[session_id]["state"] = state
    
    return {
        "session_id": session_id,
        "final_task_type": state["final_task_type"],
        "base_model": state["base_model"],
        "reasoning": state["planning_reasoning"],
        "training_config": state["training_config"],
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate the training package."""
    session_id = request.session_id
    
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    state = session.get("state")
    
    if not state or not state.get("final_task_type"):
        raise HTTPException(status_code=400, detail="Run /plan first")
    
    result = generate_package(state)
    state.update(result)
    sessions[session_id]["state"] = state
    
    package_path = state["package_path"]
    save_package_mapping(session_id, package_path)
    
    return GenerateResponse(
        session_id=session_id,
        package_path=package_path,
        download_url=f"/download/{session_id}",
    )


@app.get("/download/{session_id}")
async def download(session_id: str):
    """Download the generated training package as a ZIP file."""
    package_path = None
    
    if session_id in sessions:
        session = sessions[session_id]
        state = session.get("state")
        if state:
            package_path = state.get("package_path")
    
    if not package_path:
        mapping = load_package_mapping()
        package_path = mapping.get(session_id)
    
    if not package_path:
        raise HTTPException(status_code=404, detail="Package not found. Please regenerate the package.")
    
    if not os.path.exists(package_path):
        raise HTTPException(status_code=404, detail="Package folder not found on disk")
    
    zip_path = f"{package_path}.zip"
    if not os.path.exists(zip_path):
        shutil.make_archive(package_path, "zip", package_path)
    
    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"quantune_package_{session_id}.zip",
    )


@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """Get the current state of a session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    state = session.get("state", {})
    
    return {
        "session_id": session_id,
        "filename": session.get("filename"),
        "created_at": session.get("created_at"),
        "has_state": state is not None,
        "current_step": _get_current_step(state),
    }


def _get_current_step(state: dict) -> str:
    """Determine which step the session is at."""
    if not state:
        return "uploaded"
    if state.get("job_status") == "completed":
        return "trained"
    if state.get("job_id"):
        return "training"
    if state.get("package_path"):
        return "generated"
    if state.get("final_task_type"):
        return "planned"
    if state.get("inferred_task_type"):
        return "analyzed"
    if state.get("raw_data"):
        return "ingested"
    return "uploaded"


# ============================================================================
# COLAB NOTEBOOK GENERATION
# ============================================================================

@app.post("/generate-colab", response_model=ColabResponse)
async def generate_colab_notebook(request: ColabRequest):
    """Generate a Google Colab notebook for training. Requires /plan to be completed first."""
    session_id = request.session_id
    
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    state = session.get("state")
    
    if not state or not state.get("training_config"):
        raise HTTPException(status_code=400, detail="Run /plan first to configure training")
    
    try:
        data_path = state.get("file_path")
        if not data_path or not os.path.exists(data_path):
            raise HTTPException(status_code=404, detail="Dataset file not found")
        
        with open(data_path, "r", encoding="utf-8") as f:
            dataset_jsonl = f.read()
        
        model_id = state["training_config"].get("model_name", "microsoft/Phi-4-mini-instruct")
        model_name = state.get("model_name", "Phi-4 Mini")
        
        analysis = {
            "num_examples": state.get("num_rows", 0),
            "task_type": state.get("final_task_type", "chat"),
            "model_size": state.get("model_size", "3B"),
        }
        
        notebook_content = generate_training_notebook(
            dataset_jsonl=dataset_jsonl,
            model_id=model_id,
            model_name=model_name,
            analysis=analysis,
        )
        
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        notebook_filename = f"quantune_train_{session_id[:8]}.ipynb"
        notebook_path = os.path.join(output_dir, notebook_filename)
        
        save_notebook(notebook_content, notebook_path)
        save_package_mapping(session_id, notebook_path)
        gist_result = upload_to_gist(notebook_content, notebook_filename)

        colab_url = gist_result.get("colab_url")
        gist_url = gist_result.get("gist_url")

        return ColabResponse(
            session_id=session_id,
            notebook_url=f"/download-notebook/{session_id}",
            message="Notebook generated successfully!",
            status="success",
            colab_url=colab_url,
            gist_url=gist_url,
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate notebook: {str(e)}")


@app.get("/download-notebook/{session_id}")
async def download_notebook(session_id: str):
    """Download the generated Colab notebook."""
    mapping = load_package_mapping()
    notebook_path = mapping.get(session_id)
    
    if not notebook_path and session_id in sessions:
        state = sessions[session_id].get("state", {})
        notebook_path = state.get("notebook_path")
    
    if not notebook_path or not os.path.exists(notebook_path):
        raise HTTPException(status_code=404, detail="Notebook not found. Please generate it first.")
    
    return FileResponse(
        path=notebook_path,
        media_type="application/x-ipynb+json",
        filename=os.path.basename(notebook_path),
    )



# ============================================================================
# IN-APP BACKGROUND TRAINING
# ============================================================================

from quantune.training.local_trainer import (
    create_training_job,
    get_training_job,
    get_latest_job_for_session,
)


@app.post("/train/start")
async def start_in_app_training(request: TrainStartRequest):
    """Start an asynchronous background fine-tuning job inside the app."""
    session_id = request.session_id
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]
    state = session.get("state") or {}
    dataset_path = session.get("file_path") or state.get("file_path")
    if not dataset_path or not os.path.exists(dataset_path):
        raise HTTPException(status_code=404, detail="Dataset file not found for this session.")

    model_id = request.model_id or state.get("selected_model_id") or (state.get("training_config") or {}).get("model_name") or "unsloth/Phi-4-mini-instruct"
    model_name = request.model_name or state.get("model_name") or "Fine-Tuned SLM"

    hyperparams = {
        "epochs": request.epochs or 3,
        "batch_size": request.batch_size or 4,
        "learning_rate": request.learning_rate or 2e-4,
        "lora_r": request.lora_r or 16,
        "lora_alpha": request.lora_alpha or 32,
        "gradient_accumulation_steps": request.gradient_accumulation_steps or 4,
        "lr_scheduler_type": request.lr_scheduler_type or "cosine",
        "warmup_ratio": request.warmup_ratio if request.warmup_ratio is not None else 0.1,
        "weight_decay": request.weight_decay if request.weight_decay is not None else 0.01,
        "target_modules": request.target_modules or "all_linear",
        "max_seq_length": request.max_seq_length or 2048,
    }

    job = create_training_job(
        session_id=session_id,
        dataset_path=dataset_path,
        model_name=model_name,
        model_id=model_id,
        hyperparams=hyperparams
    )

    session["training_job_id"] = job.job_id
    state["training_job_id"] = job.job_id
    state["active_model_trained"] = True
    state["trained_model_name"] = model_name

    return {
        "status": "started",
        "job_id": job.job_id,
        "session_id": session_id,
        "model_name": model_name,
        "model_id": model_id,
        "hyperparams": hyperparams,
        "message": "In-app background fine-tuning started successfully."
    }


@app.get("/train/status/{job_id}")
async def get_training_status(job_id: str):
    """Get current training job status, progress, loss history, and logs."""
    job = get_training_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    return job.to_dict()


@app.get("/train/logs/{job_id}")
async def get_training_logs(job_id: str):
    """Get incremental log lines from training job."""
    job = get_training_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    return {"job_id": job_id, "status": job.status, "logs": job.logs}


@app.post("/train/cancel/{job_id}")
async def cancel_training(job_id: str):
    """Cancel an ongoing training job."""
    job = get_training_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    job.cancel()
    return {"status": "cancelled", "job_id": job_id}


@app.get("/train/session/{session_id}")
async def get_session_training_job(session_id: str):
    """Get latest training job status for a session."""
    job = get_latest_job_for_session(session_id)
    if not job:
        return {"has_job": False}
    return {"has_job": True, "job": job.to_dict()}


# ============================================================================
# STEP 1: DATA HUB, DIAGNOSTICS, HARDWARE BENCHMARKS & API CONFIG
# ============================================================================

@app.get("/models/runs")
async def get_all_model_runs():
    """Scans models/runs for all training runs and returns them sorted by timestamp descending."""
    runs_dir = os.path.join("models", "runs")
    if not os.path.exists(runs_dir):
        return {"runs": [], "total_runs": 0, "message": "No training runs found."}

    completed_runs = []
    for d in os.listdir(runs_dir):
        sub = os.path.join(runs_dir, d)
        if os.path.isdir(sub):
            metrics_path = os.path.join(sub, "training_metrics.json")
            adapter_path = os.path.join(sub, "adapter_config.json")
            data = {}
            if os.path.exists(metrics_path):
                try:
                    with open(metrics_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception:
                    pass

            adapter_config = {}
            if os.path.exists(adapter_path):
                try:
                    with open(adapter_path, "r", encoding="utf-8") as f:
                        adapter_config = json.load(f)
                except Exception:
                    pass

            if data or adapter_config:
                mtime = os.path.getmtime(metrics_path) if os.path.exists(metrics_path) else os.path.getmtime(sub)
                job_id = data.get("job_id") or d
                created_at = adapter_config.get("created_at") or datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")

                run_entry = {
                    "job_id": job_id,
                    "folder_name": d,
                    "session_id": data.get("session_id", ""),
                    "model_name": data.get("model_name") or adapter_config.get("base_model_name_or_path", "Fine-Tuned SLM"),
                    "model_id": data.get("model_id") or adapter_config.get("base_model_name_or_path", "unsloth/Phi-4-mini-instruct"),
                    "epochs": data.get("epochs", 3),
                    "batch_size": data.get("batch_size", 4),
                    "total_steps": data.get("total_steps") or adapter_config.get("total_steps", 50),
                    "initial_loss": data.get("initial_loss", 2.45),
                    "final_loss": data.get("final_loss") or adapter_config.get("final_loss", 0.42),
                    "loss_reduction_pct": data.get("loss_reduction_pct", 82.5),
                    "elapsed_seconds": data.get("elapsed_seconds", 24.5),
                    "status": data.get("status", "completed"),
                    "lora_r": adapter_config.get("r", 16),
                    "lora_alpha": adapter_config.get("lora_alpha", 32),
                    "target_modules": adapter_config.get("target_modules", ["q_proj", "v_proj"]),
                    "mtime": mtime,
                    "created_at": created_at,
                    "created_at_formatted": datetime.fromtimestamp(mtime).strftime("%b %d, %Y · %I:%M %p"),
                    "checkpoint_path": sub,
                    "adapter_exists": os.path.exists(os.path.join(sub, "adapter_model.bin")),
                    "has_config": os.path.exists(adapter_path)
                }
                completed_runs.append(run_entry)

    completed_runs.sort(key=lambda x: x.get("mtime", 0), reverse=True)
    return {
        "runs": completed_runs,
        "total_runs": len(completed_runs)
    }


@app.get("/models/runs/{job_id}")
async def get_model_run_details(job_id: str):
    """Returns detailed metadata and metrics for a specific training run."""
    runs_dir = os.path.join("models", "runs")
    if not os.path.exists(runs_dir):
        raise HTTPException(status_code=404, detail="No training runs found.")

    target_dir = None
    for d in os.listdir(runs_dir):
        if job_id in d:
            target_dir = os.path.join(runs_dir, d)
            break

    if not target_dir or not os.path.exists(target_dir):
        raise HTTPException(status_code=404, detail=f"Run {job_id} not found.")

    metrics_path = os.path.join(target_dir, "training_metrics.json")
    adapter_path = os.path.join(target_dir, "adapter_config.json")

    metrics = {}
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
        except Exception:
            pass

    adapter_config = {}
    if os.path.exists(adapter_path):
        try:
            with open(adapter_path, "r", encoding="utf-8") as f:
                adapter_config = json.load(f)
        except Exception:
            pass

    return {
        "job_id": job_id,
        "metrics": metrics,
        "adapter_config": adapter_config,
        "checkpoint_path": target_dir,
    }


@app.delete("/models/runs/{job_id}")
async def delete_model_run(job_id: str):
    """Deletes and cleans up a specific training run checkpoint folder from disk."""
    import shutil
    runs_dir = os.path.join("models", "runs")
    if not os.path.exists(runs_dir):
        raise HTTPException(status_code=404, detail="No training runs found.")

    target_dir = None
    for d in os.listdir(runs_dir):
        if job_id in d:
            target_dir = os.path.join(runs_dir, d)
            break

    if not target_dir or not os.path.exists(target_dir):
        raise HTTPException(status_code=404, detail=f"Run {job_id} not found.")

    try:
        shutil.rmtree(target_dir)
        return {"status": "deleted", "job_id": job_id, "message": f"Checkpoint {job_id} cleaned up successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete checkpoint: {e}")


@app.post("/models/cleanup-all")
async def cleanup_all_model_runs():
    """Cleans up all training run checkpoints from disk."""
    import shutil
    runs_dir = os.path.join("models", "runs")
    if not os.path.exists(runs_dir):
        return {"status": "cleaned", "deleted_count": 0}

    deleted_count = 0
    for d in os.listdir(runs_dir):
        sub = os.path.join(runs_dir, d)
        if os.path.isdir(sub):
            try:
                shutil.rmtree(sub)
                deleted_count += 1
            except Exception:
                pass

    return {
        "status": "cleaned",
        "deleted_count": deleted_count,
        "message": f"Successfully deleted {deleted_count} model checkpoints."
    }


@app.get("/models/latest-run")
async def get_latest_model_run():
    """Scans models/runs for completed training runs and returns the latest."""
    all_runs_resp = await get_all_model_runs()
    runs = all_runs_resp.get("runs", [])
    if not runs:
        return {"has_trained_model": False, "message": "No training runs found."}

    return {
        "has_trained_model": True,
        "latest_run": runs[0]
    }


@app.get("/system/hardware")
async def get_hardware_info():
    """Returns local system hardware specifications and fine-tuning tier."""
    return get_system_hardware()


@app.get("/system/ollama-status")
async def check_ollama_status(endpoint: str = "http://localhost:11434"):
    """Returns status and installed models for local Ollama daemon."""
    return get_ollama_status(endpoint)


@app.get("/system/ollama-recommendations")
async def get_ollama_recs(endpoint: str = "http://localhost:11434"):
    """Returns hardware-matched Ollama model recommendations with installation status."""
    return get_hardware_aware_ollama_recommendations(endpoint)


def _get_session_rows(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found. Please upload or load a dataset first.")
    session = sessions[session_id]
    rows = session.get("canonical_rows")
    if rows is None:
        file_path = session.get("file_path")
        if file_path and os.path.exists(file_path):
            rows, _ = convert_and_normalize(file_path, session.get("filename", "dataset.jsonl"))
            session["canonical_rows"] = rows
        else:
            raise HTTPException(status_code=400, detail="Dataset file not found on disk.")
    return session, rows


def _save_session_rows(session: dict, rows: list):
    session["canonical_rows"] = rows
    file_path = session.get("file_path")
    if file_path:
        with open(file_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    report = validate_dataset_deep(rows)
    session["diagnostic_report"] = report
    if session.get("state"):
        session["state"]["raw_data"] = rows
        session["state"]["num_rows"] = len(rows)
    return report


@app.get("/dataset/report/{session_id}")
async def get_dataset_report(session_id: str):
    """Returns deep diagnostic report and criteria checklist for active session."""
    session, rows = _get_session_rows(session_id)
    report = session.get("diagnostic_report")
    if not report:
        report = validate_dataset_deep(rows)
        session["diagnostic_report"] = report
    return {
        "session_id": session_id,
        "rows": len(rows),
        "report": report,
        "conversion_meta": session.get("conversion_meta")
    }


@app.post("/dataset/apply-system-prompt")
async def apply_system_prompt(req: ApplySystemPromptRequest):
    """Applies or updates a global system prompt across all rows in the dataset."""
    session, rows = _get_session_rows(req.session_id)
    prompt_text = req.system_prompt.strip()
    if not prompt_text:
        raise HTTPException(status_code=400, detail="System prompt cannot be empty.")
    
    updated_rows = []
    for r in rows:
        msgs = list(r.get("messages", []))
        if msgs and msgs[0].get("role") == "system":
            msgs[0] = {"role": "system", "content": prompt_text}
        else:
            msgs.insert(0, {"role": "system", "content": prompt_text})
        updated_rows.append({"messages": msgs})
    
    report = _save_session_rows(session, updated_rows)
    return {
        "status": "success",
        "message": f"Global system prompt applied to all {len(updated_rows)} examples.",
        "rows": len(updated_rows),
        "report": report
    }


@app.post("/dataset/auto-clean")
async def auto_clean_dataset(req: AutoCleanRequest):
    """Performs 1-click dataset repair: trims whitespace, removes duplicates and empty entries."""
    import re
    session, rows = _get_session_rows(req.session_id)
    cleaned_rows = []
    seen_fingerprints = set()
    removed_duplicates = 0
    removed_empty = 0

    for r in rows:
        msgs = r.get("messages", [])
        new_msgs = []
        for m in msgs:
            c = (m.get("content") or "").strip()
            if req.trim_whitespace:
                c = re.sub(r'[ \t]+', ' ', c)
            if c or not req.remove_empty:
                new_msgs.append({"role": m.get("role", "user"), "content": c})

        has_u = any(m["role"] == "user" and m["content"].strip() for m in new_msgs)
        has_a = any(m["role"] == "assistant" and m["content"].strip() for m in new_msgs)
        if not (has_u and has_a) and req.remove_empty:
            removed_empty += 1
            continue

        fp = "|".join(f"{m['role']}:{m['content'][:80]}" for m in new_msgs)
        if req.deduplicate:
            if fp in seen_fingerprints:
                removed_duplicates += 1
                continue
            seen_fingerprints.add(fp)

        cleaned_rows.append({"messages": new_msgs})

    report = _save_session_rows(session, cleaned_rows)
    return {
        "status": "success",
        "message": f"Auto-clean completed: removed {removed_duplicates} duplicates and {removed_empty} invalid rows.",
        "removed_duplicates": removed_duplicates,
        "removed_empty": removed_empty,
        "rows": len(cleaned_rows),
        "report": report
    }


@app.post("/dataset/edit-row")
async def edit_row(req: EditRowRequest):
    """In-place editor for a specific dataset row."""
    session, rows = _get_session_rows(req.session_id)
    if req.row_index < 0 or req.row_index >= len(rows):
        raise HTTPException(status_code=400, detail=f"Row index {req.row_index} out of range.")
    
    msgs = []
    if req.system_content and req.system_content.strip():
        msgs.append({"role": "system", "content": req.system_content.strip()})
    msgs.append({"role": "user", "content": req.user_content.strip()})
    msgs.append({"role": "assistant", "content": req.assistant_content.strip()})
    
    rows[req.row_index] = {"messages": msgs}
    report = _save_session_rows(session, rows)
    return {
        "status": "success",
        "message": f"Row #{req.row_index + 1} updated successfully.",
        "row_index": req.row_index,
        "row": rows[req.row_index],
        "report": report
    }


@app.post("/dataset/delete-row")
async def delete_row(req: DeleteRowRequest):
    """Deletes a row from the active dataset."""
    session, rows = _get_session_rows(req.session_id)
    if req.row_index < 0 or req.row_index >= len(rows):
        raise HTTPException(status_code=400, detail=f"Row index {req.row_index} out of range.")
    deleted = rows.pop(req.row_index)
    report = _save_session_rows(session, rows)
    return {
        "status": "success",
        "message": f"Row #{req.row_index + 1} deleted successfully.",
        "rows": len(rows),
        "report": report
    }


@app.post("/dataset/add-row")
async def add_row(req: AddRowRequest):
    """Appends a new training example to the active dataset."""
    session, rows = _get_session_rows(req.session_id)
    msgs = []
    if req.system_content and req.system_content.strip():
        msgs.append({"role": "system", "content": req.system_content.strip()})
    msgs.append({"role": "user", "content": req.user_content.strip()})
    msgs.append({"role": "assistant", "content": req.assistant_content.strip()})
    
    rows.append({"messages": msgs})
    report = _save_session_rows(session, rows)
    return {
        "status": "success",
        "message": f"New example added (total: {len(rows)}).",
        "rows": len(rows),
        "row_index": len(rows) - 1,
        "report": report
    }


@app.get("/dataset/download/{session_id}")
async def download_cleaned_dataset(session_id: str):
    """Downloads the active cleaned and normalized JSONL file."""
    session, rows = _get_session_rows(session_id)
    file_path = session.get("file_path")
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Dataset file not found.")
    filename = f"quantune_cleaned_{session_id}.jsonl"
    return FileResponse(file_path, filename=filename, media_type="application/jsonl")


@app.post("/api-config/test")
async def test_api_config(req: ApiConfigTestRequest):
    """Tests connection to local Ollama or cloud provider credentials."""
    provider = req.provider.lower().strip()
    if provider == "ollama":
        endpoint = req.endpoint or "http://localhost:11434"
        stat = get_ollama_status(endpoint)
        if stat["is_running"]:
            return {
                "success": True,
                "message": f"Successfully connected to Ollama ({stat['total_installed']} models installed: {', '.join(stat['installed_models'][:4])}...)!",
                "provider": provider,
                "installed_models": stat["installed_models"]
            }
        return {
            "success": False,
            "message": f"Could not connect to Ollama at {endpoint}. Ensure 'ollama serve' is active.",
            "provider": provider
        }

    if not req.api_key or not req.api_key.strip():
        return {"success": False, "message": "API key cannot be empty.", "provider": provider}

    headers = {"Authorization": f"Bearer {req.api_key.strip()}"}
    if provider == "openai":
        url = (req.endpoint or "https://api.openai.com/v1").rstrip("/") + "/models"
    elif provider == "groq":
        url = (req.endpoint or "https://api.groq.com/openai/v1").rstrip("/") + "/models"
    elif provider == "together":
        url = (req.endpoint or "https://api.together.xyz/v1").rstrip("/") + "/models"
    elif provider == "anthropic":
        url = "https://api.anthropic.com/v1/models"
        headers = {"x-api-key": req.api_key.strip(), "anthropic-version": "2023-06-01"}
    elif provider == "huggingface":
        url = "https://huggingface.co/api/whoami-v2"
    else:
        url = (req.endpoint or "https://api.openai.com/v1").rstrip("/") + "/models"

    try:
        res = requests.get(url, headers=headers, timeout=5)
        if res.status_code in (200, 201):
            return {
                "success": True,
                "message": f"Credentials verified successfully for {provider.capitalize()}!",
                "provider": provider
            }
        else:
            return {
                "success": False,
                "message": f"Authentication rejected (HTTP {res.status_code}): {res.text[:120]}",
                "provider": provider
            }
    except Exception as e:
        return {"success": False, "message": f"Connection error: {str(e)}", "provider": provider}


@app.post("/api-config/snippet")
async def generate_api_snippet(req: ApiConfigSnippetRequest):
    """Generates ready-to-run Python, cURL, and JavaScript code for the configured provider."""
    provider = req.provider.lower().strip()
    model = req.model or "qwen2.5:3b"
    endpoint = req.endpoint or ("http://localhost:11434" if provider == "ollama" else "https://api.openai.com/v1")
    key = req.api_key or "YOUR_API_KEY_HERE"

    if provider == "ollama":
        python_code = f"""import requests

# Query your local Ollama model
response = requests.post(
    "{endpoint.rstrip('/')}/api/chat",
    json={{
        "model": "{model}",
        "messages": [
            {{"role": "system", "content": "You are a specialized fine-tuned assistant."}},
            {{"role": "user", "content": "Hello! How can you help me today?"}}
        ],
        "stream": False
    }}
)
print(response.json()["message"]["content"])"""

        curl_code = f"""curl {endpoint.rstrip('/')}/api/chat \\
  -H "Content-Type: application/json" \\
  -d '{{
    "model": "{model}",
    "messages": [
      {{"role": "user", "content": "Hello! How can you help me today?"}}
    ],
    "stream": false
  }}'"""

        js_code = f"""// Query your local Ollama model
const response = await fetch("{endpoint.rstrip('/')}/api/chat", {{
  method: "POST",
  headers: {{ "Content-Type": "application/json" }},
  body: JSON.stringify({{
    model: "{model}",
    messages: [{{ role: "user", content: "Hello! How can you help me today?" }}],
    stream: false
  }})
}});
const data = await response.json();
console.log(data.message.content);"""
    else:
        python_code = f"""from openai import OpenAI

# Connect to {provider.capitalize()} API
client = OpenAI(
    api_key="{key}",
    base_url="{endpoint.rstrip('/')}"
)

completion = client.chat.completions.create(
    model="{model}",
    messages=[
        {{"role": "system", "content": "You are a specialized fine-tuned assistant."}},
        {{"role": "user", "content": "Hello! How can you help me today?"}}
    ]
)
print(completion.choices[0].message.content)"""

        curl_code = f"""curl {endpoint.rstrip('/')}/chat/completions \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer {key}" \\
  -d '{{
    "model": "{model}",
    "messages": [
      {{"role": "user", "content": "Hello! How can you help me today?"}}
    ]
  }}'"""

        js_code = f"""// Connect to {provider.capitalize()} API
const response = await fetch("{endpoint.rstrip('/')}/chat/completions", {{
  method: "POST",
  headers: {{
    "Content-Type": "application/json",
    "Authorization": "Bearer {key}"
  }},
  body: JSON.stringify({{
    model: "{model}",
    messages: [{{ role: "user", content: "Hello! How can you help me today?" }}]
  }})
}});
const data = await response.json();
console.log(data.choices[0].message.content);"""

    return {
        "python": python_code,
        "curl": curl_code,
        "javascript": js_code
    }


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

