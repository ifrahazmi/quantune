"""
TuneKit Pipeline State
======================
Single source of truth for all data flowing through the graph.
"""

from typing import TypedDict, Literal, Optional, List, Dict, Any


class TuneKitState(TypedDict):
    """
    The state object that flows through the entire TuneKit pipeline.
    """
    
    # USER INPUT
    file_path: str                                 # Path to CSV/JSONL file
    user_description: str                          # Natural language description of the task
    
    # -------------------------------------------------------------------------
    # STAGE 1: INGESTION (output of ingest_data tool)
    # -------------------------------------------------------------------------
    raw_data: Optional[List[Dict[str, Any]]]       # All rows as list of dicts
    columns: Optional[List[str]]                   # Column names (legacy, not used for JSONL)
    sample_rows: Optional[List[Dict[str, Any]]]    # First 5 rows for preview
    num_rows: Optional[int]                        # Total row count
    stats: Optional[Dict[str, Any]]                # Dataset statistics from ingest
    
    # -------------------------------------------------------------------------
    # STAGE 2: VALIDATION (output of validate_quality tool)
    # -------------------------------------------------------------------------
    quality_score: Optional[float]                 # 0.0 to 1.0 (1.0 = perfect)
    quality_issues: Optional[List[str]]            # List of problems found
    
    # -------------------------------------------------------------------------
    # STAGE 3: ANALYSIS (output of analyze_dataset tool)
    # -------------------------------------------------------------------------
    inferred_task_type: Optional[Literal["classification", "ner", "instruction_tuning"]]
    task_confidence: Optional[float]               # 0.0 to 1.0 (how confident)
    num_classes: Optional[int]                     # For classification tasks
    label_list: Optional[List[str]]                # Actual labels found
    column_analysis: Optional[Dict[str, Any]]      # Analysis of each column
    column_candidates: Optional[Dict[str, List[str]]]  # Candidates for each role
    dataset_stats: Optional[Dict[str, Any]]        # Detailed statistics
    
    # -------------------------------------------------------------------------
    # STAGE 4: PLANNING - AGENT 1 (output of planning_agent)
    # -------------------------------------------------------------------------
    final_task_type: Optional[Literal["classification", "ner", "instruction_tuning"]]
    base_model: Optional[str]                      # e.g., "bert-base-uncased"
    training_config: Optional[Dict[str, Any]]      # Full training configuration
    planning_reasoning: Optional[str]              # Agent's explanation
    
    # -------------------------------------------------------------------------
    # STAGE 5: COST ESTIMATION (output of estimate_cost tool)
    # -------------------------------------------------------------------------
    est_cost: Optional[float]                      # Estimated cost in USD
    est_time_minutes: Optional[int]                # Estimated training time
    
    # -------------------------------------------------------------------------
    # STAGE 6: HUMAN REVIEW (checkpoint)
    # -------------------------------------------------------------------------
    user_approved: Optional[bool]                  # Did user approve?
    
    # -------------------------------------------------------------------------
    # STAGE 7: TRAINING (output of launch_training tool)
    # -------------------------------------------------------------------------
    job_id: Optional[str]                          # Training job ID
    job_status: Optional[Literal["pending", "running", "completed", "failed"]]
    
    # -------------------------------------------------------------------------
    # STAGE 8: MONITORING (output of monitor_training tool)
    # -------------------------------------------------------------------------
    current_epoch: Optional[int]
    metrics: Optional[Dict[str, float]]            # loss, accuracy, etc.
    
    # -------------------------------------------------------------------------
    # ERROR HANDLING
    # -------------------------------------------------------------------------
    error_msg: Optional[str]                       # Error message if any
    retry_count: Optional[int]                     # Number of retries attempted
    
    # -------------------------------------------------------------------------
    # STAGE 9: SUMMARY - AGENT 2 (output of summary_agent)
    # -------------------------------------------------------------------------
    final_report: Optional[str]                    # Human-readable summary
    next_steps: Optional[List[str]]                # Recommended actions
    model_download_url: Optional[str]              # Link to download model

