"""
Ingest Data Tool
================
Load JSONL files with chat format validation.

Expected format:
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING, List, Dict, Any, Tuple

if TYPE_CHECKING:
    from quantune.state import QuantuneState

VALID_ROLES = {"system", "user", "assistant"}
MIN_EXAMPLES = 50


def _validate_message(message: Any, line_num: int, msg_idx: int) -> Tuple[bool, str]:
    """Validate a single message in the conversation."""
    if not isinstance(message, dict):
        return False, f"Line {line_num}, message {msg_idx}: must be an object, got {type(message).__name__}"
    
    if "role" not in message:
        return False, f"Line {line_num}, message {msg_idx}: missing 'role' field"
    
    if "content" not in message:
        return False, f"Line {line_num}, message {msg_idx}: missing 'content' field"
    
    role = message["role"]
    if role not in VALID_ROLES:
        return False, f"Line {line_num}, message {msg_idx}: invalid role '{role}'. Must be: system, user, or assistant"
    
    content = message["content"]
    if not isinstance(content, str):
        return False, f"Line {line_num}, message {msg_idx}: 'content' must be a string"
    
    if not content.strip():
        return False, f"Line {line_num}, message {msg_idx}: 'content' cannot be empty"
    
    return True, ""


def _validate_conversation(entry: Any, line_num: int) -> Tuple[bool, str]:
    """Validate a single conversation entry (one line of JSONL)."""
    if not isinstance(entry, dict):
        return False, f"Line {line_num}: must be a JSON object, got {type(entry).__name__}"
    
    if "messages" not in entry:
        return False, f"Line {line_num}: missing 'messages' array"
    
    messages = entry["messages"]
    
    if not isinstance(messages, list):
        return False, f"Line {line_num}: 'messages' must be an array"
    
    if len(messages) < 2:
        return False, f"Line {line_num}: need at least 2 messages (user + assistant)"
    
    for i, msg in enumerate(messages):
        valid, error = _validate_message(msg, line_num, i + 1)
        if not valid:
            return False, error
    
    roles = {msg["role"] for msg in messages}
    if "user" not in roles:
        return False, f"Line {line_num}: conversation must have at least one 'user' message"
    if "assistant" not in roles:
        return False, f"Line {line_num}: conversation must have at least one 'assistant' message"
    
    return True, ""


def ingest_data(state: "QuantuneState") -> dict:
    """
    Load JSONL file with chat format validation.
    
    Inputs (from state):
        - file_path: str
    
    Outputs (to state):
        - raw_data: List[Dict] - validated conversation data
        - num_rows: int - number of examples
        - sample_rows: List[Dict] - first 5 examples
        - stats: Dict - dataset statistics
        - error_msg: str (if failed)
    """
    file_path = state["file_path"]
    
    def err(msg: str) -> dict:
        return {
            "error_msg": msg,
            "raw_data": None,
            "num_rows": None,
            "sample_rows": None,
            "stats": None
        }
    
    try:
        path = Path(file_path)
        
        if not path.exists():
            return err(f"File not found: {file_path}")
        
        ext = path.suffix.lower()
        if ext != ".jsonl":
            return err(f"Unsupported format: {ext}. Please upload a .jsonl file.")
        
        # Read file with encoding detection
        content = None
        for encoding in ["utf-8", "utf-8-sig", "latin-1"]:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    content = f.read()
                break
            except UnicodeDecodeError:
                continue
        
        if content is None:
            return err("Unable to read file. Please save as UTF-8.")
        
        # Parse and validate JSONL
        raw_data: List[Dict] = []
        validation_errors: List[str] = []
        
        lines = content.strip().splitlines()
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                validation_errors.append(f"Line {line_num}: Invalid JSON - {e.msg}")
                if len(validation_errors) >= 5:
                    validation_errors.append("... (showing first 5 errors)")
                    break
                continue
            
            valid, error = _validate_conversation(entry, line_num)
            if not valid:
                validation_errors.append(error)
                if len(validation_errors) >= 5:
                    validation_errors.append("... (showing first 5 errors)")
                    break
                continue
            
            raw_data.append(entry)
        
        if validation_errors:
            error_summary = "\n".join(validation_errors[:5])
            return err(f"Validation failed:\n{error_summary}")
        
        num_rows = len(raw_data)
        
        if num_rows == 0:
            return err("File is empty or contains no valid examples.")
        
        if num_rows < 10:
            return err(f"Need at least 10 examples for fine-tuning. Found only {num_rows}.")
        
        is_quick_test = num_rows < MIN_EXAMPLES
        
        # Calculate statistics
        total_messages = 0
        system_count = 0
        user_messages = 0
        assistant_messages = 0
        single_turn_count = 0
        multi_turn_count = 0
        
        all_user_lengths = []
        all_assistant_lengths = []
        all_conversation_chars = []
        
        for entry in raw_data:
            messages = entry["messages"]
            total_messages += len(messages)
            
            user_count = 0
            asst_count = 0
            conv_chars = 0
            
            for msg in messages:
                role = msg["role"]
                content_len = len(msg["content"])
                conv_chars += content_len
                
                if role == "system":
                    system_count += 1
                elif role == "user":
                    user_count += 1
                    user_messages += 1
                    all_user_lengths.append(content_len)
                elif role == "assistant":
                    asst_count += 1
                    assistant_messages += 1
                    all_assistant_lengths.append(content_len)
            
            all_conversation_chars.append(conv_chars)
            
            # Single-turn = 1 user + 1 assistant (no multi-turn)
            if user_count == 1 and asst_count == 1:
                single_turn_count += 1
            else:
                multi_turn_count += 1
        
        # Token estimation (rough: 1 token ≈ 4 chars for English)
        total_chars = sum(all_conversation_chars)
        estimated_tokens = total_chars // 4
        avg_tokens_per_example = estimated_tokens // num_rows if num_rows > 0 else 0
        
        avg_user_len = sum(all_user_lengths) // len(all_user_lengths) if all_user_lengths else 0
        avg_assistant_len = sum(all_assistant_lengths) // len(all_assistant_lengths) if all_assistant_lengths else 0
        
        # Warnings
        warnings: List[str] = []
        if avg_tokens_per_example > 2000:
            warnings.append("Long conversations may require truncation")
        if num_rows < 100:
            warnings.append("More examples recommended for better results")
        if system_count == 0 and num_rows > 100:
            warnings.append("Consider adding system prompts for better control")
        
        quality = "excellent" if num_rows >= 500 and avg_assistant_len > 50 else \
                  "good" if num_rows >= 100 else "minimal"
        
        stats = {
            "total_examples": num_rows,
            "total_messages": total_messages,
            "avg_messages_per_example": round(total_messages / num_rows, 1) if num_rows > 0 else 0,
            "total_tokens": estimated_tokens,
            "avg_tokens_per_example": avg_tokens_per_example,
            "single_turn_pct": round(100 * single_turn_count / num_rows) if num_rows > 0 else 0,
            "multi_turn_pct": round(100 * multi_turn_count / num_rows) if num_rows > 0 else 0,
            "has_system_prompts": system_count > 0,
            "system_prompt_pct": round(100 * system_count / num_rows) if num_rows > 0 else 0,
            "avg_input_chars": avg_user_len,
            "avg_output_chars": avg_assistant_len,
            "quality": quality,
            "warnings": warnings,
        }
        
        return {
            "raw_data": raw_data,
            "num_rows": num_rows,
            "sample_rows": raw_data[:5],
            "stats": stats,
            "error_msg": None
        }
        
    except Exception as e:
        return err(f"Failed to load file: {str(e)}")
