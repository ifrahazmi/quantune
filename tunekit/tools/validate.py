"""
Validate Quality Tool
=====================
Quality checks for JSONL chat format data.
"""

from typing import TYPE_CHECKING, List, Dict, Set

if TYPE_CHECKING:
    from tunekit.state import TuneKitState


def validate_quality(state: "TuneKitState") -> dict:
    """
    Quality checks for chat format data.
    
    Inputs (from state):
        - raw_data: List[Dict] - conversation data
        - num_rows: int
    
    Outputs (to state):
        - quality_score: float (0.0 to 1.0)
        - quality_issues: List[str]
    """
    raw_data = state.get("raw_data", [])
    num_rows = state.get("num_rows", 0)
    
    if not raw_data:
        return {"quality_score": 0.0, "quality_issues": ["No data available"]}
    
    score = 1.0
    issues: List[str] = []
    
    # --- Check 1: Duplicate conversations ---
    seen_conversations: Set[str] = set()
    duplicates = 0
    for entry in raw_data:
        fingerprint = "|".join(
            f"{msg['role']}:{msg['content'][:100]}"
            for msg in entry["messages"]
        )
        if fingerprint in seen_conversations:
            duplicates += 1
        else:
            seen_conversations.add(fingerprint)
    
    if duplicates > 0:
        dup_ratio = duplicates / num_rows
        score -= min(dup_ratio * 0.4, 0.3)
        issues.append(f"⚠️ {duplicates} duplicate conversations ({dup_ratio:.0%})")
    
    # --- Check 2: Dataset size ---
    if num_rows < 50:
        score -= 0.5
        issues.append(f"❌ Too few examples ({num_rows}). Need at least 50.")
    elif num_rows < 100:
        score -= 0.2
        issues.append(f"⚠️ Small dataset ({num_rows}). 100+ recommended for better results.")
    elif num_rows < 500:
        score -= 0.1
        issues.append(f"💡 {num_rows} examples. 500+ recommended for best results.")
    
    # --- Check 3: Empty assistant responses (critical) ---
    empty_responses = 0
    for entry in raw_data:
        for msg in entry["messages"]:
            if msg["role"] == "assistant" and len(msg["content"].strip()) == 0:
                empty_responses += 1
    
    if empty_responses > 0:
        score -= 0.2
        issues.append(f"❌ {empty_responses} empty assistant responses")
    
    # --- Check 4: Short and long responses (info only) ---
    short_responses = 0
    very_long_responses = 0
    
    for entry in raw_data:
        for msg in entry["messages"]:
            if msg["role"] == "assistant":
                content_len = len(msg["content"].strip())
                if content_len > 0 and content_len < 10:
                    short_responses += 1
                elif content_len > 4000:
                    very_long_responses += 1
    
    if short_responses > 0:
        # Info only - no penalty (could be classification, short answers, etc.)
        issues.append(f"💡 {short_responses} short assistant responses (<10 chars) - fine if intentional")
    
    if very_long_responses > 0:
        # Info only - no penalty
        issues.append(f"💡 {very_long_responses} very long responses (>4000 chars) - may need truncation")
    
    # --- Check 5: Conversation structure consistency ---
    single_turn = 0
    multi_turn = 0
    
    for entry in raw_data:
        non_system = [m for m in entry["messages"] if m["role"] != "system"]
        if len(non_system) == 2:
            single_turn += 1
        else:
            multi_turn += 1
    
    # Note: Not penalizing, just informing
    if single_turn > 0 and multi_turn > 0:
        issues.append(f"💡 Mixed format: {single_turn} single-turn, {multi_turn} multi-turn conversations")
    
    # --- Check 6: System prompt consistency ---
    has_system = 0
    no_system = 0
    
    for entry in raw_data:
        if any(msg["role"] == "system" for msg in entry["messages"]):
            has_system += 1
        else:
            no_system += 1
    
    if has_system > 0 and no_system > 0:
        # Inconsistent system prompts
        score -= 0.1
        issues.append(f"⚠️ Inconsistent: {has_system} with system prompt, {no_system} without")
    
    # --- Finalize ---
    score = max(0.0, min(1.0, score))
    
    return {
        "quality_score": score,
        "quality_issues": issues if issues else ["✅ Data looks good!"]
    }
