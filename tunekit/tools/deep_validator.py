"""
TuneKit Deep Data Validator & Diagnostic Engine
===============================================
Performs deep 12-point inspection on chat-format datasets, computing:
- Token statistics (min, max, mean, median, P95)
- Turn & role distribution
- Outlier & truncation risks
- Duplicate detection
- Vocabulary richness & lexical diversity
- Minimum criteria checklist
- Weighted Fine-Tuning Readiness Score (0 - 100%) and letter grade (A+ to F)
- Actionable optimization checklist
"""

import math
import hashlib
from typing import List, Dict, Any, Tuple, Optional


def _estimate_tokens(text: str) -> int:
    """Rough token count estimation (~4 characters per token for English/code)."""
    if not text:
        return 0
    words = len(text.split())
    char_tokens = math.ceil(len(text) / 3.8)
    # Average between word-based and char-based
    return max(1, round((words * 1.3 + char_tokens) / 2))


def _percentile(values: List[int], p: float) -> int:
    """Calculates p-th percentile from sorted list."""
    if not values:
        return 0
    k = (len(values) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return values[int(k)]
    d0 = values[int(f)] * (c - k)
    d1 = values[int(c)] * (k - f)
    return round(d0 + d1)


def validate_dataset_deep(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Executes comprehensive 12-point dataset validation.
    """
    total_examples = len(rows)
    if total_examples == 0:
        return {
            "readiness_score": 0,
            "letter_grade": "F",
            "readiness_status": "No Data Found",
            "total_examples": 0,
            "valid_examples": 0,
            "corrupt_examples": 0,
            "criteria_checklist": [],
            "stats": {},
            "issues": [{"type": "error", "message": "Dataset is completely empty. Please upload valid rows."}],
            "recommendations": ["Upload a dataset containing at least 10-50 conversational examples."]
        }

    # Tracking metrics
    valid_count = 0
    corrupt_count = 0
    has_system_count = 0
    distinct_system_prompts = set()
    
    user_token_lengths: List[int] = []
    asst_token_lengths: List[int] = []
    total_conv_tokens: List[int] = []
    
    single_turn_count = 0
    multi_turn_count = 0
    turn_counts: List[int] = []
    
    empty_messages_count = 0
    ultra_short_responses = 0  # < 5 tokens
    extreme_long_examples = 0  # > 2048 tokens
    truncation_risk_count = 0  # > 4096 tokens
    
    all_words: List[str] = []
    seen_prompts: Dict[str, int] = {}
    seen_conversations: set = set()
    exact_duplicates = 0
    
    issues: List[Dict[str, Any]] = []

    for idx, row in enumerate(rows):
        msgs = row.get("messages", [])
        if not isinstance(msgs, list) or len(msgs) < 2:
            corrupt_count += 1
            issues.append({"type": "error", "row": idx + 1, "message": f"Row #{idx+1} has fewer than 2 messages."})
            continue

        roles = [m.get("role") for m in msgs if isinstance(m, dict)]
        if "user" not in roles or "assistant" not in roles:
            corrupt_count += 1
            issues.append({"type": "error", "row": idx + 1, "message": f"Row #{idx+1} missing 'user' or 'assistant' role."})
            continue

        valid_count += 1
        
        # Check system prompt
        sys_msgs = [m for m in msgs if m.get("role") == "system"]
        if sys_msgs:
            has_system_count += 1
            sys_content = sys_msgs[0].get("content", "").strip()
            if sys_content:
                distinct_system_prompts.add(sys_content[:100])

        # Turns count (user-assistant pairs)
        u_msgs = [m for m in msgs if m.get("role") == "user"]
        a_msgs = [m for m in msgs if m.get("role") == "assistant"]
        
        current_turns = len(u_msgs)
        turn_counts.append(current_turns)
        if current_turns == 1 and len(a_msgs) == 1:
            single_turn_count += 1
        else:
            multi_turn_count += 1

        # Token lengths
        row_tokens = 0
        first_user_prompt = ""
        for m in msgs:
            content = m.get("content", "")
            if not isinstance(content, str) or not content.strip():
                empty_messages_count += 1
                issues.append({"type": "error", "row": idx + 1, "message": f"Row #{idx+1} has an empty message with role '{m.get('role')}'."})
                continue
            
            t_cnt = _estimate_tokens(content)
            row_tokens += t_cnt
            words = content.lower().split()
            all_words.extend(words[:50])  # Sample vocabulary

            if m.get("role") == "user":
                user_token_lengths.append(t_cnt)
                if not first_user_prompt:
                    first_user_prompt = content.strip().lower()
            elif m.get("role") == "assistant":
                asst_token_lengths.append(t_cnt)
                if t_cnt < 5:
                    ultra_short_responses += 1

        total_conv_tokens.append(row_tokens)
        if row_tokens > 2048:
            extreme_long_examples += 1
        if row_tokens > 4096:
            truncation_risk_count += 1

        # Deduplication check
        if first_user_prompt:
            p_hash = hashlib.md5(first_user_prompt.encode("utf-8")).hexdigest()
            seen_prompts[p_hash] = seen_prompts.get(p_hash, 0) + 1

        # Conversation fingerprint
        conv_fp = "|".join(f"{m.get('role')}:{m.get('content', '')[:60]}" for m in msgs)
        if conv_fp in seen_conversations:
            exact_duplicates += 1
        else:
            seen_conversations.add(conv_fp)

    # Sort arrays for percentile calculations
    user_token_lengths.sort()
    asst_token_lengths.sort()
    total_conv_tokens.sort()

    user_stats = {
        "min": user_token_lengths[0] if user_token_lengths else 0,
        "max": user_token_lengths[-1] if user_token_lengths else 0,
        "avg": round(sum(user_token_lengths) / len(user_token_lengths), 1) if user_token_lengths else 0,
        "median": _percentile(user_token_lengths, 50),
        "p95": _percentile(user_token_lengths, 95)
    }

    asst_stats = {
        "min": asst_token_lengths[0] if asst_token_lengths else 0,
        "max": asst_token_lengths[-1] if asst_token_lengths else 0,
        "avg": round(sum(asst_token_lengths) / len(asst_token_lengths), 1) if asst_token_lengths else 0,
        "median": _percentile(asst_token_lengths, 50),
        "p95": _percentile(asst_token_lengths, 95)
    }

    total_tokens_all = sum(total_conv_tokens)
    total_vocab = len(set(all_words))
    lexical_diversity = round(total_vocab / max(1, len(all_words)), 3)

    # Duplicate prompts calculation
    repeated_prompts = sum(count - 1 for count in seen_prompts.values() if count > 1)
    duplicate_ratio = repeated_prompts / max(1, total_examples)

    # --- Criteria Checklist ---
    criteria = [
        {
            "id": "volume",
            "title": "Dataset Volume Threshold",
            "desc": "Minimum 50 examples recommended (10 minimum for testing)",
            "status": "pass" if total_examples >= 50 else ("warn" if total_examples >= 10 else "fail"),
            "value": f"{total_examples} examples",
            "impact": "High (Generalization)"
        },
        {
            "id": "integrity",
            "title": "Turn & Schema Integrity",
            "desc": "Every conversation must have alternating user & assistant messages",
            "status": "pass" if corrupt_count == 0 else "fail",
            "value": f"{valid_count}/{total_examples} valid",
            "impact": "Critical (Loss convergence)"
        },
        {
            "id": "empty_content",
            "title": "Non-Empty Content",
            "desc": "Zero empty or whitespace-only messages in user/assistant turns",
            "status": "pass" if empty_messages_count == 0 else "fail",
            "value": f"{empty_messages_count} empty",
            "impact": "Critical"
        },
        {
            "id": "system_prompt",
            "title": "System Prompt Consistency",
            "desc": "Uniform presence of a system prompt guides model behavior & safety",
            "status": "pass" if (has_system_count == total_examples or has_system_count == 0) else "warn",
            "value": f"{has_system_count}/{total_examples} ({round((has_system_count/max(1, total_examples))*100)}%)",
            "impact": "Medium (Persona consistency)"
        },
        {
            "id": "diversity",
            "title": "Non-Duplication & Diversity",
            "desc": "Duplicate conversations should be below 10%",
            "status": "pass" if duplicate_ratio < 0.10 else ("warn" if duplicate_ratio < 0.25 else "fail"),
            "value": f"{exact_duplicates} duplicates ({round(duplicate_ratio*100)}%)",
            "impact": "High (Overfitting risk)"
        },
        {
            "id": "length_suitability",
            "title": "Context Window Suitability",
            "desc": "Conversations fit within typical 2048-4096 token context windows",
            "status": "pass" if truncation_risk_count == 0 else "warn",
            "value": f"{truncation_risk_count} >4K tokens",
            "impact": "Medium (Truncation risk)"
        }
    ]

    # --- Weighted Score Calculation (0 - 100) ---
    score = 100.0
    recommendations = []

    # 1. Volume penalty (max 25 pts)
    if total_examples < 10:
        score -= 25.0
        recommendations.append("❌ Add more training examples (at least 10-50 rows needed for basic learning).")
    elif total_examples < 30:
        score -= 15.0
        recommendations.append("💡 Dataset has fewer than 30 examples. Expanding to 50+ will boost generalization (+15% score).")
    elif total_examples < 50:
        score -= 6.0
        recommendations.append("💡 Dataset has 30-49 examples. Aim for 50+ examples for production-ready tuning (+6% score).")

    # 2. Corrupt / invalid rows penalty (max 30 pts)
    if corrupt_count > 0:
        pen = min(30.0, (corrupt_count / total_examples) * 50.0)
        score -= pen
        recommendations.append(f"❌ Fix or remove {corrupt_count} malformed rows with missing roles or invalid formatting.")

    # 3. Empty messages penalty (max 20 pts)
    if empty_messages_count > 0:
        score -= min(20.0, empty_messages_count * 5.0)
        recommendations.append(f"❌ Remove {empty_messages_count} empty messages to avoid null-token divergence.")

    # 4. Duplication penalty (max 15 pts)
    if duplicate_ratio > 0.05:
        pen = min(15.0, duplicate_ratio * 30.0)
        score -= pen
        recommendations.append(f"⚠️ Remove {exact_duplicates} duplicate examples to prevent overfitting (+{round(pen)}% score).")

    # 5. Inconsistent system prompt penalty (max 10 pts)
    if 0 < has_system_count < total_examples:
        score -= 10.0
        recommendations.append("⚠️ System prompt is present in some rows but missing in others. Use the 'Apply Global System Prompt' tool to normalize (+10% score).")
    elif has_system_count == 0:
        recommendations.append("💡 Adding a standardized system prompt (e.g. 'You are an expert customer care specialist...') helps align model tone.")

    # 6. Ultra-short or extreme long penalties
    if truncation_risk_count > 0:
        score -= min(10.0, truncation_risk_count * 2.0)
        recommendations.append(f"⚠️ Truncate or split {truncation_risk_count} examples exceeding 4096 tokens.")

    score = max(0.0, min(100.0, round(score, 1)))

    # Letter Grade Assignment
    if score >= 95:
        grade = "A+"
        status = "Production Ready"
    elif score >= 90:
        grade = "A"
        status = "Excellent Quality"
    elif score >= 80:
        grade = "B"
        status = "Good / Ready to Fine-Tune"
    elif score >= 70:
        grade = "C"
        status = "Acceptable (Optimizations Recommended)"
    elif score >= 60:
        grade = "D"
        status = "Needs Improvement"
    else:
        grade = "F"
        status = "Critical Issues Detected"

    if not recommendations:
        recommendations.append("✅ Dataset meets all high-quality fine-tuning standards!")

    return {
        "readiness_score": score,
        "letter_grade": grade,
        "readiness_status": status,
        "total_examples": total_examples,
        "valid_examples": valid_count,
        "corrupt_examples": corrupt_count,
        "has_system_count": has_system_count,
        "distinct_system_prompts": len(distinct_system_prompts),
        "single_turn_count": single_turn_count,
        "multi_turn_count": multi_turn_count,
        "exact_duplicates": exact_duplicates,
        "duplicate_ratio": round(duplicate_ratio, 3),
        "total_tokens": total_tokens_all,
        "estimated_3_epoch_tokens": total_tokens_all * 3,
        "lexical_diversity": lexical_diversity,
        "user_token_stats": user_stats,
        "assistant_token_stats": asst_stats,
        "truncation_risk_count": truncation_risk_count,
        "criteria_checklist": criteria,
        "issues": issues[:25],  # top 25 issues
        "recommendations": recommendations
    }
