"""
Analyze Dataset Tool
====================
Analyze JSONL chat format dataset for statistics and characteristics.
"""

from typing import TYPE_CHECKING, Set

if TYPE_CHECKING:
    from tunekit.state import TuneKitState


def analyze_dataset(state: "TuneKitState") -> dict:
    """
    Analyze chat format dataset and extract statistics.
    
    Inputs (from state):
        - raw_data: List[Dict] - conversation data
        - num_rows: int
    
    Outputs (to state):
        - dataset_stats: Dict - comprehensive statistics
        - conversation_characteristics: Dict - patterns for model selection
    """
    raw_data = state.get("raw_data", [])
    num_rows = state.get("num_rows", 0)
    
    if not raw_data:
        return {
            "dataset_stats": {},
            "conversation_characteristics": {}
        }
    
    # Message type counts
    total_user_msgs = 0
    total_assistant_msgs = 0
    total_system_msgs = 0
    
    # Track lengths and patterns
    conversation_lengths = []
    assistant_message_lengths = []
    unique_assistant_responses: Set[str] = set()
    
    # Conversation structure counts
    single_turn_count = 0
    multi_turn_count = 0
    has_system_count = 0
    
    for entry in raw_data:
        messages = entry.get("messages", [])
        conversation_lengths.append(len(messages))
        
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            if role == "user":
                total_user_msgs += 1
            elif role == "assistant":
                total_assistant_msgs += 1
                assistant_message_lengths.append(len(content))
                unique_assistant_responses.add(content.strip().lower())
            elif role == "system":
                total_system_msgs += 1
        
        # Single vs multi-turn (exclude system messages)
        non_system = [m for m in messages if m.get("role") != "system"]
        if len(non_system) == 2:
            single_turn_count += 1
        else:
            multi_turn_count += 1
        
        if any(m.get("role") == "system" for m in messages):
            has_system_count += 1
    
    # Calculate averages
    avg_conversation_length = sum(conversation_lengths) / len(conversation_lengths) if conversation_lengths else 0
    avg_assistant_length = sum(assistant_message_lengths) / len(assistant_message_lengths) if assistant_message_lengths else 0
    
    # Classification detection (few unique responses = likely classification)
    num_unique_responses = len(unique_assistant_responses)
    looks_like_classification = 0 < num_unique_responses < 20
    
    # Multilingual detection (check for non-ASCII in responses)
    sample_responses = list(unique_assistant_responses)[:50]
    non_ascii_count = sum(1 for r in sample_responses if any(ord(c) > 127 for c in r))
    sample_size = len(sample_responses)
    is_multilingual = (non_ascii_count / sample_size) > 0.3 if sample_size > 0 else False
    
    # JSON output detection
    json_like_count = 0
    sample_assistant_responses = []
    for entry in raw_data:
        for msg in entry.get("messages", []):
            if msg.get("role") == "assistant":
                content = msg.get("content", "").strip()
                if content:
                    sample_assistant_responses.append(content)
                    if len(sample_assistant_responses) >= 50:
                        break
        if len(sample_assistant_responses) >= 50:
            break
    
    for response in sample_assistant_responses:
        if response.startswith("{") or response.startswith("["):
            json_like_count += 1
    
    looks_like_json_output = (json_like_count / len(sample_assistant_responses)) > 0.5 if sample_assistant_responses else False
    
    # Estimate total tokens (rough: 1 token ≈ 4 chars)
    total_chars = sum(
        sum(len(msg.get("content", "")) for msg in entry.get("messages", []))
        for entry in raw_data
    )
    estimated_tokens = total_chars // 4
    
    total_messages = total_user_msgs + total_assistant_msgs + total_system_msgs
    
    dataset_stats = {
        "num_examples": num_rows,
        "total_messages": total_messages,
        "avg_messages_per_example": round(avg_conversation_length, 1),
        "message_counts": {
            "user": total_user_msgs,
            "assistant": total_assistant_msgs,
            "system": total_system_msgs
        },
        "avg_lengths": {
            "assistant_messages": round(avg_assistant_length),
            "conversation": round(avg_conversation_length)
        },
        "conversation_structure": {
            "single_turn": single_turn_count,
            "multi_turn": multi_turn_count,
            "with_system": has_system_count,
            "without_system": num_rows - has_system_count
        },
        "estimated_tokens": estimated_tokens,
        "has_system_prompts": has_system_count > 0
    }
    
    # Characteristics used by model recommendation
    conversation_characteristics = {
        "is_multilingual": is_multilingual,
        "avg_response_length": round(avg_assistant_length),
        "looks_like_json_output": looks_like_json_output,
        "is_multi_turn": multi_turn_count > single_turn_count,
        "has_system_prompts": has_system_count > 0,
        "looks_like_classification": looks_like_classification,
    }
    
    return {
        "dataset_stats": dataset_stats,
        "conversation_characteristics": conversation_characteristics
    }
