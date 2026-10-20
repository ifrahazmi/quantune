"""
Model Recommendation Tool
=========================
Simple 3-factor scoring: Task (50) + Size (30) + Output (20) = 100 points
"""

from typing import Dict, List

# ════════════════════════════════════════════════════════════════════════════
# MODEL METADATA
# ════════════════════════════════════════════════════════════════════════════

MODELS = {
    'phi-4-mini': {
        'id': 'microsoft/Phi-4-mini-instruct',
        'name': 'Phi-4 Mini',
        'creator': 'Microsoft',
        'size': '3.8B',
        'context_window': 128000,
        'training_time_base': 3,
        'cost_base': 0.036,
        'gpu_tier': 'Local 4GB / Colab T4',
        'memory_gb': 12,
        'qlora_vram_gb': 2.8,
        'accuracy_baseline': 87,
        'license': 'MIT',
        'gated': False,
        'ollama_tag': 'phi4-mini',
        'benchmarks': {'mmlu': 75.8, 'gsm8k': 84.6, 'humaneval': 68.2},
        'category': ['reasoning', 'permissive', 'code'],
        'strengths': [
            'SOTA synthetic data reasoning & instruction following',
            'Massive 128K context window with RoPE scaling',
            'Permissive MIT open license for commercial fine-tuning',
            'Native function calling & structured JSON extraction'
        ]
    },
    'gemma-3-2b': {
        'id': 'google/gemma-2-2b-it',
        'name': 'Gemma 2 2B',
        'creator': 'Google DeepMind',
        'size': '2B',
        'context_window': 8192,
        'training_time_base': 2,
        'cost_base': 0.012,
        'gpu_tier': 'Local 4GB / Colab T4',
        'memory_gb': 6,
        'qlora_vram_gb': 2.2,
        'accuracy_baseline': 82,
        'license': 'Gemma Terms',
        'gated': True,
        'ollama_tag': 'gemma2:2b',
        'benchmarks': {'mmlu': 56.1, 'gsm8k': 42.8, 'humaneval': 31.1},
        'category': ['edge', 'lightweight'],
        'strengths': [
            'Ultra-lightweight 2B parameters for rapid local iteration',
            'Runs on virtually any GPU or high-end laptop CPU',
            'Knowledge-distilled from Gemini architecture',
            'Lowest latency inference on mobile and edge devices'
        ]
    },
    'llama-3.2-3b': {
        'id': 'meta-llama/Llama-3.2-3B-Instruct',
        'name': 'Llama 3.2 3B',
        'creator': 'Meta AI',
        'size': '3B',
        'context_window': 128000,
        'training_time_base': 4,
        'cost_base': 0.048,
        'gpu_tier': 'Local 4GB / Colab T4',
        'memory_gb': 10,
        'qlora_vram_gb': 2.7,
        'accuracy_baseline': 89,
        'license': 'Llama 3.2 Community',
        'gated': True,
        'ollama_tag': 'llama3.2:3b',
        'benchmarks': {'mmlu': 63.4, 'gsm8k': 77.2, 'humaneval': 55.5},
        'category': ['conversation', 'reasoning'],
        'strengths': [
            'Industry-standard fine-tuning ecosystem & Unsloth support',
            'Exceptional dialogue fluency & multi-turn coherence',
            '128K context window for long documents and history',
            'Distilled & pruned from flagship Llama 3.1 architecture'
        ]
    },
    'qwen-2.5-3b': {
        'id': 'Qwen/Qwen2.5-3B-Instruct',
        'name': 'Qwen 2.5 3B',
        'creator': 'Alibaba Cloud',
        'size': '3B',
        'context_window': 32768,
        'training_time_base': 4,
        'cost_base': 0.048,
        'gpu_tier': 'Local 4GB / Colab T4',
        'memory_gb': 10,
        'qlora_vram_gb': 2.7,
        'accuracy_baseline': 88,
        'license': 'Apache 2.0',
        'gated': False,
        'ollama_tag': 'qwen2.5:3b',
        'benchmarks': {'mmlu': 68.4, 'gsm8k': 79.2, 'humaneval': 65.2},
        'category': ['multilingual', 'permissive', 'code'],
        'strengths': [
            'Unrivaled multilingual competence across 29+ languages',
            'Top-tier code generation and structured JSON extraction',
            'Permissive Apache 2.0 open-weights license',
            'Dense 32K context window with high reasoning power'
        ]
    },
    'mistral-7b': {
        'id': 'mistralai/Mistral-7B-Instruct-v0.3',
        'name': 'Mistral 7B',
        'creator': 'Mistral AI',
        'size': '7B',
        'context_window': 32768,
        'training_time_base': 6,
        'cost_base': 0.30,
        'gpu_tier': 'Colab T4 (16GB) / Cloud',
        'memory_gb': 18,
        'qlora_vram_gb': 5.5,
        'accuracy_baseline': 91,
        'license': 'Apache 2.0',
        'gated': False,
        'ollama_tag': 'mistral:7b',
        'benchmarks': {'mmlu': 70.6, 'gsm8k': 81.0, 'humaneval': 60.4},
        'category': ['reasoning', 'permissive'],
        'strengths': [
            'Maximum parameter capacity (7B) for high-entropy tasks',
            'Sliding window attention & native function calling v3',
            'High quality creative and analytical text generation',
            'Permissive Apache 2.0 open-weights license'
        ]
    }
}

# ════════════════════════════════════════════════════════════════════════════
# DEPLOYMENT FILTERS
# ════════════════════════════════════════════════════════════════════════════

DEPLOYMENT_FILTERS = {
    'cloud_api': ['phi-4-mini', 'gemma-3-2b', 'llama-3.2-3b', 'qwen-2.5-3b', 'mistral-7b'],
    'desktop_app': ['phi-4-mini', 'gemma-3-2b', 'llama-3.2-3b', 'qwen-2.5-3b', 'mistral-7b'],
    'mobile_app': ['phi-4-mini', 'gemma-3-2b', 'llama-3.2-3b'],
    'ios_app': ['phi-4-mini', 'gemma-3-2b', 'llama-3.2-3b'],
    'android_app': ['phi-4-mini', 'gemma-3-2b', 'llama-3.2-3b'],
    'web_browser': ['phi-4-mini', 'gemma-3-2b', 'llama-3.2-3b'],
    'edge_device': ['gemma-3-2b', 'phi-4-mini'],
    'not_sure': ['phi-4-mini', 'gemma-3-2b', 'llama-3.2-3b', 'qwen-2.5-3b', 'mistral-7b']
}

# ════════════════════════════════════════════════════════════════════════════
# SCORING TABLES
# ════════════════════════════════════════════════════════════════════════════

# Factor 1: TASK TYPE (50 points max)
TASK_SCORES = {
    'classify': {
        'phi-4-mini': 50, 'gemma-3-2b': 35, 'llama-3.2-3b': 30, 'qwen-2.5-3b': 25, 'mistral-7b': 20
    },
    'qa': {
        'llama-3.2-3b': 50, 'mistral-7b': 45, 'phi-4-mini': 35, 'qwen-2.5-3b': 30, 'gemma-3-2b': 20
    },
    'conversation': {
        'llama-3.2-3b': 50, 'mistral-7b': 45, 'qwen-2.5-3b': 35, 'phi-4-mini': 25, 'gemma-3-2b': 20
    },
    'generation': {
        'mistral-7b': 50, 'llama-3.2-3b': 45, 'qwen-2.5-3b': 30, 'phi-4-mini': 20, 'gemma-3-2b': 20
    },
    'extraction': {
        'phi-4-mini': 50, 'llama-3.2-3b': 40, 'mistral-7b': 35, 'qwen-2.5-3b': 30, 'gemma-3-2b': 25
    }
}

# Factor 2: DATASET SIZE (30 points max)
SIZE_SCORES_SMALL = {  # < 500 examples
    'gemma-3-2b': 30, 'phi-4-mini': 20, 'llama-3.2-3b': 10, 'qwen-2.5-3b': 10, 'mistral-7b': 5
}
SIZE_SCORES_MEDIUM = {  # 500-2000 examples
    'llama-3.2-3b': 30, 'phi-4-mini': 25, 'mistral-7b': 20, 'qwen-2.5-3b': 20, 'gemma-3-2b': 15
}
SIZE_SCORES_LARGE = {  # > 2000 examples
    'llama-3.2-3b': 30, 'mistral-7b': 25, 'phi-4-mini': 10, 'qwen-2.5-3b': 10, 'gemma-3-2b': 5
}

# Factor 3: OUTPUT CHARACTERISTICS (20 points max)
OUTPUT_SCORES_LONG = {  # avg_response_length > 200
    'mistral-7b': 20, 'llama-3.2-3b': 15, 'qwen-2.5-3b': 10, 'phi-4-mini': 5, 'gemma-3-2b': 5
}
OUTPUT_SCORES_JSON = {  # JSON output detected
    'phi-4-mini': 20, 'llama-3.2-3b': 15, 'mistral-7b': 10, 'qwen-2.5-3b': 10, 'gemma-3-2b': 5
}

# Multi-turn bonus (+10 points max)
MULTI_TURN_BONUS = {
    'llama-3.2-3b': 10, 'mistral-7b': 8, 'qwen-2.5-3b': 6, 'phi-4-mini': 3, 'gemma-3-2b': 2
}

# Tie-breaker priorities per task
TIEBREAKER = {
    'classify': ['phi-4-mini', 'gemma-3-2b', 'llama-3.2-3b', 'mistral-7b', 'qwen-2.5-3b'],
    'extraction': ['phi-4-mini', 'llama-3.2-3b', 'mistral-7b', 'qwen-2.5-3b', 'gemma-3-2b'],
    'qa': ['llama-3.2-3b', 'mistral-7b', 'phi-4-mini', 'qwen-2.5-3b', 'gemma-3-2b'],
    'conversation': ['llama-3.2-3b', 'mistral-7b', 'qwen-2.5-3b', 'phi-4-mini', 'gemma-3-2b'],
    'generation': ['mistral-7b', 'llama-3.2-3b', 'qwen-2.5-3b', 'phi-4-mini', 'gemma-3-2b'],
    'default': ['mistral-7b', 'llama-3.2-3b', 'phi-4-mini', 'qwen-2.5-3b', 'gemma-3-2b']
}


# ════════════════════════════════════════════════════════════════════════════
# MAIN RECOMMENDATION FUNCTION
# ════════════════════════════════════════════════════════════════════════════

def recommend_model(
    user_task: str,
    conversation_characteristics: Dict,
    num_examples: int,
    deployment_target: str = 'not_sure'
) -> Dict:
    """
    Recommend best SLM model using simple 3-factor scoring.
    
    Args:
        user_task: 'classify', 'qa', 'conversation', 'generation', 'extraction'
        conversation_characteristics: Dict with is_multilingual, avg_response_length, looks_like_json_output
        num_examples: Dataset size
        deployment_target: 'cloud_api', 'mobile_app', 'edge_device', etc.
    
    Returns:
        Dict with primary_recommendation, alternatives, all_scores
    """
    is_multilingual = conversation_characteristics.get('is_multilingual', False)
    avg_response_length = conversation_characteristics.get('avg_response_length', 50)
    looks_like_json = conversation_characteristics.get('looks_like_json_output', False)
    is_multi_turn = conversation_characteristics.get('is_multi_turn', False)
    
    # Score each model and calculate 3-factor breakdown
    scores = {}
    score_breakdowns = {}
    
    # Filter models by deployment target
    allowed_models = DEPLOYMENT_FILTERS.get(deployment_target, DEPLOYMENT_FILTERS['not_sure'])
    
    for model_key in MODELS:
        # Factor 1: Task type (50 points)
        task_scores = TASK_SCORES.get(user_task, TASK_SCORES['classify'])
        task_score = task_scores.get(model_key, 25)
        
        # Factor 2: Dataset size (30 points)
        if num_examples < 500:
            size_score = SIZE_SCORES_SMALL.get(model_key, 15)
        elif num_examples >= 2000:
            size_score = SIZE_SCORES_LARGE.get(model_key, 15)
        else:
            size_score = SIZE_SCORES_MEDIUM.get(model_key, 15)
        
        # Factor 3: Output characteristics (20 points)
        if avg_response_length > 200:
            output_score = OUTPUT_SCORES_LONG.get(model_key, 10)
        elif looks_like_json:
            output_score = OUTPUT_SCORES_JSON.get(model_key, 10)
        else:
            output_score = 10
        
        # Bonus: Multi-turn conversations (+10 points)
        multi_turn_bonus = MULTI_TURN_BONUS.get(model_key, 0) if is_multi_turn else 0
        
        total = task_score + size_score + output_score + multi_turn_bonus
        if model_key in allowed_models:
            scores[model_key] = total
        else:
            scores[model_key] = max(10, total - 35) # Penalize models outside deployment target
            
        score_breakdowns[model_key] = {
            'task_fit': task_score,
            'size_fit': size_score,
            'output_fit': output_score,
            'multi_turn_bonus': multi_turn_bonus,
            'total': total
        }

    # Hard overrides for obvious cases
    if is_multilingual:
        scores['qwen-2.5-3b'] = 100
        score_breakdowns['qwen-2.5-3b'] = {'task_fit': 50, 'size_fit': 30, 'output_fit': 20, 'multi_turn_bonus': 0, 'total': 100}
        return build_response(
            primary_key='qwen-2.5-3b',
            score=100,
            all_scores=scores,
            reasons=['Best multilingual support (29 languages)', 'Optimized for non-English and code tasks', 'Permissive Apache 2.0 open-source license'],
            num_examples=num_examples,
            alternatives=[{'model': 'llama-3.2-3b', 'score': scores.get('llama-3.2-3b', 75)}, {'model': 'mistral-7b', 'score': scores.get('mistral-7b', 70)}],
            score_breakdowns=score_breakdowns
        )
    
    if deployment_target == 'edge_device':
        scores['gemma-3-2b'] = 100
        score_breakdowns['gemma-3-2b'] = {'task_fit': 50, 'size_fit': 30, 'output_fit': 20, 'multi_turn_bonus': 0, 'total': 100}
        return build_response(
            primary_key='gemma-3-2b',
            score=100,
            all_scores=scores,
            reasons=['Smallest model (2B params)', 'Optimized for low-power edge devices and laptops', 'Runs with ultra-low VRAM (<2.5 GB)'],
            num_examples=num_examples,
            alternatives=[{'model': 'phi-4-mini', 'score': scores.get('phi-4-mini', 80)}, {'model': 'llama-3.2-3b', 'score': scores.get('llama-3.2-3b', 70)}],
            score_breakdowns=score_breakdowns
        )
    
    # Pick winner with tie-breaking
    priority_order = TIEBREAKER.get(user_task, TIEBREAKER['default'])
    
    def get_priority(model_key):
        try:
            return priority_order.index(model_key)
        except ValueError:
            return 99
    
    sorted_models = sorted(scores.items(), key=lambda x: (-x[1], get_priority(x[0])))
    
    primary_key = sorted_models[0][0]
    primary_score = sorted_models[0][1]
    
    reasons = generate_reasons(primary_key, user_task, num_examples, avg_response_length, looks_like_json, is_multi_turn, deployment_target)
    
    # Build alternatives
    alternatives = []
    for model_key, score in sorted_models[1:3]:
        model = MODELS[model_key]
        alt_reasons = []
        
        if score >= primary_score - 10:
            alt_reasons.append(f"Close match ({score}/100)")
        
        if model_key in ['llama-3.2-3b', 'gemma-3-2b']:
            if model_key == 'llama-3.2-3b':
                alt_reasons.append("Requires HuggingFace approval (gated)")
            elif model_key == 'gemma-3-2b':
                alt_reasons.append("Requires terms agreement (gated)")
        
        if model['training_time_base'] < MODELS[primary_key]['training_time_base']:
            time_diff = MODELS[primary_key]['training_time_base'] - model['training_time_base']
            alt_reasons.append(f"~{time_diff} min faster training")
        
        if model.get('qlora_vram_gb', 3) < MODELS[primary_key].get('qlora_vram_gb', 3):
            alt_reasons.append(f"Lower VRAM ({model.get('qlora_vram_gb', 3)}GB vs {MODELS[primary_key].get('qlora_vram_gb', 3)}GB)")
        
        if model['context_window'] > MODELS[primary_key]['context_window']:
            alt_reasons.append(f"Larger context ({model['context_window']//1000}K tokens)")
        
        alternatives.append({
            'model': model_key,
            'score': score,
            'reasons': alt_reasons if alt_reasons else ['Good alternative']
        })
    
    return build_response(
        primary_key=primary_key,
        score=primary_score,
        all_scores=scores,
        reasons=reasons,
        num_examples=num_examples,
        alternatives=alternatives,
        score_breakdowns=score_breakdowns
    )


def generate_reasons(model_key: str, task: str, num_examples: int, avg_response: int, is_json: bool, is_multi_turn: bool, deployment: str) -> List[str]:
    """Generate human-readable reasons for the recommendation."""
    reasons = []
    model = MODELS[model_key]
    
    # Task-based reason
    task_reasons = {
        ('phi-4-mini', 'classify'): 'Best for classification tasks & structured accuracy',
        ('gemma-3-2b', 'classify'): 'Fast, efficient, and lightweight for classification',
        ('phi-4-mini', 'extraction'): 'SOTA structured JSON extraction & schema adherence',
        ('qwen-2.5-3b', 'extraction'): 'Strong structured extraction and syntax fidelity',
        ('llama-3.2-3b', 'qa'): 'Top performer for Q&A knowledge recall and coherence',
        ('mistral-7b', 'qa'): 'Deep knowledge base with powerful analytical depth',
        ('phi-4-mini', 'qa'): 'Advanced reasoning and step-by-step chain-of-thought',
        ('llama-3.2-3b', 'conversation'): 'Best for natural multi-turn conversational AI',
        ('mistral-7b', 'conversation'): 'Natural dialogue and nuanced context tracking',
        ('qwen-2.5-3b', 'conversation'): 'Multilingual conversational fluency (29+ langs)',
        ('mistral-7b', 'generation'): 'Best-in-class long-form coherent generation',
        ('llama-3.2-3b', 'generation'): 'Creative, coherent, and stylistically versatile',
        ('qwen-2.5-3b', 'generation'): 'Strong multilingual generation & code synthesis',
    }
    reason = task_reasons.get((model_key, task))
    if reason:
        reasons.append(reason)
    else:
        reasons.append(f"Strong performance for {task} tasks")
    
    # Model characteristics
    if model_key in ['phi-4-mini', 'mistral-7b']:
        reasons.append('Native support for function calling and tools')
    
    if model_key == 'qwen-2.5-3b':
        reasons.append('Multilingual fluency (29 languages) & coding strength')
    
    if model_key in ['gemma-3-2b', 'llama-3.2-3b']:
        reasons.append('Optimized for on-device deployment & fast inference')
    
    if model_key in ['phi-4-mini', 'llama-3.2-3b', 'qwen-2.5-3b']:
        context_window = model.get('context_window', 0)
        if context_window >= 128000:
            reasons.append('Massive 128K token long context window')
        elif context_window >= 32000:
            reasons.append('Expansive 32K token context window')
    
    if is_multi_turn and model_key in ['llama-3.2-3b', 'mistral-7b']:
        reasons.append('Superb multi-turn memory & dialogue coherence')
    
    # Size-based reason
    if num_examples < 500:
        if model_key == 'gemma-3-2b':
            reasons.append('Ideal parameter efficiency for small datasets')
        elif model_key == 'phi-4-mini':
            reasons.append('Resistant to overfitting on focused datasets')
    elif num_examples > 2000:
        if model_key in ['llama-3.2-3b', 'mistral-7b']:
            reasons.append(f'High capacity scales effectively with {num_examples:,} examples')
    
    # Output-based reason
    if avg_response > 200 and model_key in ['mistral-7b', 'llama-3.2-3b']:
        reasons.append('Optimized for lengthy, multi-paragraph completions')
    elif is_json and model_key in ['phi-4-mini', 'qwen-2.5-3b']:
        reasons.append('Strict JSON syntax compliance & schema validation')
    
    # Deployment reason
    if deployment in ['mobile_app', 'ios_app', 'android_app'] and model_key in ['phi-4-mini', 'gemma-3-2b']:
        reasons.append('Optimized for mobile & low-power deployment')
    elif deployment == 'web_browser' and model_key in ['gemma-3-2b', 'phi-4-mini']:
        reasons.append('Runs efficiently in browser via WebGPU / ONNX')
    elif deployment == 'edge_device' and model_key == 'gemma-3-2b':
        reasons.append('Engineered specifically for edge silicon & CPU execution')
    
    return reasons[:5]


def build_response(
    primary_key: str,
    score: int,
    all_scores: Dict,
    reasons: List[str],
    num_examples: int,
    alternatives: List[Dict] = None,
    score_breakdowns: Dict = None
) -> Dict:
    """Build the final response with model recommendation and enriched metadata."""
    model = MODELS[primary_key]
    confidence = 'high' if score >= 80 else ('medium' if score >= 60 else 'low')
    score_breakdowns = score_breakdowns or {}
    
    def format_context_window(tokens):
        if tokens >= 1000:
            return f"{tokens // 1000}K"
        return str(tokens)
    
    formatted_alternatives = []
    if alternatives:
        for alt in alternatives:
            alt_key = alt.get('model')
            alt_model = MODELS.get(alt_key)
            if alt_model:
                ctx_window = alt_model.get('context_window', 0)
                formatted_alternatives.append({
                    'key': alt_key,
                    'model_id': alt_model['id'],
                    'model_name': alt_model['name'],
                    'creator': alt_model.get('creator', 'Open Source'),
                    'size': alt_model['size'],
                    'license': alt_model.get('license', 'Open Source'),
                    'gated': alt_model.get('gated', False),
                    'qlora_vram_gb': alt_model.get('qlora_vram_gb', 2.8),
                    'ollama_tag': alt_model.get('ollama_tag', ''),
                    'benchmarks': alt_model.get('benchmarks', {}),
                    'strengths': alt_model.get('strengths', []),
                    'category': alt_model.get('category', []),
                    'score_breakdown': score_breakdowns.get(alt_key, {}),
                    'score': round(alt.get('score', 0) / 100, 2),
                    'raw_score': alt.get('score', 0),
                    'reasons': alt.get('reasons', ['Good alternative']),
                    'context_window': ctx_window,
                    'context_window_formatted': format_context_window(ctx_window)
                })
    
    full_matrix = []
    for m_key, m_info in MODELS.items():
        m_score = all_scores.get(m_key, 0)
        is_primary = (m_key == primary_key)
        is_alt = any(alt.get('model') == m_key for alt in (alternatives or []))
        if is_primary:
            status = 'Recommended'
        elif is_alt:
            status = 'Alternative'
        else:
            status = 'Compatible'
            
        full_matrix.append({
            'key': m_key,
            'model_id': m_info['id'],
            'model_name': m_info['name'],
            'creator': m_info.get('creator', 'Open Source'),
            'size': m_info['size'],
            'license': m_info.get('license', 'Open Source'),
            'gated': m_info.get('gated', False),
            'qlora_vram_gb': m_info.get('qlora_vram_gb', 2.8),
            'ollama_tag': m_info.get('ollama_tag', ''),
            'benchmarks': m_info.get('benchmarks', {}),
            'strengths': m_info.get('strengths', []),
            'category': m_info.get('category', []),
            'score_breakdown': score_breakdowns.get(m_key, {}),
            'score': round(min(m_score, 100) / 100, 2),
            'raw_score': m_score,
            'context_window': m_info.get('context_window', 0),
            'context_window_formatted': format_context_window(m_info.get('context_window', 0)),
            'memory_gb': m_info.get('memory_gb', 8),
            'gpu_tier': m_info.get('gpu_tier', 'Local 4GB / Colab T4'),
            'accuracy_baseline': m_info.get('accuracy_baseline', 85),
            'training_time_base': m_info.get('training_time_base', 3),
            'status': status,
            'is_primary': is_primary
        })
    full_matrix.sort(key=lambda x: -x['raw_score'])

    primary_breakdown = score_breakdowns.get(primary_key, {
        'task_fit': 45, 'size_fit': 25, 'output_fit': 15, 'multi_turn_bonus': 5, 'total': score
    })

    return {
        'primary_recommendation': {
            'key': primary_key,
            'model_id': model['id'],
            'model_name': model['name'],
            'creator': model.get('creator', 'Open Source'),
            'size': model['size'],
            'license': model.get('license', 'Open Source'),
            'gated': model.get('gated', False),
            'qlora_vram_gb': model.get('qlora_vram_gb', 2.8),
            'ollama_tag': model.get('ollama_tag', ''),
            'benchmarks': model.get('benchmarks', {}),
            'strengths': model.get('strengths', []),
            'category': model.get('category', []),
            'score_breakdown': primary_breakdown,
            'score': round(min(score, 100) / 100, 2),
            'confidence': confidence,
            'reasons': reasons,
            'context_window': model.get('context_window', 0),
            'context_window_formatted': format_context_window(model.get('context_window', 0))
        },
        'alternatives': formatted_alternatives,
        'all_scores': {k: round(min(v, 100) / 100, 2) for k, v in all_scores.items()},
        'full_matrix': full_matrix
    }
