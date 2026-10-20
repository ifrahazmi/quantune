"""
Quantune Training Module
=======================
Colab notebook generation for SLM fine-tuning.
"""

from .colab_generator import (
    generate_training_notebook as _generate_training_notebook,
    save_notebook,
)
from .playground_notebook import append_playground_cells

__all__ = [
    "generate_training_notebook",
    "save_notebook",
]


def generate_training_notebook(dataset_jsonl: str, model_id: str, model_name: str, analysis: dict) -> str:
    """
    Build the complete training notebook: the core template from colab_generator
    plus the chat playground cells used by Quantune's "Test Your Model" page.
    """
    notebook_json = _generate_training_notebook(
        dataset_jsonl=dataset_jsonl,
        model_id=model_id,
        model_name=model_name,
        analysis=analysis,
    )
    return append_playground_cells(notebook_json, model_name=model_name, dataset_jsonl=dataset_jsonl)
