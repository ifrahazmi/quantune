"""
TuneKit (Now Quantune) - Automated LLM Fine-Tuning Pipeline
============================================================
Note: TuneKit has been rebranded to Quantune.
This module is maintained as an alias for backwards compatibility.
"""

from quantune import (
    QuantuneState,
    TuneKitState,
    ingest_data,
    validate_quality,
    analyze_dataset,
    recommend_model,
    generate_package,
    __version__,
)

__all__ = [
    "QuantuneState",
    "TuneKitState",
    "ingest_data",
    "validate_quality",
    "analyze_dataset",
    "recommend_model",
    "generate_package",
]

