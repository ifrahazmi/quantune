"""
Quantune application entry point
================================
Provides the unified FastAPI app:

Run with:
    uvicorn api.app:app --reload
or
    uvicorn api.main:app --reload
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.main import app

__all__ = ["app"]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
