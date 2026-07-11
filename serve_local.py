"""Run the SLM-125M inference server locally for frontend testing.

Serves space/app.py (FastAPI) on http://localhost:8000. The API_SECRET here must
match web/.env.local's INFERENCE_SECRET. On first run it downloads the model
(~503MB) from Ace-2504/slm-125m-base into the HF cache.

    ./.venv/Scripts/python.exe serve_local.py
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("API_SECRET", "local-dev-secret")
os.environ.setdefault("MODEL_ID", "Ace-2504/slm-125m-base")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "space"))

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, log_level="info")
