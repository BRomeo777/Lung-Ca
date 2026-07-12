# Phase 8 CDSS — Dockerfile for Render (repo root)
FROM python:3.11-slim

WORKDIR /app

# Install CPU-only PyTorch (smaller image)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY PHASE8_CDSS/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY PHASE8_CDSS/ .

EXPOSE $PORT

CMD streamlit run app/frontend/streamlit_app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true
