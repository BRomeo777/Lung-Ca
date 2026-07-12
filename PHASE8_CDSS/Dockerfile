# Phase 8 CDSS — Dockerfile for Render
FROM python:3.11-slim

WORKDIR /app

# Install CPU-only PyTorch (smaller image)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE $PORT

ENV STREAMLIT_SERVER_PORT=$PORT
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true

CMD streamlit run app/frontend/streamlit_app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true
