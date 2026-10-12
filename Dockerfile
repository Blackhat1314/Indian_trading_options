# One image for every service (ingestor / analytics / gateway / backfill / research).
# Stage 1 builds the React terminal; stage 2 is the Python runtime that also serves it.

FROM node:22-alpine AS ui
WORKDIR /ui
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app/backend TZ=Asia/Kolkata
WORKDIR /app
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt
COPY backend/ backend/
COPY backfill/ backfill/
COPY research/ research/
COPY run.py .env.example ./
COPY --from=ui /ui/dist frontend/dist
VOLUME ["/app/data"]
EXPOSE 8000
WORKDIR /app/backend
CMD ["python", "-m", "uvicorn", "td.services.gateway:app", "--host", "0.0.0.0", "--port", "8000"]
