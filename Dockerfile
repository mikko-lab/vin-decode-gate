FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md LICENSE ./
COPY src/ src/
COPY scripts/ scripts/
COPY data/ data/
ENV PYTHONPATH=/app/src

# The corpus is fetched at build time rather than vendored: it is not ours
# to redistribute. Train here so the image ships with its artifacts.
RUN ./scripts/fetch_data.sh \
    && python -m vindecode.cli --data data/ml-engineer-challenge-redacted-data.csv --out artifacts

EXPOSE 8000
CMD ["uvicorn", "vindecode.api:app", "--host", "0.0.0.0", "--port", "8000"]
