# API Krito sur CPU, sans PyTorch ni GPU.
#
# À construire depuis la racine du dépôt, avec un dossier de modèle ONNX
# (model.opt.onnx ou model.int8.onnx, tokenizer.json, labels.json) :
#
#   hf download polymorfis/krito-nli-fr-multi --local-dir modele
#   docker build -f examples/docker_api/Dockerfile -t krito-api .
#   docker run --rm -p 8000:8000 --memory=1g --cpus=2 krito-api
#
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    KRITO_MODEL=/opt/krito/model KRITO_THREADS=2
WORKDIR /opt/krito

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir ".[onnx]" "fastapi>=0.115" "uvicorn>=0.30" \
    && rm -rf src pyproject.toml

ARG MODEL_DIR=modele
COPY ${MODEL_DIR}/tokenizer.json ${MODEL_DIR}/labels.json ${MODEL_DIR}/model.*onnx ./model/
COPY examples/_modele.py examples/04_api_fastapi.py ./app/

RUN useradd --system krito
USER krito
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "--app-dir", "app", "04_api_fastapi:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
