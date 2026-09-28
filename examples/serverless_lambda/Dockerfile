# Image conteneur AWS Lambda (limite de 10 Go ; l'archive zip classique est limitée à 250 Mo).
#
# Depuis la racine du dépôt, avec un dossier de modèle ONNX dans ./modele :
#   docker build -f examples/serverless_lambda/Dockerfile -t krito-lambda .
#   docker run --rm -p 9000:8080 --memory=1g krito-lambda
#   curl -s localhost:9000/2015-03-31/functions/function/invocations \
#        -d '{"context": "Mon colis n est jamais arrivé", "options": {"a": "la livraison", "b": "la facturation"}}'
#
# Configuration Lambda conseillée : 1024 Mo de mémoire (ce qui alloue aussi plus de CPU), délai 30 s.
FROM public.ecr.aws/lambda/python:3.12

COPY pyproject.toml README.md LICENSE /tmp/krito/
COPY src /tmp/krito/src
RUN pip install --no-cache-dir "/tmp/krito[onnx]" && rm -rf /tmp/krito

ARG MODEL_DIR=modele
COPY ${MODEL_DIR}/tokenizer.json ${MODEL_DIR}/labels.json ${MODEL_DIR}/model.*onnx /opt/krito/model/
COPY examples/serverless_lambda/handler.py ${LAMBDA_TASK_ROOT}/

CMD ["handler.handler"]
