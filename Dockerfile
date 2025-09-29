FROM python:3.10-slim

WORKDIR /workspace

RUN apt-get update && pip install --upgrade pip \
    && pip install transformers bertopic numpy pandas torch transformers datasets anthropic sentence-transformers bitsandbytes

CMD ["python3"]

VOLUME [ "/workspace"]
