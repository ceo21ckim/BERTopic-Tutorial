FROM python:3.10-slim

WORKDIR /

RUN apt-get update && pip install --upgrade pip \
    && pip install transformers bertopic numpy pandas torch transformers datasets

CMD ["python", "run.py"]
