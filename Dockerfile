FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

WORKDIR /workspace

RUN apt-get update && apt-get install git && pip install --upgrade pip && pip install jupyter\
    && pip install transformers bertopic numpy pandas transformers datasets anthropic sentence-transformers bitsandbytes accelerate

CMD ["python3"]

VOLUME [ "/workspace"]
