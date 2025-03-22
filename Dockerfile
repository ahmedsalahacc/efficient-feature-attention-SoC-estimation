FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY /datasets /app/datasets

COPY /feature_based_transformer.ipynb /app/src/feature_based_transformer.ipynb