FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    onnxruntime==1.30.0 \
    opencv-python-headless==4.14.0.94 \
    numpy==2.4.6 \
    onnx==1.19.1

WORKDIR /app
COPY bench.py convert.py ./

# モデルは初回実行時に /models へダウンロードし、2回目以降は使い回す
ENTRYPOINT ["python", "bench.py"]
