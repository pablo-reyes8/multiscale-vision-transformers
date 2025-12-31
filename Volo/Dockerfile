FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

WORKDIR /app

COPY Volo/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

ENV PYTHONPATH=/app/Volo
ENV MPLBACKEND=Agg

WORKDIR /app/Volo

CMD ["python", "scripts/infer_volo_cli.py", "--help"]
