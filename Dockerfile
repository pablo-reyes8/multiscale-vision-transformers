# syntax=docker/dockerfile:1.7

ARG PYTORCH_IMAGE=pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

FROM ${PYTORCH_IMAGE} AS builder

WORKDIR /build

RUN python -m pip install --no-cache-dir "build==1.2.2"

COPY pyproject.toml README.md LICENSE ./
COPY famous_vits ./famous_vits
COPY model_zoo/ViT/model ./model_zoo/ViT/model
COPY model_zoo/HierarchicalViT/model ./model_zoo/HierarchicalViT/model
COPY model_zoo/SwinViT/model ./model_zoo/SwinViT/model
COPY model_zoo/MaxViT/model ./model_zoo/MaxViT/model
COPY model_zoo/MaxViT/model_configurations.py ./model_zoo/MaxViT/model_configurations.py
COPY model_zoo/Volo/model ./model_zoo/Volo/model

RUN python -m build --wheel


FROM ${PYTORCH_IMAGE} AS runtime

ARG APP_UID=10001
ARG APP_GID=10001

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLCONFIGDIR=/tmp/matplotlib

RUN groupadd --gid "${APP_GID}" app \
    && useradd --uid "${APP_UID}" --gid app --create-home app

WORKDIR /app

COPY --from=builder /build/dist /wheels
RUN python -m pip install --no-cache-dir /wheels/*.whl \
    && rm -rf /wheels

COPY --chown=app:app configs ./configs
RUN mkdir -p /app/data /app/outputs /app/arena_runs /tmp/matplotlib \
    && chown -R app:app /app /tmp/matplotlib

USER app

ENTRYPOINT ["famous-vits"]
CMD ["--help"]
