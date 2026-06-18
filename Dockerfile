FROM python:3.11-slim AS base
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

FROM base AS deps
COPY pyproject.toml .
RUN pip install --upgrade pip && pip install -e ".[dev]"

FROM base AS runtime
COPY --from=deps /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=deps /usr/local/bin /usr/local/bin
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser
COPY src ./src
COPY pyproject.toml .
RUN pip install -e . --no-deps
RUN mkdir -p models data && chown -R appuser:appgroup /app
USER appuser
EXPOSE 8000
CMD ["uvicorn", "twitter_analytics.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
