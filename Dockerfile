FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY src/ ./src/

RUN uv sync --frozen

EXPOSE 8000

CMD ["uv", "run", "-m", "gtgh_team3_compliance_assistant.main", "api", "-H", "0.0.0.0", "-p", "8000"]