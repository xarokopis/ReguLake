FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen

COPY src/ ./src/

EXPOSE 8000

CMD ["uv", "run", "python", "src/gtgh_team3_compliance_assistant/main.py", "api"]
