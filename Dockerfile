FROM node:22-bookworm-slim AS frontend
WORKDIR /build
COPY frontend/package*.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM mcr.microsoft.com/dotnet/sdk:10.0-noble AS reader
ENV DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_GENERATE_ASPNET_CERTIFICATE=false
WORKDIR /build
COPY exporter/ ./exporter/
RUN dotnet publish exporter/LazerReader.csproj -c Release -o /reader
COPY difficulty/ ./difficulty/
RUN dotnet publish difficulty/BobDifficulty.csproj -c Release -o /difficulty

FROM mcr.microsoft.com/dotnet/runtime:10.0-noble
ARG COMPUTE=cpu
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv ffmpeg libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
COPY backend/requirements.txt /tmp/requirements.txt
RUN if [ "$COMPUTE" = cuda ]; then \
      pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128; \
    elif [ "$COMPUTE" = directml ]; then \
      pip install --no-cache-dir torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu \
      && pip install --no-cache-dir torch-directml==0.2.5.dev240914; \
    else pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu; fi \
    && pip install --no-cache-dir -r /tmp/requirements.txt
WORKDIR /app
COPY --from=reader /reader /opt/reader
COPY --from=reader /difficulty /opt/difficulty
COPY --from=frontend /build/dist /app/frontend/dist
COPY backend /app/backend
ENV PYTHONPATH=/app/backend BOB_DATA=/app/data BOB_READER=/opt/reader/LazerReader.dll \
    BOB_DIFFICULTY=/opt/difficulty/BobDifficulty.dll \
    BOB_FRONTEND=/app/frontend/dist BOB_DEVICE=${COMPUTE} BOB_OSU_PATH=/osu \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TMPDIR=/app/data/tmp \
    HOME=/app/data/cache/home XDG_CACHE_HOME=/app/data/cache \
    DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_GENERATE_ASPNET_CERTIFICATE=false
EXPOSE 8000
CMD ["python", "-m", "app.launch"]
