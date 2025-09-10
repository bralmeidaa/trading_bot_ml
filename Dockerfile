# --- Estágio 1: Builder ---
# Começa com a imagem completa que tem as ferramentas de build
FROM python:3.12 AS builder

# Instala as ferramentas de sistema e Node.js
RUN apt-get update && apt-get install -y \
    nodejs npm \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Atualiza pip para a versão mais recente
RUN pip install --upgrade pip

# Instala dependências Python
COPY requirements.txt .
RUN pip install --prefix="/install" --no-cache-dir -r requirements.txt

# Instala dependências do frontend e faz o build
WORKDIR /app/frontend_react

COPY frontend_react/package.json frontend_react/package-lock.json ./
RUN npm ci

COPY frontend_react/. .
RUN npm run build

# Copia o código do backend
WORKDIR /app
COPY . .

# --- Estágio 2: Imagem Final de Produção ---
# Começa de uma imagem limpa e leve
FROM python:3.12-slim

# Instala APENAS as dependências de RUNTIME necessárias
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libgfortran5 \
    libopenblas0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copia as dependências Python instaladas do estágio 'builder'
COPY --from=builder /install /usr/local

# Copia os arquivos de build do React do estágio 'builder'
COPY --from=builder /app/frontend_react/dist ./frontend_react/dist

# Copia apenas o código do backend necessário para rodar do estágio 'builder'
COPY --from=builder /app /app

# Cria o usuário não-root
RUN useradd -m -u 1000 tradingbot && chown -R tradingbot:tradingbot /app
USER tradingbot

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
EXPOSE 12000
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:12000/api/health || exit 1

CMD ["python", "api_server.py"]
