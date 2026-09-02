#!/bin/bash
# iniciar.sh
#
# Script unico para preparar e rodar a simulacao Metalflex no macOS:
#   1. Para qualquer instancia do Ollama que esteja rodando
#   2. Reinicia o Ollama do zero (evita estados travados de sessoes anteriores)
#   3. Confirma que o modelo llama3.1:8b esta disponivel (baixa se faltar)
#   4. Sobe o dashboard Streamlit
#
# Uso:
#   chmod +x iniciar.sh   (so na primeira vez)
#   ./iniciar.sh
#
# Para parar tudo: Ctrl+C no terminal onde este script esta rodando.

set -e

MODELO="llama3.2:3b"
PASTA_RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Metalflex — iniciando ambiente ==="
echo ""

# --- 1. Verifica se o Ollama esta instalado ---
if ! command -v ollama &> /dev/null; then
    echo "ERRO: Ollama não encontrado."
    echo "Instale com: brew install ollama"
    exit 1
fi

# --- 2. Para qualquer instancia do Ollama rodando ---
echo "Parando instâncias anteriores do Ollama (se houver)..."
if pgrep -x "ollama" > /dev/null; then
    pkill -x "ollama" || true
    sleep 1
    echo "  Ollama anterior encerrado."
else
    echo "  Nenhuma instância anterior encontrada."
fi

# Em alguns setups o Ollama roda como app/serviço do macOS (ollama.app) --
# tenta parar isso tambem, ignorando erro se nao existir.
osascript -e 'quit app "Ollama"' 2>/dev/null || true
sleep 1

# --- 3. Reinicia o Ollama em background ---
echo "Iniciando Ollama..."
nohup ollama serve > /tmp/ollama_metalflex.log 2>&1 &
OLLAMA_PID=$!
echo "  Ollama iniciado (PID $OLLAMA_PID), log em /tmp/ollama_metalflex.log"

# espera o servico responder antes de seguir (ate 15s)
echo "  Aguardando o servidor responder..."
for i in $(seq 1 15); do
    if curl -s http://localhost:11434 > /dev/null 2>&1; then
        echo "  Ollama respondendo em localhost:11434."
        break
    fi
    sleep 1
    if [ "$i" -eq 15 ]; then
        echo "  AVISO: Ollama não respondeu em 15s. Verifique /tmp/ollama_metalflex.log"
    fi
done
echo ""

# --- 4. Confirma que o modelo esta disponivel ---
echo "Verificando modelo $MODELO..."
if ollama list | grep -q "$MODELO"; then
    echo "  Modelo já disponível."
else
    echo "  Modelo não encontrado, baixando (isso pode levar alguns minutos)..."
    ollama pull "$MODELO"
fi
echo ""

# --- 5. Sobe o dashboard ---
echo "=== Subindo o dashboard Streamlit ==="
echo "Abra o navegador em http://localhost:8501 se ele não abrir sozinho."
echo "Pressione Ctrl+C para encerrar tudo (dashboard e Ollama)."
echo ""

cleanup() {
    echo ""
    echo "Encerrando..."
    kill "$OLLAMA_PID" 2>/dev/null || true
    exit 0
}
trap cleanup INT TERM

cd "$PASTA_RAIZ/dashboard"
streamlit run app.py
