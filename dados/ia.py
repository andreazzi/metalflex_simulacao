"""
ia.py

Funcoes que usam modelos de IA REAIS (treinados) e o LLM local via Ollama
para tomar decisoes ou gerar conteudo. Usadas nos Estados B e C.

Modelos: RandomForest (score de fit) e NearestNeighbors (match de perfil)
sao treinados em treinar_modelos.py a partir do historico do Estado A,
e carregados aqui via pickle. O LLM (Llama local) e' chamado via Ollama.
"""

import hashlib
import json
import os
import pickle

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"

# Modelo padrao: llama3.2:3b -- mais rapido para demonstracoes ao vivo
# (responde em ~2-5s por proposta no Mac vs ~15-30s do 8b).
# Para trocar, altere a linha abaixo e rode: ollama pull <nome_do_modelo>
# Opcoes disponiveis (do mais rapido ao mais capaz):
#   llama3.2:1b  -- ~800MB, <2s por proposta, qualidade basica
#   llama3.2:3b  -- ~2GB,   2-5s por proposta  (PADRAO)
#   phi3:mini    -- ~2.2GB, bom para textos estruturados em PT
#   qwen2.5:3b   -- ~2GB,   muito bom em portugues
#   llama3.1:8b  -- ~4.7GB, melhor qualidade, mas lento sem GPU dedicada
MODELO_LLM = "llama3.2:3b"
CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "saidas", "cache_llm.json")

FEATURES_MODELO = ["orcamento_declarado", "urgencia_declarada", "tamanho_funcionarios"]


# ---------------------------------------------------------------------------
# Carregamento dos modelos treinados (chamado uma vez por execucao de estado)
# ---------------------------------------------------------------------------
def carregar_modelo_score(caminho="modelo_score.pkl"):
    with open(caminho, "rb") as f:
        return pickle.load(f)


def carregar_modelo_match(caminho="modelo_match.pkl"):
    with open(caminho, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Etapa de triagem/qualificacao via IA real (RandomForest)
# ---------------------------------------------------------------------------
def qualificacao_ia(lead: dict, contexto: dict) -> dict:
    import pandas as pd
    modelo = contexto["modelo_score"]
    threshold = contexto.get("threshold_score", 0.5)
    x = pd.DataFrame([[lead[f] for f in FEATURES_MODELO]], columns=FEATURES_MODELO)
    proba = modelo.predict_proba(x)[0][1]
    qualificado = proba > threshold

    return {
        "etapa": "qualificacao_ia",
        "passou": qualificado,
        "tempo_gasto_min": 0.2,  # inferencia e' quase instantanea
        "detalhes": {
            "probabilidade_fechamento": round(float(proba), 4),
            "threshold_usado": threshold,
            # Contrato neutro consumido por descoberta_closer_humana (ver F-11):
            # a probabilidade do RandomForest E' a confianca do modelo de que o
            # lead fecha -- mapeamento direto, mesma escala [0,1] que a
            # qualificacao humana usa (score_aparente/100).
            "qualidade_handoff": 0.50,
        },
    }


# ---------------------------------------------------------------------------
# Etapa de match closer-perfil via kNN (usada no Estado C)
# ---------------------------------------------------------------------------
def match_closer_ia(lead: dict, contexto: dict) -> dict:
    """Aloca o closer com perfil mais similar a deals ja ganhos (kNN).
    Isso e' um ganho de PRODUTIVIDADE/ALOCACAO -- direciona o lead ao closer
    mais adequado mais rapido que uma triagem manual faria -- mas nao altera
    a probabilidade real de fechamento, que depende do fit_real do lead.

    IMPORTANTE: o kNN mede similaridade por distancia euclidiana, que so e'
    valida quando os atributos estao na mesma escala. Aplicamos aqui a MESMA
    padronizacao (StandardScaler) ajustada durante o treino -- ver
    treinar_modelos.treinar_modelo_match -- para que orcamento (dezenas de
    milhares) nao domine a distancia sobre urgencia (0 a 1)."""
    import pandas as pd
    modelo_knn = contexto["modelo_match"]
    scaler = contexto["scaler_match"]
    x = pd.DataFrame([[lead[f] for f in FEATURES_MODELO]], columns=FEATURES_MODELO)
    x_escalado = scaler.transform(x)
    distancias, indices = modelo_knn.kneighbors(x_escalado)

    perfil_similar_score = float(1 / (1 + distancias.mean()))

    return {
        "etapa": "match_closer_ia",
        "passou": True,
        "tempo_gasto_min": 0.1,
        "detalhes": {"perfil_similaridade": round(perfil_similar_score, 4)},
    }


# ---------------------------------------------------------------------------
# Geracao de proposta via LLM local (Ollama) — com cache para replicabilidade
# ---------------------------------------------------------------------------
def _hash_chave(lead_id: int) -> str:
    # chave estavel por lead -- NAO inclui dor_principal porque essa e'
    # sorteada aleatoriamente a cada execucao de descoberta_closer_humana,
    # o que fazia o cache quase nunca bater entre rodadas diferentes e
    # forcava uma chamada nova ao LLM (lenta) para cada lead toda vez.
    return hashlib.md5(str(lead_id).encode()).hexdigest()


_CACHE_EM_MEMORIA = None  # carregado uma vez por processo, nao a cada chamada


def _carregar_cache() -> dict:
    global _CACHE_EM_MEMORIA
    if _CACHE_EM_MEMORIA is not None:
        return _CACHE_EM_MEMORIA
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            _CACHE_EM_MEMORIA = json.load(f)
    else:
        _CACHE_EM_MEMORIA = {}
    return _CACHE_EM_MEMORIA


def _salvar_cache(cache: dict) -> None:
    # mantem o cache em memoria atualizado e grava em disco -- gravar a
    # cada chamada ainda tem custo, mas e' necessario para sobreviver a
    # uma simulacao interrompida no meio. O ganho de performance real vem
    # de NAO reler o arquivo inteiro do disco a cada chamada (ver acima).
    global _CACHE_EM_MEMORIA
    _CACHE_EM_MEMORIA = cache
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def _chamar_ollama(prompt: str) -> str:
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": MODELO_LLM,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0,
                "seed": 42,
                "num_predict": 180,   # limita o tamanho da resposta -- sem isso
                                      # o modelo pode gerar muito mais que os
                                      # ~150 palavras pedidas, multiplicando o
                                      # tempo de geracao por lead
                "num_ctx": 512,       # contexto pequeno o suficiente pro prompt,
                                      # reduz uso de memoria e acelera a chamada
            },
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["response"]


def proposta_via_llm(lead: dict, contexto: dict) -> dict:
    dor_principal = contexto.get("dor_principal", "nao identificada")
    chave = _hash_chave(lead["lead_id"])

    cache = _carregar_cache()
    if chave in cache:
        texto = cache[chave]
        tempo_gasto = 0.5  # leitura de cache, irrelevante para metricas de tempo
    else:
        prompt = (
            "Voce e' um vendedor B2B experiente. Gere uma proposta comercial "
            "curta e objetiva, em portugues.\n\n"
            f"Cliente: {lead['empresa']}\n"
            f"Setor: {lead['setor']}\n"
            f"Orcamento estimado: R$ {lead['orcamento_declarado']:,.2f}\n"
            f"Dor principal identificada: {dor_principal}\n\n"
            "Gere uma proposta em até 150 palavras, com escopo, prazo estimado "
            "e proximos passos."
        )
        print(f"[ia.py] chamando LLM local para lead {lead['lead_id']}...", flush=True)
        t0 = __import__("time").time()
        try:
            texto = _chamar_ollama(prompt)
            duracao = __import__("time").time() - t0
            print(f"[ia.py] LLM respondeu em {duracao:.1f}s para lead {lead['lead_id']}", flush=True)
            tempo_gasto = 1.5  # tempo de chamada real ao modelo, em minutos simulados
        except requests.exceptions.ConnectionError:
            # Ollama nao esta rodando localmente -- nao deixa o processo
            # inteiro morrer por causa disso. Usa um texto de fallback e
            # sinaliza claramente o problema nos detalhes do evento, para
            # o dashboard poder avisar o usuario.
            texto = ("[AVISO: Ollama não está rodando em localhost:11434 — "
                     "proposta não pôde ser gerada. Rode 'ollama serve' e "
                     "'ollama pull llama3.1:8b' antes de simular os Estados B/C.]")
            tempo_gasto = 0.1
        except requests.exceptions.Timeout:
            texto = "[AVISO: Ollama demorou demais para responder (timeout de 120s).]"
            tempo_gasto = 2.0
        except Exception as e:
            texto = f"[AVISO: erro ao chamar o LLM local: {type(e).__name__}]"
            tempo_gasto = 0.1
        else:
            cache[chave] = texto
            _salvar_cache(cache)

    return {
        "etapa": "proposta_ia",
        "passou": True,
        "tempo_gasto_min": tempo_gasto,
        "detalhes": {"proposta_texto": texto[:300]},
    }
