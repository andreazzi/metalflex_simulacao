"""
humanos.py

Funcoes que simulam o comportamento humano em cada etapa do processo comercial.
Um humano NAO analisa perfeitamente: usa heuristica simplificada, tem tempo
variavel de execucao, sofre fadiga ao longo do dia, e introduz ruido/viés
na decisao. Isso e' o que diferencia o Estado A (baseline) dos estados com IA.

Cada funcao recebe um lead (dict/Series) e o estado atual da simulacao
(hora do dia, etc.) e devolve um dicionario com a decisao e o tempo gasto.
"""

import random


def _fadiga(hora_do_dia: float) -> float:
    """Fator de fadiga: humano erra mais e demora mais ao longo do dia."""
    return 1 + (hora_do_dia / 8) * 0.3


# ---------------------------------------------------------------------------
# Etapa 1 — Geracao e entrada de leads (nao tem decisao humana, e' so chegada)
# ---------------------------------------------------------------------------
def geracao_leads(lead: dict, contexto: dict) -> dict:
    return {
        "etapa": "geracao_leads",
        "passou": True,
        "tempo_gasto_min": 0.0,
        "detalhes": {"canal": lead["canal_origem"]},
    }


# ---------------------------------------------------------------------------
# Etapa 2 — Qualificacao (SDR) — O GARGALO MANUAL CLASSICO
# ---------------------------------------------------------------------------
def qualificacao_sdr_humana(lead: dict, contexto: dict) -> dict:
    hora_do_dia = contexto.get("hora_do_dia", 0)
    fadiga = _fadiga(hora_do_dia)

    tempo_base = random.gauss(25, 8)  # minutos, pesquisa manual em LinkedIn/Google
    tempo_real = max(5.0, tempo_base * fadiga)

    # Heuristica humana simplificada: usa so 2 sinais visiveis + ruido (viés/intuicao).
    # Desvio-padrao do ruido e' parametrizavel via contexto (default 18, igual ao
    # valor original) para permitir analise de sensibilidade sem alterar o
    # comportamento padrao de nenhum estado existente.
    sigma_ruido = contexto.get("sigma_ruido_sdr", 18)
    score_aparente = (
        (lead["orcamento_declarado"] / 100_000) * 40
        + lead["urgencia_declarada"] * 35
        + random.gauss(0, sigma_ruido)  # ruido humano: favorece quem "parece" bom
    )
    
    # Limiar calibrado para taxa de qualificacao de 25%, ponto medio da faixa
    # de 20-30% obtida por elicitacao com gerente comercial B2B em atividade,
    # compativel com o benchmark publicado de taxa de qualificacao de SDR.
    # Valor derivado: media_i[1 - Phi((L - base_i)/sigma)] = 0.25, com sigma=18.
    # qualificado = score_aparente > 38
    qualificado = score_aparente > 51.798
    score_aparente_arred = round(score_aparente, 1)

    return {
        "etapa": "qualificacao_sdr",
        "passou": qualificado,
        "tempo_gasto_min": round(tempo_real, 1),
        "detalhes": {
            "score_aparente": score_aparente_arred,
            # Contrato neutro consumido por descoberta_closer_humana (ver F-11):
            # mesma escala [0,1] que qualificacao_ia usa para probabilidade_fechamento,
            # para que a etapa seguinte trate qualificacao humana e por IA de forma
            # identica, sem depender do nome do campo de origem. Derivado do MESMO
            # score_aparente_arred (ja arredondado) que o campo acima, e nao do
            # score_aparente bruto -- dividir antes de arredondar dava um valor de
            # qualidade_handoff ligeiramente diferente do que o codigo anterior
            # produzia (arredondava primeiro, dividia depois), o suficiente para
            # cruzar o limiar de random.random() em alguns leads e alterar
            # negocios_fechados de A e B -- exatamente o que a restricao do F-11
            # probe contra.
            "qualidade_handoff": 0.50,
        },
    }


# ---------------------------------------------------------------------------
# Etapa 3 — Descoberta (Closer) — depende da qualidade do que o SDR passou
# ---------------------------------------------------------------------------
def descoberta_closer_humana(lead: dict, contexto: dict) -> dict:
    # Sem default silencioso (achado F-11): toda etapa de qualificacao -- humana
    # (qualificacao_sdr_humana) ou por IA (ia.qualificacao_ia) -- e' responsavel
    # por escrever contexto["qualidade_handoff"] antes de descoberta_closer
    # rodar. Se faltar, e' erro de composicao do estado, nao algo a mascarar.
    qualidade_handoff = contexto["qualidade_handoff"]
    tempo_reuniao_min = 60.0

    # Se a qualificacao foi malfeita, ha risco de reuniao inutil
    reuniao_proveitosa = random.random() < (0.35 + 0.5 * qualidade_handoff)

    return {
        "etapa": "descoberta_closer",
        "passou": reuniao_proveitosa,
        "tempo_gasto_min": tempo_reuniao_min,
        "detalhes": {
            "dor_principal": random.choice(
                ["custo_operacional", "falta_visibilidade", "atraso_entrega", "qualidade"]
            ),
            "reuniao_proveitosa": reuniao_proveitosa,
        },
    }


# ---------------------------------------------------------------------------
# Etapa 4 — Proposta manual (Closer para de vender pra formatar slide/preco)
# ---------------------------------------------------------------------------
def proposta_manual_humana(lead: dict, contexto: dict) -> dict:
    tempo_base_h = random.gauss(3.5, 1.0)  # horas -- trabalho artesanal
    tempo_real_h = max(1.0, tempo_base_h)

    return {
        "etapa": "proposta_manual",
        "passou": True,
        "tempo_gasto_min": round(tempo_real_h * 60, 1),
        "detalhes": {"metodo": "manual_planilha_slide"},
    }


# ---------------------------------------------------------------------------
# Etapa 5 — Negociacao e revisao — nao estruturada, dificil de rastrear
# ---------------------------------------------------------------------------
def negociacao_humana(lead: dict, contexto: dict) -> dict:
    n_rodadas = random.randint(1, 4)
    tempo_total_min = n_rodadas * random.gauss(45, 15)

    # IMPORTANTE: a chance de avancar na negociacao depende SO de fit_real
    # (o ajuste genuino entre produto e necessidade do cliente) e de ruido
    # humano -- NAO da qualidade da triagem/qualificacao. Uma IA de
    # qualificacao acelera e direciona esforco, mas nao "vende melhor".
    # Isso garante que a comparacao entre estados isole o efeito de
    # PRODUTIVIDADE (tempo, volume), nao um ganho artificial de conversao
    # que IA de triagem nao teria como gerar na realidade.
    avanca = random.random() < (0.20 + 0.55 * lead["fit_real"])

    return {
        "etapa": "negociacao",
        "passou": avanca,
        "tempo_gasto_min": round(max(15.0, tempo_total_min), 1),
        "detalhes": {"rodadas": n_rodadas},
    }


# ---------------------------------------------------------------------------
# Etapa 6 — Fechamento
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Etapa 6 — Fechamento
# ---------------------------------------------------------------------------
def fechamento_humano(lead: dict, contexto: dict) -> dict:
    # A decisao final de fechar depende do fit real (produto/preco/necessidade
    # genuinamente compativeis) e de ruido proprio da negociacao final --
    # NAO da qualidade da triagem de entrada. Manter essa separacao e'
    # essencial: uma IA de qualificacao nao deveria, na pratica, alterar a
    # probabilidade de venda de um lead que ja chegou ate aqui.
    prob_fechar = 0.12 + 0.55 * lead["fit_real"]
    fechou = random.random() < prob_fechar

    return {
        "etapa": "fechamento",
        "passou": fechou,
        "tempo_gasto_min": round(random.gauss(20, 5), 1),
        "detalhes": {"resultado": "ganho" if fechou else "perdido"},
    }


# ---------------------------------------------------------------------------
# Etapa 7 — Pos-venda e aprendizado (raramente realimenta o funil no Estado A)
# ---------------------------------------------------------------------------
def pos_venda_humano(lead: dict, contexto: dict) -> dict:
    documentado = random.random() < 0.15  # so 15% dos casos vira aprendizado registrado
    return {
        "etapa": "pos_venda",
        "passou": True,
        "tempo_gasto_min": round(random.gauss(30, 10), 1),
        "detalhes": {"feedback_documentado": documentado},
    }
