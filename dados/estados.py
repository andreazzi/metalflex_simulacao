"""
estados.py

Define os tres estados do experimento como configuracoes coesas e isoladas.
Cada estado especifica: (1) a sequencia de etapas do fluxo, (2) qual funcao
executa cada etapa, e (3) parametros humanos relevantes (capacidade/dia).

Isso impede combinacoes invalidas -- voce sempre seleciona um ESTADO INTEIRO,
nunca mistura fluxo de um estado com funcoes de outro.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List

import humanos
import ia


@dataclass
class ConfiguracaoEstado:
    nome: str
    descricao: str
    etapas: List[str]
    funcoes: Dict[str, Callable]
    capacidade_sdr_dia: int = 8       # leads/dia que 1 SDR processa (None = nao se aplica)
    capacidade_closer_dia: int = 3    # reunioes/dia que 1 closer processa
    usa_modelos_ia: bool = False
    usa_llm: bool = False


# ---------------------------------------------------------------------------
# ESTADO A — Baseline. Empresa operando normalmente, 7 etapas, 100% humano.
# ---------------------------------------------------------------------------
ESTADO_A = ConfiguracaoEstado(
    nome="Estado A — Baseline",
    descricao="Processo comercial manual, sem intervencao de IA.",
    etapas=[
        "geracao_leads",
        "qualificacao_sdr",
        "descoberta_closer",
        "proposta_manual",
        "negociacao",
        "fechamento",
        "pos_venda",
    ],
    funcoes={
        "geracao_leads": humanos.geracao_leads,
        "qualificacao_sdr": humanos.qualificacao_sdr_humana,
        "descoberta_closer": humanos.descoberta_closer_humana,
        "proposta_manual": humanos.proposta_manual_humana,
        "negociacao": humanos.negociacao_humana,
        "fechamento": humanos.fechamento_humano,
        "pos_venda": humanos.pos_venda_humano,
    },
    capacidade_sdr_dia=8,
    capacidade_closer_dia=3,
    usa_modelos_ia=False,
    usa_llm=False,
)


# ---------------------------------------------------------------------------
# ESTADO B — Melhoria pontual. So a etapa de proposta usa IA (LLM real).
# O resto do fluxo permanece IDENTICO ao Estado A -- inclusive o gargalo
# de qualificacao do SDR, que e' o ponto central da tese.
# ---------------------------------------------------------------------------
ESTADO_B = ConfiguracaoEstado(
    nome="Estado B — Melhoria pontual (IA na proposta)",
    descricao="IA generativa isolada na etapa de proposta. Resto do fluxo continua manual.",
    etapas=[
        "geracao_leads",
        "qualificacao_sdr",
        "descoberta_closer",
        "proposta_ia",          # UNICA etapa substituida
        "negociacao",
        "fechamento",
        "pos_venda",
    ],
    funcoes={
        "geracao_leads": humanos.geracao_leads,
        "qualificacao_sdr": humanos.qualificacao_sdr_humana,
        "descoberta_closer": humanos.descoberta_closer_humana,
        "proposta_ia": ia.proposta_via_llm,
        "negociacao": humanos.negociacao_humana,
        "fechamento": humanos.fechamento_humano,
        "pos_venda": humanos.pos_venda_humano,
    },
    capacidade_sdr_dia=8,
    capacidade_closer_dia=3,
    usa_modelos_ia=False,
    usa_llm=True,
)


# ---------------------------------------------------------------------------
# ESTADO C — Orquestracao sistemica. Fluxo REDESENHADO com 5 etapas.
# IA entra na entrada do funil (score real treinado) e mantem a proposta via
# LLM. O fluxo encurta porque qualificacao + parte da descoberta se fundem.
# ---------------------------------------------------------------------------
ESTADO_C = ConfiguracaoEstado(
    nome="Estado C — Orquestracao sistemica",
    descricao="Fluxo redesenhado: triagem por IA treinada + match de closer + proposta via LLM.",
    etapas=[
        "geracao_leads",
        "qualificacao_ia",       # substitui qualificacao_sdr -- modelo treinado real
        "match_closer_ia",       # novo -- aloca o melhor closer por perfil (kNN)
        "descoberta_closer",     # mantida, mas agora recebe handoff de qualidade muito maior
        "proposta_ia",
        "negociacao",
        "fechamento",
        "pos_venda",
    ],
    funcoes={
        "geracao_leads": humanos.geracao_leads,
        "qualificacao_ia": ia.qualificacao_ia,
        "match_closer_ia": ia.match_closer_ia,
        "descoberta_closer": humanos.descoberta_closer_humana,
        "proposta_ia": ia.proposta_via_llm,
        "negociacao": humanos.negociacao_humana,
        "fechamento": humanos.fechamento_humano,
        "pos_venda": humanos.pos_venda_humano,
    },
    capacidade_sdr_dia=None,   # nao se aplica -- triagem e' instantanea via IA
    capacidade_closer_dia=5,   # closer agora atende mais reunioes pq leads chegam melhores
    usa_modelos_ia=True,
    usa_llm=True,
)


# ---------------------------------------------------------------------------
# ESTADO D — Teste de ablacao. Isola a qualificacao por RandomForest do
# resto das intervencoes de IA do Estado C (sem kNN, sem LLM), para atribuir
# o ganho de conversao de C especificamente a QUAL etapa recebeu IA, nao
# a QUANTAS etapas receberam. Mesma cardinalidade de intervencao que o
# Estado B (uma etapa substituida), mas em local diferente: a restricao do
# processo (qualificacao) em vez da etapa mais cara em tempo (proposta).
# ---------------------------------------------------------------------------
ESTADO_D = ConfiguracaoEstado(
    nome="Estado D — Ablacao (so qualificacao por IA)",
    descricao="RandomForest isolado na qualificacao. Proposta volta a ser manual, sem kNN de match.",
    etapas=[
        "geracao_leads",
        "qualificacao_ia",       # UNICA etapa substituida -- mesmo modelo do Estado C
        "descoberta_closer",
        "proposta_manual",       # volta a ser manual, como no Estado A -- sem LLM
        "negociacao",
        "fechamento",
        "pos_venda",
    ],
    funcoes={
        "geracao_leads": humanos.geracao_leads,
        "qualificacao_ia": ia.qualificacao_ia,
        "descoberta_closer": humanos.descoberta_closer_humana,
        "proposta_manual": humanos.proposta_manual_humana,
        "negociacao": humanos.negociacao_humana,
        "fechamento": humanos.fechamento_humano,
        "pos_venda": humanos.pos_venda_humano,
    },
    capacidade_sdr_dia=None,   # nao se aplica -- triagem e' instantanea via IA, igual ao Estado C
    capacidade_closer_dia=3,   # igual a A/B -- sem kNN de match, handoff nao melhora por alocacao
    usa_modelos_ia=True,
    usa_llm=False,
)


ESTADOS = {"A": ESTADO_A, "B": ESTADO_B, "C": ESTADO_C, "D": ESTADO_D}
