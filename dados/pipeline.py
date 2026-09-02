"""
pipeline.py

Motor de execucao: roda a base fixa de leads atraves das etapas de um
ConfiguracaoEstado, registrando cada decisao em um arquivo jsonl (append-only).

Cada lead passa pelas etapas em sequencia. Se uma etapa retorna passou=False,
o lead "morre" no funil naquela etapa (perdido) e o restante das etapas
nao e' executado para ele -- exatamente como aconteceria com um lead real
que nao avanca no funil.

Ha' duas formas de AGENDAR essa execucao -- executar_estado (lote, sequencial)
e executar_estado_escalonado (leads intercalados, para o modo "ao vivo") --
mas as duas chamam o MESMO nucleo (_processar_etapa) para semear, executar e
registrar cada etapa individual. Isso existe para eliminar uma classe inteira
de bugs: antes desta unificacao, o modo "ao vivo" reimplementava essa logica
por conta propria e foi ficando dessincronizado do lote a cada correcao (foi
assim que o pareamento por semente do F-07 nunca chegou no modo ao vivo, e
foi assim que o campo qualidade_handoff do F-11 quase repetiu o mesmo erro).
Agora so' existe UM lugar onde "como uma etapa e' executada" esta' definido.
"""

import hashlib
import json
import os
import random
import time
from datetime import datetime

import numpy as np
import pandas as pd


def _seed_pareada(repeticao: int, lead_id: int, nome_etapa: str) -> int:
    """Semente deterministica por (repeticao, lead, etapa) -- NAO usa hash()
    embutido do Python porque, com PYTHONHASHSEED randomizado (padrao desde
    o Python 3.3), hash() de uma string varia entre execucoes do processo,
    quebrando a reprodutibilidade entre maquinas/execucoes. md5 e' estavel.

    Isso implementa numeros aleatorios comuns (common random numbers) por
    etapa, nao por repeticao inteira: os estados tem sequencias de etapas
    de tamanhos diferentes (o Estado C/D pulam qualificacao_sdr, que
    consome sorteios em A/B), entao uma unica semente no inicio da
    replica desalinharia os sorteios das etapas seguintes entre estados.
    Semear por etapa garante que "negociacao do lead X na replica N" sorteie
    os mesmos numeros em A, B, C e D, independentemente do que veio antes.

    Tambem serve para semear execucoes de UM estado so' (ex: a geracao do
    historico de treino, repeticao=SEED_TREINO em
    rodar_estado_a_para_treino_modelos.py) -- a funcao nao pressupoe
    comparacao entre estados, so' precisa de um numero de repeticao valido.
    """
    chave = f"{repeticao}|{lead_id}|{nome_etapa}"
    digest = hashlib.md5(chave.encode()).hexdigest()
    return int(digest[:8], 16)


def _conversor_json(obj):
    """Converte tipos numpy (bool_, int64, float64) para tipos nativos do Python,
    necessario porque resultados de modelos sklearn frequentemente retornam
    esses tipos, que json.dumps nao serializa por padrao."""
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class EventLogger:
    """Escreve um evento por linha em jsonl. Append-only, flush imediato --
    permite que um dashboard leia o arquivo conforme ele cresce (efeito 'ao vivo')."""

    def __init__(self, caminho_arquivo: str):
        os.makedirs(os.path.dirname(caminho_arquivo), exist_ok=True)
        self.caminho = caminho_arquivo
        self._arquivo = open(caminho_arquivo, "w", encoding="utf-8")

    def registrar(self, **kwargs):
        evento = {"timestamp": datetime.now().isoformat(), **kwargs}
        self._arquivo.write(json.dumps(evento, ensure_ascii=False, default=_conversor_json) + "\n")
        self._arquivo.flush()

    def fechar(self):
        self._arquivo.close()


def _resumo_vazio(config) -> dict:
    return {
        "total_leads": 0,
        "fechados": 0,
        "perdidos_por_etapa": {etapa: 0 for etapa in config.etapas},
        "tempo_total_min": 0.0,
        "tempo_por_etapa_min": {etapa: 0.0 for etapa in config.etapas},
        "leads_processados_por_etapa": {etapa: 0 for etapa in config.etapas},
        # Diagnostico F-11: distribuicao de qualidade_handoff (0..1) recebida
        # por descoberta_closer, so' para leads que passaram na qualificacao
        # (unica populacao em que o campo existe -- ver descoberta_closer_humana).
        "qualidade_handoff_valores": [],
    }


def _processar_etapa(config, logger: EventLogger, resumo: dict, lead_dict: dict,
                      contexto: dict, nome_etapa: str, repeticao: int | None) -> dict:
    """Nucleo COMPARTILHADO entre executar_estado e executar_estado_escalonado:
    executa uma etapa para um lead. Semeia deterministicamente (se repeticao
    informado), chama a funcao da etapa, registra o evento no jsonl, atualiza
    o resumo agregado e repassa os sinais relevantes (qualidade_handoff,
    dor_principal) para o contexto da proxima etapa desse mesmo lead.

    Unico lugar do codigo onde isso acontece -- ver docstring do modulo.
    """
    funcao = config.funcoes[nome_etapa]

    if repeticao is not None:
        random.seed(_seed_pareada(repeticao, int(lead_dict["lead_id"]), nome_etapa))

    if nome_etapa == "descoberta_closer" and "qualidade_handoff" in contexto:
        resumo["qualidade_handoff_valores"].append(contexto["qualidade_handoff"])

    resultado = funcao(lead_dict, contexto)

    logger.registrar(
        estado=config.nome,
        lead_id=int(lead_dict["lead_id"]),
        etapa=nome_etapa,
        passou=resultado["passou"],
        tempo_gasto_min=resultado["tempo_gasto_min"],
        detalhes=resultado.get("detalhes", {}),
    )

    resumo["tempo_total_min"] += resultado["tempo_gasto_min"]
    resumo["tempo_por_etapa_min"][nome_etapa] += resultado["tempo_gasto_min"]
    resumo["leads_processados_por_etapa"][nome_etapa] += 1

    # repassa sinais relevantes para a proxima etapa via contexto.
    # qualidade_handoff (ver F-11): contrato neutro [0,1] alimentado
    # tanto por qualificacao_sdr_humana quanto por ia.qualificacao_ia,
    # consumido por descoberta_closer_humana para calibrar a chance de
    # reuniao proveitosa -- sem isso, qualquer estado que qualifica via
    # IA cai num default silencioso, penalizando-o (ver achado F-11).
    # NOTA: propositalmente NAO repassamos nenhum sinal de "qualidade
    # da triagem" que influencie a probabilidade de negociacao/
    # fechamento. A qualificacao (humana ou por IA) decide QUEM entra
    # no funil e melhora PRODUTIVIDADE (tempo, volume) -- mas a chance
    # de fechar negocio depende so de fit_real e ruido genuino de cada
    # etapa, igual para todos os estados. Isso evita atribuir a uma
    # IA de triagem um poder de "vender melhor" que ela nao tem na
    # pratica, e mantem o experimento reprodutivel e defensavel.
    if "qualidade_handoff" in resultado.get("detalhes", {}):
        contexto["qualidade_handoff"] = resultado["detalhes"]["qualidade_handoff"]
    if "dor_principal" in resultado.get("detalhes", {}):
        contexto["dor_principal"] = resultado["detalhes"]["dor_principal"]

    if not resultado["passou"]:
        resumo["perdidos_por_etapa"][nome_etapa] += 1

    return resultado


def _contar_fechados(caminho_saida_jsonl: str) -> int:
    """Recalcula fechados a partir do proprio jsonl para evitar duplicar logica."""
    fechados = 0
    with open(caminho_saida_jsonl, "r", encoding="utf-8") as f:
        for linha in f:
            evento = json.loads(linha)
            if evento["etapa"] == "fechamento" and evento["passou"]:
                fechados += 1
    return fechados


def executar_estado(
    config,
    leads_df: pd.DataFrame,
    caminho_saida_jsonl: str,
    contexto_extra: dict | None = None,
    pausa_entre_leads: float = 0.0,
    repeticao: int | None = None,
):
    """
    Executa um ConfiguracaoEstado sobre toda a base de leads, um lead por vez
    (todas as etapas daquele lead antes do proximo comecar). Motor usado para
    gerar os dados OFICIAIS de comparacao entre estados (lote, sequencial,
    rapido).

    contexto_extra: dict com objetos auxiliares que algumas funcoes precisam
        (ex: {"modelo_score": modelo, "modelo_match": modelo_knn} no Estado C)
    pausa_entre_leads: segundos de sleep entre cada lead -- usado SOMENTE para
        demonstracao ao vivo no dashboard; deixar 0.0 para execucao em lote real.
    repeticao: se informado, cada etapa e' semeada deterministicamente por
        (repeticao, lead_id, nome_etapa) antes de ser chamada -- ver
        _seed_pareada. Isso pareia os sorteios estocasticos das etapas
        compartilhadas (negociacao, fechamento, etc.) entre os estados na
        mesma replica, permitindo teste t pareado em vez de amostras
        independentes. Se None (padrao), preserva o comportamento anterior
        (fluxo global do modulo random, sem pareamento).
    """
    logger = EventLogger(caminho_saida_jsonl)
    contexto_extra = contexto_extra or {}
    resumo = _resumo_vazio(config)
    resumo["total_leads"] = len(leads_df)

    for _, lead in leads_df.iterrows():
        lead_dict = lead.to_dict()
        contexto = {"hora_do_dia": (lead_dict["lead_id"] % 8)}
        contexto.update(contexto_extra)

        for nome_etapa in config.etapas:
            resultado = _processar_etapa(config, logger, resumo, lead_dict, contexto, nome_etapa, repeticao)
            if not resultado["passou"]:
                break

        if pausa_entre_leads > 0:
            time.sleep(pausa_entre_leads)

    resumo["fechados"] = _contar_fechados(caminho_saida_jsonl)
    logger.fechar()
    return resumo


def executar_estado_escalonado(
    config,
    leads_df: pd.DataFrame,
    caminho_saida_jsonl: str,
    contexto_extra: dict | None = None,
    intervalo_chegada: float = 0.3,
    repeticao: int | None = None,
):
    """
    Variante do executar_estado pensada para o modo "ao vivo" do dashboard.
    Diferenca (so' de AGENDAMENTO -- a execucao de cada etapa em si vem do
    mesmo _processar_etapa que o lote usa): em vez de processar um lead por
    completo antes do proximo comecar, os leads sao intercalados -- um novo
    lead "chega" a cada `intervalo_chegada` segundos e avanca uma etapa por
    vez, revezando com os leads ja em andamento. Isso cria FILAS REAIS e
    visiveis entre etapas (varios leads podem estar aguardando a mesma
    estacao ao mesmo tempo), o que e' fisicamente mais correto para a
    visualizacao do que o processamento sequencial do lote.

    repeticao: se informado (recomendado), reproduz exatamente a mesma
        replica do lote -- mesmos sorteios estocasticos, mesmos eventos,
        so' a ORDEM de exibicao no tempo e' diferente (intercalada em vez
        de sequencial). Se None, roda em modo "exploratorio": sorteios
        genuinamente novos a cada execucao, sem correspondencia com nenhuma
        replica oficial -- NAO deve ser usado para gerar numeros reportados.
    """
    logger = EventLogger(caminho_saida_jsonl)
    contexto_extra = contexto_extra or {}
    resumo = _resumo_vazio(config)
    resumo["total_leads"] = len(leads_df)

    # cada "trabalhador" representa um lead em andamento: guarda em que
    # etapa do funil ele esta e seu proprio contexto acumulado
    leads_pendentes = list(leads_df.to_dict("records"))
    em_andamento = []  # cada item: {"lead": dict, "contexto": dict, "idx_etapa": int}

    MAX_EM_ANDAMENTO = 10  # quantos leads podem estar "na linha" ao mesmo tempo

    while leads_pendentes or em_andamento:
        # admite quantos leads novos couberem na capacidade disponivel --
        # nao so um por rodada, senao com reprovacoes rapidas (ex: SDR
        # reprova na 1a etapa) a "linha" nunca chega a ter mais de 1 lead
        # por vez e a fila nunca aparece visualmente.
        while leads_pendentes and len(em_andamento) < MAX_EM_ANDAMENTO:
            novo_lead = leads_pendentes.pop(0)
            contexto = {"hora_do_dia": (novo_lead["lead_id"] % 8)}
            contexto.update(contexto_extra)
            em_andamento.append({"lead": novo_lead, "contexto": contexto, "idx_etapa": 0})

        # avanca cada lead em andamento UMA etapa por rodada -- e' isso
        # que faz leads diferentes aparecerem entrelacados no jsonl, em
        # vez de um lead completar o funil inteiro antes do proximo comecar
        ainda_em_andamento = []
        for item in em_andamento:
            idx = item["idx_etapa"]
            if idx >= len(config.etapas):
                continue  # ja terminou, nao deveria estar aqui

            nome_etapa = config.etapas[idx]
            resultado = _processar_etapa(
                config, logger, resumo, item["lead"], item["contexto"], nome_etapa, repeticao
            )

            # pequena pausa apos CADA evento individual (nao por lote) --
            # e' isso que da o ritmo "ao vivo" visivel no dashboard
            time.sleep(intervalo_chegada / 4)

            if not resultado["passou"]:
                continue  # lead sai do sistema, nao volta para ainda_em_andamento

            item["idx_etapa"] += 1
            if item["idx_etapa"] < len(config.etapas):
                ainda_em_andamento.append(item)
            # se completou todas as etapas, tambem sai (nao precisa readicionar)

        em_andamento = ainda_em_andamento

    resumo["fechados"] = _contar_fechados(caminho_saida_jsonl)
    logger.fechar()
    return resumo
