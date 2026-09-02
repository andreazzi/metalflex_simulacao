# Guia completo do dashboard — Metalflex Simulação B2B

---

## Conceitos fundamentais

### O que é "tempo" nesta simulação?

O tempo exibido no dashboard **não é o tempo real de execução do programa** — é o **tempo simulado de trabalho humano**, acumulado em minutos ao longo de cada etapa. Cada função de etapa retorna um campo `tempo_gasto_min` que representa o esforço humano modelado (ex: uma reunião de discovery = 60 minutos simulados). O total que aparece nos KPIs é a soma de todos esses minutos de todas as etapas de todos os leads, convertido em horas.

É isso que torna o Estado C tão mais eficiente: a IA substitui etapas que custavam dezenas de minutos humanos por inferências de frações de segundo.

### Os três estados do experimento

| Estado | Descrição | Etapas |
|---|---|---|
| **A — Baseline** | 100% manual, sem IA | 7 etapas |
| **B — Melhoria pontual** | Idêntico ao A, exceto a proposta que passa a ser gerada por LLM | 7 etapas |
| **C — Orquestração sistêmica** | Fluxo redesenhado: RandomForest na entrada, kNN para match de closer, LLM na proposta | 8 etapas |

### O `fit_real` oculto

Cada lead tem um campo `fit_real` (entre 0 e 1) que representa o ajuste genuíno entre o lead e o produto — combinação de orçamento real disponível, necessidade legítima e momento de compra correto. **Nenhuma IA ou humano na simulação vê esse campo diretamente.** Ele só se manifesta nas probabilidades de conversão nas etapas de negociação e fechamento.

O `fit_real` é calculado assim:

```
fit_real = 0,40 × orçamento_normalizado
         + 0,35 × urgência_declarada
         + 0,25 × tamanho_normalizado
         + ruído aleatório (desvio padrão = 0,18)
```

Isso garante que o problema seja **genuinamente solucionável** por um modelo de ML (há correlação com as features visíveis), mas **não trivial** (há ruído que nenhum modelo elimina completamente).

**Princípio central da tese:** a IA de qualificação acelera e direciona esforço, mas não altera a probabilidade de fechamento de um lead que já chegou à negociação. O ganho mensurável está em **produtividade** (tempo, volume), não em vender melhor.

---

## Aba 1 — ⚙️ Setup do experimento

Esta é a aba de laboratório. Antes de rodar qualquer simulação ao vivo, o experimento precisa de três artefatos preparados aqui, **em ordem obrigatória**.

### Etapa 1 — Base de dados (leads_base.csv)

Gera 2.000 leads sintéticos com `seed=42` (completamente determinístico). Qualquer pessoa que execute `gerar_base.py` com os mesmos parâmetros obterá exatamente os mesmos dados — essa é a garantia de replicabilidade do experimento.

Cada lead contém:

| Campo | Descrição |
|---|---|
| `lead_id` | Identificador único (0 a 1999) |
| `empresa` | Nome sintético (Empresa_0001…) |
| `setor` | Indústria, varejo, serviços, tecnologia ou saúde |
| `canal_origem` | Inbound (55%), outbound (30%) ou indicação (15%) |
| `dia_chegada` | Dia de chegada ao funil (0 a 540 ≈ 18 meses úteis) |
| `orcamento_declarado` | Distribuição Normal(50.000, 20.000), mínimo R$ 5.000 |
| `urgencia_declarada` | Distribuição Uniforme(0, 1) |
| `tamanho_funcionarios` | Inteiro aleatório entre 5 e 500 |
| `fit_real` | **Campo oculto** — calculado conforme fórmula acima |

A aba exibe:
- Amostra dos primeiros 10 leads
- Expander explicando o conceito de `fit_real` e por que ele é oculto
- Distribuições das variáveis (histogramas)
- Correlação de cada feature visível com o `fit_real` — evidência de que o modelo tem o que aprender

### Etapa 2 — Rodar Estado A em lote (para treino)

Executa os 2.000 leads pelo processo manual completo (sem IA) usando o motor em lote `executar_estado`, que processa um lead inteiro do início ao fim antes do próximo começar. Gera dois artefatos:

- **`historico_estado_a_treino.jsonl`** — cada linha é um evento de um lead numa etapa (timestamp, lead_id, etapa, passou, tempo_gasto_min, detalhes)
- **`historico_estado_a_consolidado.csv`** — uma linha por lead com resultado final (fechou_negocio = 0 ou 1) e todas as suas features

Esses arquivos são o **material de treino dos modelos**. Usar o histórico do Estado A garante que os modelos aprendam com dados gerados por um processo sem IA — evitando contaminação (data leakage) com os dados dos estados futuros.

A aba exibe depois de concluído:
- 4 KPIs (leads, fechados, taxa de conversão, indicação de uso)
- Funil do Estado A (quantos leads passaram por cada etapa)
- Exemplos de leads que fecharam vs que não fecharam
- Scatter orçamento × urgência marcando onde estão os deals ganhos

### Etapa 3 — Treinar os modelos de IA

Treina dois modelos usando exclusivamente o histórico do Estado A:

#### RandomForest — qualificação de leads (`modelo_score.pkl`)

Classificador binário: features visíveis → probabilidade de fechamento de negócio.

- **Entrada:** orçamento declarado, urgência declarada, tamanho da empresa (em funcionários)
- **Saída:** probabilidade de 0 a 1 de o lead fechar negócio
- Treinado com `class_weight="balanced"` para compensar o desbalanceamento natural (~7% de fechamentos)
- Avaliado por **AUC-ROC** (não por acurácia, que seria artificialmente alta por prever sempre a classe majoritária)
- O **threshold de corte** é calibrado no percentil 50 da distribuição de probabilidades previstas → `threshold_score.pkl` — isso mantém o volume de leads qualificados comparável ao Estado A (~48–50%), isolando o efeito de *qualidade* da decisão, não de *volume*

#### kNN — match de closer por perfil (`modelo_match.pkl`)

Treinado apenas com os leads que fecharam negócio (os "deals ganhos" do histórico do Estado A).

- **Entrada:** as mesmas 3 features do RandomForest
- **Saída:** os 5 deals ganhos mais próximos no espaço de features (k=5, distância euclidiana)
- Usado no Estado C para alocar o closer mais adequado por perfil de cliente

A aba exibe após o treinamento:
- **Métricas de qualidade:** AUC-ROC, correlação com `fit_real`, threshold calibrado, quantidade de deals ganhos usados no treino
- **Distribuição de scores** com a linha de corte do threshold em vermelho tracejado
- **Importância de features** (barra horizontal) — as três variáveis contribuem de forma equilibrada
- **Matriz de confusão** com interpretação dos Verdadeiros Positivos, Falsos Positivos e Falsos Negativos
- **Curva ROC** comparada com a linha de acaso (diagonal)
- **Análise do kNN:** boxplot de distâncias (leads que fecharam estão mais próximos dos deals ganhos) e exemplo concreto dos 5 vizinhos de um lead real

---

## Aba 2 — ▶ Rodar ao vivo

É aqui que as simulações acontecem visualmente em tempo real.

### Motor de execução ao vivo vs. motor em lote

O modo ao vivo usa um motor diferente do modo em lote (`executar_estado_escalonado` vs. `executar_estado`). A diferença chave:

- **Motor em lote:** processa um lead inteiro (todas as etapas) antes do próximo começar. Mais simples, sequencial, usado para gerar dados oficiais.
- **Motor ao vivo:** admite até **10 leads simultaneamente** no funil. Cada lead avança **uma etapa por vez**, revezando com os outros. Isso cria **filas reais e visíveis** quando uma estação é mais lenta que as vizinhas — que é o que torna a demonstração visual informativa.

> ⚠️ Os dados dos jsonl gerados pelo modo ao vivo são usados no dashboard de comparação, mas o motor escalonado pode diferir ligeiramente do motor em lote em edge cases. Para os dados oficiais de referência do TCC, use os scripts `rodar_estado_b.py` e `rodar_estado_c.py` diretamente.

### Controles

| Campo | Opções | Observação |
|---|---|---|
| Estado | A, B ou C | C fica bloqueado se os modelos não foram treinados |
| Velocidade | Lenta (1,0s/lead), Normal (0,3s/lead), Rápida (0,05s/lead) | Controla o sleep entre eventos individuais (= intervalo / 4) |
| Leads | 10 a 2.000 (default 300) | Para banca: 200–300 em Normal dura ~1–2 min de simulação |

O botão **▶ Iniciar** lança o script `rodar_ao_vivo.py` como subprocesso separado — o stdout/stderr vai para um arquivo de log (`log_ultima_execucao_<estado>.txt`) em vez de um pipe, evitando travamentos em execuções longas.

### Canvas do processo (bolinhas animadas)

O centro visual da aba. Cada "bolinha" representa um lead em movimento:

- 🟢 **Verde** — avançou para a próxima etapa
- 🔴 **Vermelho** — foi descartado naquela etapa (saiu do funil)
- ⬜ **Cinza empilhada** — leads na fila aguardando aquela estação processar

As estações são desenhadas em sequência horizontal com trilhas entre elas. A diferença de velocidade de processamento entre estações cria filas visíveis — por exemplo, a etapa de discovery (60 min simulados) fica congestionada porque é mais lenta que a qualificação por IA (0,2 min).

### Painel "Modelos de IA em ação" (apenas B e C)

Aparece logo abaixo do canvas e mostra as decisões dos modelos em tempo real, com os últimos leads processados:

**Coluna 1 — RandomForest (apenas Estado C)**
- Score de probabilidade de fechamento para cada lead recente
- Barra de progresso visual (`█████░░░░░` = proporção da probabilidade)
- Resultado: ✅ qualificado ou ❌ descartado

**Coluna 2 — kNN (apenas Estado C)**
- Similaridade calculada para os últimos leads que passaram pelo match de closer
- 🟢 perfil alto (similaridade > 0,003) ou 🟡 perfil baixo

**Coluna 3 — LLM local Llama (B e C)**
- As últimas 3 propostas geradas pelo modelo, com texto completo em textarea
- Se o lead tinha proposta em cache, o texto vem do `cache_llm.json` instantaneamente

### Dashboard secundário (números e funil)

Abaixo do canvas, mais discreto visualmente:
- 4 KPIs: leads que entraram, negócios fechados, conversão parcial, tempo simulado acumulado
- Gráfico de funil Plotly (quantos leads passaram por cada etapa até agora)
- Expander com os últimos 15 eventos brutos em tabela

O dashboard se atualiza automaticamente a cada 1 segundo enquanto a simulação está rodando (`time.sleep(1)` + `st.rerun()`). Quando o processo filho termina, o auto-refresh para.

### Diagnóstico de erros

Se o subprocesso terminar com código != 0 (erro), o dashboard exibe automaticamente as últimas 1.500 linhas do arquivo de log com sugestão de correção. A causa mais comum é o Ollama não estar rodando — os Estados B e C dependem de uma chamada HTTP para `localhost:11434`.

---

## Aba 3 — 📋 Ver histórico

Mostra o resultado consolidado de uma simulação já concluída, **sem replay**. Lê o arquivo `.jsonl` correspondente ao estado selecionado e exibe:

- 4 KPIs: leads processados, negócios fechados, taxa de conversão, tempo total investido
- Gráfico de funil completo com todas as etapas do estado
- Expander com os últimos 50 eventos brutos em tabela

Útil para revisar uma execução anterior sem precisar rodar de novo. Como o jsonl é append-only, toda a execução fica preservada mesmo que o dashboard seja fechado no meio.

---

## Aba 4 — 📊 Comparar A vs B vs C

Lê os três arquivos `.jsonl` simultaneamente e exibe uma comparação lado a lado. Mostra quais estados foram executados e quando (data/hora da última modificação do arquivo).

### Tabela comparativa de referência

| Métrica | Estado A | Estado B | Estado C |
|---|---|---|---|
| Leads processados | 2.000 | 2.000 | 2.000 |
| Leads qualificados | 960 (48%) | 941 (47%) | 1.000 (50%) |
| Negócios fechados | ~121 (6,0%) | ~121 (6,05%) | ~133 (6,65%) |
| Tempo total do processo | ~5.256 h | ~3.146 h | ~2.274 h |
| Tempo só na triagem | ~946 h | ~942 h | ~7 h |

### Dois gráficos de barra

**Taxa de conversão por estado** — deve permanecer estatisticamente equivalente entre os três. Pequenas flutuações de ruído amostral são esperadas (6,0% → 6,05% → 6,65%), mas nenhuma tendência sistemática deve aparecer. Esse é o princípio central da tese: a IA não vende melhor, apenas acelera e direciona.

**Tempo total investido por estado** — aqui aparece o ganho real. A queda de ~57% do Estado A ao C é atribuível exclusivamente à substituição de etapas humanas por inferência de modelos.

A nota metodológica abaixo dos gráficos lembra que os três estados rodam sobre a mesma base fixa (`leads_base.csv`, `seed=42`), garantindo que as diferenças sejam atribuíveis ao processo e não aos dados de entrada.

---

## Aba 5 — 🔬 Análise dos modelos de IA

A aba mais técnica. Usa os mesmos artefatos que a simulação usa de verdade (os `.pkl` treinados e o `leads_base.csv`) — não são números ilustrativos, são os valores reais por trás de cada decisão do Estado C.

### Painel ao vivo (se o Estado C foi executado)

Se `historico_estado_c.jsonl` existe, mostra em tempo real:
- Histograma de scores do RandomForest (qualificados em verde, descartados em vermelho)
- Histograma das similaridades de perfil calculadas pelo kNN

### 🌳 Como o modelo toma a decisão — árvore de decisão

Expander que exibe visualmente **uma** das 200 árvores do RandomForest, com profundidade limitada a 3 (de ~6 reais).

Cada nó mostra:
- A **condição testada** (`Orçamento declarado ≤ R$ X`)
- A **proporção de leads** que chegou até ali
- A **classe majoritária** naquele nó (Qualificar / Descartar)

Nós em **verde** tendem a qualificar; nós em **roxo** tendem a descartar. Para ler: percorra de cima para baixo seguindo Verdadeiro ou Falso em cada bifurcação. No RandomForest real, a decisão final é a **votação da maioria** entre todas as 200 árvores — esta é apenas uma representante.

### Distribuição de scores e separação

**Histograma sobreposto:** compara a distribuição de scores previstos entre quem fechou negócio (verde, deslocado para a direita) e quem não fechou (cinza). A separação entre as curvas é a evidência visual de que o modelo discrimina. Se estivessem completamente sobrepostas, o modelo não capturaria sinal algum.

**Importância de features (barra horizontal):** mostra quanto cada variável contribuiu para as decisões do RandomForest. As três variáveis contribuem de forma relativamente equilibrada — nenhuma domina sozinha, o que reduz o risco de viés em uma única dimensão.

**3 métricas resumidas:**
- **AUC-ROC** — capacidade discriminativa global (0,5 = acaso, 1,0 = perfeito, >0,7 = bom)
- **Correlação com fit_real** — o quanto o score previsto captura a verdade oculta que nenhum modelo observa diretamente
- **Threshold de qualificação** — o corte usado no Estado C para decidir quem avança

### Análise do kNN

**Boxplot de distâncias:** compara a distância média aos 5 deals ganhos mais próximos entre leads que fecharam vs que não fecharam. Leads que fecharam estão sistematicamente mais próximos dos deals ganhos anteriores — essa é a base estatística do match de closer.

**Scatter orçamento × urgência:** mostra onde estão os deals ganhos (diamantes verdes) no espaço de features. A concentração não-aleatória em regiões de orçamento e urgência mais altos é o que o kNN explora para encontrar leads parecidos.

### Justificativa metodológica (expander)

Texto pronto para apresentação em banca cobrindo:

- **Por que RandomForest:** captura interações não-lineares entre variáveis sem precisar especificá-las manualmente (orçamento alto sozinho não garante fechamento, mas certa combinação de orçamento + urgência sim)
- **Por que kNN:** é uma tarefa de recuperação de casos similares — o kNN preserva os casos individuais e mede distância direta, e é **interpretável por design** ("este lead foi direcionado a este closer porque é parecido com estes N negócios que já fechamos antes")
- **Honestidade metodológica:** o kNN não normaliza variáveis antes de calcular distância, o que faz o orçamento (variando em dezenas de milhares) dominar sobre a urgência (0 a 1). Isso foi testado — a alternativa normalizada resultou em poder de separação menor, sugerindo que o orçamento carrega sinal real nesta base.

---

## Detalhamento dos tempos por etapa

A tabela abaixo mostra o custo simulado de cada etapa em cada estado. É daqui que vêm os totais de horas exibidos no dashboard.

| Etapa | Estado A | Estado B | Estado C |
|---|---|---|---|
| Geração de leads | 0 min | 0 min | 0 min |
| **Qualificação (SDR humano)** | **~25 min/lead × fadiga** | **~25 min/lead × fadiga** | — |
| **Qualificação (IA — RandomForest)** | — | — | **0,2 min/lead** |
| **Match de closer (kNN)** | — | — | **0,1 min/lead** |
| Discovery (reunião com Closer) | 60 min/lead (fixo) | 60 min/lead (fixo) | 60 min/lead (fixo) |
| **Proposta manual** | **~210 min/lead (~3,5h)** | — | — |
| **Proposta via LLM** | — | **1,5 min/lead** (0,5 se cache) | **1,5 min/lead** (0,5 se cache) |
| Negociação | 1–4 rodadas × ~45 min | idem | idem |
| Fechamento | ~20 min | ~20 min | ~20 min |
| Pós-venda | ~30 min | ~30 min | ~30 min |

### Como o gargalo de triagem desaparece do A para o C

No **Estado A**, o SDR gasta em média 25 minutos por lead (distribuição Normal com desvio padrão de 8 minutos), com um fator de **fadiga** que aumenta o tempo e os erros conforme o dia avança (`1 + hora_do_dia/8 × 0,3`). Para 2.000 leads, isso resulta em ~946 horas de esforço humano só na triagem.

No **Estado C**, a inferência do RandomForest custa 0,2 minutos por lead. Para os mesmos 2.000 leads: 2.000 × 0,2 min = 400 min = **~7 horas**. Uma redução de 99%.

### Como a proposta contribui para o ganho do B para o C

A proposta manual (Estado A) custa em média 3,5 horas por lead — trabalho artesanal de formatação de slide e precificação. A proposta via LLM (Estados B e C) custa 1,5 minuto de chamada ao modelo (ou 0,5 minuto se já está em cache). Para os leads que chegam à etapa de proposta (~700 no Estado A), isso representa uma economia de centenas de horas — que explica boa parte do salto de 5.256h para 3.146h do A para o B.

### Por que a taxa de conversão não melhora proporcionalmente

A decisão de fechar negócio depende de `fit_real` e de ruído aleatório da negociação — **não** da qualidade da triagem de entrada. Nas etapas de negociação e fechamento:

```python
# Negociação
avanca = random.random() < (0.20 + 0.55 × lead["fit_real"])

# Fechamento
fechou = random.random() < (0.12 + 0.55 × lead["fit_real"])
```

Nenhuma variável de "qualidade da triagem" entra nessas fórmulas. Uma IA que qualifica melhor coloca leads com `fit_real` ligeiramente mais alto na fila — mas o ruído amostral de 2.000 leads faz essa diferença ser estatisticamente marginal, como os números confirmam (6,0% → 6,05% → 6,65%).

---

## Arquivos gerados e onde cada aba os usa

| Arquivo | Gerado por | Usado por |
|---|---|---|
| `dados/leads_base.csv` | Etapa 1 do Setup | Todos os estados, Análise dos modelos |
| `saidas/historico_estado_a_treino.jsonl` | Etapa 2 do Setup | Etapa 3 do Setup (treino) |
| `saidas/historico_estado_a_consolidado.csv` | Etapa 2 do Setup | Etapa 3 do Setup, Análise dos modelos |
| `modelos/modelo_score.pkl` | Etapa 3 do Setup | Estado C (ao vivo e em lote), Análise dos modelos |
| `modelos/modelo_match.pkl` | Etapa 3 do Setup | Estado C (ao vivo e em lote), Análise dos modelos |
| `modelos/threshold_score.pkl` | Etapa 3 do Setup | Estado C, Análise dos modelos |
| `saidas/historico_estado_a.jsonl` | Aba "Rodar ao vivo" | Aba "Ver histórico", Comparar A vs B vs C |
| `saidas/historico_estado_b.jsonl` | Aba "Rodar ao vivo" | Aba "Ver histórico", Comparar A vs B vs C |
| `saidas/historico_estado_c.jsonl` | Aba "Rodar ao vivo" | Aba "Ver histórico", Comparar A vs B vs C, Análise dos modelos |
| `saidas/cache_llm.json` | Primeira execução com LLM (B ou C) | Todas as execuções seguintes de B e C (evita rechamar o modelo) |
