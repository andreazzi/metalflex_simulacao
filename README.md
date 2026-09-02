# Metalflex — Simulação de Processo Comercial B2B

Simulação reprodutível de um funil comercial B2B (geração de leads até fechamento),
comparando três estados de maturidade operacional:

- **Estado A** — baseline. Processo 100% manual, com comportamento humano sintético
  (heurística simplificada + ruído + fadiga).
- **Estado B** — melhoria pontual. Uma IA generativa (LLM local) é inserida isoladamente
  na etapa de geração de proposta. O resto do fluxo permanece manual.
- **Estado C** — orquestração sistêmica. O fluxo é redesenhado: um modelo de classificação
  treinado (RandomForest) qualifica leads na entrada, um modelo de similaridade (kNN)
  aloca o closer ideal por perfil, e o LLM gera a proposta — agora sobre leads
  melhor qualificados.

Os três estados rodam sobre **a mesma base fixa de 2.000 leads** (semente determinística),
o que isola o efeito do processo/IA, sem variação atribuível aos dados de entrada.

## Por que isso importa para a tese

**Princípio central de design:** os modelos de IA (RandomForest e kNN) aceleram e
direcionam o trabalho — não vendem melhor. Eles aumentam produtividade (mais leads
processados por período, triagem em segundos em vez de minutos) e qualidade de
alocação (closer certo para o perfil certo), mas a probabilidade de um lead fechar
negócio depende exclusivamente do `fit_real` (ajuste genuíno entre produto, preço e
necessidade do cliente) e do ruído inerente à negociação — fatores que nenhuma IA de
triagem ou geração de proposta controla. Esse é um princípio deliberado da simulação,
necessário para que o experimento seja reprodutível e defensável: se a taxa de
conversão melhorasse simplesmente por inserir um classificador na entrada do funil,
isso atribuiria à IA um poder de venda que ela não tem na prática.

Por isso, a métrica central de comparação entre os três estados é **produtividade**
(tempo total investido, throughput de leads processados por período), não taxa de
conversão. A taxa de conversão deve permanecer estatisticamente equivalente entre A,
B e C — pequenas flutuações de ruído amostral são esperadas, mas nenhuma tendência
sistemática de melhora ou piora deve aparecer entre os estados nessa métrica.

O Estado B demonstra a "ilusão da melhoria pontual": a IA acelera uma etapa isolada
(proposta gerada em minutos, não horas), reduzindo o tempo total do processo, mas sem
mudar a taxa de conversão — porque a qualificação de entrada continua manual e o
funil não foi redesenhado.

O Estado C demonstra o ganho de uma reestruturação sistêmica: a IA entra na
qualificação (eliminando praticamente todo o tempo do gargalo clássico do SDR — de
~945h para ~7h de esforço de triagem) e o match de closer melhora a alocação. O ganho
mensurável e defensável é de **eficiência operacional**: o mesmo volume de leads é
processado com uma fração do esforço humano, liberando tempo para as etapas que de
fato dependem de julgamento humano (descoberta, negociação).

## Estrutura do projeto

```
metalflex_simulacao/
├── iniciar.sh                  # comando único: reinicia Ollama e sobe o dashboard
├── dados/
│   ├── gerar_base.py          # gera leads_base.csv (roda 1x)
│   ├── humanos.py              # funções de comportamento humano sintético
│   ├── ia.py                   # funções de IA real (RandomForest, kNN, LLM via Ollama)
│   ├── estados.py              # define os três ConfiguracaoEstado (A, B, C)
│   ├── pipeline.py             # motor de execução + event logger (jsonl)
│   ├── rodar_estado_a.py       # executa Estado A, gera histórico para treino
│   ├── treinar_modelos.py      # treina RandomForest + kNN no histórico do Estado A
│   ├── rodar_estado_b.py       # executa Estado B em lote (sem pausa)
│   ├── rodar_estado_c.py       # executa Estado C em lote (sem pausa)
│   ├── rodar_ao_vivo.py        # executa qualquer estado COM pausa entre leads —
│   │                           # usado pelo dashboard para o modo "ao vivo"
│   └── leads_base.csv          # base fixa gerada (versionar este arquivo)
├── modelos/
│   ├── modelo_score.pkl        # RandomForest treinado
│   ├── threshold_score.pkl     # threshold calibrado (percentil 50)
│   └── modelo_match.pkl        # kNN treinado
├── saidas/
│   ├── historico_estado_a.jsonl
│   ├── historico_estado_a_consolidado.csv
│   ├── historico_estado_b.jsonl
│   ├── historico_estado_c.jsonl
│   └── cache_llm.json          # cache de respostas do LLM (garante replicabilidade)
├── dashboard/
│   └── app.py                  # dashboard Streamlit (funil ao vivo + comparação)
├── requirements.txt
└── README.md
```

## Pré-requisitos

- Python 3.10+
- [Ollama](https://ollama.com) instalado localmente, com o modelo `llama3.2:3b`

```bash
brew install ollama
ollama serve &
ollama pull llama3.2:3b
```

> **Nota sobre o modelo:** o padrão é `llama3.2:3b` (~2GB, 2-5s por proposta),
> escolhido por ser significativamente mais rápido que o `llama3.1:8b` para
> demonstrações ao vivo. Para trocar, edite a variável `MODELO_LLM` em
> `dados/ia.py`, apague `saidas/cache_llm.json` e rode `ollama pull <modelo>`.

## Como reproduzir o experimento — passo a passo

A ordem é obrigatória. Os Estados B e C dependem de artefatos gerados pelas etapas
anteriores (base fixa, histórico do Estado A, modelos treinados).

```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Gerar a base fixa de leads (executar uma única vez)
cd dados
python gerar_base.py

# 3. Rodar o Estado A — gera o histórico que servirá de base de treino
python rodar_estado_a.py

# 4. Treinar os modelos de IA (RandomForest + kNN) sobre o histórico do Estado A
python treinar_modelos.py

# 5. Abrir o dashboard
cd ../dashboard
streamlit run app.py
```

A partir do passo 5, os Estados B e C podem ser rodados de duas formas, direto pelo
dashboard.

## Comando único para abrir tudo — iniciar.sh

Depois de já ter rodado os passos 1 a 4 pelo menos uma vez, o script `iniciar.sh`
(na raiz do projeto) automatiza o resto: para qualquer instância anterior do
Ollama, reinicia o serviço do zero, confirma que o modelo `llama3.1:8b` está
disponível (baixa se faltar) e sobe o dashboard — tudo em um único comando.

```bash
chmod +x iniciar.sh   # só na primeira vez
./iniciar.sh
```

Reiniciar o Ollama do zero a cada execução evita estados travados de sessões
anteriores, que foi uma causa real de simulações parando sozinhas no modo ao
vivo. Para encerrar tudo (dashboard e Ollama), use Ctrl+C no terminal onde o
script está rodando.

## Modo ao vivo — acompanhando a simulação rodando

A aba **"Rodar ao vivo"** do dashboard lança o motor de simulação como um processo
em segundo plano e mostra uma cena visual com o fluxo acontecendo: estações
desenhadas em sequência horizontal, ligadas por uma trilha, com bolinhas (leads)
viajando fisicamente de uma estação para a outra — verde quando avançam, vermelho
quando não avançam. Cada estação mostra dois números sempre atualizados: quantos
leads já passaram por ali (contagem real, calculada do histórico completo da
execução) e quantos estão na fila aguardando processamento.

Esse modo usa um motor de execução diferente do motor em lote
(`dados/rodar_ao_vivo.py`, função `executar_estado_escalonado` em `pipeline.py`):
em vez de processar um lead inteiro do início ao fim antes do próximo começar
(como o motor oficial de geração de dados faz), vários leads são admitidos ao
mesmo tempo e avançam uma etapa por vez, revezando entre si — isso é o que cria
filas reais e visíveis quando uma estação é mais lenta que as vizinhas. Essa
variante existe apenas para a demonstração visual; os dados oficiais usados nas
métricas de comparação A/B/C continuam vindo do motor em lote padrão
(`executar_estado`), que processa sequencialmente e já foi validado.

```bash
cd dados
python rodar_ao_vivo.py --estado C --velocidade normal --n-leads 300
```

Para uma demonstração em banca, recomenda-se rodar com `--n-leads 200` a 300 e
velocidade "normal" — a simulação leva 1 a 2 minutos e é possível acompanhar
visualmente o funil se formando, lead por lead, com filas reais aparecendo entre
estações de capacidade diferente.

A aba **"Ver histórico de uma execução"** mostra o resultado consolidado de uma
simulação já concluída, e a aba **"Comparar A vs B vs C"** mostra os três estados
lado a lado — use essa depois de já ter rodado os três pelo menos uma vez.

## Vendo os modelos de IA em ação (Estados B e C)

Logo abaixo da cena visual, quando o estado simulado é B ou C, aparece um painel
**"Modelos de IA em ação"** com três colunas: as últimas decisões do RandomForest
(score de qualificação calculado para cada lead recente, com o resultado de
qualificado/descartado), as últimas decisões do kNN (a similaridade de perfil
calculada para o match de closer), e as últimas propostas geradas de fato pelo
LLM local, com o texto completo gerado — não apenas um resumo cortado. Isso deixa
explícito que os modelos estão sendo consultados de verdade a cada lead, com
valores concretos, não apenas um número genérico no funil.

## Análise dos modelos de IA — evidência para a banca

A aba **"Análise dos modelos de IA"** mostra a evidência estatística por trás das
decisões do RandomForest e do kNN usados no Estado C — não apenas que os modelos
rodam, mas que efetivamente capturam sinal real nos dados.

Para o RandomForest: um histograma sobreposto comparando a distribuição de scores
previstos entre quem fechou negócio e quem não fechou (a separação visual entre as
duas curvas é a evidência de que o modelo discrimina), o gráfico de importância de
cada variável, e duas métricas de validação — AUC-ROC e a correlação do score
previsto com o `fit_real` (a variável oculta que nem o SDR humano nem o próprio
modelo observam diretamente).

Para o kNN: um boxplot comparando a distância média aos 5 vizinhos mais próximos
entre quem fechou e quem não fechou negócio (leads que fecharam estão
consistentemente mais próximos dos deals ganhos anteriores), e um scatter plot
mostrando onde os deals ganhos se concentram no espaço de orçamento × urgência.

A aba também inclui um bloco de **justificativa metodológica** pronto para
apresentação, explicando por que RandomForest foi escolhido para a qualificação
(captura interações não-lineares entre variáveis sem precisar especificá-las) e
por que kNN foi escolhido para o match de closer (é uma tarefa de recuperação de
casos similares, não de previsão de probabilidade, e tem a vantagem de ser
interpretável por design — "este lead foi direcionado a este perfil porque é
parecido com estes negócios que já fechamos"). O texto também reconhece
explicitamente uma limitação metodológica real: a distância do kNN não normaliza
as variáveis antes de calcular, o que faz o orçamento dominar sobre a urgência —
e mostra que essa alternativa foi testada, não ignorada.

## Nota sobre performance do LLM local

A etapa de geração de proposta via LLM (Llama local, Estados B e C) é a parte
mais lenta da simulação ao vivo — cada chamada ao modelo leva alguns segundos,
dependendo do hardware. Duas otimizações foram aplicadas para reduzir esse
impacto: a resposta do modelo é limitada a um tamanho menor (`num_predict=180`),
e o cache de propostas (`cache_llm.json`) usa uma chave estável por lead, o que
significa que, ao re-rodar a mesma base de leads, propostas já geradas
anteriormente são reaproveitadas em vez de chamar o LLM de novo. Ainda assim,
para uma demonstração ao vivo mais ágil, recomenda-se reduzir `--n-leads` para
50-100 ao simular os Estados B ou C — isso é suficiente para ver o processo, o
funil e os modelos funcionando sem esperar muitos minutos.

Se a simulação dos Estados B ou C parar sozinha pouco depois de iniciar (sem você
ter clicado em Parar), a causa mais provável é o Ollama não estar rodando ou não
ter o modelo baixado. A etapa de geração de proposta (`proposta_ia`) depende de uma
chamada HTTP para `localhost:11434`; se essa chamada falhar, o dashboard agora exibe
o erro completo na tela (com sugestão de correção), em vez de simplesmente mostrar
"simulação parada" sem explicação. Confirme que `ollama serve` está ativo em outro
terminal e que `ollama pull llama3.1:8b` já foi executado antes de rodar B ou C.

## Notas metodológicas importantes

**Replicabilidade do LLM.** Modelos locais via Ollama não garantem determinismo
bit-a-bit mesmo com `temperature=0` e `seed` fixa (depende de otimizações internas
de hardware). Por isso, as respostas geradas são armazenadas em `cache_llm.json` na
primeira execução — reproduções subsequentes leem o cache, garantindo que as métricas
finais sejam idênticas a cada nova rodada.

**Desbalanceamento de classes.** A taxa de fechamento de negócio é baixa (~7-8%),
típica de funis B2B reais. Por isso o RandomForest é treinado com
`class_weight="balanced"` e avaliado por AUC-ROC, não acurácia simples — acurácia
seria artificialmente alta apenas por prever sempre a classe majoritária.

**Calibração do threshold de qualificação.** O corte de qualificação do Estado C é
calibrado pelo percentil 50 da distribuição de probabilidades prevista pelo modelo,
de forma a manter volume de leads qualificados comparável ao Estado A (~48-50%).
Isso isola o efeito de **qualidade** da decisão de qualificação, em vez de medir
apenas o efeito de **cortar volume**.

**Prevenção de vazamento de dados (data leakage).** Os modelos de IA usados nos
Estados B e C são treinados exclusivamente sobre o histórico do Estado A — um
processo que não usa IA. Isso garante que a inferência feita nos Estados B e C seja
genuína (sobre dados que o modelo nunca viu), não circular.

**Correlação entre features visíveis e "fit real".** O campo oculto `fit_real`
(verdade não observável diretamente por humanos ou IA) é parcialmente derivado das
features visíveis (orçamento, urgência, tamanho da empresa) com ruído idiossincrático.
Isso é necessário para que o problema de qualificação seja genuinamente solucionável:
sem essa correlação, nem um humano perfeito nem uma IA perfeita poderiam prever
fechamento de negócio.

## Resultados obtidos nesta execução de referência

| Métrica | Estado A | Estado B | Estado C |
|---|---|---|---|
| Leads processados | 2.000 | 2.000 | 2.000 |
| Leads qualificados | 960 (48%) | 941 (47%) | 1.000 (50%) |
| Negócios fechados | 121 (6,0%) | 121 (6,05%) | 133 (6,65%) |
| Tempo total do processo | 5.256 h | 3.146 h | 2.274 h |
| Tempo só na triagem/qualificação | ~946 h | ~942 h | ~7 h |

Note que a **taxa de conversão** permanece estatisticamente equivalente entre os três
estados (6,0% → 6,05% → 6,65%, dentro da faixa esperada de ruído amostral) — como
deveria ser, já que nenhum modelo de triagem altera o fit real entre produto e cliente.
O ganho mensurável e atribuível à IA está em **produtividade**: o tempo total do
processo cai 57% do Estado A ao C, e o tempo investido especificamente na etapa de
qualificação cai de ~946h para ~7h (redução de 99%) ao substituir o SDR humano por
inferência de um modelo treinado.

Estes números foram gerados com `seed=42` e devem ser reproduzidos de forma idêntica
por qualquer pessoa seguindo os passos acima, com a mesma versão do `leads_base.csv`
e do cache do LLM fornecidos.
