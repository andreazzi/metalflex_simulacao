# Metalflex — experimento de simulação de funil comercial B2B

Código, dados e resultados do experimento que sustenta a monografia:

> ANDREAZZI, A. **Redesenho sistêmico de processos como condição para a adoção de
> inteligência artificial: um experimento simulado em um funil comercial B2B.**
> Monografia (MBA em Inteligência Artificial e Big Data) — Instituto de Ciências
> Matemáticas e de Computação, Universidade de São Paulo, São Carlos, 2026.

## A pergunta

Intervenções de Inteligência Artificial de mesma magnitude produzem resultados
diferentes conforme o ponto do processo em que incidem?

O experimento simula o mesmo funil comercial B2B sob quatro configurações, com
**100 réplicas pareadas** cada, e compara duas variáveis dependentes: o tempo total
de esforço humano e a taxa de conversão.

## As quatro configurações

| Estado | Intervenções de IA | Onde incide |
|---|---|---|
| **A** | nenhuma | — (referência manual) |
| **B** | 1 — proposta por modelo de linguagem | etapa de maior custo em tempo |
| **D** | 1 — qualificação por RandomForest | restrição do sistema |
| **C** | 3 — qualificação, alocação por kNN e proposta | ambas |

O **Estado D** é a configuração de controle: aplica o mesmo número de intervenções
que o Estado B, mudando apenas o **lugar**. É o par B–D que torna a hipótese
testável — sem ele, a superioridade do Estado C seria esperada por construção.

## Resultado

Com 100 réplicas pareadas por estado, base de 2.000 leads, taxa de qualificação de 25%:

| Estado | Tempo | vs. A | Conversão |
|---|---|---|---|
| A | 3.139,8 h ± 96,8 | — | 3,374% ± 0,395 |
| B | 2.091,2 h ± 47,9 | −33,4% | 3,374% ± 0,395 |
| D | 2.208,8 h ± 58,1 | −29,7% | **3,812% ± 0,350** |
| C | 1.164,3 h ± 28,0 | −62,9% | **3,812% ± 0,350** |

**Diferença de conversão D − B: +0,438 p.p.** (t = 11,13; p < 0,001; d = 1,11;
IC 95% [0,360; 0,517]).

Com custo operacional praticamente equivalente — 33,4% contra 29,7% de redução de
tempo —, a intervenção na etapa de maior custo não produziu ganho comercial algum, e
a intervenção na restrição produziu a totalidade do ganho disponível.

O achado é robusto: mantém-se positivo e significativo em cinco cenários de
parametrização (σ ∈ {12, 18, 24} e taxa de qualificação ∈ {20%, 25%, 30%}), com
tamanhos de efeito entre 0,73 e 1,51.

## Reprodução

```bash
./replicar.sh 100
```

O script executa a sequência completa e, ao final, verifica automaticamente as duas
identidades exatas do desenho — interrompendo com erro se alguma falhar.

A ordem das etapas é obrigatória por encadeamento de dependências:

```
gerar_base.py                        seed=42  -> base de avaliação (2.000 leads)
gerar_base_treino.py                 seed=142 -> base de treino (10.000, disjunta)
rodar_estado_a_para_treino_modelos.py         -> histórico rotulado
treinar_modelos.py                            -> RandomForest + kNN + limiar
experimento_30x.py --n-repeticoes 100         -> as 400 execuções
```

Rodadas anteriores são arquivadas em `saidas/_arquivo_<timestamp>/`, não apagadas.

### Requisitos

```bash
pip install -r requirements.txt
```

O modelo de linguagem (Llama 3.2, 3B) roda localmente via [Ollama](https://ollama.com).
Como as propostas geradas ficam em cache indexado por `lead_id`, uma reexecução sobre
a mesma base **não aciona o Ollama** — ele só é necessário para gerar propostas de
leads ainda não cacheados.

## Reprodutibilidade

Toda fonte de aleatoriedade é controlada. O inventário completo está em
`dados/sementes.py`, que também roda um conjunto de verificações:

```bash
python3 dados/sementes.py
```

O ponto central é o **pareamento por sementes**: antes de cada etapa estocástica, o
gerador é reinicializado com uma semente derivada da tripla `(repetição, lead, etapa)`
por hash MD5. Como a semente não depende do estado que executa a etapa, a réplica *i*
do Estado A consome exatamente os mesmos sorteios que a réplica *i* dos Estados B, C e
D em todas as etapas compartilhadas. A única fonte de divergência é o tratamento
aplicado — condição que legitima o teste t pareado.

MD5 em vez de `hash()` porque, desde o Python 3.3, `PYTHONHASHSEED` é aleatorizado por
padrão e `hash()` de uma string varia entre execuções do interpretador.

Duas **identidades exatas** decorrem do desenho e servem como verificação de
integridade: A ≡ B e C ≡ D em número de negócios fechados, em todas as 100 réplicas.

## Parâmetros calibrados

| Parâmetro | Valor | Origem |
|---|---|---|
| Taxa de qualificação | 25% | elicitação com gerente comercial B2B (faixa 20–30%), compatível com benchmark publicado de ~30% |
| Limiar do SDR humano | 51,798 | derivado da taxa-alvo e de σ |
| σ do ruído de julgamento | 18 | ponto central da faixa que a taxa elicitada admite (12 a 24) |
| Limiar do RandomForest | 0,4982 | percentil 75, para igualar o volume qualificado ao da triagem humana |
| Qualidade do repasse | 0,50 constante | ver nota abaixo |

**Sobre o repasse.** Em versões preliminares, a qualidade da informação transmitida ao
vendedor era derivada do escore do qualificador — `score/100` no caso humano, a
probabilidade prevista no caso do modelo. Como o escore humano incorpora o ruído de
julgamento, a formulação fazia o *erro* do avaliador *aumentar* a qualidade do repasse.
Adotou-se o valor constante, que isola o efeito da seleção de leads. A monografia
reporta os quatro tratamentos avaliados e o efeito de cada um.

## Estrutura

```
dados/          código da simulação, geradores de base, treino, orquestração
dashboard/      painel interativo (Streamlit)
modelos/        RandomForest, kNN, scaler e limiar (.pkl)
saidas/         CSVs das rodadas, cache do LLM, figuras
replicar.sh     execução ponta a ponta com verificação de integridade
```

## Limitações

Os dados são sintéticos e o processo é único, o que restringe a generalização. O
experimento estabelece que a **localização** da intervenção determina o resultado, mas
não estabelece que a análise sistêmica **encontra** a localização correta: no
simulador, a restrição foi definida pela parametrização adotada, não descoberta. Esse
elo permanece sustentado pela fundamentação teórica, não por evidência empírica deste
estudo.

## Licença

MIT — ver [LICENSE](LICENSE).
