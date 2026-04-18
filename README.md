# activetextclassification

Biblioteca Python para **classificação de texto com Aprendizado Ativo** (*Active Learning*).

O objetivo é reduzir o custo de rotulação manual de dados, selecionando de forma inteligente quais amostras devem ser anotadas a cada iteração — maximizando o desempenho do classificador com o menor número possível de exemplos rotulados.

---

## Índice

- [Instalação](#instalação)
- [Visão Geral](#visão-geral)
- [Uso Rápido](#uso-rápido)
- [Fluxo de Trabalho](#fluxo-de-trabalho)
- [Módulos](#módulos)
  - [ActiveLearner](#activelearner)
  - [Preparação de Dados](#preparação-de-dados)
  - [Embeddings](#embeddings)
  - [Modelos](#modelos)
  - [Estratégias de Seleção](#estratégias-de-seleção)
  - [Cold Start](#cold-start)
  - [Oráculos](#oráculos)
  - [Otimização](#otimização)
  - [Gerenciamento de Experimentos](#gerenciamento-de-experimentos)
  - [Visualização e Análise](#visualização-e-análise)
- [Configuração via JSON](#configuração-via-json)
- [Dependências](#dependências)
- [Licença](#licença)

---

## Instalação

```bash
# Instalar a partir do código fonte
pip install -e .
```

**Dependências opcionais** (para oráculos LLM):

```bash
pip install openai google-generativeai ollama anthropic python-dotenv
```

---

## Visão Geral

O aprendizado ativo é um paradigma de aprendizado de máquina onde o modelo pode consultar um "oráculo" (humano ou LLM) para obter rótulos das amostras mais informativas. Isso é especialmente útil quando a rotulação é cara ou demorada.

Esta biblioteca implementa o ciclo completo de aprendizado ativo para classificação de texto:

```
Dataset → Cold Start (L0) → Loop AL → Análise
              ↑                   |
              |    [treinar → avaliar → selecionar → rotular]
              +-------------------------------------------+
```

---

## Uso Rápido

```python
from activetextclassification import ActiveLearner

config = {
    "experiment_name": "meu_experimento",
    "active": True,
    "data_params": {
        "file_path": "dados.csv",
        "text_column": "texto",
        "label_column": "categoria",
        "population_size": 0.5,
        "min_samples_per_class": 2
    },
    "al_params": {
        "cold_start_config": {"type": "RND", "params": {"n_initial": 50}},
        "classifier_config": {"type": "PVBin", "params": {}},
        "query_strategy_config": {"type": "ENT", "params": {"batch_size": 10}},
        "target_budget_pct": 0.30,
        "max_iterations": 50,
        "internal_test_size": 0.2
    },
    "general_params": {"random_seed": 42}
}

learner = ActiveLearner(config)
learner.setup()
learner.run()

# Acessar histórico de métricas
import pandas as pd
history_df = pd.DataFrame(learner.history)
print(history_df[["iteration", "L_size", "external_acc", "external_f1"]])
```

---

## Fluxo de Trabalho

### 1. Preparação dos Dados

```python
from activetextclassification.data_preparation import load_and_prepare_data

P_df, U_df, label_to_id, id_to_label, all_labels = load_and_prepare_data(
    file_path="dados.csv",
    text_column="texto",
    label_column="categoria",
    population_size=0.50,     # 50% para população de avaliação
    min_samples_per_class=2,  # agrupa classes raras
    random_seed=42
)
```

### 2. Cold Start — Seleção do Lote Inicial (L0)

```python
from activetextclassification.cold_start import select_initial_batch

# Seleção aleatória
selected_indices = select_initial_batch(
    cold_start_config={"type": "RND", "params": {"n_initial": 50}},
    U_df=U_df
)

# Seleção por K-Medians (baseada em embeddings)
selected_indices = select_initial_batch(
    cold_start_config={"type": "KMD", "params": {"n_clusters": 10}},
    U_df=U_df,
    embeddings=U_embeddings
)
```

### 3. Embeddings

```python
from activetextclassification.embeddings import get_embedder

embedder = get_embedder({"type": "PV", "params": {"method": "tfidf"}})
embedder.fit(texts=train_texts, labels=train_labels)
X_features = embedder.transform(new_texts)
```

### 4. Classificadores

```python
from activetextclassification.models import get_model

# Baseado em texto (ProductVectorizer binário)
model = get_model({"type": "PVBin", "params": {}})

# Baseado em features (requer embedder)
model = get_model({"type": "LR", "params": {"C": 1.0, "max_iter": 1000}})
model = get_model({"type": "GNB", "params": {}})
model = get_model({"type": "LSVC", "params": {"C": 0.5}})
model = get_model({"type": "SGD", "params": {}})

model.fit(X_train, y_train_labels)
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

### 5. Estratégias de Seleção (Query Strategies)

```python
from activetextclassification.selection import select_query_batch

# Seleção de lote por Máxima Entropia
indices = select_query_batch(
    query_strategy_config={"type": "ENT", "params": {"batch_size": 10}},
    probabilities=model.predict_proba(U_features)
)
```

| Código | Estratégia             | Descrição                                              |
|--------|------------------------|--------------------------------------------------------|
| `RND`  | Random Sampling        | Seleção aleatória                                      |
| `ENT`  | Max Entropy            | Seleciona amostras com maior entropia preditiva        |
| `LC`   | Least Confidence       | Seleciona amostras com menor confiança na predição     |
| `SM`   | Smallest Margin        | Seleciona amostras com menor margem entre 2 top classes|
| `HYB`  | Hybrid Entropy+Random  | Mistura entropia e seleção aleatória (`entropy_fraction`)|

### 6. Oráculos

```python
from activetextclassification.oracle import SimulatedOracle, get_oracle

# Oráculo simulado (usa labels verdadeiros existentes — para experimentos)
oracle = SimulatedOracle(label_column="categoria")
labels = oracle.query(df_to_label)

# Oráculo via OpenAI (rotulação real)
oracle = get_oracle({
    "type": "OpenAI",
    "params": {
        "model": "gpt-4o-mini",
        "labels": ["cat1", "cat2", "cat3"],
        "api_key_env": "OPENAI_API_KEY"
    }
})
```

---

## Módulos

### `activetextclassification.ActiveLearner`

Classe principal que orquestra todo o ciclo de aprendizado ativo.

| Método        | Descrição                                                    |
|---------------|--------------------------------------------------------------|
| `__init__(config)` | Inicializa o learner com dicionário de configuração     |
| `setup()`     | Carrega dados, embedder, baseline e L0 inicial               |
| `run()`       | Executa o loop de aprendizado ativo                          |
| `history`     | Lista de dicionários com métricas de cada iteração           |
| `L_df`        | DataFrame do conjunto rotulado atual                         |
| `U_df`        | DataFrame do pool não rotulado atual                         |
| `P_df`        | DataFrame da população                                       |

### Preparação de Dados

**`load_and_prepare_data`**: carrega CSV/Excel, pré-processa labels, agrupa classes raras e divide em P/U.

**`load_split_and_preprocess_data`**: versão estendida com suporte a divisão treino/teste persistida em disco.

### Embeddings

| Tipo      | Descrição                                              |
|-----------|--------------------------------------------------------|
| `PV`      | `ProductVectorizerEmbedder` — TF-IDF com softmax       |
| `TFIDF`   | `TFIDFEmbedder` — TF-IDF padrão do sklearn             |

### Modelos

| Tipo     | Classe                          | Tipo de entrada  |
|----------|---------------------------------|------------------|
| `PVBin`  | `ProductVectorizerClassifier`   | Texto            |
| `GNB`    | Gaussian Naive Bayes (sklearn)  | Features/Embeddings |
| `LSVC`   | Linear SVC (sklearn)            | Features/Embeddings |
| `LR`     | Logistic Regression (sklearn)   | Features/Embeddings |
| `SGD`    | SGD Classifier (sklearn)        | Features/Embeddings |

### Estratégias de Seleção

Módulo `activetextclassification.selection`. Funções disponíveis:
- `random_sampling`
- `max_entropy_sampling`
- `least_confidence_sampling`
- `smallest_margin_sampling`
- `hybrid_entropy_random_sampling`
- `select_query_batch` ← função fábrica principal

### Cold Start

Módulo `activetextclassification.cold_start`. Métodos:
- `RND` — seleção aleatória
- `KMD` — K-Medians nos embeddings (via `sklearn-extra`)

### Oráculos

Módulo `activetextclassification.oracle`. Disponíveis:
- `SimulatedOracle` — usa labels existentes (experimentos)
- `OpenAIOracle` — via API OpenAI

Submódulo `activetextclassification.oracle` (avançado):
- `GeminiOracle` — Google Gemini via `google-generativeai`
- `OllamaOracle` — modelos locais via Ollama
- `AnthropicOracle` — via API Anthropic

### Otimização

**`GeneticL0Optimizer`** (`activetextclassification.optimization`): otimiza a seleção do L0 usando algoritmos genéticos, buscando o conjunto inicial que maximiza (ou minimiza) o desempenho do classificador.

```python
from activetextclassification.optimization import GeneticL0Optimizer

optimizer = GeneticL0Optimizer(
    df_full=df_train_pool,
    text_column="texto",
    label_column="categoria",
    classifier_config={"type": "PVBin", "params": {}},
    initial_l0_size=50,
    all_possible_labels=all_labels,
    population_size=50,
    n_generations=100,
    df_evaluation_set=df_eval
)
best_l0_df, best_fitness = optimizer.run()
```

### Gerenciamento de Experimentos

**`ExperimentManager`** (`activetextclassification.management`): executa múltiplos experimentos sequencialmente a partir de um arquivo `experiments.json`, evitando reexecuções com base no histórico salvo.

```python
from activetextclassification.management import ExperimentManager

manager = ExperimentManager(
    config_file_path="experiments.json",
    history_log_path="history_log.jsonl"
)
manager.load_and_prepare()
manager.run_pending()
```

### Visualização e Análise

```python
from activetextclassification.utils import load_and_flatten_experiment_history, calculate_lce

# Carregar histórico de experimentos para análise
history_df = load_and_flatten_experiment_history("history_log.jsonl")

# Calcular Learning Curve Efficiency
lce = calculate_lce(
    l_sizes=history_df["L_size"],
    performance_scores=history_df["external_f1"],
    baseline_performance=0.85
)
```

---

## Configuração via JSON

Cada experimento é definido por um dicionário (ou entrada no `experiments.json`):

```json
{
  "experiment_name": "exp_ent_lr_kmd",
  "active": true,
  "data_params": {
    "file_path": "data/produtos.csv",
    "text_column": "descricao",
    "label_column": "categoria",
    "population_size": 0.5,
    "min_samples_per_class": 2
  },
  "al_params": {
    "cold_start_config": {
      "type": "KMD",
      "params": { "n_clusters": 20 }
    },
    "classifier_config": {
      "type": "LR",
      "params": { "C": 1.0, "max_iter": 500 }
    },
    "query_strategy_config": {
      "type": "ENT",
      "params": { "batch_size": 20 }
    },
    "target_budget_pct": 0.30,
    "max_iterations": 100,
    "internal_test_size": 0.2,
    "embedder_config": { "type": "PV", "params": { "method": "tfidf" } }
  },
  "general_params": {
    "random_seed": 42,
    "results_dir": "results/"
  }
}
```

---

## Dependências

| Pacote              | Uso                                         |
|---------------------|---------------------------------------------|
| `numpy`             | Operações numéricas                         |
| `pandas`            | Manipulação de DataFrames                   |
| `scikit-learn`      | Classificadores, métricas, TF-IDF           |
| `scikit-learn-extra`| K-Medoids para cold start                   |
| `scipy`             | Softmax, integrais numéricas                |
| `unidecode`         | Normalização de texto (labels)              |
| `matplotlib`        | Visualização de curvas de aprendizado       |
| `seaborn`           | Visualizações estatísticas                  |
| `tqdm`              | Barras de progresso                         |

**Opcionais** (oráculos LLM):

| Pacote                | Oráculo             |
|-----------------------|---------------------|
| `openai`              | OpenAI (GPT)        |
| `google-generativeai` | Google Gemini       |
| `ollama`              | Modelos locais      |
| `anthropic`           | Anthropic (Claude)  |
| `python-dotenv`       | Variáveis de ambiente |

---

## Licença

MIT License — veja o arquivo [LICENSE](LICENSE) para detalhes.
