# activetextclassification/baseline.py

import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Importar a fábrica de modelos e as bases (se necessário para type hints)
from .models import get_model, BaseTextClassifier, BaseFeatureClassifier

def calculate_baseline_metrics(
    df_full,                   # DataFrame completo (P+U concatenado)
    classifier_config,         # Dicionário com tipo e params, ex: {'type': 'PVBin', 'params': {...}}
    text_column,
    label_column,
    test_size=0.05,
    random_seed=42,
    all_possible_labels=None, # Lista de labels para F1
    # --- Argumentos específicos para modelos baseados em features ---
    embedder=None,             # Instância AJUSTADA de um BaseEmbedder (opcional)
    # features_column=None     # Não precisamos mais passar a coluna, passamos o embedder
):
    """
    Calcula métricas de baseline treinando um modelo em (1-test_size)%
    do dataset completo e avaliando no restante. Reutiliza a fábrica
    get_model e as interfaces de modelo.

    Args:
        df_full (pd.DataFrame): DataFrame completo.
        classifier_config (dict): Configuração para get_model.
        text_column (str): Nome da coluna de texto.
        label_column (str): Nome da coluna de rótulos string.
        test_size (float): Proporção para teste do baseline.
        random_seed (int): Semente aleatória.
        all_possible_labels (list, optional): Lista global de labels para F1.
        embedder (BaseEmbedder, optional): Uma instância JÁ AJUSTADA (fitted)
                                           de um BaseEmbedder. Necessário se
                                           classifier_config['type'] for um modelo
                                           baseado em features (GNB, LSVC, etc.).

    Returns:
        dict: {'baseline_acc': float, 'baseline_f1': float, 'baseline_train_time_sec': float}
    """
    print(f"\n--- Calculando Baseline (Modelo Config: {classifier_config.get('type', 'N/A')}) ---")
    results = {'baseline_acc': np.nan, 'baseline_f1': np.nan, 'baseline_train_time_sec': np.nan}

    if df_full is None or df_full.empty:
        print("AVISO: DataFrame vazio. Pulando baseline.")
        return results

    model_type = classifier_config.get('type')
    is_feature_based = model_type in ['GNB', 'LSVC', 'LR', 'SGD'] # Assumindo estes requerem features

    # Validar se temos o embedder quando necessário
    if is_feature_based and embedder is None:
         print(f"ERRO: Modelo '{model_type}' requer um 'embedder' ajustado, mas nenhum foi fornecido. Pulando baseline.")
         return results
    if is_feature_based and not hasattr(embedder, 'transform'):
         print(f"ERRO: O 'embedder' fornecido não parece ter um método 'transform'. Pulando baseline.")
         return results


    # --- Dividir Dados ---
    print(f"Dividindo dados ({1-test_size:.0%} treino / {test_size:.0%} teste)...")
    stratify_col = df_full[label_column]
    try:
        if stratify_col.nunique() > 1 and stratify_col.value_counts().min() < 2:
             print("AVISO: Split não estratificado para baseline.")
             stratify_col = None
        X_train_base_df, X_test_base_df = train_test_split(
            df_full, test_size=test_size, random_state=random_seed, stratify=stratify_col
        )
    except Exception as e:
        print(f"Erro ao dividir dados para baseline: {e}. Pulando.")
        return results


    # --- Preparar Dados de Entrada para o Modelo ---
    y_train_labels = X_train_base_df[label_column].tolist()
    y_test_labels = X_test_base_df[label_column].tolist()
    X_train_input = None
    X_test_input = None

    try:
        if is_feature_based:
            print("Gerando features para treino/teste do baseline usando o embedder fornecido...")
            # Gerar embeddings para treino e teste
            train_texts = X_train_base_df[text_column].tolist()
            test_texts = X_test_base_df[text_column].tolist()
            X_train_input = embedder.transform(train_texts) # Usa o embedder ajustado
            X_test_input = embedder.transform(test_texts)
            print(f"Shapes das features: Treino={X_train_input.shape}, Teste={X_test_input.shape}")
        else: # Modelo baseado em texto (PVBin)
            X_train_input = X_train_base_df[text_column].tolist()
            X_test_input = X_test_base_df[text_column].tolist()
    except Exception as e:
         print(f"Erro ao preparar dados de entrada para o modelo baseline: {e}")
         return results


    # --- Instanciar Modelo usando a Fábrica ---
    try:
         # Passar o mapeamento global para consistência de predict_proba? Não necessário aqui.
         baseline_model = get_model(classifier_config)
    except Exception as e:
        print(f"Erro ao instanciar modelo baseline via get_model: {e}")
        return results

    # --- Treinar Modelo ---
    print(f"Treinando modelo baseline ({type(baseline_model).__name__})...")
    start_train_time = time.time()
    try:
        # A interface do modelo (fit) decide se usa texto ou features
        baseline_model.fit(X_train_input, y_train_labels)
        results['baseline_train_time_sec'] = time.time() - start_train_time
        print(f"Tempo de treinamento baseline: {results['baseline_train_time_sec']:.2f} seg")
    except Exception as e:
        print(f"Erro durante o treinamento do baseline: {e}")
        return results # Retorna tempo parcial

    # --- Avaliar Modelo ---
    print(f"Avaliando modelo baseline...")
    try:
        # A interface do modelo (predict) decide se usa texto ou features
        y_pred_labels = baseline_model.predict(X_test_input) # Retorna strings

        results['baseline_acc'] = accuracy_score(y_test_labels, y_pred_labels)
        results['baseline_f1'] = f1_score(y_test_labels, y_pred_labels, average='macro', zero_division=0, labels=all_possible_labels)
        print(f"Baseline - Acurácia: {results['baseline_acc']:.4f}")
        print(f"Baseline - Macro F1-Score: {results['baseline_f1']:.4f}")

    except Exception as e:
        print(f"Erro durante a avaliação do baseline: {e}")
        # Mantém tempo de treino, mas acc/f1 ficam NaN

    return results