# activetextclassification/cold_start.py

import numpy as np
import pandas as pd
from sklearn_extra.cluster import KMedoids # Se usar K-Medians

def random_cold_start(U_df, n_initial, random_seed=42):
    """
    Seleciona as amostras iniciais L0 aleatoriamente de U.

    Args:
        U_df (pd.DataFrame): DataFrame do pool não rotulado.
        n_initial (int): Número de amostras a serem selecionadas.
        random_seed (int): Semente para reprodutibilidade.

    Returns:
        np.ndarray: Array de índices (baseados na posição em U_df) das amostras selecionadas.
                    Retorna array vazio se n_initial <= 0 ou U_df for vazio.
    """
    np.random.seed(random_seed)
    num_available = len(U_df)
    if n_initial <= 0 or num_available == 0:
        return np.array([], dtype=int)

    # Não selecionar mais do que o disponível
    actual_n_initial = min(n_initial, num_available)
    if actual_n_initial < n_initial:
        print(f"Aviso (Random Cold Start): Solicitado {n_initial}, mas apenas {num_available} disponíveis. Selecionando {actual_n_initial}.")

    print(f"Cold Start: Selecionando {actual_n_initial} amostras aleatórias.")
    selected_indices = np.random.choice(num_available, size=actual_n_initial, replace=False)
    return selected_indices

def kmedians_cold_start(U_df, U_embeddings, n_clusters, random_seed=42):
    """
    Seleciona as amostras iniciais L0 usando K-Medians nos embeddings de U.
    Retorna os índices dos medoides.

    Args:
        U_df (pd.DataFrame): DataFrame do pool não rotulado.
        U_embeddings (np.ndarray): Matriz de embeddings para as amostras em U_df.
        n_clusters (int): Número de clusters (geralmente o número de classes).
        random_seed (int): Semente para reprodutibilidade.

    Returns:
        np.ndarray: Array de índices (baseados na posição em U_df) dos medoides selecionados.
                    Retorna array vazio se U_df ou U_embeddings forem vazios ou n_clusters <= 0.
    """
    num_available = len(U_df)
    if n_clusters <= 0 or num_available == 0 or U_embeddings is None or U_embeddings.shape[0] == 0:
        print("Aviso (K-Medians Cold Start): Dados insuficientes ou inválidos. Retornando seleção vazia.")
        return np.array([], dtype=int)
    if U_embeddings.shape[0] != num_available:
        raise ValueError(f"Número de embeddings ({U_embeddings.shape[0]}) não corresponde ao número de amostras em U_df ({num_available}).")

    # K-Medians requer n_samples >= n_clusters
    if num_available < n_clusters:
        print(f"AVISO (K-Medians Cold Start): Menos amostras em U ({num_available}) que clusters solicitados ({n_clusters}). Retornando todas as amostras disponíveis como 'medoides'.")
        # Retorna todos os índices disponíveis
        return np.arange(num_available)

    print(f"Cold Start: Calculando K-Medians ({n_clusters} clusters) em {num_available} amostras...")
    try:
        kmedoids = KMedoids(n_clusters=n_clusters, random_state=random_seed, method='pam')
        kmedoids.fit(U_embeddings) # Clusterizar os embeddings
        selected_indices = kmedoids.medoid_indices_
        print(f"Medoides encontrados nos índices: {selected_indices}")
        return selected_indices
    except Exception as e:
        print(f"Erro durante K-Medians: {e}. Retornando seleção vazia.")
        # Pode acontecer por diversos motivos, incluindo problemas numéricos
        return np.array([], dtype=int)

# --- Função Fábrica Atualizada ---
def select_initial_batch(
    U_df,
    cold_start_config,   # Recebe o dicionário de config
    n_classes,           # Número de classes (usado como n_initial padrão para KM/RND)
    random_seed=42,
    U_embeddings=None
):
    """
    Seleciona o lote inicial L0 usando a estratégia e parâmetros da configuração.

    Args:
        U_df (pd.DataFrame): Pool não rotulado.
        cold_start_config (dict): Dicionário de configuração, ex:
                                  {'type': 'KM', 'params': {'method':'pam'}}
                                  {'type': 'RND', 'params': {'n_samples': 10}} (opcional)
        n_classes (int): Número de classes no dataset (default para n_initial).
        random_seed (int): Semente aleatória.
        U_embeddings (np.ndarray, optional): Embeddings de U, necessários para 'KM'.

    Returns:
        np.ndarray: Array de índices (posição em U_df) das amostras selecionadas.
    """
    method = cold_start_config.get('type', 'RND') # Default para Random
    params = cold_start_config.get('params', {})

    print(f"\n--- Selecionando Lote Inicial L0 (Config: {cold_start_config}) ---")

    # Determinar número de amostras iniciais
    # Usar n_classes como padrão, mas permitir override nos params
    n_initial = params.get('n_samples', n_classes)

    if method == 'RND':
        # Passar params extras para random_cold_start se houver (nenhum por enquanto)
        return random_cold_start(U_df, n_initial, random_seed)
    elif method == 'KM':
        if U_embeddings is None:
            raise ValueError("Embeddings (U_embeddings) são necessários para K-Medians ('KM').")
        # n_initial para K-Medians é o número de clusters
        # Passar outros params do config['params'] para a função kmedians
        return kmedians_cold_start(U_df, U_embeddings, n_clusters=n_initial, random_seed=random_seed, **params)
    else:
        raise ValueError(f"Método de cold start desconhecido: '{method}'. Use 'KM' ou 'RND'.")