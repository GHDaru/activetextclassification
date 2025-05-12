# activetextclassification/selection.py

import numpy as np
import pandas as pd

def random_sampling(pool_size, n_instances, random_seed=None):
    if random_seed is not None: np.random.seed(random_seed)
    if pool_size == 0 or n_instances <= 0 : return np.array([], dtype=int)
    actual_n_instances = min(n_instances, pool_size)
    if actual_n_instances < n_instances: print(f"Aviso (Random): Solicitado {n_instances}, disponíveis {pool_size}. Selecionando {actual_n_instances}.")
    # print(f"Seleção (Random): Escolhendo {actual_n_instances} amostras.")
    return np.random.choice(pool_size, size=actual_n_instances, replace=False)

def max_entropy_sampling(probabilities, n_instances):
    pool_size = probabilities.shape[0]
    if pool_size == 0 or n_instances <= 0: return np.array([], dtype=int)
    actual_n_instances = min(n_instances, pool_size)
    if actual_n_instances < n_instances: print(f"Aviso (Max Entropy): Solicitado {n_instances}, disponíveis {pool_size}. Selecionando {actual_n_instances}.")
    probs_clipped = np.clip(probabilities, 1e-9, 1 - 1e-9)
    entropy = -np.sum(probs_clipped * np.log2(probs_clipped), axis=1)
    selected_indices = np.argsort(entropy)[-actual_n_instances:] # Maior entropia
    # print(f"Seleção (Max Entropy): Escolhendo {actual_n_instances} amostras.")
    return selected_indices

# --- NOVAS ESTRATÉGIAS ---
def least_confidence_sampling(probabilities, n_instances):
    """ Seleciona amostras onde a confiança na classe mais provável é a MENOR. """
    pool_size = probabilities.shape[0]
    if pool_size == 0 or n_instances <= 0: return np.array([], dtype=int)
    actual_n_instances = min(n_instances, pool_size)
    if actual_n_instances < n_instances: print(f"Aviso (Least Confidence): Solicitado {n_instances}, disponíveis {pool_size}. Selecionando {actual_n_instances}.")

    # Confiança é a probabilidade da classe mais provável
    confidence_scores = np.max(probabilities, axis=1)
    # Incerteza = 1 - confiança (ou simplesmente ordenar por confiança e pegar os menores)
    # Queremos os que têm menor confiança, então ordenamos 'confidence_scores' e pegamos os primeiros
    selected_indices = np.argsort(confidence_scores)[:actual_n_instances]
    # print(f"Seleção (Least Confidence): Escolhendo {actual_n_instances} amostras.")
    return selected_indices

def smallest_margin_sampling(probabilities, n_instances):
    """ Seleciona amostras onde a margem entre as duas classes mais prováveis é a MENOR. """
    pool_size = probabilities.shape[0]
    if pool_size == 0 or n_instances <= 0: return np.array([], dtype=int)
    actual_n_instances = min(n_instances, pool_size)
    if actual_n_instances < n_instances: print(f"Aviso (Smallest Margin): Solicitado {n_instances}, disponíveis {pool_size}. Selecionando {actual_n_instances}.")

    if probabilities.shape[1] < 2:
        print("Aviso (Smallest Margin): Menos de 2 classes, não é possível calcular margem. Usando aleatório como fallback.")
        # Fallback para amostragem aleatória se não houver pelo menos 2 classes
        return random_sampling(pool_size, actual_n_instances) # Precisa passar seed se quiser

    # Ordenar probabilidades para cada amostra para pegar as duas maiores
    sorted_probs = np.sort(probabilities, axis=1)
    # Margem = Probabilidade da 1ª mais provável - Probabilidade da 2ª mais provável
    margins = sorted_probs[:, -1] - sorted_probs[:, -2]
    # Selecionar os índices com as menores margens
    selected_indices = np.argsort(margins)[:actual_n_instances]
    # print(f"Seleção (Smallest Margin): Escolhendo {actual_n_instances} amostras.")
    return selected_indices

def hybrid_entropy_random_sampling(
    probabilities,
    pool_size, # Necessário para a parte aleatória
    n_instances,
    entropy_fraction=0.5, # Proporção de amostras por entropia
    random_seed=None
):
    """
    Seleciona uma fração de n_instances usando Max Entropy e o restante aleatoriamente.
    Garante que não haja sobreposição entre as seleções.
    """
    if pool_size == 0 or n_instances <= 0: return np.array([], dtype=int)
    actual_n_instances = min(n_instances, pool_size)
    if actual_n_instances < n_instances: print(f"Aviso (Hybrid): Solicitado {n_instances}, disponíveis {pool_size}. Selecionando {actual_n_instances}.")

    n_entropy = int(np.round(actual_n_instances * entropy_fraction))
    n_random = actual_n_instances - n_entropy

    print(f"Seleção (Hybrid): {n_entropy} por Entropia, {n_random} Aleatórias.")

    selected_by_entropy = np.array([], dtype=int)
    if n_entropy > 0:
        selected_by_entropy = max_entropy_sampling(probabilities, n_entropy)

    # Para seleção aleatória, precisamos dos índices que NÃO foram selecionados por entropia
    remaining_indices = np.setdiff1d(np.arange(pool_size), selected_by_entropy, assume_unique=True)

    selected_by_random = np.array([], dtype=int)
    if n_random > 0 and len(remaining_indices) > 0:
        # Garantir que não tentamos selecionar mais do que o disponível nos restantes
        actual_n_random = min(n_random, len(remaining_indices))
        if actual_n_random < n_random: print(f"Aviso (Hybrid-Random): Solicitado {n_random}, disponíveis {len(remaining_indices)} para aleatório. Selecionando {actual_n_random}.")

        random_choices_from_remaining = np.random.choice(
            remaining_indices, size=actual_n_random, replace=False
        )
        selected_by_random = random_choices_from_remaining
    elif n_random > 0:
        print("Aviso (Hybrid-Random): Não há mais amostras restantes para seleção aleatória.")


    final_selected_indices = np.concatenate((selected_by_entropy, selected_by_random)).astype(int)
    # Em caso raro de n_entropy ser maior que o pool_size, etc.
    if len(final_selected_indices) > actual_n_instances:
         final_selected_indices = final_selected_indices[:actual_n_instances]

    return final_selected_indices


# --- Função Fábrica Atualizada ---
def select_query_batch(
    query_strategy_config,
    pool_size=None,
    probabilities=None,
    random_seed=None
):
    strategy_type = query_strategy_config.get('type', 'RND')
    params = query_strategy_config.get('params', {})
    n_instances = params.get('batch_size')

    if n_instances is None or n_instances <= 0:
        raise ValueError("Parâmetro 'batch_size' não encontrado/inválido na config.")

    # print(f"\n--- Seleção de Lote (Config: {query_strategy_config}) ---") # Mover print para dentro da função específica

    if strategy_type == 'RND':
        if pool_size is None: raise ValueError("'pool_size' necessário para 'RND'.")
        return random_sampling(pool_size, n_instances, random_seed)
    elif strategy_type == 'ENT':
        if probabilities is None: raise ValueError("'probabilities' necessário para 'ENT'.")
        return max_entropy_sampling(probabilities, n_instances)
    elif strategy_type == 'LCO': # Least Confidence
        if probabilities is None: raise ValueError("'probabilities' necessário para 'LCO'.")
        return least_confidence_sampling(probabilities, n_instances)
    elif strategy_type == 'SMA': # Smallest Margin
        if probabilities is None: raise ValueError("'probabilities' necessário para 'SMA'.")
        return smallest_margin_sampling(probabilities, n_instances)
    elif strategy_type == 'HYB': # Hybrid Entropy-Random
        if probabilities is None or pool_size is None:
            raise ValueError("'probabilities' e 'pool_size' necessários para 'HYB'.")
        entropy_fraction = params.get('entropy_fraction', 0.5) # Default para 50% entropia
        if not (0 <= entropy_fraction <= 1):
             raise ValueError("'entropy_fraction' deve estar entre 0 e 1 para 'HYB'.")
        return hybrid_entropy_random_sampling(probabilities, pool_size, n_instances, entropy_fraction, random_seed)
    else:
        raise ValueError(f"Estratégia de consulta desconhecida: '{strategy_type}'.")