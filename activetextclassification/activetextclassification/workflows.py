# activetextclassification/workflows.py

import pandas as pd
import numpy as np
import time
from tqdm.notebook import tqdm # Ou tqdm padrão

# Imports de outros módulos da biblioteca
from .models import get_model, BaseTextClassifier, BaseFeatureClassifier
from .embeddings import BaseEmbedder # Para type hints e checagem
from .selection import select_query_batch
# from .oracle import get_labels_from_oracle # Vamos comentar por enquanto se não estiver usando

# Imports de ML/Avaliação
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

def run_active_learning(
    P_df,                     # População (DataFrame)
    U_df_initial,             # Pool U inicial (DataFrame, após cold start)
    L_df_initial,             # Lote L0 inicial (DataFrame)
    classifier_config,        # Config do modelo a ser treinado no loop
    query_strategy_config,    # Config da estratégia de query
    # query_batch_size,       # <<< REMOVIDO DESTA LINHA
    target_budget_pct,        # % do U original
    max_iterations,
    internal_test_size,       # % de L para teste interno
    text_column,
    label_column,
    all_possible_labels,      # Lista global de labels para F1
    random_seed=42,
    # --- Argumentos Opcionais ---
    embedder=None,            # Embedder AJUSTADO (necessário se classificador for baseado em features)
    early_stopping_metric=None,
    early_stopping_patience=None,
    early_stopping_tolerance=0.001
):
    """
    Orquestra e executa o loop principal de Aprendizado Ativo.
    (O resto da Docstring permanece igual)
    """
    np.random.seed(random_seed)
    print("--- Iniciando Workflow de Aprendizado Ativo ---")

    # --- Validações Iniciais ---
    model_type = classifier_config.get('type')
    is_feature_based_model = model_type in ['GNB', 'LSVC', 'LR', 'SGD']
    if is_feature_based_model and embedder is None:
        raise ValueError(f"Modelo '{model_type}' requer um 'embedder', mas nenhum foi fornecido.")

    # --- Inicialização ---
    L_df = L_df_initial.copy()
    U_df = U_df_initial.copy()
    original_U_size = len(U_df_initial)
    history_list = []
    active_model = None
    best_metric_value = -np.inf
    patience_counter = 0

    # --- Loop Principal ---
    for iteration in range(max_iterations):
        iter_start_time = time.time()
        current_L_size = len(L_df)
        labeled_percentage = current_L_size / original_U_size if original_U_size > 0 else 1.0

        print(f"\n--- Iteração {iteration + 1}/{max_iterations} ---")
        print(f"L: {current_L_size} ({labeled_percentage*100:.1f}% U original), U: {len(U_df)}")

        if labeled_percentage >= target_budget_pct or len(U_df) == 0:
            print("Critério de parada (Budget ou U vazio) atingido.")
            break

        try:
            active_model = get_model(classifier_config)
        except Exception as e:
            print(f"Erro ao instanciar modelo na iteração {iteration+1}: {e}")
            break

        # Split L interno
        train_L_df = L_df; test_L_df = None
        internal_acc, internal_f1 = np.nan, np.nan
        if current_L_size >= 10 and internal_test_size > 0:
             try: train_L_df, test_L_df = train_test_split(L_df, test_size=internal_test_size, random_state=random_seed + iteration, stratify=L_df[label_column])
             except ValueError: print("Aviso: Split interno não estratificado."); train_L_df, test_L_df = train_test_split(L_df, test_size=internal_test_size, random_state=random_seed + iteration)
        else: print("Split interno pulado.")

# --- Preparar Dados de Treino/Teste para o Modelo ---
        y_train_labels = train_L_df[label_column].tolist()
        X_train_input = None
        X_test_input = None
        X_P_input = None # Input para avaliação externa (P)
        X_U_input = None # Input para seleção de query (U)

        try:
            # --- CORREÇÃO DA VERIFICAÇÃO ---
            if isinstance(active_model, BaseFeatureClassifier):
                if embedder is None: raise ValueError("Embedder necessário para BaseFeatureClassifier.")
                print("Gerando features para Treino/Teste/P/U...")
                X_train_input = embedder.transform(train_L_df[text_column].tolist())
                # Verificar se test_L_df existe antes de transformar
                X_test_input = embedder.transform(test_L_df[text_column].tolist()) if test_L_df is not None and not test_L_df.empty else None
                X_P_input = embedder.transform(P_df[text_column].tolist())
                 # Verificar se U_df existe antes de transformar
                X_U_input = embedder.transform(U_df[text_column].tolist()) if not U_df.empty else None
            elif isinstance(active_model, BaseTextClassifier): # <--- Usar elif aqui
                print("Preparando texto bruto para Treino/Teste/P/U...")
                X_train_input = train_L_df[text_column].tolist()
                # Verificar se test_L_df existe
                X_test_input = test_L_df[text_column].tolist() if test_L_df is not None and not test_L_df.empty else None
                X_P_input = P_df[text_column].tolist()
                 # Verificar se U_df existe
                X_U_input = U_df[text_column].tolist() if not U_df.empty else None
            else:
                 # Este else agora não deve ser atingido se get_model funciona
                 raise TypeError(f"Tipo de modelo não suportado durante preparação de dados: {type(active_model)}")
            # --- FIM DA CORREÇÃO ---
        except Exception as e:
             print(f"Erro ao preparar dados de entrada na iteração {iteration+1}: {e}")
             # Adicionar traceback para mais detalhes
            #  traceback.print_exc()
             break # Abortar

        # Treinar Modelo Ativo
        fit_start_time = time.time(); fit_duration = np.nan
        try:
            print(f"Treinando {type(active_model).__name__}..."); active_model.fit(X_train_input, y_train_labels)
            fit_duration = time.time() - fit_start_time; print(f"Tempo de Treinamento: {fit_duration:.2f} seg")
        except Exception as e: print(f"Erro durante o treinamento: {e}"); break

        # Avaliar Modelo
        eval_start_time = time.time(); eval_duration = np.nan
        external_acc, external_f1 = np.nan, np.nan
        try:
            if X_test_input is not None and test_L_df is not None: # Avaliação Interna
                y_pred_internal_str = active_model.predict(X_test_input)
                y_true_internal_str = test_L_df[label_column].tolist()
                internal_acc = accuracy_score(y_true_internal_str, y_pred_internal_str)
                internal_f1 = f1_score(y_true_internal_str, y_pred_internal_str, average='macro', zero_division=0, labels=all_possible_labels)
            # Avaliação Externa
            y_pred_external_str = active_model.predict(X_P_input)
            y_true_external_str = P_df[label_column].tolist()
            external_acc = accuracy_score(y_true_external_str, y_pred_external_str)
            external_f1 = f1_score(y_true_external_str, y_pred_external_str, average='macro', zero_division=0, labels=all_possible_labels)
            eval_duration = time.time() - eval_start_time
            print(f"Interna: Acc={internal_acc:.4f} F1={internal_f1:.4f} | Externa: Acc={external_acc:.4f} F1={external_f1:.4f} (Eval Time: {eval_duration:.2f}s)")
        except Exception as e: print(f"Erro durante avaliação: {e}"); eval_duration = time.time() - eval_start_time


        # Seleção de Lote
        query_start_time = time.time(); query_duration = np.nan
        query_indices = np.array([], dtype=int)
        if len(U_df) > 0:
            pool_size_u = len(U_df)
            probabilities_u = None
            # --- CORREÇÃO: Obter batch_size de query_strategy_config ---
            n_instances_to_query = query_strategy_config.get('params',{}).get('batch_size')
            if n_instances_to_query is None:
                 print("AVISO: 'batch_size' não encontrado em query_strategy_config['params']. Usando default 10.")
                 n_instances_to_query = 10 # Ou um valor default configurável
            # --- FIM CORREÇÃO ---
            strategy_type = query_strategy_config.get('type', 'RND')
            needs_proba = strategy_type in ['ENT', 'LC', 'MAR']

            if needs_proba:
                 try:
                      print(f"Calculando probabilidades para U ({pool_size_u} amostras)...")
                      probabilities_u = active_model.predict_proba(X_U_input)
                 except Exception as e:
                      print(f"Erro predict_proba U: {e}. Fallback para Random.")
                      query_strategy_config = {'type': 'RND', 'params': query_strategy_config.get('params', {})} # Mantém params
                      probabilities_u = None

            try:
                query_indices = select_query_batch(
                    query_strategy_config=query_strategy_config,
                    # --- CORREÇÃO: n_instances vem da config ---
                    # n_instances=n_instances_to_query, # << select_query_batch já pega de dentro da config
                    # --- FIM CORREÇÃO ---
                    pool_size=pool_size_u,
                    probabilities=probabilities_u,
                    random_seed=random_seed + iteration
                )
            except Exception as e: print(f"Erro durante a seleção: {e}")
        query_duration = time.time() - query_start_time
        print(f"Seleção ({query_strategy_config.get('type')}, {len(query_indices)} amostras): {query_duration:.2f} seg")


        # Oráculo e Atualização L/U
        update_start_time = time.time(); update_duration = np.nan
        if len(query_indices) > 0:
             queried_df = U_df.iloc[query_indices].copy()
             # labels = get_labels_from_oracle(queried_df, label_column) # Usaria oracle.py se não fosse simulação
             L_df = pd.concat([L_df, queried_df], ignore_index=True)
             U_df = U_df.drop(U_df.index[query_indices]).reset_index(drop=True)
        else: print("Nenhuma amostra selecionada.")
        update_duration = time.time() - update_start_time

        # Registrar Histórico
        iter_duration = time.time() - iter_start_time
        history_list.append({
            'iteration': iteration + 1, 'L_size': current_L_size,
            'internal_acc': internal_acc, 'internal_f1': internal_f1,
            'external_acc': external_acc, 'external_f1': external_f1,
            'iteration_duration_sec': iter_duration,
            'train_duration_sec': fit_duration, 'eval_duration_sec': eval_duration,
            'query_duration_sec': query_duration, 'update_duration_sec': update_duration
        })
        print(f"Duração Iteração Total: {iter_duration:.2f} seg")

        # Checar Parada Antecipada
        if early_stopping_metric and early_stopping_patience:
             # ... (lógica de early stopping igual) ...
             pass


    print("\n--- Workflow de Aprendizado Ativo Concluído ---")
    history_df = pd.DataFrame(history_list)
    return history_df, active_model