# activetextclassification/active_learning.py

import pandas as pd
import numpy as np
import time
import os
import json # Para log/debug talvez
import traceback
from tqdm.notebook import tqdm # Ou tqdm padrão
from datetime import datetime

# Imports da própria biblioteca
# Presume-se que config.py exista para validate_experiment_config
from .config import validate_experiment_config
from .data_preparation import load_and_prepare_data
from .embeddings import get_embedder, BaseEmbedder
from .baseline import calculate_baseline_metrics
from .cold_start import select_initial_batch
from .models import get_model, BaseTextClassifier, BaseFeatureClassifier
from .selection import select_query_batch
from .oracle import BaseOracle, get_oracle # <- ADICIONAR


# Imports de ML/Avaliação
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


class ActiveLearner:
    """
    Orquestra o processo de Aprendizado Ativo para classificação de texto.

    Gerencia o estado (L, U, P), modelos, embeddings e o ciclo de iteração.

    Atributos Principais:
        config (dict): A configuração completa do experimento.
        status (str): Estado atual do learner ('INITIALIZED', 'READY', 'RUNNING', 'COMPLETED', 'FAILED', etc.).
        P_df (pd.DataFrame): DataFrame da População.
        U_df (pd.DataFrame): DataFrame do Pool Não Rotulado atual.
        L_df (pd.DataFrame): DataFrame do Conjunto Rotulado atual.
        history (list): Lista de dicionários com métricas e tempos de cada iteração.
        baseline_metrics (dict): Métricas calculadas para o baseline.
        active_model: A última instância treinada do modelo classificador.
        global_embedder (BaseEmbedder): Instância do embedder global (se usado).
    """

    def __init__(self, config):
        """
        Inicializa o ActiveLearner com uma configuração de experimento.

        Args:
            config (dict): Dicionário contendo toda a configuração do experimento,
                           seguindo a estrutura de 'experiments.json'.
        """
        print("--- Inicializando ActiveLearner ---")
        if not isinstance(config, dict):
            raise TypeError("Configuração deve ser um dicionário.")

        self.config = config
        # Tenta obter nome da config, senão usa default
        self.experiment_name = config.get("experiment_name", f"Exp_{int(time.time())}")
        self.status = "INITIALIZED"

        # Extrair parâmetros principais com validação básica ou defaults
        self.data_params = config.get("data_params")
        self.embedder_global_config = config.get("embedder_global_config") # Pode ser None
        self.baseline_config = config.get("baseline_classifier_config") # Pode ser None
        self.al_params = config.get("al_params")
        self.general_params = config.get("general_params", {})
        self.oracle_config = self.al_params.get("oracle_config", {'type': 'Simulated'}) # Default para Simulado

        # Validar parâmetros essenciais
        if not self.data_params: raise ValueError("Seção 'data_params' ausente na configuração.")
        if not self.al_params: raise ValueError("Seção 'al_params' ausente na configuração.")
        self.text_column = self.data_params.get('text_column')
        self.label_column = self.data_params.get('label_column')
        if not self.text_column or not self.label_column:
             raise ValueError("'text_column' e 'label_column' devem ser definidos em 'data_params'.")

        self.random_seed = self.general_params.get('random_seed', int(time.time()))
        np.random.seed(self.random_seed)

        # --- NOVOS ATRIBUTOS PARA LOGGING DETALHADO ---
        self.verbose = self.general_params.get('verbose', False) # Default é não ser verboso
        self.selection_log_file = self.general_params.get('selection_log_file', 'detailed_selection_log.csv') # Default filename
        self._last_query_config = None # Para armazenar a config da query na iteração
        self._last_uncertainty_scores = None # Para armazenar scores (opcional)
        # --- FIM NOVOS ATRIBUTOS ---

        # Inicializar variáveis de estado
        self.P_df, self.U_df, self.L_df = None, None, None
        self.label_to_id, self.id_to_label, self.all_possible_labels = None, None, None
        self.num_classes, self.original_U_size = None, None
        self.global_embedder, self.active_model = None, None
        self.history = []
        self.baseline_metrics = {}
        self.total_duration_sec = None
        self.error_message = None
        self._is_setup = False
        self._current_iteration = 0
        self._max_iterations = self.al_params.get("max_iterations", 1000) # Default alto
        self._target_budget_pct = self.al_params.get("target_budget_pct", 1.0) # Default 100%
        # Atributos para Early Stopping
        self._best_metric_value = -np.inf
        self._patience_counter = 0

        # Validação da Config (Opcional, mas útil)
        if not validate_experiment_config(config):
            print("AVISO: Validação básica da configuração falhou (verifique 'config.py').")

        print(f"ActiveLearner para '{self.experiment_name}' inicializado. Seed: {self.random_seed}")


    def setup(self, data_source=None):
        """
        Executa a preparação inicial: carrega dados, prepara, calcula baseline,
        prepara embedder global (se necessário) e realiza cold start.

        Args:
            data_source (str, optional): Caminho para o arquivo de dados (CSV/Excel).
                                        Se não fornecido, usa self.data_params['file_path'].
        """
        if self._is_setup:
            print("AVISO: Setup já foi executado. Pulando.")
            return
        if self.status != "INITIALIZED":
             print(f"AVISO: Estado inválido para setup ('{self.status}').")
             return # Ou levantar erro?

        print("\n--- Executando Setup ---")
        setup_start_time = time.time()
        self.status = "SETTING_UP"

        try:
            # --- 1. Preparação de Dados ---
            print("\n--- 1.1 Preparação de Dados ---")
            file_path = data_source or self.data_params.get('file_path')
            if not file_path: raise ValueError("Caminho do arquivo de dados não fornecido.")

            P_df_temp, U_df_temp, self.label_to_id, self.id_to_label, self.all_possible_labels = load_and_prepare_data(
                file_path=file_path, # Passar explicitamente
                text_column=self.text_column,
                label_column=self.label_column,
                min_samples_per_class=self.data_params.get('min_samples_per_class', 1),
                rare_group_label=self.data_params.get('rare_group_label', '_RARE_GROUP_'),
                population_size=self.data_params.get('population_size', 0.5),
                random_seed=self.random_seed,
                sheet_name=self.data_params.get('sheet_name', 0)
            )
            self.P_df = P_df_temp.copy()
            self.U_df = U_df_temp.copy() # U inicial antes do cold start
            self.num_classes = len(self.all_possible_labels)
            self.original_U_size = len(self.U_df)
            print(f"Dados preparados: {len(self.P_df)} P, {len(self.U_df)} U inicial, {self.num_classes} classes.")
            if self.original_U_size == 0:
                 print("AVISO: Pool U inicial está vazio após preparação.")

            # --- 2. Preparação Embedder Global ---
            cold_start_config = self.al_params.get("cold_start_config", {'type':'RND'})
            baseline_model_type = self.baseline_config.get('type') if self.baseline_config else None
            active_model_type = self.al_params.get("classifier_config",{}).get('type')
            needs_global_embedder = (cold_start_config.get('type') == 'KM' or
                                     baseline_model_type in ['GNB', 'LSVC', 'LR', 'SGD'] or
                                     active_model_type in ['GNB', 'LSVC', 'LR', 'SGD'])

            if needs_global_embedder:
                if not self.embedder_global_config: raise ValueError("Config 'embedder_global_config' necessária.")
                print("\n--- 1.2 Preparando Embedder Global ---")
                self.global_embedder = get_embedder(self.embedder_global_config)
                print("Ajustando embedder global (P+U)...")
                fit_texts_emb = pd.concat([self.P_df[self.text_column], self.U_df[self.text_column]], ignore_index=True).tolist()
                fit_labels_emb = pd.concat([self.P_df[self.label_column], self.U_df[self.label_column]], ignore_index=True).tolist()
                self.global_embedder.fit(fit_texts_emb, fit_labels_emb)
                print("Embedder global ajustado.")
                del fit_texts_emb, fit_labels_emb
            else:
                print("\n--- 1.2 Embedder Global não necessário ---")

            # --- 3. Cálculo Baseline ---
            print("\n--- 1.3 Cálculo do Baseline ---")
            if not self.baseline_config:
                print("AVISO: 'baseline_classifier_config' não definida. Pulando baseline.")
                self.baseline_metrics = {'status': 'skipped', 'reason': 'Config not provided'}
            else:
                df_full_bl = pd.concat([self.P_df, self.U_df], ignore_index=True)
                baseline_req_embed = baseline_model_type in ['GNB', 'LSVC', 'LR', 'SGD']
                embedder_for_bl = self.global_embedder if baseline_req_embed else None

                if baseline_req_embed and embedder_for_bl is None:
                    print(f"AVISO: Baseline '{baseline_model_type}' requer embedder, mas não disponível.")
                    self.baseline_metrics = {'error': 'Embedder global não disponível/ajustado', 'status': 'skipped'}
                else:
                    self.baseline_metrics = calculate_baseline_metrics(
                        df_full=df_full_bl,
                        classifier_config=self.baseline_config,
                        text_column=self.text_column,
                        label_column=self.label_column,
                        test_size=self.baseline_config.get('test_size', 0.05),
                        random_seed=self.random_seed,
                        all_possible_labels=self.all_possible_labels,
                        embedder=embedder_for_bl
                    )
                del df_full_bl
            print(f"Métricas Baseline: {self.baseline_metrics}")

            try:
                # Passar label_column para SimulatedOracle ou all_labels/text_col para OpenAI
                oracle_params = self.oracle_config.get('params', {})
                if self.oracle_config['type'] == 'Simulated' and 'label_column' not in oracle_params:
                    oracle_params['label_column'] = self.label_column # Usar label_column global
                if self.oracle_config['type'] == 'OpenAI' and 'text_column' not in oracle_params:
                    oracle_params['text_column'] = self.text_column # Usar text_column global

                self.oracle = get_oracle(
                    config=self.oracle_config,
                    all_possible_labels=self.all_possible_labels # Necessário para OpenAI
                )
                print("Oráculo instanciado com sucesso.")
            except Exception as e:
                print(f"ERRO ao instanciar Oráculo: {e}")
                # Propagar erro ou marcar setup como falho? Propagar é mais seguro.
                self.status = "SETUP_FAILED"
                self.error_message = f"Oracle Instantiation Error: {e}"
                raise e
        # --- FIM Instanciar Oráculo ---

            # --- 4. Cold Start ---
            print("\n--- 1.4 Cold Start ---")
            if self.U_df.empty:
                 print("AVISO: Pool U está vazio. Não é possível fazer cold start. L0 estará vazio.")
                 self.L_df = pd.DataFrame(columns=self.P_df.columns) # Usa colunas de P como ref
            else:
                U_embed_km = None
                if cold_start_config.get('type') == 'KM':
                    if self.global_embedder is None: raise ValueError("Embedder global necessário para KM.")
                    print("Gerando embeddings U para KM...")
                    U_embed_km = self.global_embedder.transform(self.U_df[self.text_column].tolist())

                initial_indices = select_initial_batch(
                    U_df=self.U_df,
                    cold_start_config=cold_start_config,
                    n_classes=self.num_classes,
                    random_seed=self.random_seed,
                    U_embeddings=U_embed_km
                )
                if len(initial_indices) > 0 and len(initial_indices) <= len(self.U_df):
                    self.L_df = self.U_df.iloc[initial_indices].copy()
                    self.U_df.drop(self.U_df.index[initial_indices], inplace=True)
                    self.U_df.reset_index(drop=True, inplace=True)
                    if U_embed_km is not None: del U_embed_km
                else:
                    print("AVISO: Nenhum índice válido retornado pelo cold start. L0 estará vazio.")
                    self.L_df = pd.DataFrame(columns=self.P_df.columns)
            print(f"Tamanho L0: {len(self.L_df)}, U restante: {len(self.U_df)}")

            # --- Finalizar Setup ---
            self._is_setup = True
            self.status = "READY" # Pronto para rodar o loop
            setup_duration = time.time() - setup_start_time
            print(f"--- Setup Concluído ({setup_duration:.2f} seg) ---")

        except Exception as e:
            print(f"\nERRO DURANTE O SETUP do experimento {self.experiment_name}: {type(e).__name__} - {e}")
            traceback.print_exc()
            self.status = "SETUP_FAILED"
            self.error_message = f"Setup Error: {type(e).__name__}: {e}"
            # Propagar o erro para o notebook saber que falhou
            raise e


    def run(self):
        """Executa o loop completo de Aprendizado Ativo."""
        if not self._is_setup or self.status != "READY":
            print(f"ERRO: Setup não concluído ou estado inválido ('{self.status}'). Chame learner.setup() primeiro.")
            # Se já falhou antes, não tenta rodar de novo
            if self.status != "READY": return self.get_history_dataframe()
            self.status = "ERROR"
            return None

        print(f"\n--- Iniciando Run Completo (Máx: {self._max_iterations} iterações, Budget: {self._target_budget_pct*100:.0f}%) ---")
        self.status = "RUNNING"
        run_start_time = time.time()
        self.history = [] # Limpar histórico de execuções anteriores da *mesma instância*
        self._current_iteration = 0 # Resetar contador de iteração
        self._best_metric_value = -np.inf # Resetar early stopping
        self._patience_counter = 0

        try:
            # Usar tqdm para barra de progresso no loop
            with tqdm(total=self._max_iterations, desc="AL Iterations") as pbar:
                while self._current_iteration < self._max_iterations:
                    # Chamar step interno, que retorna True se deve continuar
                    should_continue = self._execute_one_iteration()
                    pbar.update(1) # Atualiza barra de progresso
                    if not should_continue:
                         # Ajustar total da barra se parou antes
                         pbar.total = self._current_iteration + 1
                         pbar.refresh()
                         break # Sai do loop while

        except Exception as e:
            print(f"\nERRO DURANTE O RUN do experimento {self.experiment_name}: {type(e).__name__} - {e}")
            traceback.print_exc()
            self.status = "FAILED"
            self.error_message = f"Run Error: {type(e).__name__}: {e}"
            # Histórico parcial será retornado

        # Marcar como COMPLETED apenas se não falhou e saiu normalmente do loop (ou por critério de parada)
        if self.status == "RUNNING":
            self.status = "COMPLETED"

        run_end_time = time.time()
        self.total_duration_sec = round(run_end_time - run_start_time, 2)
        print(f"--- Run Concluído (Status: {self.status}, Duração Loop: {self.total_duration_sec:.2f} seg) ---")
        return self.get_history_dataframe()


    def step(self):
        """Executa UMA iteração do loop de Aprendizado Ativo."""
        if not self._is_setup or self.status not in ["READY", "STEPPING"]:
            print(f"ERRO/AVISO: Setup não concluído ou estado inválido ('{self.status}').")
            return None

        self.status = "STEPPING" # Indica que está em modo step
        should_continue = self._execute_one_iteration()
        if not should_continue and self.status == "STEPPING": # Se parou normalmente no step
             self.status = "COMPLETED"

        return self.history[-1] if self.history else None


    def _execute_one_iteration(self):
        """ Lógica interna para executar uma única iteração. Retorna False se deve parar. """
        if not self._is_setup: return False

        iter_start_time = time.time()
        current_L_size = len(self.L_df)
        labeled_percentage = current_L_size / self.original_U_size if self.original_U_size > 0 else 1.0

        print(f"\n--- Iteração {self._current_iteration + 1}/{self._max_iterations} ---")
        print(f"L: {current_L_size} ({labeled_percentage*100:.1f}% U original), U: {len(self.U_df)}")

        # --- Critério de Parada ---
        if labeled_percentage >= self._target_budget_pct:
            print("Critério de parada (Budget) atingido.")
            return False
        if len(self.U_df) == 0:
             print("Critério de parada (U vazio) atingido.")
             return False
        # Max iterations é verificado no loop 'run' ou externamente no 'step'

        # --- Tentar executar os passos da iteração ---
        fit_duration, eval_duration, query_duration, update_duration = np.nan, np.nan, np.nan, np.nan
        internal_acc, internal_f1, external_acc, external_f1 = np.nan, np.nan, np.nan, np.nan
        query_indices = np.array([], dtype=int)

        # Limpar info da iteração anterior
        self._last_query_config = None
        self._last_uncertainty_scores = None

        try:
            # 1. Treinar Modelo
            train_L_df, test_L_df = self._split_L_internal()
            # Verificar se train_L_df não está vazio (pode acontecer com L0 pequeno)
            if train_L_df.empty:
                 print("AVISO: Conjunto de treino interno vazio. Pulando iteração.")
                 # Não incrementa _current_iteration, mas retorna True para tentar na próxima
                 # Ou talvez devesse parar? Vamos pular a iteração.
                 self.history.append({ # Registrar iteração pulada
                      'iteration': self._current_iteration + 1, 'L_size': current_L_size,
                      'status': 'SKIPPED_EMPTY_TRAIN', 'iteration_duration_sec': time.time() - iter_start_time
                 })
                 self._current_iteration += 1
                 return True # Tentar próxima iteração

            X_train_input, X_test_input, X_P_input, X_U_input = self._prepare_model_inputs(train_L_df, test_L_df)
            fit_duration = self._train_step(X_train_input, train_L_df[self.label_column].tolist())

            # 2. Avaliar Modelo
            eval_results = self._evaluation_step(X_test_input, test_L_df, X_P_input)
            internal_acc, internal_f1 = eval_results['internal_acc'], eval_results['internal_f1']
            external_acc, external_f1 = eval_results['external_acc'], eval_results['external_f1']
            eval_duration = eval_results['duration']

            # 3. Selecionar Lote (Query)
            query_result = self._query_step(X_U_input) # X_U_input pode ser None se U for vazio
            query_indices = query_result['indices']
            query_duration = query_result['duration']

            # 4. Oráculo e Atualizar L/U
            update_duration = self._update_step(query_indices)

        except Exception as e:
            print(f"\nERRO DURANTE ITERAÇÃO {self._current_iteration + 1}: {type(e).__name__} - {e}")
            traceback.print_exc()
            self.status = "FAILED"
            self.error_message = f"Iteration {self._current_iteration + 1} Error: {type(e).__name__}: {e}"
            # Registrar histórico parcial da iteração com erro
            iter_duration = time.time() - iter_start_time
            self.history.append({
                'iteration': self._current_iteration + 1, 'L_size': current_L_size,
                'internal_acc': internal_acc, 'internal_f1': internal_f1, # Podem ser NaN
                'external_acc': external_acc, 'external_f1': external_f1, # Podem ser NaN
                'status': 'FAILED_ITERATION', 'error': str(e),
                'iteration_duration_sec': iter_duration,
                'train_duration_sec': fit_duration, 'eval_duration_sec': eval_duration,
                'query_duration_sec': query_duration, 'update_duration_sec': update_duration,
                'U_size': len(self.U_df)
            })
            return False # Indica que deve parar devido a erro

        # --- Registrar Histórico da Iteração Bem Sucedida---
        iter_duration = time.time() - iter_start_time
        self.history.append({
            'iteration': self._current_iteration + 1, 'L_size': current_L_size,
            'internal_acc': internal_acc, 'internal_f1': internal_f1,
            'external_acc': external_acc, 'external_f1': external_f1,
            'iteration_duration_sec': iter_duration,
            'train_duration_sec': fit_duration, 'eval_duration_sec': eval_duration,
            'query_duration_sec': query_duration, 'update_duration_sec': update_duration,
            'U_size': len(self.U_df),
            'status': 'COMPLETED_ITERATION' # Adicionar status da iteração
        })
        print(f"Duração Iteração Total: {iter_duration:.2f} seg")

        # --- Checar Parada Antecipada ---
        should_stop_early = self._check_early_stopping(external_acc, external_f1)
        if should_stop_early:
            return False

        # Incrementar iteração
        self._current_iteration += 1
        return True # Indica para continuar

    # --- Métodos Auxiliares Internos ---
    # ( _split_L_internal, _prepare_model_inputs, _train_step,
    #   _evaluation_step, _query_step, _update_step, _check_early_stopping
    #   permanecem praticamente os mesmos da resposta anterior, com pequenas
    #   melhorias de robustez adicionadas abaixo )

    def _split_L_internal(self):
        train_L_df = self.L_df; test_L_df = None
        internal_test_size = self.al_params.get("internal_test_size", 0.0)
        min_samples_for_split = 10
        if len(self.L_df) >= min_samples_for_split and internal_test_size > 0:
            try:
                train_L_df, test_L_df = train_test_split(
                    self.L_df, test_size=internal_test_size,
                    random_state=self.random_seed + self._current_iteration,
                    stratify=self.L_df[self.label_column]
                )
            except ValueError:
                print("Aviso: Split interno não estratificado.")
                train_L_df, test_L_df = train_test_split(
                    self.L_df, test_size=internal_test_size,
                    random_state=self.random_seed + self._current_iteration
                )
        else: print("Split interno pulado.")
        return train_L_df, test_L_df

    def _prepare_model_inputs(self, train_df, test_df=None):
        X_train_input, X_test_input, X_P_input, X_U_input = None, None, None, None
        active_model_type = self.al_params.get("classifier_config",{}).get('type')
        is_feature_based = active_model_type in ['GNB', 'LSVC', 'LR', 'SGD']

        # Helper para transformar texto em features com segurança
        def safe_transform(embedder, df, col):
             if embedder and df is not None and not df.empty:
                  return embedder.transform(df[col].tolist())
             return None # Ou np.empty com shape (0, dim)? None é mais fácil de checar

        if is_feature_based:
            if self.global_embedder is None: raise ValueError("Embedder global necessário.")
            print("Gerando features para Treino/Teste/P/U...")
            X_train_input = safe_transform(self.global_embedder, train_df, self.text_column)
            X_test_input = safe_transform(self.global_embedder, test_df, self.text_column)
            X_P_input = safe_transform(self.global_embedder, self.P_df, self.text_column)
            X_U_input = safe_transform(self.global_embedder, self.U_df, self.text_column)
        else: # Baseado em texto
            print("Preparando texto bruto para Treino/Teste/P/U...")
            X_train_input = train_df[self.text_column].tolist() if train_df is not None else []
            X_test_input = test_df[self.text_column].tolist() if test_df is not None and not test_df.empty else []
            X_P_input = self.P_df[self.text_column].tolist() if self.P_df is not None else []
            X_U_input = self.U_df[self.text_column].tolist() if self.U_df is not None else []

        # Validar se inputs necessários foram criados
        if X_train_input is None or (isinstance(X_train_input, list) and not X_train_input):
             raise ValueError("Falha ao preparar dados de treino (X_train_input está vazio ou None).")

        return X_train_input, X_test_input, X_P_input, X_U_input

    def _train_step(self, X_train_input, y_train_labels):
        start_time = time.time()
        classifier_config = self.al_params.get("classifier_config")
        self.active_model = get_model(classifier_config)
        print(f"Treinando {type(self.active_model).__name__} em {len(y_train_labels)} amostras...")
        # Fit pode falhar se X_train_input for vazio ou formato errado
        if (isinstance(X_train_input, np.ndarray) and X_train_input.size == 0) or \
           (isinstance(X_train_input, list) and not X_train_input):
             raise ValueError("Input de treino (X_train_input) está vazio.")
        self.active_model.fit(X_train_input, y_train_labels)
        duration = time.time() - start_time
        print(f"Tempo Treino: {duration:.2f} seg")
        return duration

    def _evaluation_step(self, X_test_input, test_df, X_P_input):
        start_time = time.time()
        results = {'internal_acc': np.nan, 'internal_f1': np.nan, 'external_acc': np.nan, 'external_f1': np.nan, 'duration': np.nan}
        if self.active_model is None: return results # Se treino falhou

        try:
            # Interna
            if X_test_input is not None and test_df is not None and len(X_test_input) > 0:
                 y_pred_internal_str = self.active_model.predict(X_test_input)
                 y_true_internal_str = test_df[self.label_column].tolist()
                 results['internal_acc'] = accuracy_score(y_true_internal_str, y_pred_internal_str)
                 results['internal_f1'] = f1_score(y_true_internal_str, y_pred_internal_str, average='macro', zero_division=0, labels=self.all_possible_labels)

            # Externa
            if X_P_input is not None and self.P_df is not None and len(X_P_input) > 0:
                 y_pred_external_str = self.active_model.predict(X_P_input)
                 y_true_external_str = self.P_df[self.label_column].tolist()
                 results['external_acc'] = accuracy_score(y_true_external_str, y_pred_external_str)
                 results['external_f1'] = f1_score(y_true_external_str, y_pred_external_str, average='macro', zero_division=0, labels=self.all_possible_labels)
        except Exception as e:
             print(f"Erro durante avaliação: {e}")
             # Métricas não calculadas permanecerão NaN

        results['duration'] = time.time() - start_time
        print(f"Interna: Acc={results['internal_acc']:.4f} F1={results['internal_f1']:.4f} | Externa: Acc={results['external_acc']:.4f} F1={results['external_f1']:.4f} (Eval Time: {results['duration']:.2f}s)")
        return results

    def _query_step(self, X_U_input):
        """ Seleciona o próximo lote de U. Retorna dict com índices e duração. """
        start_time = time.time()
        query_indices = np.array([], dtype=int)
        # Usar cópia da config para evitar modificar a original no fallback
        query_strat_config = self.al_params.get("query_strategy_config", {}).copy()
        self._last_query_config = query_strat_config # Armazenar config usada
        self._last_uncertainty_scores = None # Resetar scores

        if len(self.U_df) > 0 and X_U_input is not None and \
           ((isinstance(X_U_input,np.ndarray) and X_U_input.size > 0) or (isinstance(X_U_input, list) and X_U_input)):

            pool_size_u = len(self.U_df)
            probabilities_u = None
            strategy_type = query_strat_config.get('type', 'RND')
            needs_proba = strategy_type in ['ENT', 'LCO', 'SMA', 'HYB']

            if needs_proba:
                try:
                    print(f"Calculando probabilidades para U ({pool_size_u} amostras)...")
                    probabilities_u = self.active_model.predict_proba(X_U_input)
                    if probabilities_u is None or probabilities_u.shape[0] != pool_size_u:
                         raise ValueError(f"Shape de predict_proba ({probabilities_u.shape if probabilities_u is not None else 'None'}) inválido.")
                     # --- Opcional: Armazenar scores de incerteza para log ---
                    if self.verbose:
                         if strategy_type == 'ENT':
                              probs_clipped = np.clip(probabilities_u, 1e-9, 1 - 1e-9)
                              self._last_uncertainty_scores = -np.sum(probs_clipped * np.log2(probs_clipped), axis=1)
                         # Adicionar cálculo para LC, MAR se implementar
                except Exception as e:
                    print(f"Erro predict_proba U: {e}. Fallback para Random.")
                    # Atualizar a config *usada* nesta iteração para RND
                    query_strat_config = {'type': 'RND', 'params': query_strat_config.get('params', {})}
                    self._last_query_config = query_strat_config # Armazenar config de fallback
                    probabilities_u = None

            try:
                query_indices = select_query_batch(
                    query_strategy_config=query_strat_config, # Passa config (pode ser a de fallback)
                    pool_size=pool_size_u,
                    probabilities=probabilities_u,
                    random_seed=self.random_seed + self._current_iteration
                )
            except Exception as e:
                 print(f"Erro durante a seleção do lote: {e}.")
                 query_indices = np.array([], dtype=int)
        else:
             print("Pool U vazio ou input X_U indisponível, nenhuma seleção a fazer.")

        duration = time.time() - start_time
        print(f"Seleção ({query_strat_config.get('type', 'N/A')}, {len(query_indices)} amostras): {duration:.2f} seg")
        # Retorna os índices e a duração
        return {'indices': query_indices, 'duration': duration}

    def _update_step(self, query_indices):
        """ Simula oráculo, atualiza L e U, e LOGA SE VERBOSE. Retorna duração. """
        start_time = time.time()
        if self.oracle is None:
             print("ERRO: Oráculo não foi instanciado no setup. Pulando update.")
             return 0.0 # Duração zero

        if len(query_indices) > 0 and len(query_indices) <= len(self.U_df):
            try:
                # Obter as amostras selecionadas
                queried_df = self.U_df.iloc[query_indices].copy()

                # --- CHAMAR O ORÁCULO ---
                print(f"Consultando oráculo ({self.oracle_config.get('type')}) para {len(queried_df)} amostras...")
                # Passar o DataFrame para o oráculo (ele sabe qual coluna usar)
                labels_from_oracle = self.oracle.query(queried_df)
                # Verificar se o retorno é uma lista do tamanho esperado
                if not isinstance(labels_from_oracle, list) or len(labels_from_oracle) != len(queried_df):
                     print(f"AVISO: Oráculo retornou dados inesperados (tipo {type(labels_from_oracle)}, tamanho {len(labels_from_oracle)}). Pulando atualização.")
                     return time.time() - start_time # Retorna duração mesmo com erro

                # Filtrar Nones (se o oráculo falhou em algumas)
                valid_labels_mask = [label is not None for label in labels_from_oracle]
                queried_df_valid = queried_df[valid_labels_mask].copy()
                labels_valid = [label for label, valid in zip(labels_from_oracle, valid_labels_mask) if valid]
                query_indices_valid = query_indices[valid_labels_mask] # Índices originais válidos

                if not labels_valid:
                     print("AVISO: Oráculo não retornou nenhum rótulo válido. Nenhuma atualização.")
                     return time.time() - start_time

                print(f"Oráculo retornou {len(labels_valid)} rótulos válidos.")
                # Adicionar os rótulos obtidos ao DataFrame consultado válido
                # Usar a coluna de label definida globalmente
                queried_df_valid[self.label_column] = labels_valid
                # --- FIM CHAMADA ORÁCULO ---


                # --- LOGGING DETALHADO (usando labels obtidos) ---
                if self.verbose:
                    print(f"Verbose: Logando {len(queried_df_valid)} amostras rotuladas...")
                    log_data = []
                    strategy_type = self._last_query_config.get('type', 'N/A') if self._last_query_config else 'N/A'
                    uncertainty = self._last_uncertainty_scores[query_indices_valid] if self._last_uncertainty_scores is not None and strategy_type == 'ENT' and len(query_indices_valid)>0 else np.nan

                    for i, idx in enumerate(query_indices_valid): # Iterar sobre índices válidos
                        row = queried_df_valid.iloc[i] # Usar df com labels do oráculo
                        log_entry = {
                            "experiment_name": self.experiment_name,
                            "iteration": self._current_iteration + 1,
                            "query_strategy": strategy_type,
                            "selected_index_in_U": idx,
                            "text": row[self.text_column],
                            "oracle_label": row[self.label_column] # <-- Label do oráculo
                        }
                        # Adicionar score se existir
                        if isinstance(uncertainty, np.ndarray) and i < len(uncertainty) and not np.isnan(uncertainty[i]):
                            log_entry["uncertainty_score"] = uncertainty[i]
                        log_data.append(log_entry)

                    if log_data:
                        log_df = pd.DataFrame(log_data)
                        log_file_exists = os.path.exists(self.selection_log_file)
                        try:
                            log_df.to_csv(self.selection_log_file, mode='a', header=not log_file_exists, index=False, encoding='utf-8')
                        except Exception as log_e: print(f"ERRO ao salvar log de seleção: {log_e}")
                # --- FIM LOGGING ---

                # Atualizar L e U (APENAS com dados validados pelo oráculo)
                self.L_df = pd.concat([self.L_df, queried_df_valid], ignore_index=True)
                # Remover os índices que *foram consultados* de U (mesmo os que falharam no oráculo)
                self.U_df.drop(self.U_df.index[query_indices], inplace=True)
                self.U_df.reset_index(drop=True, inplace=True)

            except IndexError as e: print(f"Erro de índice ao atualizar L/U: {e}")
            except Exception as e: print(f"Erro inesperado no update step: {e}"); traceback.print_exc()

        else:
            print("Nenhuma amostra válida selecionada para atualizar L/U.")

        duration = time.time() - start_time
        return duration

    def _check_early_stopping(self, external_acc, external_f1):
        metric = self.al_params.get("early_stopping_metric")
        patience = self.al_params.get("early_stopping_patience")
        tolerance = self.al_params.get("early_stopping_tolerance", 0.001)
        if not metric or not patience: return False

        if metric == 'external_acc': current_value = external_acc
        elif metric == 'external_f1': current_value = external_f1
        else: current_value = None

        if current_value is not None and not np.isnan(current_value):
            if current_value > self._best_metric_value + tolerance:
                self._best_metric_value = current_value
                self._patience_counter = 0
                print(f"Early Stopping: Melhora em '{metric}' ({self._best_metric_value:.4f}).")
            else:
                self._patience_counter += 1
                print(f"Early Stopping: Sem melhora sig. Paciência: {self._patience_counter}/{patience}")
            if self._patience_counter >= patience:
                print(f"Critério de parada (Early Stopping) atingido.")
                return True
        else: print(f"Aviso Early Stopping: Métrica '{metric}' não disponível.")
        return False


    # --- Métodos Públicos para Acesso ---
    def get_history_dataframe(self):
        return pd.DataFrame(self.history)

    def get_current_model(self):
        return self.active_model

    def get_L_set(self):
        return self.L_df.copy() if self.L_df is not None else pd.DataFrame()

    def get_U_set(self):
        return self.U_df.copy() if self.U_df is not None else pd.DataFrame()

    def get_P_set(self):
        return self.P_df.copy() if self.P_df is not None else pd.DataFrame()

    def get_results_summary(self):
        # Atualizar timestamp para o momento da chamada
        summary = {
            "experiment_name": self.experiment_name,
            "execution_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": self.status,
            "config": self.config,
            "baseline_metrics": self.baseline_metrics,
            "history_data": self.history,
            "total_duration_sec": self.total_duration_sec, # Duração apenas do run()
            "error_message": self.error_message
        }
        return summary