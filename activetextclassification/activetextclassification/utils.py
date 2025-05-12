# activetextclassification/utils.py
import pandas as pd
import os
import json
import re
import unidecode
from datetime import datetime
import numpy as np

def preprocess_label(label):
    if not isinstance(label, str): label = str(label)
    label = label.lower()
    label = unidecode.unidecode(label)
    label = re.sub(r'\s+', ' ', label).strip()
    return label

def save_history_to_csv(history_df, base_filename="active_learning_history.csv"):
    """
    Anexa um DataFrame de histórico a um arquivo CSV.

    Adiciona o cabeçalho apenas se o arquivo não existir.

    Args:
        history_df (pd.DataFrame): DataFrame contendo os dados do histórico.
        base_filename (str): Nome base do arquivo CSV onde o histórico será salvo.
    """
    if history_df is None or history_df.empty:
        print("Aviso: DataFrame de histórico está vazio. Nada para salvar.")
        return

    try:
        file_exists = os.path.exists(base_filename)
        print(f"Anexando histórico em: {base_filename} (Arquivo existe: {file_exists})")
        # Garantir ordem consistente das colunas pode ser bom
        # history_df = history_df.sort_index(axis=1) # Ordena colunas alfabeticamente
        history_df.to_csv(base_filename, mode='a', index=False, header=not file_exists)
        print("Histórico anexado com sucesso.")
    except Exception as e:
        print(f"Erro ao salvar/anexar histórico no CSV: {e}")

# --- NOVA FUNÇÃO ---
def load_and_flatten_experiment_history(history_log_path="history_log.jsonl"):
    """
    Carrega o log de experimentos de um arquivo JSON Lines,
    filtra por execuções 'COMPLETED', e "achata" os dados de iteração
    em um único DataFrame Pandas para fácil análise e plotagem.

    Args:
        history_log_path (str): Caminho para o arquivo de log JSONL.

    Returns:
        pd.DataFrame: DataFrame contendo todos os registros de iteração de
                      todas as execuções 'COMPLETED', com informações
                      do experimento adicionadas a cada linha de iteração.
                      Retorna DataFrame vazio se o arquivo não existir ou
                      não houver dados válidos.
    """
    all_iteration_records = []

    if not os.path.exists(history_log_path):
        print(f"AVISO (load_history): Arquivo de log '{history_log_path}' não encontrado.")
        return pd.DataFrame(all_iteration_records) # Retorna DF vazio

    print(f"Processando arquivo de log: {history_log_path}")
    processed_executions = 0
    malformed_lines = 0
    skipped_status = 0
    lines_read = 0

    try:
        with open(history_log_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                lines_read += 1
                try:
                    log_entry = json.loads(line)

                    if log_entry.get('status') == 'COMPLETED' and \
                       'history_data' in log_entry and \
                       isinstance(log_entry['history_data'], list) and \
                       log_entry['history_data']: # Garantir que history_data não está vazio

                        processed_executions += 1
                        exp_name = log_entry.get('experiment_name', f'UnknownExp_Line{line_num+1}')
                        timestamp_str = log_entry.get('execution_timestamp', '')
                        try: timestamp_dt = pd.to_datetime(timestamp_str)
                        except: timestamp_dt = pd.NaT # Usar NaT para timestamp inválido

                        baseline_metrics = log_entry.get('baseline_metrics', {})
                        baseline_acc = baseline_metrics.get('baseline_acc')
                        baseline_f1 = baseline_metrics.get('baseline_f1')
                        # Duração total do experimento (setup + run)
                        overall_duration = log_entry.get('overall_experiment_duration_sec')
                        # Duração apenas do loop AL (run)
                        loop_run_duration = log_entry.get('total_duration_sec')

                        config = log_entry.get('config', {})
                        al_params = config.get('al_params', {})
                        cs_config = al_params.get('cold_start_config', {})
                        clf_config = al_params.get('classifier_config', {})
                        qs_config = al_params.get('query_strategy_config', {})
                        data_params_config = config.get('data_params', {})
                        general_params_config = config.get('general_params', {})


                        cs_type = cs_config.get('type', 'N/A')
                        clf_type = clf_config.get('type', 'N/A')
                        clf_params_str = str(clf_config.get('params', {}))
                        qs_type = qs_config.get('type', 'N/A')
                        qs_batch_size = qs_config.get('params', {}).get('batch_size', np.nan)
                        qs_entropy_fraction = qs_config.get('params', {}).get('entropy_fraction', np.nan)


                        for iteration_data in log_entry['history_data']:
                            if isinstance(iteration_data, dict):
                                record = {
                                    'experiment_name': exp_name,
                                    'timestamp_str': timestamp_str,
                                    'timestamp_dt': timestamp_dt,
                                    'overall_experiment_duration_sec': overall_duration,
                                    'loop_run_duration_sec': loop_run_duration,
                                    'baseline_acc': baseline_acc,
                                    'baseline_f1': baseline_f1,
                                    'cold_start_type': cs_type,
                                    'classifier_type': clf_type,
                                    'classifier_params_str': clf_params_str, # Salvar como string
                                    'query_strategy_type': qs_type,
                                    'query_batch_size': qs_batch_size,
                                    'query_entropy_fraction': qs_entropy_fraction,
                                    'min_samples_per_class': data_params_config.get('min_samples_per_class'),
                                    'population_size_pct': data_params_config.get('population_size'),
                                    'random_seed': general_params_config.get('random_seed'),
                                    **iteration_data # Desempacota dados da iteração
                                }
                                all_iteration_records.append(record)
                            else:
                                 print(f"Aviso: Item inválido em 'history_data' para {exp_name} na linha {line_num+1}")
                    elif log_entry.get('status') != 'COMPLETED':
                        skipped_status += 1

                except json.JSONDecodeError:
                    malformed_lines += 1
                    # print(f"Aviso: Pulando linha mal formada ({line_num+1}) no histórico.")
                except Exception as e:
                    print(f"Erro ao processar entrada de log para {log_entry.get('experiment_name','N/A')}: {e}")

    except Exception as e:
         print(f"ERRO CRÍTICO ao ler arquivo de log '{history_log_path}': {e}")
         return pd.DataFrame() # Retorna DF vazio em caso de erro de leitura


    if not all_iteration_records:
        print("Nenhum dado de iteração válido ('COMPLETED' com history_data) encontrado nos logs.")
        return pd.DataFrame()

    print(f"\nCriando DataFrame com {len(all_iteration_records)} registros de {processed_executions} execuções.")
    history_plot_df = pd.DataFrame(all_iteration_records)

    # Converter colunas numéricas
    numeric_cols = [
        'baseline_acc', 'baseline_f1', 'overall_experiment_duration_sec', 'loop_run_duration_sec',
        'iteration', 'L_size', 'internal_acc', 'internal_f1', 'external_acc', 'external_f1',
        'iteration_duration_sec', 'train_duration_sec', 'eval_duration_sec',
        'query_duration_sec', 'update_duration_sec', 'U_size',
        'query_batch_size', 'query_entropy_fraction', 'min_samples_per_class', 'population_size_pct',
        'random_seed'
    ]
    for col in numeric_cols:
        if col in history_plot_df.columns:
             history_plot_df[col] = pd.to_numeric(history_plot_df[col], errors='coerce')

    # Reordenar colunas (opcional, mas bom para consistência)
    preferred_order = [
        'experiment_name', 'timestamp_str', 'iteration', 'L_size', 'U_size',
        'external_acc', 'external_f1', 'internal_acc', 'internal_f1',
        'baseline_acc', 'baseline_f1',
        'query_strategy_type', 'query_batch_size', 'query_entropy_fraction',
        'classifier_type', 'classifier_params_str', 'cold_start_type',
        'iteration_duration_sec', 'train_duration_sec', 'eval_duration_sec',
        'query_duration_sec', 'update_duration_sec',
        'loop_run_duration_sec', 'overall_experiment_duration_sec',
        'min_samples_per_class', 'population_size_pct', 'random_seed',
        'status', # Status da iteração
        'error' # Erro da iteração
        # 'timestamp_dt' # Pode adicionar se quiser usar no Pandas para time-series
    ]
    # Manter apenas colunas que existem e adicionar as restantes
    final_cols = [col for col in preferred_order if col in history_plot_df.columns]
    remaining_cols = [col for col in history_plot_df.columns if col not in final_cols]
    history_plot_df = history_plot_df[final_cols + remaining_cols]

    if malformed_lines > 0: print(f"Aviso: {malformed_lines}/{lines_read} linhas mal formadas foram ignoradas durante o processamento do log.")
    if skipped_status > 0: print(f"{skipped_status} execuções não 'COMPLETED' foram ignoradas.")

    return history_plot_df
# --- FIM NOVA FUNÇÃO ---

# Em activetextclassification/utils.py

import numpy as np
import pandas as pd
# ... (outros imports e funções: preprocess_label, append_experiment_result_to_jsonl, load_and_flatten_experiment_history) ...

def calculate_lce(l_sizes, performance_scores, baseline_performance):
    """
    Calcula a Learning Curve Efficiency (LCE).

    LCE = Area Under Actual Curve / Area Under Ideal Curve (Baseline Rectangle)

    Args:
        l_sizes (list or np.ndarray or pd.Series): Lista ou array dos tamanhos
                                                   do conjunto rotulado (L) em cada
                                                   iteração (eixo X). Deve estar ordenado.
        performance_scores (list or np.ndarray or pd.Series): Lista ou array das
                                                               performances correspondentes
                                                               (ex: external_acc, external_f1)
                                                               (eixo Y).
        baseline_performance (float): O valor da métrica de performance do baseline.

    Returns:
        float: O valor da métrica LCE (geralmente entre 0 e ~1, pode ser > 1).
               Retorna np.nan se os dados forem inválidos ou não for possível calcular.
    """
    # --- Validações Iniciais ---
    if baseline_performance is None or pd.isna(baseline_performance):
        print("AVISO (LCE): Baseline performance inválido (None ou NaN). Impossível calcular LCE.")
        return np.nan
    if baseline_performance <= 0:
        # A métrica perde sentido se o baseline for não-positivo, pois a área ideal seria <= 0.
        print(f"AVISO (LCE): Baseline performance ({baseline_performance:.4f}) não é positivo. LCE pode não ser significativo.")
        # Poderia retornar NaN ou 0, ou continuar e ver o resultado? Vamos retornar NaN.
        return np.nan

    try:
        # Garantir que são arrays numpy e remover NaNs (importante para trapz)
        x = np.array(l_sizes)
        y = np.array(performance_scores)
        valid_mask = ~np.isnan(x) & ~np.isnan(y)
        x = x[valid_mask]
        y = y[valid_mask]

        if len(x) < 2 or len(y) < 2 or len(x) != len(y):
            print(f"AVISO (LCE): Dados de entrada insuficientes ou com tamanhos diferentes após limpar NaNs (len x: {len(x)}, len y: {len(y)}). Impossível calcular LCE.")
            return np.nan

        # Garantir ordenação por l_size (necessário para np.trapz)
        sort_indices = np.argsort(x)
        x_sorted = x[sort_indices]
        y_sorted = y[sort_indices]

        # --- Calcular Área Ideal (Retângulo) ---
        x_min = x_sorted[0]
        x_max = x_sorted[-1]
        delta_x = x_max - x_min

        if delta_x <= 0:
            print(f"AVISO (LCE): Intervalo de L size inválido (min={x_min}, max={x_max}). Impossível calcular LCE.")
            return np.nan

        area_ideal = baseline_performance * delta_x
        if area_ideal <= 0: # Segurança extra
             print(f"AVISO (LCE): Área ideal calculada ({area_ideal:.4f}) não é positiva. LCE pode não ser significativo.")
             return np.nan


        # --- Calcular Área Real (Integral Numérica com Regra do Trapézio) ---
        # np.trapz calcula a área sob os pontos (y) dados os pontos x correspondentes
        area_actual = np.trapz(y=y_sorted, x=x_sorted)

        # --- Calcular LCE ---
        lce = area_actual / area_ideal

        print(f"LCE Calculado: AUC_Actual={area_actual:.4f}, AUC_Ideal={area_ideal:.4f}, LCE={lce:.4f}")
        return lce

    except Exception as e:
        print(f"ERRO inesperado ao calcular LCE: {e}")
        traceback.print_exc() # Adicionado para depuração
        return np.nan

# --- FIM NOVA FUNÇÃO ---