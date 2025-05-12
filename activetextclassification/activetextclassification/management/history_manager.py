# activetextclassification/management/history_manager.py

import json
import os
import pandas as pd # Para o método get_experiment_history_df (opcional)
from datetime import datetime # Para timestamps, se necessário no futuro

class HistoryManager:
    """
    Gerencia a leitura e escrita do log de histórico de execuções de experimentos.
    Opera sobre um arquivo JSON Lines (.jsonl).
    """
    def __init__(self, log_file_path="history_log.jsonl"):
        """
        Inicializa o HistoryManager.

        Args:
            log_file_path (str): Caminho para o arquivo de log JSONL.
        """
        self.log_file_path = log_file_path
        print(f"HistoryManager inicializado para o arquivo: {os.path.abspath(self.log_file_path)}")
        # Criar diretório do log se não existir (apenas o diretório, não o arquivo)
        log_dir = os.path.dirname(self.log_file_path)
        if log_dir and not os.path.exists(log_dir): # Checa se log_dir não é vazio
            try:
                os.makedirs(log_dir, exist_ok=True)
                print(f"Diretório de log criado: {log_dir}")
            except Exception as e:
                print(f"AVISO: Não foi possível criar diretório de log {log_dir}: {e}")


    def get_completed_experiment_names(self):
        """
        Lê o arquivo de log e retorna um conjunto com os nomes dos experimentos
        que foram concluídos com sucesso ('status': 'COMPLETED').

        Returns:
            set: Conjunto de nomes de experimentos concluídos.
        """
        completed_experiments = set()
        if not os.path.exists(self.log_file_path):
            print(f"Arquivo de histórico '{self.log_file_path}' não encontrado. Nenhum experimento concluído carregado.")
            return completed_experiments

        print(f"Lendo histórico de execuções de: {self.log_file_path}")
        lines_read = 0
        malformed_lines = 0
        try:
            with open(self.log_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    lines_read += 1
                    try:
                        log_entry = json.loads(line)
                        exp_name_log = log_entry.get('experiment_name')
                        exp_status_log = log_entry.get('status')
                        if exp_name_log and exp_status_log == 'COMPLETED':
                            completed_experiments.add(exp_name_log)
                    except json.JSONDecodeError:
                        malformed_lines +=1
                        # print(f"Aviso: Pulando linha mal formada ({line_num+1}) no histórico: {line.strip()}")
            if malformed_lines > 0:
                 print(f"Aviso: {malformed_lines}/{lines_read} linhas mal formadas encontradas no histórico.")
            print(f"Arquivo de histórico lido. Encontrados {len(completed_experiments)} experimentos 'COMPLETED'.")
        except Exception as e:
            print(f"Erro ao ler o arquivo de histórico {self.log_file_path}: {e}")
            # Retorna conjunto vazio em caso de erro maior na leitura
            return set()
        return completed_experiments


    def log_experiment_result(self, result_data_dict):
        """
        Anexa o dicionário de resultado de uma execução de experimento
        ao arquivo de log JSON Lines (.jsonl).

        Args:
            result_data_dict (dict): Dicionário contendo todas as informações e resultados
                                     da execução do experimento.
        """
        if not isinstance(result_data_dict, dict) or not result_data_dict:
            print("AVISO (HistoryManager): Dados de resultado inválidos ou vazios. Nada para logar.")
            return

        exp_name_log = result_data_dict.get('experiment_name', 'N/A_Result')
        try:
            print(f"HistoryManager: Logando resultado para '{exp_name_log}' em: {self.log_file_path}")
            # Usar default=str para lidar com tipos não serializáveis (ex: numpy dtypes, datetime)
            json_string = json.dumps(result_data_dict, default=str, ensure_ascii=False, indent=None) # indent=None para uma linha

            # Anexar a string JSON como uma nova linha no arquivo
            with open(self.log_file_path, 'a', encoding='utf-8') as f:
                f.write(json_string + '\n')

            print(f"Resultado para '{exp_name_log}' anexado com sucesso ao log.")
        except Exception as e:
            print(f"ERRO (HistoryManager) ao anexar resultado para '{exp_name_log}' no JSONL: {e}")


    def get_experiment_log_entries(self, experiment_name_filter=None):
        """
        Lê o arquivo de log e retorna uma lista de todas as entradas (dicionários).
        Pode filtrar por nome de experimento.

        Args:
            experiment_name_filter (str, optional): Se fornecido, retorna apenas
                                                    entradas para este nome de experimento.

        Returns:
            list: Lista de dicionários, onde cada dicionário é uma entrada de log.
        """
        log_entries = []
        if not os.path.exists(self.log_file_path):
            return log_entries

        try:
            with open(self.log_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if experiment_name_filter:
                            if entry.get('experiment_name') == experiment_name_filter:
                                log_entries.append(entry)
                        else:
                            log_entries.append(entry)
                    except json.JSONDecodeError:
                        continue # Pula linhas mal formadas
        except Exception as e:
            print(f"Erro ao ler entradas do log {self.log_file_path}: {e}")
        return log_entries

    def get_latest_completed_run(self, experiment_name):
        """
        Encontra a entrada de log mais recente para um experimento que foi 'COMPLETED'.

        Args:
            experiment_name (str): Nome do experimento.

        Returns:
            dict or None: O dicionário da entrada de log mais recente e completa, ou None se não encontrado.
        """
        all_runs = self.get_experiment_log_entries(experiment_name_filter=experiment_name)
        completed_runs = [r for r in all_runs if r.get('status') == 'COMPLETED']
        if not completed_runs:
            return None
        # Ordenar por timestamp (assumindo que está no formato ISO e pode ser comparado como string)
        # Ou converter para datetime se o formato for garantido
        try:
             return sorted(completed_runs, key=lambda x: x.get('execution_timestamp', ''), reverse=True)[0]
        except: # Fallback se timestamp for problemático
             return completed_runs[-1] # Pega a última na ordem do arquivo

    def get_experiment_history_df(self, experiment_name):
        """
        Obtém o histórico de iterações ('history_data') da execução
        COMPLETED mais recente de um experimento específico como DataFrame.

        Args:
            experiment_name (str): Nome do experimento.

        Returns:
            pd.DataFrame or None: DataFrame do histórico, ou None se não encontrado/sem dados.
        """
        latest_run = self.get_latest_completed_run(experiment_name)
        if latest_run and 'history_data' in latest_run and isinstance(latest_run['history_data'], list):
             if latest_run['history_data']: # Checar se a lista não está vazia
                  return pd.DataFrame(latest_run['history_data'])
             else:
                  print(f"Histórico de iterações para '{experiment_name}' está vazio.")
                  return pd.DataFrame() # Retorna DataFrame vazio
        print(f"Nenhum histórico de iterações 'COMPLETED' encontrado para '{experiment_name}'.")
        return None