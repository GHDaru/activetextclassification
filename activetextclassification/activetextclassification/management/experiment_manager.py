# activetextclassification/management/experiment_manager.py

import time
import traceback
import os # Para verificar arquivos
from datetime import datetime

# Imports da própria biblioteca
# Usar imports absolutos DENTRO da biblioteca é geralmente mais robusto
from activetextclassification.config import load_experiments_config, validate_experiment_config
from activetextclassification.active_learner import ActiveLearner
from activetextclassification.management.history_manager import HistoryManager
# Importar função de plotagem se o manager for responsável por ela
# from activetextclassification.visualization import plot_learning_curves


class ExperimentManager:
    """
    Gerencia a execução de múltiplos experimentos de Aprendizado Ativo.

    Lê configurações, interage com o histórico para evitar reexecuções,
    instancia e executa ActiveLearner para cada experimento pendente,
    e loga os resultados.
    """
    def __init__(self, config_file_path="experiments.json", history_log_path="history_log.jsonl"):
        """
        Inicializa o ExperimentManager.

        Args:
            config_file_path (str): Caminho para o arquivo JSON de configurações.
            history_log_path (str): Caminho para o arquivo JSONL de log de histórico.
        """
        self.config_file_path = config_file_path
        self.history_log_path = history_log_path
        self.history_mgr = HistoryManager(log_file_path=self.history_log_path)
        self.all_configs = []
        self.completed_experiments = set()
        self.executed_in_session = 0
        self.failed_in_session = []

        print(f"ExperimentManager inicializado.")
        print(f" - Arquivo de Configuração: {os.path.abspath(self.config_file_path)}")
        print(f" - Arquivo de Histórico: {os.path.abspath(self.history_log_path)}")


    def load_and_prepare(self):
        """ Carrega configurações e histórico anterior. """
        print("\n--- Carregando Configurações e Histórico ---")
        self.all_configs = load_experiments_config(self.config_file_path)
        self.completed_experiments = self.history_mgr.get_completed_experiment_names()
        if not self.all_configs:
             print("AVISO: Nenhuma configuração carregada.")


    def run_pending_experiments(self, force_rerun=False):
        """
        Itera sobre as configurações carregadas e executa os experimentos
        que estão ativos e ainda não foram concluídos (ou todos se force_rerun=True).

        Args:
            force_rerun (bool): Se True, ignora o histórico de concluídos e
                                tenta executar todos os experimentos ativos.
        """
        if not self.all_configs:
            print("Nenhuma configuração para executar.")
            return

        self.executed_in_session = 0 # Resetar contadores da sessão
        self.failed_in_session = []
        initial_completed_count = len(self.completed_experiments)

        print(f"\n{'='*15} Iniciando Execução de Experimentos {'='*15}")
        print(f"Total de configurações encontradas: {len(self.all_configs)}")
        if not force_rerun:
             print(f"Execuções 'COMPLETED' anteriores a serem puladas: {initial_completed_count}")
        else:
             print("AVISO: Forçando re-execução de todos os experimentos ativos (force_rerun=True).")


        for i, config in enumerate(self.all_configs):
            exp_name = config.get("experiment_name", f"UnnamedExp_{i+1}")
            is_active = config.get("active", False)
            print(f"\n>>> Verificando Experimento {i+1}/{len(self.all_configs)}: {exp_name}")

            # --- Checagens ---
            if not is_active:
                print("Status: INATIVO. Pulando.")
                continue
            if not validate_experiment_config(config):
                print("Status: CONFIGURAÇÃO INVÁLIDA. Pulando.")
                continue
            if not force_rerun and exp_name in self.completed_experiments:
                print("Status: JÁ EXECUTADO (COMPLETED). Pulando.")
                continue

            # --- Executar Experimento ---
            print(f"Status: EXECUTANDO (Ativo: {is_active}, Concluído Anteriormente: {exp_name in self.completed_experiments}, Forçar Re-run: {force_rerun})")
            learner = None # Resetar para cada experimento
            exp_start_time = time.time()

            try:
                print(f"Instanciando ActiveLearner para '{exp_name}'...")
                learner = ActiveLearner(config)

                print(f"Executando setup para '{exp_name}'...")
                learner.setup() # Executa data prep, baseline, embedder, cold start

                print(f"Executando run para '{exp_name}'...")
                if learner.status == "READY":
                     learner.run() # Executa o loop AL
                else:
                     print(f"Setup falhou ou estado inválido ({learner.status}). Pulando run.")
                     # Status de falha já foi definido no learner se setup falhou

            except KeyboardInterrupt:
                 print(f"\nEXECUÇÃO INTERROMPIDA PELO USUÁRIO para {exp_name}.")
                 if learner: learner.status = "INTERRUPTED"; learner.error_message = "Interrupção manual."
                 self.failed_in_session.append(exp_name)
                 # Logar resultado parcial e parar tudo
                 if learner: self.history_mgr.log_experiment_result(learner.get_results_summary())
                 raise KeyboardInterrupt
            except Exception as e:
                 print(f"\nERRO INESPERADO no experimento {exp_name}: {type(e).__name__} - {e}")
                 traceback.print_exc()
                 if learner: learner.status = "FAILED"; learner.error_message = f"Manager Error: {type(e).__name__}: {e}"
                 else: # Se falhou antes de instanciar learner
                      error_summary = {"experiment_name": exp_name, "status": "FAILED", "config": config, "error_message": f"Instantiation/Setup Error: {type(e).__name__}: {e}"}
                      self.history_mgr.log_experiment_result(error_summary) # Logar erro mínimo
                 self.failed_in_session.append(exp_name)
                 # Continuar para o próximo experimento

            finally:
                 # Registrar resultado final via HistoryManager
                 if learner:
                      # Atualizar duração total se necessário
                      if learner.total_duration_sec is None:
                           learner.total_duration_sec = round(time.time() - exp_start_time, 2)
                      result_summary = learner.get_results_summary()
                      # Adicionar duração total do manager se diferente da do learner
                      result_summary["manager_wall_time_sec"] = round(time.time() - exp_start_time, 2)
                      self.history_mgr.log_experiment_result(result_summary)
                      if learner.status == "COMPLETED":
                           self.executed_in_session += 1
                           # Adicionar aos concluídos da sessão para evitar re-rodar na *mesma* chamada de run_pending_experiments
                           self.completed_experiments.add(exp_name)
                 # Se learner não foi criado, já logamos no except


        print(f"\n{'='*15} Execução de Experimentos Concluída {'='*15}")
        self.display_summary()


    def display_summary(self):
        """ Imprime um sumário da execução atual. """
        print("\n--- Sumário da Execução ---")
        print(f"Total de configurações no arquivo: {len(self.all_configs)}")
        print(f"Novas execuções concluídas com sucesso nesta sessão: {self.executed_in_session}")
        if self.failed_in_session:
            print(f"Experimentos que FALHARAM ou foram INTERROMPIDOS nesta sessão: {self.failed_in_session}")
        else:
            print("Nenhum experimento falhou ou foi interrompido nesta sessão.")
        print(f"Resultados completos logados em: {os.path.abspath(self.history_log_path)}")


    # --- Métodos Adicionais (Opcionais) ---
    def get_experiment_result(self, experiment_name):
         """ Obtém o resultado da última execução completa de um experimento. """
         return self.history_mgr.get_latest_completed_run(experiment_name)

    def get_experiment_history_df(self, experiment_name):
         """ Obtém o DataFrame de histórico de iterações para um experimento. """
         return self.history_mgr.get_experiment_history_df(experiment_name)

    # Poderia adicionar um método de plotagem aqui que usa get_experiment_history_df
    def plot_experiment_results(self, experiment_name, **plot_kwargs):
        """ Plota as curvas de aprendizado para um experimento específico. """
        print(f"--- Plotando resultados para: {experiment_name} ---")
        log_entry = self.get_experiment_result(experiment_name)
        history_df = self.get_experiment_history_df(experiment_name)
    
        if history_df is not None and not history_df.empty:
             baseline_metrics = log_entry.get('baseline_metrics', {}) if log_entry else {}
             # Precisa importar a função de plotagem
             from activetextclassification.visualization import plot_learning_curves
             plot_learning_curves(
                 history_df=history_df,
                 baseline_metrics=baseline_metrics,
                 experiment_name=experiment_name,
                 **plot_kwargs # Passar argumentos extras como figsize
             )
        elif log_entry:
             print("Log encontrado, mas sem dados de histórico de iteração para plotar.")
        else:
             print(f"Nenhum resultado 'COMPLETED' encontrado para '{experiment_name}'.")