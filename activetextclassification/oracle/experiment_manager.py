# -*- coding: utf-8 -*-
"""Módulo para gerenciar a execução de experimentos de classificação de texto com LLMs.

Este módulo contém a classe OraculoExperimentManager e funções de relatório.
A classe orquestra a avaliação de múltiplos modelos de linguagem, e as funções
de relatório operam sobre os dados gerados para criar análises sob demanda.
"""

import os
import json
import hashlib
import logging
import sys
import time
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# Importações com tratamento de erro
try: import google.generativeai as genai
except ImportError: genai = None
try: from ollama import Client
except ImportError: Client = None
try: import openai
except ImportError: openai = None
try: import anthropic
except ImportError: anthropic = None

try:
    from .oracles import get_oracle
except ImportError:
    print("AVISO CRÍTICO: Não foi possível importar 'oracles.py'.")
    get_oracle = None

class OraculoExperimentManager:
    """Orquestra a execução de experimentos usando uma fábrica de oráculos."""

    def __init__(self, control_path, calls_path, experiments_path, data_path, labels_path):
        """Inicializa o gerenciador de experimentos."""
        self.paths = locals(); del self.paths['self']
        self._configure_logging(); load_dotenv()
        
        self.COLUNAS_CONTROLE = ["item_hash", "call_id", "descrição_original", "descrição_retornada_pelo_llm", "descrição_expandida", "descricao_machine_learning", "categoria", "racional", "ground_truth_categoria", "modelo", "temperatura_usada", "prompt_version_key", "target_run_id", "batch_size"]
        self.COLUNAS_API_CALLS = ["call_id", "timestamp", "model_name", "provider", "num_items_in_call", "prompt_tokens", "completion_tokens", "total_tokens", "duration_sec", "api_status", "error_message"]
        
        self.lista_categorias_py_mestra = self._load_master_categories()
        self.lista_categorias_str_mestra = json.dumps(self.lista_categorias_py_mestra, ensure_ascii=False)
        self.experimentos_a_rodar = self._load_experiments()
        self.map_hash_para_resultado_existente = self._load_control_file_to_cache()

    def _configure_logging(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s', handlers=[logging.FileHandler("experiment_manager_run.log", mode='a'), logging.StreamHandler()]); self.logger = logging.getLogger(__name__)

    def _load_master_categories(self):
        try:
            with open(self.paths["labels_path"], 'r', encoding='utf-8') as f: return sorted(list(set(json.load(f) + ["_RARE_"])))
        except: self.logger.exception("Erro ao carregar labels."); return ["_RARE_"]

    def _load_experiments(self):
        try: return json.load(open(self.paths["experiments_path"], 'r', encoding='utf-8'))
        except: self.logger.exception("Erro ao carregar experimentos."); return []

    def _load_control_file_to_cache(self):
        cache = {}; path = self.paths["control_path"]
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, dtype=str)
                if 'item_hash' in df.columns:
                    for _, row in df.iterrows():
                        if pd.notna(row['item_hash']): cache[row['item_hash']] = dict(row)
                    self.logger.info(f"Cache de resultados populado com {len(cache)} registros.")
            except Exception as e: self.logger.exception(f"Erro ao ler arquivo de controle para cache: {e}")
        return cache

    def _append_to_csv(self, file_path, record_dict, columns):
        """Adiciona um registro a um arquivo CSV de forma segura."""
        try:
            df_row = pd.DataFrame([record_dict]).reindex(columns=columns)
            file_exists = os.path.exists(file_path)
            df_row.to_csv(file_path, mode='a', header=not file_exists, index=False)
        except Exception as e: self.logger.exception(f"Falha CRÍTICA ao salvar registro em {file_path}: {e}")

    def _load_dataset_for_experiment(self, target_run_id):
        """Carrega o conjunto de dados para um experimento específico."""
        try:
            df = pd.read_csv(self.paths["data_path"]); df = df[df['run_id'] == target_run_id].copy()
            df.dropna(subset=['text_sample', 'true_label_sample'], inplace=True); return df.to_dict('records')
        except Exception: self.logger.exception("Erro ao carregar dataset."); return []

    def _process_batch(self, batch_to_process, config):
        """Delega o processamento de um lote para a instância de oráculo apropriada."""
        if not batch_to_process: return []
        
        try:
            oracle_instance = get_oracle(config, self.lista_categorias_str_mestra)
            results, call_log = oracle_instance.query(batch_to_process)
            self._append_to_csv(self.paths["calls_path"], call_log, self.COLUNAS_API_CALLS)
            
            final_records = []
            for res_dict in results:
                item_hash = hashlib.sha256(f"{res_dict['descrição_original']}|{config['model_name']}|{config['temperature']:.2f}|{config['prompt_version_key']}|{config['target_run_id']}|{config.get('batch_size', 1)}".encode()).hexdigest()
                full_record = { "item_hash": item_hash, "call_id": res_dict.get("call_id"), "descrição_original": res_dict.get("descrição_original"), "descrição_retornada_pelo_llm": res_dict.get("descrição_retornada_pelo_llm"), "descrição_expandida": res_dict.get("descrição_expandida"), "descricao_machine_learning": res_dict.get("descricao_machine_learning"), "categoria": res_dict.get("categoria"), "racional": res_dict.get("racional"), "ground_truth_categoria": res_dict.get("ground_truth_categoria"), "modelo": config.get("model_name"), "temperatura_usada": config.get("temperature"), "prompt_version_key": config.get("prompt_version_key"), "target_run_id": config.get("target_run_id"), "batch_size": config.get("batch_size")}
                final_records.append(full_record)
            return final_records
        except (ValueError, RuntimeError, ImportError) as e:
            self.logger.error(f"Não foi possível processar o lote para o modelo '{config['model_name']}': {e}"); return []

    def run_experiments(self):
        """Executa o loop principal de experimentos, gerando os arquivos de dados."""
        if not self.experimentos_a_rodar: self.logger.critical("Nenhum experimento para rodar."); return
        self.logger.info(f"--- INICIANDO EXECUÇÃO DE {len(self.experimentos_a_rodar)} EXPERIMENTOS ---")
        
        for exp_config in self.experimentos_a_rodar:
            batch_size_exp = exp_config.get("batch_size", 1)
            desc_exp = f"Mod={exp_config['model_name']}, Temp={exp_config['temperature']:.1f}, Lote='{exp_config['target_run_id']}', Batch={batch_size_exp}"
            self.logger.info(f"--- Iniciando Experimento: {desc_exp} ---")
            
            dataset = self._load_dataset_for_experiment(exp_config['target_run_id'])
            if not dataset: continue
            
            items_to_process = []
            for item in dataset:
                item_hash = hashlib.sha256(f"{item['text_sample']}|{exp_config['model_name']}|{exp_config['temperature']:.2f}|{exp_config['prompt_version_key']}|{exp_config['target_run_id']}|{batch_size_exp}".encode()).hexdigest()
                if item_hash not in self.map_hash_para_resultado_existente: items_to_process.append(item)
            
            self.logger.info(f"Dataset de {len(dataset)} itens. {len(items_to_process)} para processar via API.")
            
            with tqdm(total=len(items_to_process), desc=desc_exp) as pbar:
                for i in range(0, len(items_to_process), batch_size_exp):
                    batch = items_to_process[i:i + batch_size_exp]
                    if not batch: continue
                    results = self._process_batch(batch, exp_config)
                    for record in results:
                        if record.get('item_hash'):
                            self._append_to_csv(self.paths["control_path"], record, self.COLUNAS_CONTROLE)
                            self.map_hash_para_resultado_existente[record['item_hash']] = record
                    pbar.update(len(batch))
        
        self.logger.info("--- EXECUÇÃO DE EXPERIMENTOS FINALIZADA ---")

# --- FUNÇÕES DE RELATÓRIO (para serem chamadas independentemente) ---

def _format_duration_static(seconds):
    """Versão estática da função de formatação de tempo."""
    if pd.isna(seconds): return "N/A"
    secs = int(seconds); h = secs//3600; m = (secs%3600)//60; s = secs%60
    return f"{h}h {m}m {s}s" if h > 0 else (f"{m}m {s}s" if m > 0 else f"{s}s")

def generate_summary_report(control_csv_path, calls_csv_path, output_path_no_ext, output_format='excel'):
    """Gera um relatório de sumário com acurácia, F1-Score e taxa de replicação."""
    print("--- Iniciando Geração de Relatório de Sumário ---")
    try:
        if not os.path.exists(control_csv_path): print(f"Erro: Arquivo '{control_csv_path}' não encontrado."); return
        df_results = pd.read_csv(control_csv_path, dtype=str).drop_duplicates(subset=['item_hash'], keep='last')
        if df_results.empty: print("Aviso: 'control_results.csv' está vazio."); return
        print(f"Resultados únicos carregados: {len(df_results)} linhas.")
        
        exp_cols = ['modelo', 'temperatura_usada', 'prompt_version_key', 'target_run_id', 'batch_size']
        df_results['categoria'] = df_results['categoria'].fillna('_NULL_')
        df_results['ground_truth_categoria'] = df_results['ground_truth_categoria'].fillna('_NULL_')
        for col in ['temperatura_usada', 'batch_size']: df_results[col] = pd.to_numeric(df_results[col], errors='coerce')
        
        summary_list = []
        for name, group in df_results.groupby(exp_cols):
            exp_config = dict(zip(exp_cols, name))
            
            # Cálculo das métricas de classificação
            ground_truth_labels = group['ground_truth_categoria'].unique()
            exp_config['acuracia'] = accuracy_score(group['ground_truth_categoria'], group['categoria'])
            exp_config['f1_score_macro'] = f1_score(group['ground_truth_categoria'], group['categoria'], labels=ground_truth_labels, average='macro', zero_division=0)
            
            # --- NOVA MÉTRICA AQUI ---
            # Compara a descrição original com a retornada pelo LLM.
            replicated_count = (group['descrição_original'].fillna('') == group['descrição_retornada_pelo_llm'].fillna('')).sum()
            total_items = len(group)
            exp_config['taxa_replicacao_descricao'] = replicated_count / total_items if total_items > 0 else 0
            # --- FIM DA NOVA MÉTRICA ---

            exp_config['total_itens_unicos'] = total_items
            summary_list.append(exp_config)
            
        if not summary_list: print("Nenhum grupo de experimento encontrado."); return
        summary_df = pd.DataFrame(summary_list)

        if os.path.exists(calls_csv_path):
            df_calls = pd.read_csv(calls_csv_path)
            for col in ['prompt_tokens', 'completion_tokens', 'total_tokens', 'duration_sec']: df_calls[col] = pd.to_numeric(df_calls[col], errors='coerce').fillna(0)
            
            call_summary = df_calls.groupby(['model_name', 'num_items_in_call']).agg(
                duracao_total_segundos=('duration_sec', 'sum'),
                total_chamadas_api=('call_id', 'nunique'),
                total_geral_tokens=('total_tokens', 'sum')
            ).reset_index()
            
            summary_df = pd.merge(summary_df, call_summary, left_on=['modelo', 'batch_size'], right_on=['model_name', 'num_items_in_call'], how='left')
            summary_df.drop(columns=['model_name', 'num_items_in_call'], inplace=True, errors='ignore')
            summary_df['duracao_formatada'] = summary_df.get('duracao_total_segundos', 0).apply(_format_duration_static)
        
        output_file = f"{output_path_no_ext}.{'xlsx' if output_format == 'excel' else 'csv'}"
        if output_format == 'excel': summary_df.to_excel(output_file, index=False)
        else: summary_df.to_csv(output_file, index=False)
        print(f"Relatório de sumário gerado com sucesso em '{output_file}'")
    except Exception as e: print(f"Ocorreu um erro ao gerar o relatório de sumário: {e}"); import traceback; traceback.print_exc()

def generate_error_report(control_csv_path, output_path_no_ext, output_format='excel'):
    """Gera um relatório contendo apenas os erros de classificação."""
    print("--- Iniciando Geração de Relatório de Erros ---")
    try:
        df = pd.read_csv(control_csv_path, dtype=str).drop_duplicates(subset=['item_hash'], keep='last')
        df_errors = df[df['categoria'] != df['ground_truth_categoria']].copy()
        output_file = f"{output_path_no_ext}.{'xlsx' if output_format == 'excel' else 'csv'}"
        if output_format == 'excel': df_errors.to_excel(output_file, index=False)
        else: df_errors.to_csv(output_file, index=False)
        print(f"Relatório de erros com {len(df_errors)} registros gerado em '{output_file}'")
    except Exception as e: print(f"Ocorreu um erro ao gerar o relatório de erros: {e}")

def convert_csv_to_excel(csv_path, excel_path_no_ext):
    """Converte um arquivo CSV em um arquivo Excel."""
    print(f"--- Convertendo {os.path.basename(csv_path)} para Excel ---")
    try:
        if not os.path.exists(csv_path): print(f"Erro: Arquivo CSV não encontrado em '{csv_path}'."); return
        df = pd.read_csv(csv_path, dtype=str).drop_duplicates(subset=['item_hash'], keep='last')
        output_file = f"{excel_path_no_ext}.xlsx"
        df.to_excel(output_file, index=False, engine='openpyxl')
        print(f"Arquivo convertido para Excel com sucesso em '{output_file}'")
    except Exception as e: print(f"Ocorreu um erro durante a conversão para Excel: {e}")