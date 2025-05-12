# activetextclassification/data_preparation.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os # Para verificar extensão do arquivo

# Importar função utilitária do mesmo pacote
from .utils import preprocess_label

def load_and_prepare_data(
    file_path,                  # Caminho para o arquivo (CSV ou Excel)
    text_column,
    label_column,
    min_samples_per_class=2,    # Novo: Mínimo de amostras para manter uma classe
    rare_group_label="_RARE_GROUP_", # Novo: Rótulo para classes agrupadas
    population_size=0.50,
    random_seed=42,
    sheet_name=0 # Para arquivos Excel: nome ou índice da planilha
):
    """
    Carrega dados de CSV ou Excel, pré-processa, agrupa classes raras e divide em P/U.

    Args:
        file_path (str): Caminho para o arquivo CSV ou Excel (.xlsx, .xls).
        text_column (str): Nome da coluna de texto.
        label_column (str): Nome da coluna de rótulos.
        min_samples_per_class (int): Número mínimo de amostras para uma classe não ser agrupada.
                                     Se <= 1, nenhum agrupamento é feito.
        rare_group_label (str): Rótulo a ser atribuído às classes com menos de min_samples_per_class.
        population_size (float): Proporção para População (P).
        random_seed (int): Semente para reprodutibilidade.
        sheet_name (int or str, optional): Nome ou índice da planilha para arquivos Excel. Padrão é 0 (primeira planilha).


    Returns:
        tuple: Contendo:
            - pd.DataFrame: DataFrame da População (P).
            - pd.DataFrame: DataFrame do Pool Não Rotulado inicial (U).
            - dict: Mapeamento label_to_id criado APÓS agrupamento.
            - dict: Mapeamento id_to_label criado APÓS agrupamento.
            - list: Lista de todos os labels únicos (strings) APÓS agrupamento.
    """
    print(f"--- Iniciando Preparação de Dados ---")
    print(f"Carregando dados de: {file_path}")

    # --- Carregar Dados (CSV ou Excel) ---
    try:
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext == '.csv':
            df = pd.read_csv(file_path)
        elif file_ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
        else:
            raise ValueError(f"Extensão de arquivo não suportada: {file_ext}. Use .csv, .xlsx ou .xls.")

        print(f"Shape inicial: {df.shape}")
        if text_column not in df.columns: raise ValueError(f"Coluna de texto '{text_column}' não encontrada.")
        if label_column not in df.columns: raise ValueError(f"Coluna de label '{label_column}' não encontrada.")

        initial_rows = len(df)
        df.dropna(subset=[text_column, label_column], inplace=True)
        rows_after_na = len(df)
        print(f"Shape após remover NaNs: {df.shape} ({initial_rows - rows_after_na} linhas removidas)")
        if df.empty: raise ValueError("DataFrame vazio após remover NaNs.")

    except FileNotFoundError: print(f"Erro: Arquivo '{file_path}' não encontrado."); raise
    except Exception as e: print(f"Erro ao carregar ou processar o arquivo: {e}"); raise

    # --- Pré-processar Rótulos ---
    print("Pré-processando rótulos...")
    df[label_column] = df[label_column].apply(preprocess_label).astype(str) # Garantir que são strings

    # --- Agrupar Classes Raras ---
    if min_samples_per_class is not None and min_samples_per_class > 1:
        print(f"Verificando classes com menos de {min_samples_per_class} amostras...")
        label_counts = df[label_column].value_counts()
        rare_labels = label_counts[label_counts < min_samples_per_class].index.tolist()

        if rare_labels:
            print(f"Agrupando {len(rare_labels)} classes raras em '{rare_group_label}': {rare_labels[:10]}...") # Mostra até 10
            # Substitui os rótulos raros pelo rótulo de grupo
            df[label_column] = df[label_column].replace(rare_labels, rare_group_label)
            print(f"Distribuição de classes após agrupamento:\n{df[label_column].value_counts().head()}") # Mostra as mais frequentes
        else:
            print("Nenhuma classe rara encontrada para agrupar.")
    else:
        print("Agrupamento de classes raras desativado (min_samples_per_class <= 1).")


    # --- Criar Mapeamento de Rótulos para IDs (APÓS agrupamento) ---
    all_possible_labels = pd.unique(df[label_column]).tolist()
    label_to_id = {label: i for i, label in enumerate(all_possible_labels)}
    id_to_label = {i: label for label, i in label_to_id.items()}
    num_classes = len(all_possible_labels)
    print(f"Mapeamento de rótulos finalizado: {num_classes} classes (incluindo '{rare_group_label}' se criado).")

    # Adicionar coluna 'label_id' final
    df['label_id'] = df[label_column].map(label_to_id)

    # --- Dividir em População (P) e Pool (U) ---
    print(f"Dividindo em População (P = {population_size*100:.0f}%) e Pool (U)...")
    if len(df) < 2: raise ValueError("Dataset muito pequeno para dividir.")

    # Verificar se todas as classes *restantes* têm amostras suficientes para estratificar (pelo menos 2)
    final_label_counts = df['label_id'].value_counts()
    if (final_label_counts < 2).any():
        labels_lt_2 = final_label_counts[final_label_counts < 2].index.map(id_to_label).tolist()
        print(f"AVISO FINAL: As seguintes classes têm menos de 2 amostras e não podem ser estratificadas: {labels_lt_2}")
        print("Realizando divisão não estratificada.")
        stratify_labels = None
    else:
        stratify_labels = df['label_id'] # Estratificar por ID numérico

    P_df, U_df = train_test_split(
        df,
        test_size=(1.0 - population_size),
        random_state=random_seed,
        stratify=stratify_labels # Usa None se não puder estratificar
    )

    print(f"Tamanho População P: {len(P_df)}")
    print(f"Tamanho Pool U inicial: {len(U_df)}")
    print("--- Preparação de Dados Concluída ---")

    # Retornar os dataframes e os mapeamentos FINAIS
    return P_df.copy(), U_df.copy(), label_to_id, id_to_label, all_possible_labels