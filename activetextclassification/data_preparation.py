# activetextclassification/data_preparation.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import json

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

def load_split_and_preprocess_data(
    file_path,
    text_column,
    label_column,
    min_samples_per_class=2,
    rare_group_label="_RARE_",
    test_set_size=0.30,
    random_state_split=42,
    output_dir=".", # Diretório para salvar/carregar os splits
    force_split=False, # Forçar nova divisão mesmo que arquivos existam
    sheet_name=0
):
    """
    Carrega dados, pré-processa (incluindo agrupamento de classes raras no dataset completo),
    divide em conjuntos de treino/otimização e teste (estratificado),
    e opcionalmente salva/carrega esses conjuntos.

    Args:
        file_path (str): Caminho para o arquivo de dados (CSV ou Excel).
        text_column (str): Nome da coluna de texto.
        label_column (str): Nome da coluna de rótulos.
        min_samples_per_class (int): Mínimo de amostras para uma classe não ser rara.
        rare_group_label (str): Rótulo para classes raras agrupadas.
        test_set_size (float): Proporção do dataset a ser usada para o conjunto de teste.
        random_state_split (int): Semente para a divisão treino/teste.
        output_dir (str): Diretório para salvar/carregar os arquivos CSV dos splits.
        force_split (bool): Se True, força a re-divisão mesmo que os arquivos existam.
        sheet_name (int or str): Nome ou índice da planilha para arquivos Excel.

    Returns:
        tuple: (df_train_opt, df_test_T, all_possible_labels_original)
               Retorna (None, None, None) em caso de erro crítico.
    """
    print(f"--- Iniciando Carga, Pré-processamento e Divisão de Dados ---")
    os.makedirs(output_dir, exist_ok=True)

    train_file = os.path.join(output_dir, "df_train_opt.csv")
    test_file = os.path.join(output_dir, "df_test_T.csv")
    labels_file = os.path.join(output_dir, "all_possible_labels_original.json")

    if not force_split and os.path.exists(train_file) and os.path.exists(test_file) and os.path.exists(labels_file):
        print(f"Carregando conjuntos de dados divididos e labels de '{output_dir}'...")
        try:
            df_train_opt = pd.read_csv(train_file)
            df_test_T = pd.read_csv(test_file)
            with open(labels_file, 'r') as f:
                all_possible_labels_original = json.load(f)
            print("Dados carregados com sucesso.")
            # Garantir tipos corretos após carregar do CSV
            df_train_opt[text_column] = df_train_opt[text_column].astype(str)
            df_train_opt[label_column] = df_train_opt[label_column].astype(str)
            df_test_T[text_column] = df_test_T[text_column].astype(str)
            df_test_T[label_column] = df_test_T[label_column].astype(str)
            return df_train_opt, df_test_T, all_possible_labels_original
        except Exception as e:
            print(f"Erro ao carregar arquivos divididos: {e}. Prosseguindo com nova divisão.")

    print(f"Processando arquivo original: {file_path}")
    try:
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext == '.csv':
            df_original = pd.read_csv(file_path)
        elif file_ext in ['.xlsx', '.xls']:
            df_original = pd.read_excel(file_path, sheet_name=sheet_name)
        else:
            raise ValueError(f"Extensão de arquivo não suportada: {file_ext}.")

        if text_column not in df_original.columns or label_column not in df_original.columns:
            raise ValueError("Colunas de texto ou rótulo não encontradas no arquivo.")

        df_original.dropna(subset=[text_column, label_column], inplace=True)
        df_original[text_column] = df_original[text_column].astype(str) # Garantir texto como string
        df_original[label_column] = df_original[label_column].apply(preprocess_label).astype(str)

        if df_original.empty:
            raise ValueError("Dataset vazio após remover NaNs ou pré-processar labels.")

        # Agrupar classes raras no dataset completo ANTES da divisão
        if min_samples_per_class > 1:
            print(f"Agrupando classes raras (threshold: {min_samples_per_class})...")
            label_counts = df_original[label_column].value_counts()
            rare_labels = label_counts[label_counts < min_samples_per_class].index.tolist()
            if rare_labels:
                df_original[label_column] = df_original[label_column].replace(rare_labels, rare_group_label)
                print(f"  {len(rare_labels)} classes raras agrupadas em '{rare_group_label}'.")
        
        all_possible_labels_original = sorted(list(df_original[label_column].unique()))
        print(f"Total de classes únicas (após agrupamento, se houver): {len(all_possible_labels_original)}")


        # Verificar se ainda há classes com menos de 2 amostras após o agrupamento de raros
        # Isso é importante para a estratificação
        final_label_counts_for_split = df_original[label_column].value_counts()
        labels_with_one_sample = final_label_counts_for_split[final_label_counts_for_split < 2].index.tolist()

        if labels_with_one_sample:
            print(f"AVISO: As seguintes classes têm apenas 1 amostra e serão removidas ANTES da divisão para permitir estratificação: {labels_with_one_sample}")
            df_original = df_original[~df_original[label_column].isin(labels_with_one_sample)]
            # Recalcular all_possible_labels_original se classes foram removidas
            all_possible_labels_original = sorted(list(df_original[label_column].unique()))
            print(f"Total de classes únicas (após remover classes com 1 amostra): {len(all_possible_labels_original)}")
            if df_original.empty:
                 raise ValueError("Dataset ficou vazio após remover classes com apenas 1 amostra.")


        if len(df_original) < 2: # Checar novamente após possível remoção
            raise ValueError("Dataset muito pequeno para dividir após pré-processamento.")

        print(f"Dividindo em treino ({100*(1-test_set_size):.0f}%) e teste ({100*test_set_size:.0f}%)...")
        df_train_opt, df_test_T = train_test_split(
            df_original,
            test_size=test_set_size,
            random_state=random_state_split,
            stratify=df_original[label_column] # Estratificar pelos labels processados
        )
        
        df_train_opt = df_train_opt.reset_index(drop=True)
        df_test_T = df_test_T.reset_index(drop=True)

        print(f"  Tamanho df_train_opt: {len(df_train_opt)}")
        print(f"  Tamanho df_test_T: {len(df_test_T)}")

        # Salvar os conjuntos divididos
        df_train_opt.to_csv(train_file, index=False)
        df_test_T.to_csv(test_file, index=False)
        with open(labels_file, 'w') as f:
            json.dump(all_possible_labels_original, f)
        print(f"Conjuntos divididos e labels salvos em '{output_dir}'.")
        
        return df_train_opt, df_test_T, all_possible_labels_original

    except Exception as e:
        print(f"ERRO CRÍTICO em load_split_and_preprocess_data: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None # Retornar None em caso de erro