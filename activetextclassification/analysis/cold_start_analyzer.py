# activetextclassification/analysis/cold_start_analyzer.py

import pandas as pd
import numpy as np

def calculate_l0_sensitivity_stats(results_df, l0_size_column='l0_size'):
    """
    Calcula estatísticas descritivas (média, dp, min, max, percentis) para
    'accuracy_on_population' e 'f1_macro_on_population' agrupadas por l0_size.

    Args:
        results_df (pd.DataFrame): DataFrame com os resultados brutos das simulações
                                   (deve conter l0_size_column, accuracy_on_population,
                                    f1_macro_on_population).
        l0_size_column (str): Nome da coluna que identifica o tamanho do L0.

    Returns:
        pd.DataFrame: DataFrame com as estatísticas sumarizadas.
    """
    if results_df is None or results_df.empty:
        print("AVISO (calculate_l0_stats): DataFrame de resultados vazio.")
        return pd.DataFrame()

    required_metrics = ['accuracy_on_population', 'f1_macro_on_population']
    if not all(col in results_df.columns for col in required_metrics + [l0_size_column]):
        missing = [col for col in required_metrics + [l0_size_column] if col not in results_df.columns]
        print(f"AVISO (calculate_l0_stats): Colunas necessárias ausentes: {missing}")
        return pd.DataFrame()

    stats_summary_list = []
    for metric_col_base, metric_name_base in [
        ('accuracy_on_population', 'Acurácia'),
        ('f1_macro_on_population', 'F1-Macro')
    ]:
        valid_metric_df = results_df.dropna(subset=[metric_col_base])
        if valid_metric_df.empty:
            print(f"Sem dados válidos para a métrica '{metric_name_base}' após remover NaNs.")
            continue

        stats = valid_metric_df.groupby(l0_size_column)[metric_col_base].agg(
            Média='mean',
            DesvioPadrão='std',
            Mínimo='min',
            P25=lambda x: x.quantile(0.25),
            Mediana='median',
            P75=lambda x: x.quantile(0.75),
            Máximo='max'
        ).reset_index()

        stats['IQR'] = stats['P75'] - stats['P25']
        stats['CV (Mediana)'] = stats['DesvioPadrão'] / (stats['Mediana'].replace(0, np.nan) + 1e-9) # Evitar divisão por zero
        stats['Métrica'] = metric_name_base
        stats_summary_list.append(stats)

    if not stats_summary_list:
        print("Nenhuma estatística calculada.")
        return pd.DataFrame()

    full_stats_df = pd.concat(stats_summary_list, ignore_index=True)
    # Reordenar colunas para melhor visualização
    cols_order_stats = ['Métrica', l0_size_column, 'Média', 'Mediana', 'DesvioPadrão', 'Mínimo', 'P25', 'P75', 'Máximo', 'IQR', 'CV (Mediana)']
    # Manter apenas colunas que existem
    final_cols = [col for col in cols_order_stats if col in full_stats_df.columns]
    full_stats_df = full_stats_df[final_cols]
    return full_stats_df


def generate_l0_stats_latex_table(
    stats_summary_df,
    l0_size_column='l0_size',
    caption="Estatísticas Descritivas da Performance em Função do Tamanho de L0 Aleatório.",
    label="tab:l0_random_stats_summary",
    float_format="%.3f"
):
    """
    Gera o código LaTeX para uma tabela de estatísticas descritivas.

    Args:
        stats_summary_df (pd.DataFrame): DataFrame retornado por calculate_l0_sensitivity_stats.
        l0_size_column (str): Nome da coluna de tamanho L0.
        caption (str): Legenda da tabela.
        label (str): Label LaTeX para a tabela.
        float_format (str): Formato para números float na tabela.

    Returns:
        str: String contendo o código LaTeX da tabela.
    """
    if stats_summary_df is None or stats_summary_df.empty:
        return "% DataFrame de estatísticas vazio, nenhuma tabela gerada."

    latex_df = stats_summary_df.copy()
    latex_df.rename(columns={
        l0_size_column: 'Tam. L0 (I)',
        'DesvioPadrão': 'DP',
        'Métrica': 'Métrica Base',
        'CV (Mediana)': 'CV (Med.)' # Abreviação para caber
    }, inplace=True)

    float_cols = ['Média', 'Mediana', 'DP', 'Mínimo', 'P25', 'P75', 'Máximo', 'IQR', 'CV (Med.)']
    for col in float_cols:
        if col in latex_df.columns:
            latex_df[col] = latex_df[col].apply(lambda x: float_format % x if pd.notna(x) else "-")

    # Selecionar e reordenar colunas para a tabela
    cols_for_latex = ['Métrica Base', 'Tam. L0 (I)', 'Média', 'Mediana', 'DP', 'Mínimo', 'P25', 'P75', 'Máximo', 'IQR', 'CV (Med.)']
    # Garantir que só usamos colunas que existem
    final_latex_cols = [col for col in cols_for_latex if col in latex_df.columns]
    latex_df_subset = latex_df[final_latex_cols]

    # Gerar código LaTeX
    # Ajustar column_format conforme o número de colunas e o que você quer alinhar
    num_data_cols = len(final_latex_cols) - 2 # Métrica e Tam L0 são 'l'
    col_format = 'll|' + 'r' * num_data_cols

    latex_string = latex_df_subset.to_latex(
        index=False,
        escape=False,
        column_format=col_format,
        multicolumn_format='c', # Para headers de multicoluna, se usar
        caption=caption,
        label=label,
        header=True # Incluir o header
    )
    # Ajustes comuns
    latex_string = latex_string.replace("\\begin{tabular}", "\\centering\n\\begin{tabular}")
    latex_string = latex_string.replace("NaN", "-")
    latex_string = "\\begin{table}[htbp]\n" + latex_string + "\\end{table}\n"

    return latex_string