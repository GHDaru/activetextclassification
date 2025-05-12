# activetextclassification/visualization/ag_plots.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

def plot_ag_convergence(history_df, title_prefix="", l0_size=None, figsize=(10,8)):
    """
    Plota a evolução das métricas (max, avg, min para Acc e F1) ao longo das gerações do AG.
    """
    if history_df is None or history_df.empty:
        print(f"AVISO (plot_ag_convergence): Histórico para '{title_prefix}' está vazio. Pulando plot.")
        return

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    main_title = f"{title_prefix} - Evolução do AG"
    if l0_size: main_title += f" (L0 Tam: {l0_size})"
    fig.suptitle(main_title, fontsize=16)

    # Plot Acurácia
    if 'max_acc' in history_df.columns:
        axes[0].plot(history_df['generation'], history_df['max_acc'], label='Max Acc (Real)', marker='.')
        axes[0].plot(history_df['generation'], history_df['avg_acc'], label='Avg Acc (Real)', marker='.')
        axes[0].plot(history_df['generation'], history_df['min_acc'], label='Min Acc (Real)', linestyle='--', marker='.')
        axes[0].set_ylabel('Acurácia (Real)')
        axes[0].legend()
        axes[0].grid(True)
        axes[0].set_title('Evolução da Acurácia da População')

    # Plot F1-Score
    if 'max_f1' in history_df.columns:
        axes[1].plot(history_df['generation'], history_df['max_f1'], label='Max F1 (Real)', marker='.')
        axes[1].plot(history_df['generation'], history_df['avg_f1'], label='Avg F1 (Real)', marker='.')
        axes[1].plot(history_df['generation'], history_df['min_f1'], label='Min F1 (Real)', linestyle='--', marker='.')
        axes[1].set_ylabel('F1-Macro (Real)')
        axes[1].legend()
        axes[1].grid(True)
        axes[1].set_title('Evolução do F1-Score da População')

    axes[1].set_xlabel('Geração')
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Ajustar para super título
    plt.show()


def plot_ag_population_evolution_boxplots(
    detailed_fitness_log_df,
    metric_to_plot='accuracy_on_full', # ou 'f1_macro_on_full'
    title_prefix="",
    l0_size=None,
    figsize=(15, 7),
    generation_step=5 # Plotar boxplot a cada N gerações para não poluir
):
    """
    Plota a evolução da distribuição da performance da população do AG
    ao longo das gerações usando boxplots.

    Args:
        detailed_fitness_log_df (pd.DataFrame): DataFrame do log detalhado de fitness.
        metric_to_plot (str): Coluna da métrica a ser usada para os boxplots.
        title_prefix (str): Prefixo para o título do gráfico.
        l0_size (int, optional): Tamanho do L0 para incluir no título.
        figsize (tuple): Tamanho da figura.
        generation_step (int): Intervalo entre as gerações para plotar o boxplot.
    """
    if detailed_fitness_log_df is None or detailed_fitness_log_df.empty:
        print(f"AVISO (plot_ag_population_evolution_boxplots): Log detalhado para '{title_prefix}' está vazio. Pulando plot.")
        return
    if metric_to_plot not in detailed_fitness_log_df.columns:
        print(f"AVISO: Métrica '{metric_to_plot}' não encontrada no log detalhado. Pulando plot.")
        return

    # Converter métrica para numérico, tratando 'Error' como NaN
    plot_data_df = detailed_fitness_log_df.copy()
    plot_data_df[metric_to_plot] = pd.to_numeric(plot_data_df[metric_to_plot], errors='coerce')
    plot_data_df.dropna(subset=[metric_to_plot], inplace=True) # Remover linhas onde a métrica é NaN

    if plot_data_df.empty:
        print(f"AVISO: Sem dados válidos para a métrica '{metric_to_plot}' após limpeza. Pulando plot.")
        return

    # Selecionar gerações para plotar
    generations = sorted(plot_data_df['generation'].unique())
    gens_to_plot = generations[::generation_step]
    if generations[-1] not in gens_to_plot: # Garantir que a última geração seja plotada
        gens_to_plot.append(generations[-1])
    gens_to_plot = sorted(list(set(gens_to_plot))) # Remover duplicatas e ordenar

    plot_data_subset = plot_data_df[plot_data_df['generation'].isin(gens_to_plot)]

    if plot_data_subset.empty:
        print(f"AVISO: Sem dados após filtrar por generation_step para '{metric_to_plot}'. Pulando plot.")
        return

    plt.figure(figsize=figsize)
    sns.boxplot(data=plot_data_subset, x='generation', y=metric_to_plot, order=gens_to_plot)

    main_title = f"{title_prefix} - Evolução da População AG ({metric_to_plot})"
    if l0_size: main_title += f" (L0 Tam: {l0_size})"
    plt.title(main_title, fontsize=14)
    plt.xlabel('Geração')
    plt.ylabel(f'{metric_to_plot.replace("_on_full","").replace("_"," ").title()} (Real)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_ag_time_stats(optimization_history_df, title_prefix="", l0_size=None, figsize=(8,5)):
    """
    Plota estatísticas de tempo de execução por geração do AG.
    """
    if optimization_history_df is None or optimization_history_df.empty or \
       'generation_time_sec' not in optimization_history_df.columns:
        print(f"AVISO (plot_ag_time_stats): Histórico para '{title_prefix}' sem dados de tempo. Pulando plot.")
        return

    plt.figure(figsize=figsize)
    sns.boxplot(y=optimization_history_df['generation_time_sec'])
    # Ou: plt.plot(optimization_history_df['generation'], optimization_history_df['generation_time_sec'], marker='o')

    main_title = f"{title_prefix} - Tempo de Execução por Geração do AG"
    if l0_size: main_title += f" (L0 Tam: {l0_size})"
    plt.title(main_title, fontsize=14)
    plt.ylabel('Tempo da Geração (segundos)')
    # plt.xlabel('Geração') # Se usar plot de linha
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    print(f"Estatísticas do Tempo por Geração ({title_prefix}):")
    print(optimization_history_df['generation_time_sec'].describe())