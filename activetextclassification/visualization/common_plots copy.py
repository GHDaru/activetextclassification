# activetextclassification/visualization/common_plots.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns


def plot_performance_trend_lines_combined( # Nome ligeiramente alterado para clareza
    stats_df_long, # Espera DataFrame no formato LONGO com coluna 'Métrica'
    l0_size_column='l0_size',
    title="Tendência da Performance por Tamanho de L0",
    xlabel="Tamanho da Amostra Inicial (L0)",
    ylabel="Performance",
    figsize=(14, 7),
    use_log_scale_x=False,
    show_markers=True,
    plot_min_max_fill=True,
    y_min=0.0, y_max=1.01, # Default para métricas 0-1
    metrics_to_plot=None # Dicionário {'Nome Métrica no DF': 'Label Legenda'}
                        # Ex: {'Acurácia': 'Acurácia Média', 'F1-Macro': 'F1-Macro Médio'}
):
    """
    Plota linhas de tendência (Média) para múltiplas métricas (ex: Acurácia e F1)
    no mesmo gráfico, com bandas Min-Max opcionais.

    Args:
        stats_df_long (pd.DataFrame): DataFrame com estatísticas agregadas por l0_size
                                     e uma coluna 'Métrica' para diferenciar Acc/F1.
                                     Deve ter colunas 'Média', 'Mínimo', 'Máximo'.
        metrics_to_plot (dict, optional): Dicionário especificando quais métricas plotar
                                          e seus labels. Se None, tenta plotar 'Acurácia' e 'F1-Macro'.
    """
    print(f"\n--- Gerando Gráfico Combinado de Tendências ---")
    if stats_df_long is None or stats_df_long.empty:
        print("AVISO: DataFrame de estatísticas vazio.")
        return
    if l0_size_column not in stats_df_long.columns:
        print(f"AVISO: Coluna '{l0_size_column}' não encontrada.")
        return

    if metrics_to_plot is None:
        metrics_to_plot = {'Acurácia': 'Acurácia Média', 'F1-Macro': 'F1-Macro Médio'}

    plt.figure(figsize=figsize)
    marker_style = '.' if show_markers else None
    
    # Usar uma paleta de cores do Seaborn
    palette = sns.color_palette("husl", n_colors=len(metrics_to_plot))
    color_idx = 0

    for metric_name_in_df, legend_label_mean in metrics_to_plot.items():
        metric_data = stats_df_long[stats_df_long['Métrica'] == metric_name_in_df].sort_values(by=l0_size_column)
        if metric_data.empty:
            print(f"AVISO: Sem dados para a métrica '{metric_name_in_df}'.")
            continue

        color = palette[color_idx % len(palette)]
        color_idx += 1

        if 'Média' in metric_data.columns:
             plt.plot(metric_data[l0_size_column], metric_data['Média'],
                      label=legend_label_mean, marker=marker_style, linewidth=2, color=color)

        if plot_min_max_fill and 'Mínimo' in metric_data.columns and 'Máximo' in metric_data.columns:
             min_vals = metric_data['Mínimo'].fillna(method='ffill').fillna(method='bfill')
             max_vals = metric_data['Máximo'].fillna(method='ffill').fillna(method='bfill')
             plt.fill_between(metric_data[l0_size_column], min_vals, max_vals,
                              color=color, alpha=0.15, label=f'Intervalo Min-Max {metric_name_in_df}')
        elif 'Mínimo' in metric_data.columns: # Plotar linhas se não preencher
             plt.plot(metric_data[l0_size_column], metric_data['Mínimo'],
                      label=f'{metric_name_in_df} Mínima', marker=marker_style, linestyle=':', color=color, alpha=0.7)
        elif 'Máximo' in metric_data.columns:
             plt.plot(metric_data[l0_size_column], metric_data['Máximo'],
                      label=f'{metric_name_in_df} Máxima', marker=marker_style, linestyle=':', color=color, alpha=0.7)


    plt.title(title, fontsize=14)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)

    unique_x_values_plot = sorted(list(stats_df_long[l0_size_column].unique()))

    if use_log_scale_x:
        plt.xscale('log')
        # Ticks para escala logarítmica
        if unique_x_values_plot:
            log_ticks_candidates = [10, 20, 30, 50, 70, 100, 200, 300, 500, 700, 1000,
                                    2000, 3000, 5000, 7000, 10000, 20000, 50000, 100000, 200000]
            tick_positions_log = [t for t in log_ticks_candidates if t >= unique_x_values_plot[0] and t <= unique_x_values_plot[-1]]
            if not tick_positions_log or (tick_positions_log and tick_positions_log[-1] < unique_x_values_plot[-1] and unique_x_values_plot[-1] not in tick_positions_log):
                 tick_positions_log.append(unique_x_values_plot[-1])
            if tick_positions_log and tick_positions_log[0] > unique_x_values_plot[0] and unique_x_values_plot[0] not in tick_positions_log :
                 tick_positions_log.insert(0,unique_x_values_plot[0])
            tick_positions_log = sorted(list(set(tick_positions_log)))
            plt.xticks(tick_positions_log, rotation=45, ha='right')
            plt.gca().xaxis.set_major_formatter(mticker.ScalarFormatter())
            plt.gca().xaxis.get_major_formatter().set_scientific(False)
            plt.gca().tick_params(axis='x', which='minor', bottom=False)
    else: # Escala Linear
        plt.xticks(unique_x_values_plot, rotation=45, ha='right') # Mostrar todos os L0 sizes

    if y_min is not None or y_max is not None:
        plt.ylim(bottom=y_min if y_min is not None else plt.ylim()[0],
                   top=y_max if y_max is not None else plt.ylim()[1])
    else:
        plt.ylim(bottom=0)

    plt.tight_layout()
    plt.show()


def plot_filtered_evolutionary_boxplots(
    results_df, # DataFrame com os resultados BRUTOS das N_REPETITIONS
    metric_column,
    l0_size_column='l0_size',
    title_suffix="",
    xlabel="Tamanho da Amostra Inicial (L0)",
    ylabel=None,
    figsize=(15, 7),
    x_value_min=None,
    x_value_max=None,
    y_min=0.0, y_max=1.01, # Default para métricas 0-1
    width_factor=0.03, # Fator para largura dos boxplots em relação ao valor de L0
    min_width=5 # Largura mínima absoluta
):
    """
    Plota boxplots evolutivos para UMA métrica, filtrando por um intervalo de l0_size,
    e posicionando os boxplots numericamente no eixo X.
    """
    print(f"\n--- Gerando Boxplot Filtrado para: {metric_column} ---")
    if results_df is None or results_df.empty: print("AVISO (boxplot): DF vazio."); return
    if metric_column not in results_df.columns or l0_size_column not in results_df.columns:
        print(f"AVISO (boxplot): Coluna '{metric_column}' ou '{l0_size_column}' ausente."); return

    plot_data_filtered = results_df.copy()
    plot_data_filtered[metric_column] = pd.to_numeric(plot_data_filtered[metric_column], errors='coerce')

    # Aplicar filtros de intervalo
    if x_value_min is not None: plot_data_filtered = plot_data_filtered[plot_data_filtered[l0_size_column] >= x_value_min]
    if x_value_max is not None: plot_data_filtered = plot_data_filtered[plot_data_filtered[l0_size_column] <= x_value_max]
    plot_data_filtered.dropna(subset=[metric_column, l0_size_column], inplace=True)

    if plot_data_filtered.empty: print(f"AVISO (boxplot): Sem dados para '{metric_column}' após filtros."); return

    unique_l0_sizes_to_plot = sorted(plot_data_filtered[l0_size_column].unique())
    if not unique_l0_sizes_to_plot: print(f"AVISO (boxplot): Sem tamanhos L0 válidos para '{metric_column}' após filtros."); return

    data_for_matplotlib_boxplot = [
        plot_data_filtered[plot_data_filtered[l0_size_column] == size][metric_column].values
        for size in unique_l0_sizes_to_plot
    ]
    # Filtrar arrays vazios que podem surgir se algum l0_size não tiver dados válidos após dropna
    valid_data_bp = []
    valid_positions_bp = []
    for i, data_arr in enumerate(data_for_matplotlib_boxplot):
        if len(data_arr) > 0:
            valid_data_bp.append(data_arr)
            valid_positions_bp.append(unique_l0_sizes_to_plot[i])
    
    if not valid_data_bp: print(f"AVISO (boxplot): Nenhum dado válido para plotar após filtrar NaNs por grupo."); return


    plt.figure(figsize=figsize)
    # Calcular larguras dinâmicas baseadas nas posições
    # Se as posições são muito próximas, as larguras podem precisar ser menores
    widths_bp = np.array(valid_positions_bp) * width_factor + min_width
    # Para evitar sobreposição, a largura não deve ser maior que a menor diferença entre posições
    if len(valid_positions_bp) > 1:
        min_diff = np.min(np.diff(valid_positions_bp))
        widths_bp = np.clip(widths_bp, min_width, min_diff * 0.8 if min_diff > 0 else min_width)


    bp = plt.boxplot(valid_data_bp,
                     positions=valid_positions_bp, # Posições numéricas
                     widths=widths_bp,
                     manage_ticks=False, # Gerenciaremos os ticks
                     patch_artist=True,
                     medianprops=dict(color="red", linewidth=1.5),
                     showfliers=True) # Mostrar outliers

    final_ylabel = ylabel if ylabel else metric_column.replace('_on_population', '').replace('_', ' ').title()
    plt.title(f"Distribuição de {final_ylabel} por Tamanho de L0 (Filtrado)\n{title_suffix}", fontsize=14)
    plt.xlabel(xlabel)
    plt.ylabel(final_ylabel)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Ajustar ticks e limites do eixo X
    plt.xticks(valid_positions_bp, rotation=45, ha='right')
    if len(valid_positions_bp) > 1 :
        padding = (valid_positions_bp[-1] - valid_positions_bp[0]) * 0.05 # 5% de padding
        plt.xlim(valid_positions_bp[0] - padding, valid_positions_bp[-1] + padding)
    elif len(valid_positions_bp) == 1:
         plt.xlim(valid_positions_bp[0] - widths_bp[0], valid_positions_bp[0] + widths_bp[0])


    if y_min is not None or y_max is not None: plt.ylim(bottom=y_min, top=y_max)
    else: plt.ylim(bottom=0) # Default para performance

    # Colorir boxplots
    colors = sns.color_palette(palette, n_colors=len(valid_positions_bp))
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(colors[i % len(colors)])

    plt.tight_layout()
    plt.show()


# Manter plot_variability_trend como antes, pois ele já lida com múltiplas métricas e subplots
def plot_variability_trend(
    stats_df, variability_metrics, metric_names, l0_size_column='l0_size',
    title="Variabilidade da Performance por Tamanho de L0", xlabel="Tamanho L0",
    figsize=(12, 6), use_log_scale_x=True, show_markers=True, palette=None
):
    # ... (código igual ao da sua última versão para esta função) ...
    print(f"\n--- Gerando Gráfico de Tendência de Variabilidade para: {metric_names} ---")
    if stats_df is None or stats_df.empty: print("AVISO: DataFrame de estatísticas vazio."); return
    if not isinstance(metric_names, list): metric_names = [metric_names]
    if not isinstance(variability_metrics, list): variability_metrics = [variability_metrics]

    fig, axes = plt.subplots(len(variability_metrics), 1, figsize=figsize, sharex=True)
    if len(variability_metrics) == 1: axes = [axes]

    colors = sns.color_palette(palette if palette else "Set1", n_colors=len(metric_names))
    marker_style = '.' if show_markers else None

    for i, var_metric_col in enumerate(variability_metrics):
        ax = axes[i]; plot_data_for_var_metric = False
        for j, metric_base_name in enumerate(metric_names):
            metric_data = stats_df[stats_df['Métrica'] == metric_base_name].sort_values(by=l0_size_column)
            if not metric_data.empty and var_metric_col in metric_data.columns:
                plot_values = metric_data.dropna(subset=[var_metric_col])
                if not plot_values.empty:
                    ax.plot(plot_values[l0_size_column], plot_values[var_metric_col],
                            label=f'{var_metric_col} ({metric_base_name})',
                            marker=marker_style, color=colors[j % len(colors)])
                    plot_data_for_var_metric = True
        if plot_data_for_var_metric:
            ax.set_title(f'Evolução de {var_metric_col}'); ax.set_ylabel(var_metric_col)
            ax.legend(); ax.grid(True, linestyle='--', alpha=0.7); ax.set_ylim(bottom=0)
        else: ax.set_title(f'{var_metric_col} (Dados Indisponíveis)'); ax.text(0.5,0.5, "Dados não encontrados.", ha='center',va='center')

    axes[-1].set_xlabel(xlabel)
    if use_log_scale_x:
        axes[-1].set_xscale('log'); unique_x_values = sorted(stats_df[l0_size_column].unique())
        if unique_x_values:
            ticks = [t for t in [10,20,30,50,70,100,200,300,500,700,1000,2000,3000,5000,7000,10000,20000,50000,100000,200000] if t >= unique_x_values[0] and t <= unique_x_values[-1]]
            if not ticks or (ticks and ticks[-1] < unique_x_values[-1] and unique_x_values[-1] not in ticks): ticks.append(unique_x_values[-1])
            if ticks and ticks[0] > unique_x_values[0] and unique_x_values[0] not in ticks: ticks.insert(0,unique_x_values[0])
            ticks = sorted(list(set(ticks))) if ticks else unique_x_values
            axes[-1].set_xticks(ticks); axes[-1].set_xticklabels(ticks, rotation=45, ha='right')
            axes[-1].xaxis.set_major_formatter(mticker.ScalarFormatter()); axes[-1].xaxis.get_major_formatter().set_scientific(False)
            axes[-1].tick_params(axis='x', which='minor', bottom=False)
    else: axes[-1].tick_params(axis='x', rotation=45, ha='right')
    fig.suptitle(title, fontsize=16); plt.tight_layout(rect=[0, 0, 1, 0.95]); plt.show()

def plot_metric_boxplot_evolution(
    results_df,
    metric_column,
    l0_size_column='l0_size',
    title_suffix="",
    xlabel="Tamanho da Amostra Inicial (L0)",
    ylabel=None, # Será inferido da metric_column se None
    figsize=(15, 7),
    order=None,
    y_min=None, y_max=None,
    palette="pastel"
):
    """
    Plota um boxplot evolutivo para UMA métrica específica em função do l0_size.
    """
    if results_df is None or results_df.empty:
        print(f"AVISO (boxplot): DF vazio para {metric_column}.")
        return
    if metric_column not in results_df.columns or l0_size_column not in results_df.columns:
        print(f"AVISO (boxplot): Coluna '{metric_column}' ou '{l0_size_column}' ausente.")
        return

    plot_data = results_df.copy()
    plot_data[metric_column] = pd.to_numeric(plot_data[metric_column], errors='coerce')
    plot_data.dropna(subset=[metric_column, l0_size_column], inplace=True)
    if plot_data.empty:
        print(f"AVISO (boxplot): Sem dados válidos para '{metric_column}'.")
        return

    actual_order = order if order is not None else sorted(plot_data[l0_size_column].unique())
    if not actual_order:
        print(f"AVISO (boxplot): Sem tamanhos L0 válidos para '{metric_column}'.")
        return

    plt.figure(figsize=figsize)
    sns.boxplot(data=plot_data, x=l0_size_column, y=metric_column, order=actual_order, palette=palette)

    final_ylabel = ylabel if ylabel else metric_column.replace('_on_population', '').replace('_', ' ').title()
    plt.title(f"Distribuição de {final_ylabel} por Tamanho de L0\n{title_suffix}", fontsize=14)
    plt.xlabel(xlabel)
    plt.ylabel(final_ylabel)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    if y_min is not None or y_max is not None:
        ymin_val = y_min if y_min is not None else plot_data[metric_column].min() - 0.05 * (plot_data[metric_column].max() - plot_data[metric_column].min())
        ymax_val = y_max if y_max is not None else plot_data[metric_column].max() + 0.05 * (plot_data[metric_column].max() - plot_data[metric_column].min())
        # Garantir que ymin é pelo menos 0 para métricas de performance
        if final_ylabel.lower() in ["acurácia", "f1-macro", "performance"] and ymin_val < 0: ymin_val = 0
        plt.ylim(bottom=ymin_val, top=ymax_val)
    elif final_ylabel.lower() in ["acurácia", "f1-macro", "performance"]:
         plt.ylim(bottom=max(0, plot_data[metric_column].min() - 0.05), top=min(1.01, plot_data[metric_column].max() + 0.05) )


    plt.tight_layout()
    plt.show()


def plot_metric_lines_evolution(
    stats_df,
    metric_names, # Lista de nomes de métricas base, ex: ['Acurácia', 'F1-Macro']
    l0_size_column='l0_size',
    stat_types=['Média', 'Mediana', 'Mínimo', 'Máximo'], # Quais estatísticas plotar
    title="Tendência da Performance por Tamanho de L0",
    xlabel="Tamanho da Amostra Inicial (L0)",
    ylabel="Performance",
    figsize=(14, 7),
    use_log_scale_x=True,
    show_markers=True,
    plot_min_max_fill=True,
    y_min=None, y_max=None,
    palette=None # Dicionário opcional {metric_name: color}
):
    """
    Plota linhas de tendência (Max, Min, Média, Mediana) para uma ou duas métricas
    em função do tamanho de L0, no mesmo gráfico ou em subplots.
    Assume que stats_df tem colunas como 'MédiaAcurácia', 'MínimoF1-Macro', etc.
    OU que stats_df tem uma coluna 'Métrica' e as estatísticas são valores.
    VAMOS ASSUMIR O SEGUNDO FORMATO (longo) para stats_df, vindo de calculate_l0_sensitivity_stats.
    """
    print(f"\n--- Gerando Gráfico de Linhas de Tendência para: {metric_names} ---")
    if stats_df is None or stats_df.empty:
        print("AVISO: DataFrame de estatísticas vazio.")
        return
    if not isinstance(metric_names, list): metric_names = [metric_names]

    plt.figure(figsize=figsize)
    colors = sns.color_palette(palette if palette else "husl", n_colors=len(metric_names))

    for i, metric_base_name in enumerate(metric_names):
        metric_data = stats_df[stats_df['Métrica'] == metric_base_name].sort_values(by=l0_size_column)
        if metric_data.empty:
            print(f"AVISO: Sem dados para a métrica '{metric_base_name}'.")
            continue

        marker_style = '.' if show_markers else None
        color = colors[i % len(colors)] # Ciclar cores

        if 'Média' in metric_data.columns:
             plt.plot(metric_data[l0_size_column], metric_data['Média'], label=f'{metric_base_name} Média', marker=marker_style, linewidth=2, color=color)
        if 'Mediana' in metric_data.columns:
             plt.plot(metric_data[l0_size_column], metric_data['Mediana'], label=f'{metric_base_name} Mediana', marker=marker_style, linestyle='--', color=color)
        if 'Mínimo' in metric_data.columns and 'Máximo' in metric_data.columns and plot_min_max_fill:
             # Remover NaNs antes de fill_between
             min_vals = metric_data['Mínimo'].fillna(method='ffill').fillna(method='bfill')
             max_vals = metric_data['Máximo'].fillna(method='ffill').fillna(method='bfill')
             plt.fill_between(metric_data[l0_size_column], min_vals, max_vals, color=color, alpha=0.1, label=f'Intervalo Min-Max {metric_base_name}')
        elif 'Mínimo' in metric_data.columns: # Plotar linha se não for preencher
             plt.plot(metric_data[l0_size_column], metric_data['Mínimo'], label=f'{metric_base_name} Mínima', marker=marker_style, linestyle=':', color=color, alpha=0.7)
        elif 'Máximo' in metric_data.columns:
             plt.plot(metric_data[l0_size_column], metric_data['Máximo'], label=f'{metric_base_name} Máxima', marker=marker_style, linestyle=':', color=color, alpha=0.7)


    plt.title(title, fontsize=14)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    if use_log_scale_x:
        plt.xscale('log')
        # Tentar obter todos os tamanhos L0 únicos de todas as métricas plotadas para os ticks
        all_l0_sizes_for_ticks = sorted(list(stats_df[l0_size_column].unique()))
        if all_l0_sizes_for_ticks:
            tick_positions_log = all_l0_sizes_for_ticks
            if len(all_l0_sizes_for_ticks) > 12: # Simplificar se muitos ticks
                log_ticks_candidates = [t for t in [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000] if t >= all_l0_sizes_for_ticks[0] and t <= all_l0_sizes_for_ticks[-1]]
                if not log_ticks_candidates or (log_ticks_candidates and log_ticks_candidates[-1] < all_l0_sizes_for_ticks[-1] and all_l0_sizes_for_ticks[-1] not in log_ticks_candidates) : log_ticks_candidates.append(all_l0_sizes_for_ticks[-1])
                if log_ticks_candidates and log_ticks_candidates[0] > all_l0_sizes_for_ticks[0] and all_l0_sizes_for_ticks[0] not in log_ticks_candidates : log_ticks_candidates.insert(0,all_l0_sizes_for_ticks[0])
                tick_positions_log = sorted(list(set(log_ticks_candidates))) if log_ticks_candidates else all_l0_sizes_for_ticks

            plt.xticks(tick_positions_log, rotation=45, ha='right')
            plt.gca().xaxis.set_major_formatter(mticker.ScalarFormatter())
            plt.gca().xaxis.get_major_formatter().set_scientific(False)
            plt.gca().tick_params(axis='x', which='minor', bottom=False)
    else:
        plt.xticks(rotation=45, ha='right')

    if y_min is not None or y_max is not None:
        plt.ylim(bottom=y_min if y_min is not None else plt.ylim()[0],
                   top=y_max if y_max is not None else plt.ylim()[1])
    elif ylabel.lower() in ["performance", "acurácia", "f1-macro"]:
         plt.ylim(bottom=0) # Garantir que o eixo Y comece em 0 para performance

    plt.tight_layout()
    plt.show()


def plot_variability_trend(
    stats_df, # DataFrame vindo de calculate_l0_sensitivity_stats (formato longo)
    variability_metrics, # Lista de colunas de variabilidade, ex: ['DesvioPadrão', 'CV (Mediana)']
    metric_names, # Lista de nomes de métricas base, ex: ['Acurácia', 'F1-Macro']
    l0_size_column='l0_size',
    title="Variabilidade da Performance por Tamanho de L0",
    xlabel="Tamanho da Amostra Inicial (L0)",
    figsize=(12, 6), # Ajustar altura baseada no nº de métricas de variação
    use_log_scale_x=True,
    show_markers=True,
    palette=None
):
    """
    Plota a tendência das métricas de variabilidade (DP, CV) para Acurácia e F1-Score.
    Cria um subplot para cada métrica de variabilidade.
    """
    print(f"\n--- Gerando Gráfico de Tendência de Variabilidade para: {metric_names} ---")
    if stats_df is None or stats_df.empty:
        print("AVISO: DataFrame de estatísticas vazio.")
        return
    if not isinstance(metric_names, list): metric_names = [metric_names]
    if not isinstance(variability_metrics, list): variability_metrics = [variability_metrics]

    fig, axes = plt.subplots(len(variability_metrics), 1, figsize=figsize, sharex=True)
    if len(variability_metrics) == 1: axes = [axes] # Garantir que axes é iterável

    colors = sns.color_palette(palette if palette else "Set1", n_colors=len(metric_names))
    marker_style = '.' if show_markers else None

    for i, var_metric_col in enumerate(variability_metrics):
        ax = axes[i]
        plot_data_for_var_metric = False
        for j, metric_base_name in enumerate(metric_names):
            # Filtrar o stats_df para a métrica base (Acurácia ou F1-Macro)
            metric_data = stats_df[stats_df['Métrica'] == metric_base_name].sort_values(by=l0_size_column)
            if not metric_data.empty and var_metric_col in metric_data.columns:
                # Remover NaNs da coluna de variabilidade
                plot_values = metric_data.dropna(subset=[var_metric_col])
                if not plot_values.empty:
                    ax.plot(plot_values[l0_size_column], plot_values[var_metric_col],
                            label=f'{var_metric_col} ({metric_base_name})',
                            marker=marker_style, color=colors[j % len(colors)])
                    plot_data_for_var_metric = True
            else:
                print(f"AVISO: Sem dados para '{var_metric_col}' da métrica '{metric_base_name}'.")

        if plot_data_for_var_metric:
            ax.set_title(f'Evolução de {var_metric_col}')
            ax.set_ylabel(var_metric_col)
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.set_ylim(bottom=0) # DP e CV geralmente não são negativos
        else:
            ax.set_title(f'{var_metric_col} (Dados Indisponíveis)')
            ax.text(0.5,0.5, f"Dados para '{var_metric_col}' não encontrados.", ha='center', va='center')


    # Configurar eixo X para o último subplot (ou todos se sharex=False)
    axes[-1].set_xlabel(xlabel)
    if use_log_scale_x:
        axes[-1].set_xscale('log')
        unique_x_values = sorted(stats_df[l0_size_column].unique())
        if unique_x_values:
            tick_positions_log = unique_x_values
            if len(unique_x_values) > 12:
                log_ticks_candidates = [t for t in [10,20,50,100,200,500,1000,2000,5000,10000,20000,50000,100000,200000] if t >= unique_x_values[0] and t <= unique_x_values[-1]]
                if not log_ticks_candidates or (log_ticks_candidates and log_ticks_candidates[-1] < unique_x_values[-1] and unique_x_values[-1] not in log_ticks_candidates): log_ticks_candidates.append(unique_x_values[-1])
                if log_ticks_candidates and log_ticks_candidates[0] > unique_x_values[0] and unique_x_values[0] not in log_ticks_candidates : log_ticks_candidates.insert(0,unique_x_values[0])
                tick_positions_log = sorted(list(set(log_ticks_candidates))) if log_ticks_candidates else unique_x_values
            axes[-1].set_xticks(tick_positions_log)
            axes[-1].set_xticklabels(tick_positions_log, rotation=45, ha='right')
            axes[-1].xaxis.set_major_formatter(mticker.ScalarFormatter())
            axes[-1].xaxis.get_major_formatter().set_scientific(False)
            axes[-1].tick_params(axis='x', which='minor', bottom=False)
    else:
        axes[-1].tick_params(axis='x', rotation=45, ha='right')

    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# activetextclassification/visualization/ag_plots.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob # Removido, não usado diretamente nas funções de plotagem aqui

# Manter plot_ag_convergence se você ainda a usa para o histórico do AG
# Se não, pode remover ou adaptar. Vamos focar nos novos plots.

def plot_evolutionary_boxplots(
    results_df,
    metric_column,
    l0_size_column='l0_size',
    title="Distribuição da Performance por Tamanho de L0",
    xlabel="Tamanho da Amostra Inicial (L0)",
    ylabel="Performance",
    figsize=(15, 7),
    x_min=None, x_max=None, # Para definir limites do eixo X se necessário
    y_min=None, y_max=None, # Para definir limites do eixo Y
    palette="viridis", # Paleta de cores Seaborn
    order=None # Ordem explícita das categorias no eixo X
):
    """
    Plota boxplots evolutivos da performance (métrica) em função do tamanho de L0.
    Os boxplots são posicionados de acordo com os valores numéricos de l0_size.

    Args:
        results_df (pd.DataFrame): DataFrame com os resultados, contendo l0_size_column e metric_column.
        metric_column (str): Nome da coluna com a métrica de performance a ser plotada.
        l0_size_column (str): Nome da coluna com os tamanhos de L0.
        title (str): Título do gráfico.
        xlabel (str): Rótulo do eixo X.
        ylabel (str): Rótulo do eixo Y.
        figsize (tuple): Tamanho da figura.
        x_min (float, optional): Limite mínimo para o eixo X.
        x_max (float, optional): Limite máximo para o eixo X.
        y_min (float, optional): Limite mínimo para o eixo Y.
        y_max (float, optional): Limite máximo para o eixo Y.
        palette (str or list, optional): Paleta de cores para Seaborn.
        order (list, optional): Ordem específica para as categorias do eixo X (l0_size).
    """
    print(f"\n--- Gerando Boxplot Evolutivo para: {metric_column} ---")
    if results_df is None or results_df.empty:
        print("AVISO: DataFrame de resultados vazio. Não é possível plotar.")
        return
    if metric_column not in results_df.columns:
        print(f"AVISO: Coluna da métrica '{metric_column}' não encontrada. Não é possível plotar.")
        return
    if l0_size_column not in results_df.columns:
        print(f"AVISO: Coluna de tamanho L0 '{l0_size_column}' não encontrada. Não é possível plotar.")
        return

    # Garantir que a métrica é numérica e remover NaNs
    plot_data = results_df.copy()
    plot_data[metric_column] = pd.to_numeric(plot_data[metric_column], errors='coerce')
    plot_data.dropna(subset=[metric_column, l0_size_column], inplace=True)

    if plot_data.empty:
        print(f"AVISO: Sem dados válidos para '{metric_column}' após limpeza. Não é possível plotar.")
        return

    unique_l0_sizes = sorted(plot_data[l0_size_column].unique())
    if not unique_l0_sizes:
        print(f"AVISO: Sem tamanhos de L0 únicos válidos para '{metric_column}'.")
        return

    plt.figure(figsize=figsize)

    # Usar Seaborn para melhor estética, e order para os tamanhos
    # O Seaborn tratará x como categórico se order for fornecido,
    # mas a ordem será a especificada.
    # Se o espaçamento proporcional for CRÍTICO, matplotlib.pyplot.boxplot com 'positions' é melhor.
    # Vamos manter Seaborn por simplicidade e boa aparência, aceitando o espaçamento uniforme
    # entre as categorias ordenadas de l0_size.
    sns.boxplot(data=plot_data, x=l0_size_column, y=metric_column, order=order if order else unique_l0_sizes, palette=palette)

    plt.title(title, fontsize=14)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    if y_min is not None or y_max is not None:
        current_ymin, current_ymax = plt.ylim()
        plt.ylim(bottom=y_min if y_min is not None else current_ymin,
                   top=y_max if y_max is not None else current_ymax)

    # x_min e x_max são mais difíceis de controlar com boxplot categórico do seaborn.
    # Se for necessário, teria que filtrar os dados antes.

    plt.tight_layout()
    plt.show()


def plot_performance_trend_lines(
    stats_df,
    metric_name, # Ex: "Acurácia" ou "F1-Macro"
    l0_size_column='l0_size',
    mean_col='Média', median_col='Mediana', min_col='Mínimo', max_col='Máximo',
    title="Tendência da Performance por Tamanho de L0",
    xlabel="Tamanho da Amostra Inicial (L0)",
    ylabel="Performance",
    figsize=(12, 7),
    use_log_scale_x=False,
    show_markers=True,
    plot_min_max_fill=True,
    y_min=None, y_max=None
):
    """
    Plota linhas de tendência (max, min, média, mediana) da performance
    em função do tamanho de L0.

    Args:
        stats_df (pd.DataFrame): DataFrame com estatísticas agregadas por l0_size.
                                 Deve conter colunas para l0_size, mean, median, min, max.
        metric_name (str): Nome da métrica base (para títulos e legendas).
        l0_size_column (str): Nome da coluna com os tamanhos de L0.
        mean_col, median_col, min_col, max_col (str): Nomes das colunas de estatísticas.
        title (str): Título do gráfico.
        xlabel (str): Rótulo do eixo X.
        ylabel (str): Rótulo do eixo Y.
        figsize (tuple): Tamanho da figura.
        use_log_scale_x (bool): Se True, usa escala logarítmica para o eixo X.
        show_markers (bool): Se True, mostra marcadores nos pontos de dados.
        plot_min_max_fill (bool): Se True, preenche a área entre Min e Max.
        y_min (float, optional): Limite mínimo para o eixo Y.
        y_max (float, optional): Limite máximo para o eixo Y.
    """
    print(f"\n--- Gerando Gráfico de Linhas de Tendência para: {metric_name} ---")
    if stats_df is None or stats_df.empty:
        print("AVISO: DataFrame de estatísticas vazio. Não é possível plotar.")
        return

    # Validar colunas
    required_cols = [l0_size_column, mean_col, median_col, min_col, max_col]
    if not all(col in stats_df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in stats_df.columns]
        print(f"AVISO: Colunas necessárias ausentes no DataFrame de estatísticas: {missing}. Não é possível plotar.")
        return

    # Ordenar por l0_size para plotagem correta
    plot_data = stats_df.sort_values(by=l0_size_column).copy()
    # Remover NaNs para evitar problemas com fill_between
    plot_data.dropna(subset=[min_col, max_col], inplace=True)


    plt.figure(figsize=figsize)
    marker_style = '.' if show_markers else None

    plt.plot(plot_data[l0_size_column], plot_data[max_col], label=f'{metric_name} Máxima', marker=marker_style, linestyle=':')
    plt.plot(plot_data[l0_size_column], plot_data[mean_col], label=f'{metric_name} Média', marker=marker_style, linewidth=2)
    plt.plot(plot_data[l0_size_column], plot_data[median_col], label=f'{metric_name} Mediana', marker=marker_style, linestyle='--')
    plt.plot(plot_data[l0_size_column], plot_data[min_col], label=f'{metric_name} Mínima', marker=marker_style, linestyle=':')

    if plot_min_max_fill and not plot_data.empty:
        plt.fill_between(plot_data[l0_size_column], plot_data[min_col], plot_data[max_col], alpha=0.15, label=f'Intervalo Min-Max {metric_name}')

    plt.title(title, fontsize=14)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    if use_log_scale_x:
        plt.xscale('log')
        # Ajustar ticks para escala log se necessário, ou deixar automático
        unique_x_values = sorted(plot_data[l0_size_column].unique())
        tick_positions_log = unique_x_values
        if len(unique_x_values) > 10: # Simplificar se muitos ticks
            # Selecionar ticks que são potências de 10 ou intermediários significativos
            log_ticks = [t for t in [10, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000] if t >= unique_x_values[0] and t <= unique_x_values[-1]]
            if not log_ticks or log_ticks[-1] < unique_x_values[-1]: log_ticks.append(unique_x_values[-1])
            if log_ticks[0] > unique_x_values[0] and unique_x_values[0] not in log_ticks : log_ticks.insert(0,unique_x_values[0])
            tick_positions_log = sorted(list(set(log_ticks)))

        plt.xticks(tick_positions_log, rotation=45, ha='right')
        # Para formatar os ticks em escala log como números normais:
        # from matplotlib.ticker import ScalarFormatter
        # plt.gca().xaxis.set_major_formatter(ScalarFormatter())
    else:
        plt.xticks(rotation=45, ha='right')


    if y_min is not None or y_max is not None:
        current_ymin, current_ymax = plt.ylim()
        plt.ylim(bottom=y_min if y_min is not None else current_ymin,
                   top=y_max if y_max is not None else current_ymax)

    plt.tight_layout()
    plt.show()


def plot_std_dev_trend(
    stats_df,
    metric_name, # "Acurácia" ou "F1-Macro"
    l0_size_column='l0_size',
    std_dev_col='DesvioPadrão', # Ou 'CV (Mediana)'
    title="Evolução do Desvio Padrão da Performance por Tamanho de L0",
    xlabel="Tamanho da Amostra Inicial (L0)",
    ylabel="Desvio Padrão da Performance",
    figsize=(10, 6),
    use_log_scale_x=False,
    show_markers=True,
    y_min=None, y_max=None
):
    """
    Plota a tendência do desvio padrão (ou CV) da performance
    em função do tamanho de L0.
    """
    print(f"\n--- Gerando Gráfico de Tendência do Desvio Padrão para: {metric_name} ---")
    # Filtrar para a métrica específica
    metric_specific_stats_df = stats_df[stats_df['Métrica'] == metric_name].copy()

    if metric_specific_stats_df is None or metric_specific_stats_df.empty:
        print(f"AVISO: DataFrame de estatísticas vazio para a métrica '{metric_name}'. Não é possível plotar.")
        return
    if std_dev_col not in metric_specific_stats_df.columns:
        print(f"AVISO: Coluna de desvio padrão '{std_dev_col}' não encontrada. Não é possível plotar.")
        return

    plot_data = metric_specific_stats_df.sort_values(by=l0_size_column).copy()
    plot_data.dropna(subset=[std_dev_col], inplace=True)

    if plot_data.empty:
        print(f"AVISO: Sem dados válidos para desvio padrão de '{metric_name}' após limpeza. Não é possível plotar.")
        return

    plt.figure(figsize=figsize)
    marker_style = '.' if show_markers else None

    plt.plot(plot_data[l0_size_column], plot_data[std_dev_col], label=f'Desvio Padrão ({metric_name})', marker=marker_style)

    plt.title(title, fontsize=14)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    if use_log_scale_x:
        plt.xscale('log')
        unique_x_values = sorted(plot_data[l0_size_column].unique())
        tick_positions_log = unique_x_values # Default
        if len(unique_x_values) > 10:
            log_ticks = [t for t in [10, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000] if t >= unique_x_values[0] and t <= unique_x_values[-1]]
            if not log_ticks or (log_ticks and log_ticks[-1] < unique_x_values[-1]): log_ticks.append(unique_x_values[-1])
            if log_ticks and log_ticks[0] > unique_x_values[0] and unique_x_values[0] not in log_ticks : log_ticks.insert(0,unique_x_values[0])
            tick_positions_log = sorted(list(set(log_ticks))) if log_ticks else unique_x_values
        plt.xticks(tick_positions_log, rotation=45, ha='right')
    else:
        plt.xticks(rotation=45, ha='right')

    if y_min is not None or y_max is not None:
        current_ymin, current_ymax = plt.ylim()
        plt.ylim(bottom=y_min if y_min is not None else current_ymin,
                   top=y_max if y_max is not None else current_ymax)
    else: # Garantir que o eixo Y comece em 0 para desvio padrão
        plt.ylim(bottom=0)


    plt.tight_layout()
    plt.show()



# Função plot_evolutionary_boxplots pode ser removida ou mantida se usada em outro lugar.
# Vamos criar uma nova para os intervalos.

def plot_interval_boxplots(
    results_df,
    intervals, # Lista de tuplas, ex: [(10, 100), (101, 1000), ...]
    interval_names, # Lista de nomes para os intervalos no eixo X
    metric_columns=['accuracy_on_population', 'f1_macro_on_population'],
    metric_names=['Acurácia', 'F1-Macro'],
    l0_size_column='l0_size',
    title="Distribuição da Performance por Intervalo de Tamanho de L0",
    xlabel="Intervalo de Tamanho de L0",
    ylabel="Performance",
    figsize=(12, 7),
    palette="Set2"
):
    """
    Plota boxplots da performance para diferentes métricas, agrupados por intervalos de l0_size.
    Cada intervalo no eixo X terá N boxplots (um para cada métrica).
    """
    print(f"\n--- Gerando Boxplots por Intervalo para Métricas: {metric_names} ---")
    if results_df is None or results_df.empty:
        print("AVISO: DataFrame de resultados vazio.")
        return
    if not all(mc in results_df.columns for mc in metric_columns):
        print(f"AVISO: Nem todas as colunas de métrica ({metric_columns}) encontradas.")
        return
    if l0_size_column not in results_df.columns:
        print(f"AVISO: Coluna '{l0_size_column}' não encontrada.")
        return
    if len(intervals) != len(interval_names):
        print("AVISO: Número de intervalos e nomes de intervalos não coincidem.")
        return

    plot_data_list = []
    for i, (start, end) in enumerate(intervals):
        interval_df = results_df[
            (results_df[l0_size_column] >= start) & (results_df[l0_size_column] <= end)
        ].copy()
        if not interval_df.empty:
            for mc_idx, mc in enumerate(metric_columns):
                metric_data = pd.to_numeric(interval_df[mc], errors='coerce').dropna()
                if not metric_data.empty:
                    for val in metric_data:
                        plot_data_list.append({
                            'Intervalo': interval_names[i],
                            'Métrica': metric_names[mc_idx],
                            'Valor': val
                        })

    if not plot_data_list:
        print("Nenhum dado válido para plotar nos intervalos especificados.")
        return

    plot_df_melted = pd.DataFrame(plot_data_list)

    plt.figure(figsize=figsize)
    sns.boxplot(data=plot_df_melted, x='Intervalo', y='Valor', hue='Métrica', palette=palette, order=interval_names)

    plt.title(title, fontsize=14)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=15, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Métrica')
    plt.tight_layout()
    plt.show()


def plot_combined_performance_trends(
    stats_df, # DataFrame vindo da Célula 5 do notebook (full_stats_df)
    l0_size_column='l0_size',
    acc_metric_prefix='Acurácia', # Prefixo para colunas de Acurácia
    f1_metric_prefix='F1-Macro',   # Prefixo para colunas de F1
    title="Tendência da Performance (Acurácia e F1-Score) por Tamanho de L0",
    xlabel="Tamanho da Amostra Inicial (L0)",
    ylabel="Performance",
    figsize=(14, 7),
    use_log_scale_x=True,
    show_markers=True,
    y_min=None, y_max=None
):
    """
    Plota linhas de tendência (Média) para Acurácia e F1-Score no mesmo gráfico.
    Opcionalmente, pode incluir bandas Min-Max.
    """
    print(f"\n--- Gerando Gráfico Combinado de Tendências (Acurácia e F1) ---")
    if stats_df is None or stats_df.empty:
        print("AVISO: DataFrame de estatísticas vazio.")
        return

    stats_acc_df = stats_df[stats_df['Métrica'] == acc_metric_prefix].sort_values(by=l0_size_column)
    stats_f1_df = stats_df[stats_df['Métrica'] == f1_metric_prefix].sort_values(by=l0_size_column)

    if stats_acc_df.empty and stats_f1_df.empty:
        print("AVISO: Sem dados de estatísticas para Acurácia ou F1-Score.")
        return

    plt.figure(figsize=figsize)
    marker_style = '.' if show_markers else None

    # Plot Acurácia
    if not stats_acc_df.empty:
        plt.plot(stats_acc_df[l0_size_column], stats_acc_df['Média'], label=f'{acc_metric_prefix} Média', marker=marker_style, linewidth=2, color='blue')
        if 'Mínimo' in stats_acc_df.columns and 'Máximo' in stats_acc_df.columns:
            plt.fill_between(stats_acc_df[l0_size_column], stats_acc_df['Mínimo'], stats_acc_df['Máximo'], color='blue', alpha=0.1, label=f'Intervalo Min-Max {acc_metric_prefix}')

    # Plot F1-Score
    if not stats_f1_df.empty:
        plt.plot(stats_f1_df[l0_size_column], stats_f1_df['Média'], label=f'{f1_metric_prefix} Média', marker=marker_style, linewidth=2, color='green')
        if 'Mínimo' in stats_f1_df.columns and 'Máximo' in stats_f1_df.columns:
            plt.fill_between(stats_f1_df[l0_size_column], stats_f1_df['Mínimo'], stats_f1_df['Máximo'], color='green', alpha=0.1, label=f'Intervalo Min-Max {f1_metric_prefix}')


    plt.title(title, fontsize=14)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    if use_log_scale_x:
        plt.xscale('log')
        # Ajustar ticks para escala log
        combined_l0_sizes = sorted(list(set(stats_acc_df[l0_size_column].tolist() + stats_f1_df[l0_size_column].tolist())))
        if combined_l0_sizes:
            tick_positions_log = combined_l0_sizes
            if len(combined_l0_sizes) > 12:
                log_ticks_candidates = [10, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]
                tick_positions_log = [t for t in log_ticks_candidates if t >= combined_l0_sizes[0] and t <= combined_l0_sizes[-1]]
                if not tick_positions_log or (tick_positions_log and tick_positions_log[-1] < combined_l0_sizes[-1]):
                    tick_positions_log.append(combined_l0_sizes[-1])
                if tick_positions_log and tick_positions_log[0] > combined_l0_sizes[0] and combined_l0_sizes[0] not in tick_positions_log :
                    tick_positions_log.insert(0, combined_l0_sizes[0])
                tick_positions_log = sorted(list(set(tick_positions_log)))
            plt.xticks(tick_positions_log, rotation=45, ha='right')
            plt.gca().xaxis.set_major_formatter(mticker.ScalarFormatter()) # Formatar como números normais
            plt.gca().xaxis.get_major_formatter().set_scientific(False)    # Evitar notação científica
            plt.gca().tick_params(axis='x', which='minor', bottom=False)   # Remover ticks menores se houver
    else:
        plt.xticks(rotation=45, ha='right')


    if y_min is not None or y_max is not None:
        plt.ylim(bottom=y_min if y_min is not None else plt.ylim()[0],
                   top=y_max if y_max is not None else plt.ylim()[1])
    else: # Garantir que começa em 0 se for métrica de performance
        if ylabel.lower() in ["performance", "acurácia", "f1-macro"]:
            plt.ylim(bottom=0)


    plt.tight_layout()
    plt.show()


def plot_variability_metrics_trend(
    stats_df,
    l0_size_column='l0_size',
    acc_std_col='DesvioPadrãoAcc', # Nome esperado da coluna de DP da Acurácia
    f1_std_col='DesvioPadrãoF1',   # Nome esperado da coluna de DP do F1
    acc_cv_col='CV_Acc(Mediana)',   # Nome esperado da coluna de CV da Acurácia
    f1_cv_col='CV_F1(Mediana)',    # Nome esperado da coluna de CV do F1
    title="Variabilidade da Performance por Tamanho de L0",
    xlabel="Tamanho da Amostra Inicial (L0)",
    figsize=(14, 10),
    use_log_scale_x=True,
    show_markers=True
):
    """
    Plota a tendência do Desvio Padrão e do Coeficiente de Variação
    para Acurácia e F1-Score em função do tamanho de L0.
    Requer que stats_df tenha colunas separadas para DP e CV de cada métrica.
    """
    print(f"\n--- Gerando Gráfico de Variabilidade (DP e CV) ---")
    if stats_df is None or stats_df.empty:
        print("AVISO: DataFrame de estatísticas vazio.")
        return

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    marker_style = '.' if show_markers else None

    # Plot Desvio Padrão
    plot_dp = False
    if acc_std_col in stats_df.columns:
        axes[0].plot(stats_df[l0_size_column], stats_df[acc_std_col], label=f'DP Acurácia', marker=marker_style, color='royalblue')
        plot_dp = True
    if f1_std_col in stats_df.columns:
        axes[0].plot(stats_df[l0_size_column], stats_df[f1_std_col], label=f'DP F1-Macro', marker=marker_style, color='darkorange')
        plot_dp = True
    
    if plot_dp:
        axes[0].set_title('Desvio Padrão da Performance')
        axes[0].set_ylabel('Desvio Padrão')
        axes[0].legend()
        axes[0].grid(True, linestyle='--', alpha=0.7)
        axes[0].set_ylim(bottom=0) # DP não é negativo
    else:
        axes[0].set_title('Desvio Padrão (Dados Indisponíveis)')
        axes[0].text(0.5,0.5, "Dados de Desvio Padrão não encontrados.", ha='center', va='center')


    # Plot Coeficiente de Variação
    plot_cv = False
    if acc_cv_col in stats_df.columns:
        axes[1].plot(stats_df[l0_size_column], stats_df[acc_cv_col], label=f'CV Acurácia (Mediana)', marker=marker_style, color='forestgreen')
        plot_cv = True
    if f1_cv_col in stats_df.columns:
        axes[1].plot(stats_df[l0_size_column], stats_df[f1_cv_col], label=f'CV F1-Macro (Mediana)', marker=marker_style, color='firebrick')
        plot_cv = True

    if plot_cv:
        axes[1].set_title('Coeficiente de Variação da Performance (baseado na Mediana)')
        axes[1].set_ylabel('Coeficiente de Variação')
        axes[1].legend()
        axes[1].grid(True, linestyle='--', alpha=0.7)
        axes[1].set_ylim(bottom=0) # CV (performance/DP) geralmente positivo
    else:
        axes[1].set_title('Coeficiente de Variação (Dados Indisponíveis)')
        axes[1].text(0.5,0.5, "Dados de Coeficiente de Variação não encontrados.", ha='center', va='center')


    axes[1].set_xlabel(xlabel)
    if use_log_scale_x:
        axes[1].set_xscale('log')
        # Ajustar ticks para escala log
        unique_x_values = sorted(stats_df[l0_size_column].unique())
        if unique_x_values:
            tick_positions_log = unique_x_values
            if len(unique_x_values) > 12:
                log_ticks_candidates = [10, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]
                tick_positions_log = [t for t in log_ticks_candidates if t >= unique_x_values[0] and t <= unique_x_values[-1]]
                if not tick_positions_log or (tick_positions_log and tick_positions_log[-1] < unique_x_values[-1]): tick_positions_log.append(unique_x_values[-1])
                if tick_positions_log and tick_positions_log[0] > unique_x_values[0] and unique_x_values[0] not in tick_positions_log : tick_positions_log.insert(0,unique_x_values[0])
                tick_positions_log = sorted(list(set(tick_positions_log)))
            axes[1].set_xticks(tick_positions_log)
            axes[1].set_xticklabels(tick_positions_log, rotation=45, ha='right')
            axes[1].xaxis.set_major_formatter(mticker.ScalarFormatter())
            axes[1].xaxis.get_major_formatter().set_scientific(False)
            axes[1].tick_params(axis='x', which='minor', bottom=False)
    else:
        axes[1].tick_params(axis='x', rotation=45, ha='right')


    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def plot_grouped_interval_boxplots(
    results_df,
    intervals, # Lista de tuplas (min_l0, max_l0)
    interval_names, # Lista de nomes para os intervalos
    metric_info, # Dicionário como {'accuracy_on_population': 'Acurácia', 'f1_macro_on_population': 'F1-Macro'}
    l0_size_column='l0_size',
    title="Distribuição da Performance por Intervalos de Tamanho de L0",
    xlabel="Intervalo de Tamanho de L0",
    ylabel="Performance",
    figsize=(12, 8),
    palette="pastel",
    y_min=None, # Opcional: Mínimo para eixo Y
    y_max=None  # Opcional: Máximo para eixo Y
):
    """
    Plota boxplots agrupados para múltiplas métricas dentro de intervalos definidos de l0_size.

    Args:
        results_df (pd.DataFrame): DataFrame com os resultados.
        intervals (list of tuples): Lista de tuplas definindo os limites (min, max) de cada intervalo de l0_size.
        interval_names (list of str): Nomes correspondentes para cada intervalo (para o eixo X).
        metric_info (dict): Dicionário mapeando nomes de colunas de métrica para nomes amigáveis.
                            Ex: {'accuracy_on_population': 'Acurácia', 'f1_macro_on_population': 'F1-Macro'}
        l0_size_column (str): Nome da coluna com os tamanhos de L0.
        title (str): Título do gráfico.
        xlabel (str): Rótulo do eixo X.
        ylabel (str): Rótulo do eixo Y.
        figsize (tuple): Tamanho da figura.
        palette (str or list): Paleta de cores Seaborn.
        y_min (float, optional): Valor mínimo para o eixo Y.
        y_max (float, optional): Valor máximo para o eixo Y.
    """
    print(f"\n--- Gerando Boxplots Agrupados por Intervalo para: {list(metric_info.values())} ---")
    if results_df is None or results_df.empty:
        print("AVISO: DataFrame de resultados vazio.")
        return
    if not all(mc in results_df.columns for mc in metric_info.keys()):
        missing_cols = [mc for mc in metric_info.keys() if mc not in results_df.columns]
        print(f"AVISO: Colunas de métrica ausentes: {missing_cols}.")
        return
    if l0_size_column not in results_df.columns:
        print(f"AVISO: Coluna '{l0_size_column}' não encontrada.")
        return
    if len(intervals) != len(interval_names):
        print("AVISO: Número de intervalos e nomes de intervalos não coincidem.")
        return

    # Criar uma coluna 'Intervalo' no DataFrame
    # e preparar dados para melting
    plot_data_all_metrics = []

    for i, (start, end) in enumerate(intervals):
        # Selecionar dados dentro do intervalo atual
        interval_mask = (results_df[l0_size_column] >= start) & (results_df[l0_size_column] <= end)
        df_current_interval = results_df[interval_mask].copy()

        if not df_current_interval.empty:
            for metric_col, metric_name in metric_info.items():
                # Converter para numérico e remover NaNs para esta métrica
                metric_values = pd.to_numeric(df_current_interval[metric_col], errors='coerce').dropna()
                if not metric_values.empty:
                    for val in metric_values:
                        plot_data_all_metrics.append({
                            'Intervalo': interval_names[i], # Nome do intervalo para o eixo X
                            'Métrica': metric_name,        # Para o 'hue'
                            'Valor': val
                        })
    if not plot_data_all_metrics:
        print("Nenhum dado válido encontrado para os intervalos e métricas especificados.")
        return

    plot_df_melted = pd.DataFrame(plot_data_all_metrics)

    plt.figure(figsize=figsize)
    sns.boxplot(data=plot_df_melted, x='Intervalo', y='Valor', hue='Métrica',
                order=interval_names, palette=palette) # 'order' garante a ordem dos intervalos no eixo X

    plt.title(title, fontsize=14)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=15, ha='right') # Melhorar legibilidade dos nomes dos intervalos
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Métrica')

    # Ajustar limites do eixo Y
    if y_min is not None or y_max is not None:
        current_ymin, current_ymax = plt.ylim()
        final_ymin = y_min if y_min is not None else current_ymin
        final_ymax = y_max if y_max is not None else current_ymax
        # Garantir que y_min não seja maior que os dados mínimos e y_max não menor que os máximos
        # para evitar cortar os boxplots, a menos que explicitamente desejado.
        # Uma abordagem mais segura é deixar o Seaborn/Matplotlib definir se min/max não forem extremos.
        # Se y_min/y_max forem para zoom, tudo bem. Se for para escala fixa (0-1), também.
        data_min_val = plot_df_melted['Valor'].min()
        data_max_val = plot_df_melted['Valor'].max()
        if y_min is not None: final_ymin = min(final_ymin, data_min_val) if pd.notna(data_min_val) else final_ymin
        if y_max is not None: final_ymax = max(final_ymax, data_max_val) if pd.notna(data_max_val) else final_ymax

        plt.ylim(bottom=final_ymin, top=final_ymax)
    elif not plot_df_melted.empty : # Se não especificou, mas quer garantir que começa em 0 se for performance
         min_val = plot_df_melted['Valor'].min()
         max_val = plot_df_melted['Valor'].max()
         if pd.notna(min_val) and min_val >= 0 and pd.notna(max_val) and max_val <=1.1 : # Típico para Acc/F1
              plt.ylim(bottom=max(0, min_val - 0.05), top=min(1.01, max_val + 0.05) ) # Um pouco de margem
         # else, deixar automático


    plt.tight_layout()
    plt.show()