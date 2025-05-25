# activetextclassification/visualization/common_plots.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker # Para formatar ticks em log e porcentagem
import seaborn as sns

# --- FUNÇÃO MODIFICADA: plot_filtered_evolutionary_boxplots ---
def plot_filtered_evolutionary_boxplots(
    results_df,
    metric_column,
    l0_size_column='l0_size',
    title_suffix="",
    xlabel="Tamanho da Amostra Inicial (L0)",
    ylabel=None,
    figsize=(15, 7),
    x_value_min=None,
    x_value_max=None,
    y_min=0.0, y_max=1.01,
    width_factor=0.03,
    min_width=5,
    palette="pastel",
    format_y_as_percent=True # NOVO PARÂMETRO
):
    """
    Plota boxplots evolutivos para UMA métrica, filtrando por um intervalo de l0_size,
    e posicionando os boxplots numericamente no eixo X.
    Permite formatar o eixo Y como porcentagem.
    """
    print(f"\n--- Gerando Boxplot Filtrado para: {metric_column} ---")
    # ... (validações e preparação de plot_data_filtered como antes) ...
    if results_df is None or results_df.empty: print("AVISO (boxplot): DF vazio."); return
    if metric_column not in results_df.columns or l0_size_column not in results_df.columns:
        print(f"AVISO (boxplot): Coluna '{metric_column}' ou '{l0_size_column}' ausente."); return
    plot_data_filtered = results_df.copy()
    plot_data_filtered[metric_column] = pd.to_numeric(plot_data_filtered[metric_column], errors='coerce')
    if x_value_min is not None: plot_data_filtered = plot_data_filtered[plot_data_filtered[l0_size_column] >= x_value_min]
    if x_value_max is not None: plot_data_filtered = plot_data_filtered[plot_data_filtered[l0_size_column] <= x_value_max]
    plot_data_filtered.dropna(subset=[metric_column, l0_size_column], inplace=True)
    if plot_data_filtered.empty: print(f"AVISO (boxplot): Sem dados para '{metric_column}' após filtros."); return
    unique_l0_sizes_to_plot = sorted(plot_data_filtered[l0_size_column].unique())
    if not unique_l0_sizes_to_plot: print(f"AVISO (boxplot): Sem tamanhos L0 válidos para '{metric_column}'."); return

    data_for_matplotlib_boxplot = [plot_data_filtered[plot_data_filtered[l0_size_column] == size][metric_column].values for size in unique_l0_sizes_to_plot]
    valid_data_bp, valid_positions_bp = [], []
    for i, data_arr in enumerate(data_for_matplotlib_boxplot):
        if len(data_arr) > 0: valid_data_bp.append(data_arr); valid_positions_bp.append(unique_l0_sizes_to_plot[i])
    if not valid_data_bp: print(f"AVISO (boxplot): Nenhum dado válido para plotar."); return

    plt.figure(figsize=figsize)
    widths_bp = np.array(valid_positions_bp) * width_factor + min_width
    if len(valid_positions_bp) > 1:
        min_diff = np.min(np.diff(valid_positions_bp)); widths_bp = np.clip(widths_bp, min_width, min_diff * 0.8 if min_diff > 0 else min_width)

    bp = plt.boxplot(valid_data_bp, positions=valid_positions_bp, widths=widths_bp, manage_ticks=False, patch_artist=True, medianprops=dict(color="red", linewidth=1.5), showfliers=True)

    final_ylabel = ylabel if ylabel else metric_column.replace('_on_population', '').replace('_', ' ').title()
    plt.title(f"Distribuição de {final_ylabel} por Tamanho de L0 (Filtrado)\n{title_suffix}", fontsize=14)
    plt.xlabel(xlabel); plt.ylabel(final_ylabel); plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(valid_positions_bp, rotation=45, ha='right')
    if len(valid_positions_bp) > 1 : padding = (valid_positions_bp[-1] - valid_positions_bp[0]) * 0.05; plt.xlim(valid_positions_bp[0] - padding, valid_positions_bp[-1] + padding)
    elif len(valid_positions_bp) == 1: plt.xlim(valid_positions_bp[0] - widths_bp[0], valid_positions_bp[0] + widths_bp[0])

    # --- FORMATAÇÃO DO EIXO Y COMO PORCENTAGEM ---
    if format_y_as_percent:
        plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0)) # xmax=1.0 pois os dados estão em 0-1
    # --- FIM FORMATAÇÃO ---

    if y_min is not None or y_max is not None: plt.ylim(bottom=y_min, top=y_max)
    # else: plt.ylim(bottom=0) # Deixar o auto-scale se não for percentual ou se min/max não forçados

    colors = sns.color_palette(palette, n_colors=len(valid_positions_bp))
    for i, patch in enumerate(bp['boxes']): patch.set_facecolor(colors[i % len(colors)])
    plt.tight_layout(); plt.show()


# --- FUNÇÃO MODIFICADA: plot_performance_trend_lines_combined ---
def plot_performance_trend_lines_combined(
    stats_df_long,
    l0_size_column='l0_size',
    title="Tendência da Performance por Tamanho de L0",
    xlabel="Tamanho da Amostra Inicial (L0)",
    ylabel="Performance",
    figsize=(14, 7),
    use_log_scale_x=False,
    show_markers=True,
    plot_min_max_fill=True,
    y_min=0.0, y_max=1.01,
    metrics_to_plot=None,
    format_y_as_percent=True # NOVO PARÂMETRO
):
    """ Plota linhas de tendência para múltiplas métricas... """
    print(f"\n--- Gerando Gráfico Combinado de Tendências ---")
    # ... (validações como antes) ...
    if stats_df_long is None or stats_df_long.empty: print("AVISO: DF stats vazio."); return
    if metrics_to_plot is None: metrics_to_plot = {'Acurácia': 'Acurácia Média', 'F1-Macro': 'F1-Macro Médio'}

    plt.figure(figsize=figsize)
    # ... (lógica de cores e loop sobre metrics_to_plot como antes) ...
    palette = sns.color_palette("husl", n_colors=len(metrics_to_plot)); color_idx = 0
    for metric_name_in_df, legend_label_mean in metrics_to_plot.items():
        metric_data = stats_df_long[stats_df_long['Métrica'] == metric_name_in_df].sort_values(by=l0_size_column)
        if metric_data.empty: print(f"AVISO: Sem dados para '{metric_name_in_df}'."); continue
        color = palette[color_idx % len(palette)]; color_idx += 1; marker_style = '.' if show_markers else None
        if 'Média' in metric_data.columns: plt.plot(metric_data[l0_size_column], metric_data['Média'], label=legend_label_mean, marker=marker_style, linewidth=2, color=color)
        if plot_min_max_fill and 'Mínimo' in metric_data.columns and 'Máximo' in metric_data.columns:
             min_v = metric_data['Mínimo'].ffill().bfill(); max_v = metric_data['Máximo'].ffill().bfill()
             plt.fill_between(metric_data[l0_size_column], min_v, max_v, color=color, alpha=0.15, label=f'Intervalo Min-Max {metric_name_in_df}')
        # ... (plots de Min/Max como linhas se plot_min_max_fill for False, como antes) ...


    plt.title(title, fontsize=14); plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.legend(loc='lower right'); plt.grid(True, linestyle='--', alpha=0.7)

    # --- FORMATAÇÃO DO EIXO Y COMO PORCENTAGEM ---
    if format_y_as_percent:
        plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    # --- FIM FORMATAÇÃO ---

    # ... (lógica de xscale, xticks, ylim como antes) ...
    # Garantir que os ticks do eixo X logarítmico sejam formatados como números inteiros
    unique_x_values_plot = sorted(list(stats_df_long[l0_size_column].unique()))
    if use_log_scale_x:
        plt.xscale('log')
        if unique_x_values_plot:
            log_ticks_candidates = [10,20,30,50,70,100,200,300,500,700,1000,2000,3000,5000,7000,10000,20000,50000,100000,200000]
            tick_pos = [t for t in log_ticks_candidates if t >= unique_x_values_plot[0] and t <= unique_x_values_plot[-1]]
            if not tick_pos or (tick_pos and tick_pos[-1] < unique_x_values_plot[-1] and unique_x_values_plot[-1] not in tick_pos): tick_pos.append(unique_x_values_plot[-1])
            if tick_pos and tick_pos[0] > unique_x_values_plot[0] and unique_x_values_plot[0] not in tick_pos : tick_pos.insert(0,unique_x_values_plot[0])
            tick_pos = sorted(list(set(tick_pos))) if tick_pos else unique_x_values_plot
            plt.xticks(tick_pos, rotation=45, ha='right'); plt.gca().xaxis.set_major_formatter(mticker.ScalarFormatter());
            plt.gca().xaxis.get_major_formatter().set_scientific(False); plt.gca().tick_params(axis='x',which='minor',bottom=False)
    else: plt.xticks(unique_x_values_plot[10:] if unique_x_values_plot else None, rotation=45, ha='right') # Usar unique_x_values_plot para ticks lineares

    if y_min is not None or y_max is not None: plt.ylim(bottom=y_min, top=y_max)
    elif not format_y_as_percent: plt.ylim(bottom=0) # Se não for percentual, default para 0

    plt.tight_layout(); plt.show()

def plot_variability_trend(
    stats_df,
    variability_metrics,
    metric_names,
    l0_size_column='l0_size',
    title="Variabilidade da Performance por Tamanho de L0",
    xlabel="Tamanho da Amostra Inicial (L0)",
    figsize=(12, 6),
    use_log_scale_x=True,
    show_markers=True,
    palette=None
):
    print(f"\n--- Gerando Gráfico de Tendência de Variabilidade para: {metric_names} ---")
    # ... (validações iniciais como antes) ...
    if stats_df is None or stats_df.empty: print("AVISO: DataFrame de estatísticas vazio."); return
    if not isinstance(metric_names, list): metric_names = [metric_names]
    if not isinstance(variability_metrics, list): variability_metrics = [variability_metrics]
    num_var_metrics = len(variability_metrics)
    if num_var_metrics == 0: print("AVISO: Nenhuma métrica de variabilidade."); return
    fig, axes = plt.subplots(num_var_metrics, 1, figsize=figsize, sharex=True)
    if num_var_metrics == 1: axes = [axes]

    colors = sns.color_palette(palette if palette else "Set1", n_colors=len(metric_names))
    marker_style = '.' if show_markers else None

    for i, var_metric_col in enumerate(variability_metrics):
        # ... (loop interno e plotagem das linhas como antes) ...
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
        else:
            ax.set_title(f'{var_metric_col} (Dados Indisponíveis)'); ax.text(0.5,0.5, "Dados não encontrados.", ha='center',va='center')
            ax.set_ylabel(var_metric_col); ax.grid(True, linestyle='--', alpha=0.7)


    # Configurar eixo X para o último subplot
    ax_to_configure_x = axes[-1]
    ax_to_configure_x.set_xlabel(xlabel)
    unique_x_values = sorted(stats_df[l0_size_column].unique())
    tick_positions_to_set = unique_x_values

    if use_log_scale_x:
        ax_to_configure_x.set_xscale('log')
        if unique_x_values: # Garantir que unique_x_values não está vazio
            log_ticks_candidates = [t for t in [10,20,30,50,70,100,200,300,500,700,1000,2000,3000,5000,7000,10000,20000,50000,100000,200000] if t >= unique_x_values[0] and t <= unique_x_values[-1]]
            # Garantir que o primeiro e último valor original estejam nos ticks se forem significativos
            if not log_ticks_candidates or (log_ticks_candidates and log_ticks_candidates[-1] < unique_x_values[-1] and unique_x_values[-1] not in log_ticks_candidates): log_ticks_candidates.append(unique_x_values[-1])
            if log_ticks_candidates and log_ticks_candidates[0] > unique_x_values[0] and unique_x_values[0] not in log_ticks_candidates : log_ticks_candidates.insert(0,unique_x_values[0])
            tick_positions_to_set = sorted(list(set(log_ticks_candidates))) if log_ticks_candidates else unique_x_values
        # else: tick_positions_to_set já é unique_x_values (que seria vazio)

        ax_to_configure_x.set_xticks(tick_positions_to_set)
        # --- CORREÇÃO APLICADA AQUI ---
        tick_labels_log = []
        for val in tick_positions_to_set:
            if pd.notna(val):
                if isinstance(val, float) and val.is_integer(): # Checa se float pode ser int
                    tick_labels_log.append(str(int(val)))
                elif isinstance(val, (int, np.integer)): # Se já é int
                    tick_labels_log.append(str(val))
                else: # Outros floats
                    tick_labels_log.append(f"{val:.0f}") # Formatar float sem decimais para ticks log
            else:
                tick_labels_log.append("")
        ax_to_configure_x.set_xticklabels(tick_labels_log, rotation=45, ha='right')
        # --- FIM DA CORREÇÃO ---
        ax_to_configure_x.xaxis.set_major_formatter(mticker.ScalarFormatter()) # Usar ScalarFormatter para números normais
        ax_to_configure_x.xaxis.get_major_formatter().set_scientific(False) # Desabilitar notação científica
        ax_to_configure_x.tick_params(axis='x', which='minor', bottom=False) # Remover minor ticks se desejado
    else: # Escala Linear
        if len(unique_x_values) > 15:
             step = max(1, len(unique_x_values) // 10)
             tick_positions_to_set = unique_x_values[::step]
             if unique_x_values[-1] not in tick_positions_to_set:
                  tick_positions_to_set = np.append(tick_positions_to_set, unique_x_values[-1])
        # else: tick_positions_to_set já é unique_x_values

        ax_to_configure_x.set_xticks(tick_positions_to_set)
        # --- CORREÇÃO APLICADA AQUI TAMBÉM ---
        tick_labels_linear = []
        for val in tick_positions_to_set:
            if pd.notna(val):
                if isinstance(val, float) and val.is_integer():
                    tick_labels_linear.append(str(int(val)))
                elif isinstance(val, (int, np.integer)):
                    tick_labels_linear.append(str(val))
                else:
                    tick_labels_linear.append(f"{val:.1f}") # Pode precisar de decimais aqui
            else:
                tick_labels_linear.append("")
        ax_to_configure_x.set_xticklabels(tick_labels_linear, rotation=45, ha='right')
        # --- FIM DA CORREÇÃO ---

    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()