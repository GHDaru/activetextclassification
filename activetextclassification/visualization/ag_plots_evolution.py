# activetextclassification/visualization/ag_plots_evolution.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

def plot_population_evolution_combined(
    l0_size,
    metric_name_display, # e.g., "Acurácia"
    df_agg_max, # DataFrame agregado do log detalhado para MAXIMIZE
    df_agg_min, # DataFrame agregado do log detalhado para MINIMIZE
    colors_max, # Lista de cores para maximização
    colors_min  # Lista de cores para minimização
):
    """
    Plota a evolução da população (faixa, média, melhor/pior) para cenários de
    maximização e minimização de uma métrica, no mesmo gráfico.

    Args:
        l0_size (int): Tamanho do L0.
        metric_name_display (str): Nome da métrica para o título e eixo Y (e.g., "Acurácia").
        df_agg_max (pd.DataFrame): Dados agregados para o cenário de maximização.
                                   Deve conter 'generation', 'min_metric', 'avg_metric', 'max_metric'.
        df_agg_min (pd.DataFrame): Dados agregados para o cenário de minimização.
                                   Deve conter 'generation', 'min_metric', 'avg_metric', 'max_metric'.
        colors_max (list): Lista de 3 cores para maximização (média, faixa, melhor).
        colors_min (list): Lista de 3 cores para minimização (média, faixa, pior).
    """
    if df_agg_max is None and df_agg_min is None:
        print(f"  Sem dados de {metric_name_display} para L0={l0_size} para plotar.")
        return

    plt.figure(figsize=(12, 7))
    
    # Plot Maximização
    if df_agg_max is not None and not df_agg_max.empty:
        gens_max = df_agg_max['generation']
        plt.fill_between(gens_max, df_agg_max['min_metric'], df_agg_max['max_metric'], 
                         color=colors_max[1], alpha=0.2, label=f'Maximização (Faixa Pop.)')
        plt.plot(gens_max, df_agg_max['avg_metric'], 
                 color=colors_max[0], linestyle='--', label=f'Maximização (Média Pop.)')
        plt.plot(gens_max, df_agg_max['max_metric'], 
                 color=colors_max[2], linewidth=2, marker='.', markersize=4, label=f'Maximização (Melhor Indivíduo)')

    # Plot Minimização
    if df_agg_min is not None and not df_agg_min.empty:
        gens_min = df_agg_min['generation']
        plt.fill_between(gens_min, df_agg_min['min_metric'], df_agg_min['max_metric'], 
                         color=colors_min[1], alpha=0.2, label=f'Minimização (Faixa Pop.)')
        plt.plot(gens_min, df_agg_min['avg_metric'], 
                 color=colors_min[0], linestyle='--', label=f'Minimização (Média Pop.)')
        plt.plot(gens_min, df_agg_min['min_metric'], 
                 color=colors_min[2], linewidth=2, marker='.', markersize=4, label=f'Minimização (Pior Indivíduo)')
        
    plt.title(f"Evolução da {metric_name_display} da População para L0 = {l0_size}")
    plt.xlabel("Geração")
    plt.ylabel(f"{metric_name_display} (%)") # Eixo Y em percentual
    
    # Formatar eixo Y como percentual
    plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True, which="both", ls="--", alpha=0.7)
    
    # Ajustar limites do eixo Y para 0-100% se os dados estiverem nesse range, com pequena margem
    current_ymin, current_ymax = plt.gca().get_ylim()
    final_ymin = max(0, current_ymin - 0.05)
    final_ymax = min(1, current_ymax + 0.05)
    if final_ymax <= final_ymin: # Evitar ymax < ymin
        final_ymax = final_ymin + 0.1 
    plt.ylim(bottom=final_ymin, top=final_ymax)

    plt.tight_layout(rect=[0, 0, 0.82, 1]) # Ajuste para legenda
    plt.show()