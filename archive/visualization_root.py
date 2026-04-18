# activetextclassification/visualization.py

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_learning_curves(
    history_df,
    baseline_metrics=None,
    experiment_name=None,
    figsize=(15, 5)
):
    """
    Plota as curvas de aprendizado (Acurácia e F1-Score) internas e externas
    ao longo do tamanho do conjunto rotulado (L_size).

    Args:
        history_df (pd.DataFrame): DataFrame contendo o histórico do loop AL.
                                   Deve ter colunas como 'L_size', 'internal_acc',
                                   'internal_f1', 'external_acc', 'external_f1'.
        baseline_metrics (dict, optional): Dicionário com métricas do baseline,
                                           espera chaves 'baseline_acc' e 'baseline_f1'.
        experiment_name (str, optional): Nome do experimento para o título do gráfico.
        figsize (tuple, optional): Tamanho da figura para o plot.
    """
    print("\n--- Gerando Gráficos de Curvas de Aprendizado ---")

    if history_df is None or history_df.empty:
        print("AVISO: Histórico vazio ou não fornecido. Não é possível plotar.")
        return

    # Verificar colunas essenciais
    required_cols = ['L_size', 'external_acc', 'external_f1'] # Externas são mais importantes
    if not all(col in history_df.columns for col in required_cols):
         missing = [col for col in required_cols if col not in history_df.columns]
         print(f"AVISO: Colunas necessárias ausentes no histórico ({missing}). Plotagem pode estar incompleta.")
         # Continuar mesmo assim, plotando o que for possível

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # --- Gráfico de Acurácia ---
    plot_acc = False
    if 'internal_acc' in history_df.columns:
         axes[0].plot(history_df['L_size'], history_df['internal_acc'].dropna(), marker='o', linestyle='-', label=f'Acc Interna')
         plot_acc = True
    if 'external_acc' in history_df.columns:
         axes[0].plot(history_df['L_size'], history_df['external_acc'].dropna(), marker='x', linestyle='--', label='Acc Externa (P)')
         plot_acc = True

    # Linha do Baseline (Acurácia)
    bl_acc = baseline_metrics.get('baseline_acc', np.nan) if baseline_metrics else np.nan
    if not pd.isna(bl_acc):
        axes[0].axhline(bl_acc, color='r', linestyle=':', label=f'Baseline Acc ({bl_acc:.3f})')
        plot_acc = True

    if plot_acc:
         axes[0].set_title('Evolução da Acurácia')
         axes[0].set_xlabel('Tamanho do Conjunto Rotulado (L)')
         axes[0].set_ylabel('Acurácia')
         axes[0].grid(True); axes[0].legend(); axes[0].set_ylim(bottom=0)
    else:
         axes[0].set_title('Acurácia (Dados Indisponíveis)')
         axes[0].text(0.5, 0.5, 'Dados de Acurácia não encontrados no histórico.', ha='center', va='center')


    # --- Gráfico de F1-Score ---
    plot_f1 = False
    if 'internal_f1' in history_df.columns:
         axes[1].plot(history_df['L_size'], history_df['internal_f1'].dropna(), marker='o', linestyle='-', label=f'F1 Interno')
         plot_f1 = True
    if 'external_f1' in history_df.columns:
         axes[1].plot(history_df['L_size'], history_df['external_f1'].dropna(), marker='x', linestyle='--', label='F1 Externo (P)')
         plot_f1 = True

    # Linha do Baseline (F1-Score)
    bl_f1 = baseline_metrics.get('baseline_f1', np.nan) if baseline_metrics else np.nan
    if not pd.isna(bl_f1):
        axes[1].axhline(bl_f1, color='r', linestyle=':', label=f'Baseline F1 ({bl_f1:.3f})')
        plot_f1 = True

    if plot_f1:
         axes[1].set_title('Evolução do Macro F1-Score')
         axes[1].set_xlabel('Tamanho do Conjunto Rotulado (L)')
         axes[1].set_ylabel('Macro F1-Score')
         axes[1].grid(True); axes[1].legend(); axes[1].set_ylim(bottom=0)
    else:
         axes[1].set_title('F1-Score (Dados Indisponíveis)')
         axes[1].text(0.5, 0.5, 'Dados de F1-Score não encontrados no histórico.', ha='center', va='center')

    # --- Título Geral e Exibição ---
    if experiment_name:
         fig.suptitle(f"Experimento: {experiment_name}", fontsize=14)
         plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Ajustar layout
    else:
         plt.tight_layout()

    plt.show()