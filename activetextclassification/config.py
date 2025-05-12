# activetextclassification/config.py
import json
import os

def load_experiments_config(config_path="experiments.json"):
    """
    Carrega a lista de configurações de experimento de um arquivo JSON.

    Args:
        config_path (str): Caminho para o arquivo JSON de configuração.

    Returns:
        list: Lista de dicionários, onde cada dicionário é uma configuração
              de experimento. Retorna lista vazia se o arquivo não for encontrado
              ou for inválido.
    """
    if not os.path.exists(config_path):
        print(f"ERRO: Arquivo de configuração não encontrado: {config_path}")
        return []
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            experiments = json.load(f)
        if not isinstance(experiments, list):
            print(f"ERRO: Conteúdo de {config_path} não é uma lista JSON válida.")
            return []
        print(f"Configurações de {len(experiments)} experimentos carregadas de {config_path}.")
        return experiments
    except json.JSONDecodeError as e:
        print(f"ERRO: Falha ao decodificar JSON em {config_path}: {e}")
        return []
    except Exception as e:
        print(f"ERRO inesperado ao carregar {config_path}: {e}")
        return []

# Poderíamos adicionar funções de validação de schema aqui no futuro
def validate_experiment_config(config):
    # Placeholder para validação futura
    required_keys = ["experiment_name", "active", "data_params", "al_params"]
    for key in required_keys:
        if key not in config:
            print(f"AVISO: Chave obrigatória '{key}' ausente na configuração: {config.get('experiment_name', 'N/A')}")
            return False
    return True