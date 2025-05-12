# activetextclassification/oracle.py

import pandas as pd
import os
import json
import time
from abc import ABC, abstractmethod
try:
    from tqdm.notebook import tqdm # Usar versão de notebook se disponível
except ImportError:
    try:
        from tqdm import tqdm # Usar versão padrão como fallback
    except ImportError:
        print("AVISO (oracle.py): tqdm não instalado. Barra de progresso desativada.")
        # Criar uma função dummy que apenas retorna o iterável
        tqdm = lambda x, *args, **kwargs: x

# Importar o cliente OpenAI (se instalado)
try:
    from openai import OpenAI, RateLimitError, APIError
    OPENAI_AVAILABLE = True
except ImportError:
    print("AVISO (oracle.py): Biblioteca 'openai' não encontrada. Oracle OpenAI não funcionará.")
    OPENAI_AVAILABLE = False
    # Definir classes de erro dummy para evitar erros de NameError abaixo
    class OpenAI: pass
    class RateLimitError(Exception): pass
    class APIError(Exception): pass


# Importar templates e schemas do módulo de prompts
from .prompts import CLASSIFICATION_PROMPT_FSTRING, OPENAI_CLASSIFICATION_SCHEMA # Usar o schema simples

# --- Interface Base para Oráculos ---
class BaseOracle(ABC):
    """ Classe base abstrata para oráculos de rotulação. """
    @abstractmethod
    def query(self, data_to_label):
        """
        Consulta o oráculo para obter rótulos para os dados fornecidos.

        Args:
            data_to_label (list or pd.DataFrame or similar): Os dados a serem rotulados.
                                                             O formato exato pode variar.

        Returns:
            list: Uma lista de rótulos (strings) correspondentes aos dados de entrada.
        """
        pass

# --- Oráculo Simulado (Pega labels existentes) ---
class SimulatedOracle(BaseOracle):
    """ Oráculo simulado que retorna labels verdadeiros pré-existentes. """
    def __init__(self, label_column):
        self.label_column = label_column
        print(f"SimulatedOracle inicializado (usará coluna '{label_column}').")

    def query(self, data_to_label):
        """ Retorna os labels da coluna especificada. """
        print(f"Oráculo Simulado: Obtendo rótulos para {len(data_to_label)} amostras...")
        if isinstance(data_to_label, pd.DataFrame):
            if self.label_column not in data_to_label.columns:
                raise ValueError(f"Coluna de label '{self.label_column}' não encontrada no DataFrame.")
            return data_to_label[self.label_column].tolist()
        elif isinstance(data_to_label, list) and all(isinstance(item, dict) for item in data_to_label):
             # Assume lista de dicionários
             try:
                 return [item[self.label_column] for item in data_to_label]
             except KeyError:
                  raise ValueError(f"Coluna de label '{self.label_column}' não encontrada em um dos dicts.")
        else:
             raise TypeError("SimulatedOracle espera pd.DataFrame ou lista de dicionários.")



class OpenaiOracle(BaseOracle):
    """ Oráculo que consulta a API da OpenAI para obter rótulos. """
    def __init__(self, available_labels, text_column,
                    model="gpt-4o-mini", # Modelo padrão OpenAI (pode ser ajustado)
                    api_key=None,        # Opcional: passar chave diretamente
                    max_retries=3,       # Número de tentativas em caso de erro
                    retry_delay=5        # Delay entre tentativas (segundos)
                    ):
        """
        Inicializa o oráculo OpenAI.

        Args:
            available_labels (list): Lista de TODAS as categorias possíveis para classificação.
            text_column (str): Nome da coluna/chave contendo a descrição do produto.
            model (str): Nome do modelo OpenAI a ser usado.
            api_key (str, optional): Chave da API OpenAI. Se None, tenta usar a variável de ambiente OPENAI_API_KEY.
            max_retries (int): Máximo de tentativas em caso de erros da API (ex: RateLimit).
            retry_delay (int): Delay em segundos entre as tentativas.
        """
        self.text_column = text_column
        self.available_labels = sorted(list(set(available_labels))) # Garantir lista única ordenada
        self.labels_str = ", ".join(f"'{lbl}'" for lbl in self.available_labels) # Formatar para prompt
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        try:
            self.client = OpenAI(api_key=api_key) # api_key=None usa variável de ambiente
            # Testar conexão/autenticação (opcional, mas bom)
            # self.client.models.list()
            print(f"OpenaiOracle inicializado com modelo '{self.model}'. {len(self.available_labels)} labels disponíveis.")
        except Exception as e:
                raise RuntimeError(f"Erro ao inicializar cliente OpenAI: {e}. Verifique a chave da API.") from e

        # Usar o schema JSON simples definido em prompts.py
        self.response_schema = OPENAI_CLASSIFICATION_SCHEMA


    def _call_openai_with_retry(self, description):
        """ Chama a API com retentativas para uma única descrição. """
        conversation_history = [
            {"role": "system", "content": "Você é um assistente útil especialista em categorizar produtos."},
            {"role": "user", "content": CLASSIFICATION_PROMPT_FSTRING.format(
                product_description=description,
                category_list_str=self.labels_str
                )
            }
        ]
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=conversation_history,
                    response_format={ # Usando o formato estruturado
                        "type": "json_schema", # <-- IMPORTANTE: json_object para garantir JSON
                        "json_schema": self.response_schema
                    },
                    temperature=0 # Baixa temperatura para respostas mais consistentes/determinísticas
                )
                content = response.choices[0].message.content
                # Tentar decodificar o JSON retornado
                structured_response = json.loads(content)
                predicted_category = structured_response.get("predicted_category")

                # --- Validação da Resposta ---
                if predicted_category and predicted_category in self.available_labels:
                    return structured_response # Sucesso! Retorna a categoria válida
                else:
                    print(f"AVISO (OpenAI): Resposta inválida ou categoria não reconhecida: '{predicted_category}'. Desc: '{description[:50]}...'")
                    # O que fazer aqui? Retornar None? Tentar de novo? Por ora, None.
                    last_exception = ValueError(f"Categoria inválida recebida: {predicted_category}")
                    # Não tentar de novo por resposta inválida, apenas por erros da API
                    break # Sai do loop de retry

            except (RateLimitError, APIError) as api_e:
                print(f"AVISO (OpenAI): Erro da API (tentativa {attempt + 1}/{self.max_retries}): {api_e}. Tentando novamente em {self.retry_delay}s...")
                last_exception = api_e
                time.sleep(self.retry_delay)
            except json.JSONDecodeError as json_e:
                print(f"AVISO (OpenAI): Falha ao decodificar JSON da resposta: {json_e}. Content: '{content}'")
                last_exception = json_e
                # Pode não adiantar tentar de novo se o modelo retorna JSON inválido consistentemente
                break # Sai do loop de retry
            except Exception as e:
                print(f"ERRO (OpenAI): Erro inesperado na chamada (tentativa {attempt + 1}/{self.max_retries}): {e}")
                last_exception = e
                time.sleep(self.retry_delay) # Tentar de novo para erros inesperados

        # Se saiu do loop sem sucesso
        print(f"ERRO (OpenAI): Falha ao obter rótulo válido para '{description[:50]}...' após {self.max_retries} tentativas. Último erro: {last_exception}")
        return None # Retorna None em caso de falha persistente


    def query(self, data_to_label):
        """
        Consulta a API OpenAI para cada item em data_to_label.

        Args:
            data_to_label (pd.DataFrame or list of dict): Dados contendo
                                                            a coluna `self.text_column`.

        Returns:
            list: Lista de rótulos string preditos pela OpenAI (pode conter None).
        """
        print(f"Oráculo OpenAI: Consultando modelo '{self.model}' para {len(data_to_label)} amostras...")
        labels = []
        
        # Garantir que temos uma lista de descrições
        descriptions = []
        if isinstance(data_to_label, pd.DataFrame):
                if self.text_column not in data_to_label.columns: raise ValueError(f"Coluna '{self.text_column}' não encontrada.")
                descriptions = data_to_label[self.text_column].tolist()
        elif isinstance(data_to_label, list) and all(isinstance(item, dict) for item in data_to_label):
                try: descriptions = [item[self.text_column] for item in data_to_label]
                except KeyError: raise ValueError(f"Coluna '{self.text_column}' não encontrada em um dos dicts.")
        else:
                raise TypeError("OpenaiOracle espera pd.DataFrame ou lista de dicionários.")

        # Iterar e chamar API para cada descrição (pode ser otimizado com batching se a API suportar)
        # Usar tqdm para barra de progresso
        for desc in tqdm(descriptions, desc="Consultando OpenAI"):
            predicted_label = self._call_openai_with_retry(desc)
            labels.append(predicted_label) # Adiciona o label (ou None se falhar)

        print(f"Oráculo OpenAI: Consulta concluída. {sum(1 for lbl in labels if lbl is not None)} rótulos obtidos.")
        return labels

# --- Função Fábrica para Oráculos ---
def get_oracle(config, all_possible_labels=None):
    """
    Instancia um oráculo baseado na configuração.

    Args:
        config (dict): Dicionário de configuração do oráculo.
                       Ex: {'type': 'Simulated', 'params': {'label_column': 'categoria'}}
                       Ex: {'type': 'OpenAI', 'params': {'text_column': 'desc', 'model': 'gpt-4o-mini'}}
        all_possible_labels (list, optional): Necessário para OpenaiOracle.

    Returns:
        BaseOracle: Instância do oráculo solicitado.
    """
    oracle_type = config.get('type')
    params = config.get('params', {})

    print(f"Oracle Factory: Criando tipo '{oracle_type}' com params: {params}")

    if oracle_type == 'Simulated':
         if 'label_column' not in params: raise ValueError("Parâmetro 'label_column' necessário para SimulatedOracle.")
         return SimulatedOracle(label_column=params['label_column'])
    elif oracle_type == 'OpenAI':
         if not OPENAI_AVAILABLE: raise ImportError("Biblioteca 'openai' não instalada.")
         if all_possible_labels is None: raise ValueError("'all_possible_labels' necessário para OpenaiOracle.")
         if 'text_column' not in params: raise ValueError("'text_column' necessário para OpenaiOracle.")
         # Passar all_possible_labels e outros params para o construtor
         return OpenaiOracle(available_labels=all_possible_labels, **params)
    else:
         raise ValueError(f"Tipo de oráculo desconhecido: {oracle_type}")