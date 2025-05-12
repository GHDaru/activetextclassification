# activetextclassification/embeddings.py

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import os
import hashlib # Para gerar hash
import json    # Para serializar parâmetros para hash
import pickle  # Para salvar/carregar objetos numpy/python do cache

class BaseEmbedder(ABC):
    """
    Classe base abstrata para geradores de embeddings/features de texto.
    Define a interface comum para transformar listas de texto em matrizes numéricas.
    """

    @abstractmethod
    def fit(self, texts):
        """
        Ajusta o embedder aos textos fornecidos (se necessário).
        Alguns métodos (como TF-IDF) precisam de um 'fit', outros
        (como BERT pré-treinado) podem não precisar ou ter um fit trivial.

        Args:
            texts (list of str): Lista de textos para ajustar o embedder.
        """
        pass

    @abstractmethod
    def transform(self, texts):
        """
        Transforma uma lista de textos em uma matriz de embeddings/features.

        Args:
            texts (list of str): Lista de textos a serem transformados.

        Returns:
            np.ndarray: Matriz numérica onde cada linha corresponde a um texto
                        e as colunas são as features/dimensões do embedding.
                        Shape: (n_samples, n_embedding_dims).
        """
        pass

    def fit_transform(self, texts):
        """
        Combina fit e transform em uma única chamada.

        Args:
            texts (list of str): Lista de textos para ajustar e transformar.

        Returns:
            np.ndarray: Matriz de embeddings/features.
        """
        self.fit(texts)
        return self.transform(texts)

    # Opcional: Método para obter a dimensionalidade do embedding
    def get_embedding_dimension(self):
        """ Retorna a dimensionalidade do vetor gerado (número de colunas). """
        # Implementação padrão pode tentar inferir ou retornar None/NotImplementedError
        # Cada subclasse pode implementar de forma mais específica.
        return getattr(self, '_embedding_dim', None)
    

# continuacao de activetextclassification/embeddings.py

from .vectorizers import ProductVectorizer # Import relativo

class ProductVectorizerEmbedder(BaseEmbedder):
    """
    Usa ProductVectorizer para gerar vetores de probabilidade como embeddings.
    Inclui cache baseado em hash dos textos e parâmetros.
    """
    def __init__(self, vectorizer_params=None, cache_dir=".pv_embedder_cache"):
        """
        Inicializa o embedder.

        Args:
            vectorizer_params (dict, optional): Parâmetros para ProductVectorizer.
            cache_dir (str, optional): Diretório para armazenar arquivos de cache.
                                       Se None, o cache é desativado.
        """
        if vectorizer_params is None:
             vectorizer_params = {'method':'tfidf', 'query':'tfidf', 'norm':'l2', 'query_norm':'l2'}
        self._vectorizer_params = vectorizer_params
        self.pv_instance = None
        self._embedding_dim = None
        self._fitted_texts_hash = None # Hash dos dados usados no último fit
        self.cache_dir = cache_dir

        # Criar diretório de cache se não existir
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            print(f"Cache para ProductVectorizerEmbedder habilitado em: {os.path.abspath(self.cache_dir)}")
        else:
            print("Cache para ProductVectorizerEmbedder desabilitado.")

    def _generate_data_hash(self, data):
        """Gera um hash SHA256 para uma lista de strings ou dados serializáveis."""
        # Serializar para string consistente (ordenar se for dicionário)
        if isinstance(data, (list, tuple)):
            # Para listas de texto, juntar com um separador improvável
            data_str = "<SEP>".join(sorted(data))
        elif isinstance(data, dict):
            data_str = json.dumps(data, sort_keys=True)
        else:
            # Tentar converter para string diretamente
            data_str = str(data)

        return hashlib.sha256(data_str.encode('utf-8')).hexdigest()

    def _get_cache_filename(self, operation_type, texts_hash):
        """ Gera o nome do arquivo de cache. """
        if not self.cache_dir:
            return None
        # Hash dos parâmetros para garantir que o cache é específico da config
        params_hash = self._generate_data_hash(self._vectorizer_params)
        # Nome do arquivo inclui tipo (fit/transform), hash dos parâmetros e hash dos textos
        filename = f"{operation_type}_params_{params_hash}_texts_{texts_hash}.pkl"
        return os.path.join(self.cache_dir, filename)

    def fit(self, texts, labels):
        """
        Ajusta o ProductVectorizer interno, usando cache se possível
        para o estado ajustado (pv_instance).
        """
        texts_hash = self._generate_data_hash(texts)
        cache_file = self._get_cache_filename("fit_state", texts_hash)

        # Tentar carregar estado do fit do cache
        if cache_file and os.path.exists(cache_file):
            try:
                print(f"Fit: Carregando estado ajustado do cache: {cache_file}")
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                # Restaurar o estado
                self.pv_instance = cached_data['pv_instance']
                self._embedding_dim = cached_data['embedding_dim']
                self._fitted_texts_hash = texts_hash # Marcar que este hash foi ajustado
                print(f"Fit: Estado carregado. Dimensão: {self._embedding_dim}")
                return self
            except Exception as e:
                print(f"Fit: Erro ao carregar cache {cache_file}: {e}. Reajustando...")
                # Limpar estado possivelmente corrompido
                self.pv_instance = None
                self._embedding_dim = None
                self._fitted_texts_hash = None


        # Se não carregou do cache, fazer o fit
        print(f"Fit: Ajustando ProductVectorizer com {len(texts)} amostras (Hash: {texts_hash[:8]}...).")
        self.pv_instance = ProductVectorizer(**self._vectorizer_params)
        self.pv_instance.fit(texts, labels)
        self._embedding_dim = len(self.pv_instance.category_index)
        self._fitted_texts_hash = texts_hash # Armazenar hash dos dados de fit
        print(f"Fit: Concluído. Dimensão: {self._embedding_dim}")

        # Salvar estado do fit no cache
        if cache_file:
            try:
                print(f"Fit: Salvando estado ajustado no cache: {cache_file}")
                state_to_save = {
                    'pv_instance': self.pv_instance,
                    'embedding_dim': self._embedding_dim
                }
                with open(cache_file, 'wb') as f:
                    pickle.dump(state_to_save, f)
            except Exception as e:
                 print(f"Fit: Erro ao salvar estado no cache {cache_file}: {e}")

        return self

    def transform(self, texts):
        """
        Gera os vetores de probabilidade, usando cache para os resultados.
        """
        if self.pv_instance is None or self._embedding_dim is None:
            raise RuntimeError("Embedder não foi ajustado. Chame 'fit' primeiro.") 

        texts_hash = self._generate_data_hash(texts)
        cache_file = self._get_cache_filename("transform_output", texts_hash)

        # Tentar carregar resultado do transform do cache
        if cache_file and os.path.exists(cache_file):
             try:
                print(f"Transform: Carregando embeddings do cache: {cache_file}")
                with open(cache_file, 'rb') as f:
                    embeddings = pickle.load(f)
                # Validar shape (simples)
                if isinstance(embeddings, np.ndarray) and embeddings.shape[1] == self._embedding_dim:
                     print(f"Transform: Embeddings carregados. Shape: {embeddings.shape}")
                     return embeddings
                else:
                     print("Transform: Dados do cache inválidos. Recalculando...")
             except Exception as e:
                  print(f"Transform: Erro ao carregar cache {cache_file}: {e}. Recalculando...")


        # Se não carregou do cache, fazer o transform
        print(f"Transform: Gerando embeddings para {len(texts)} textos (Hash: {texts_hash[:8]}...).")
        if not texts:
            return np.empty((0, self._embedding_dim))

        proba_raw = self.pv_instance.predict_proba(texts) # PV retorna (n_classes, n_samples)
        embeddings = proba_raw.T # Transpor para (n_samples, n_classes)
        print(f"Transform: Concluído. Shape: {embeddings.shape}")


        # Salvar resultado do transform no cache
        if cache_file:
             try:
                print(f"Transform: Salvando embeddings no cache: {cache_file}")
                with open(cache_file, 'wb') as f:
                    pickle.dump(embeddings, f)
             except Exception as e:
                  print(f"Transform: Erro ao salvar embeddings no cache {cache_file}: {e}")

        return embeddings

    # fit_transform e get_embedding_dimension permanecem os mesmos da BaseEmbedder

# --- Função Fábrica ATUALIZADA ---
def get_embedder(config):
    embedder_type = config.get('type')
    # Trabalhar com cópia dos parâmetros para evitar modificar o dict original
    params = config.get('params', {}).copy()

    print(f"Embedder Factory: Criando tipo '{embedder_type}' com params: {params}")

    if embedder_type == 'PVProb':
        # Acessar os parâmetros específicos do vetorizador interno
        vectorizer_params = params.get('vectorizer_params', {}).copy() # Pega e copia params do PV

        # --- CONVERSÃO DO NGRAM_RANGE ---
        if 'ngram_range' in vectorizer_params and isinstance(vectorizer_params['ngram_range'], list):
            original_ngram = vectorizer_params['ngram_range']
            vectorizer_params['ngram_range'] = tuple(original_ngram)
            print(f"Embedder Factory (PVProb): Convertido ngram_range {original_ngram} para tupla {vectorizer_params['ngram_range']}")
        # --- FIM DA CONVERSÃO ---

        # Passar os parâmetros MODIFICADOS (se necessário) para o construtor
        return ProductVectorizerEmbedder(
            vectorizer_params=vectorizer_params, # Passa o dict (potencialmente modificado)
            cache_dir=params.get('cache_dir', ".pv_embedder_cache")
        )
    # Adicionar outros tipos de embedder aqui no futuro
    # elif embedder_type == 'ST':
    #    return SentenceTransformerEmbedder(...)
    else:
        raise ValueError(f"Tipo de embedder desconhecido: {embedder_type}")