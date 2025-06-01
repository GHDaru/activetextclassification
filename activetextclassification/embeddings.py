# activetextclassification/embeddings.py

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import os
import hashlib # Para gerar hash
import json    # Para serializar parâmetros para hash
import pickle  # Para salvar/carregar objetos numpy/python do cache

# Import relativo, ajuste se a estrutura do seu projeto for diferente
# Ex: from ..vectorizers import ProductVectorizer se embeddings.py estiver em um subdiretório
try:
    from .vectorizers import ProductVectorizer
except ImportError:
    # Fallback para import absoluto se estiver rodando scripts de fora do pacote
    from activetextclassification.vectorizers import ProductVectorizer


class BaseEmbedder(ABC):
    """
    Classe base abstrata para geradores de embeddings/features de texto.
    Define a interface comum para transformar listas de texto em matrizes numéricas.
    """

    @abstractmethod
    def fit(self, texts, labels=None): # Modificado para aceitar labels opcionalmente
        """
        Ajusta o embedder aos textos fornecidos (e labels, se necessário).
        Alguns métodos (como TF-IDF ou ProductVectorizerEmbedder) precisam de um 'fit',
        outros (como BERT pré-treinado) podem não precisar ou ter um fit trivial.

        Args:
            texts (list of str): Lista de textos para ajustar o embedder.
            labels (list of str, optional): Lista de labels correspondentes,
                                            usada por embedders como ProductVectorizerEmbedder.
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

    def fit_transform(self, texts, labels=None): # Modificado para aceitar labels
        """
        Combina fit e transform em uma única chamada.

        Args:
            texts (list of str): Lista de textos para ajustar e transformar.
            labels (list of str, optional): Labels para o fit.

        Returns:
            np.ndarray: Matriz de embeddings/features.
        """
        self.fit(texts, labels) # Passar labels para o fit
        return self.transform(texts)

    # Adicionando o método encode aqui na BaseEmbedder por consistência,
    # embora não seja abstrato. Subclasses podem sobrescrever se necessário.
    def encode(self, texts, show_progress_bar=None):
        """
        Método genérico para gerar embeddings, geralmente chamando transform.
        Assume que o embedder já foi ajustado (fit), se necessário.
        O argumento show_progress_bar é para compatibilidade com SentenceTransformer.
        """
        # A implementação padrão apenas chama transform.
        # Se uma subclasse precisa de 'fit' antes de 'encode' e 'fit' não foi chamado,
        # ela deve levantar um erro dentro de seu 'transform' ou 'encode'.
        return self.transform(texts)


    def get_embedding_dimension(self):
        """ Retorna a dimensionalidade do vetor gerado (número de colunas). """
        return getattr(self, '_embedding_dim', None)


class ProductVectorizerEmbedder(BaseEmbedder):
    """
    Usa ProductVectorizer para gerar vetores de probabilidade como embeddings.
    Inclui cache baseado em hash dos textos e parâmetros.
    O método 'fit' DEVE ser chamado com 'texts' e 'labels' antes de 'transform' ou 'encode'.
    """
    def __init__(self, vectorizer_params=None, cache_dir=".pv_embedder_cache"):
        if vectorizer_params is None:
             vectorizer_params = {'method':'tfidf', 'query':'tfidf', 'norm':'l2', 'query_norm':'l2'}
        self._vectorizer_params = vectorizer_params
        self.pv_instance = None
        self._embedding_dim = None
        self._fitted_texts_hash = None
        self.cache_dir = cache_dir

        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            # print(f"Cache para ProductVectorizerEmbedder habilitado em: {os.path.abspath(self.cache_dir)}")
        # else:
            # print("Cache para ProductVectorizerEmbedder desabilitado.")

    def _generate_data_hash(self, data):
        if isinstance(data, (list, tuple)):
            data_str = "<SEP>".join(sorted([str(item) for item in data])) # Garantir que todos são strings
        elif isinstance(data, dict):
            data_str = json.dumps(data, sort_keys=True)
        else:
            data_str = str(data)
        return hashlib.sha256(data_str.encode('utf-8')).hexdigest()

    def _get_cache_filename(self, operation_type, texts_hash, labels_hash=None):
        if not self.cache_dir:
            return None
        params_hash = self._generate_data_hash(self._vectorizer_params)
        # Para 'fit_state', o cache também depende dos labels usados no fit
        if operation_type == "fit_state" and labels_hash:
            filename = f"{operation_type}_params_{params_hash}_texts_{texts_hash}_labels_{labels_hash}.pkl"
        else:
            filename = f"{operation_type}_params_{params_hash}_texts_{texts_hash}.pkl"
        return os.path.join(self.cache_dir, filename)

    def fit(self, texts, labels): # `labels` é obrigatório aqui
        """
        Ajusta o ProductVectorizer interno usando textos e labels.
        """
        if labels is None:
            raise ValueError("ProductVectorizerEmbedder requer 'labels' para o método 'fit'.")

        texts_hash = self._generate_data_hash(texts)
        labels_hash = self._generate_data_hash(labels) # Hash dos labels
        cache_file = self._get_cache_filename("fit_state", texts_hash, labels_hash) # Cache depende de texts e labels

        if cache_file and os.path.exists(cache_file):
            try:
                # print(f"Fit: Carregando estado ajustado do cache: {cache_file}")
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                self.pv_instance = cached_data['pv_instance']
                self._embedding_dim = cached_data['embedding_dim']
                self._fitted_texts_hash = texts_hash # Hash dos textos para referência
                # print(f"Fit: Estado carregado. Dimensão: {self._embedding_dim}")
                return self
            except Exception as e:
                print(f"Fit: Erro ao carregar cache {cache_file}: {e}. Reajustando...")
                self.pv_instance = None; self._embedding_dim = None; self._fitted_texts_hash = None

        # print(f"Fit: Ajustando ProductVectorizer com {len(texts)} amostras...")
        self.pv_instance = ProductVectorizer(**self._vectorizer_params)
        self.pv_instance.fit(texts, labels) # PV usa labels
        self._embedding_dim = len(self.pv_instance.category_index) if self.pv_instance.category_index else 0
        self._fitted_texts_hash = texts_hash
        # print(f"Fit: Concluído. Dimensão: {self._embedding_dim}")

        if cache_file and self._embedding_dim > 0: # Só salva se o fit foi bem-sucedido
            try:
                # print(f"Fit: Salvando estado ajustado no cache: {cache_file}")
                state_to_save = {'pv_instance': self.pv_instance, 'embedding_dim': self._embedding_dim}
                with open(cache_file, 'wb') as f: pickle.dump(state_to_save, f)
            except Exception as e:
                 print(f"Fit: Erro ao salvar estado no cache {cache_file}: {e}")
        return self

    def transform(self, texts):
        """
        Gera os vetores de probabilidade. O método 'fit' deve ter sido chamado antes.
        """
        if self.pv_instance is None or self._embedding_dim is None:
            raise RuntimeError("Embedder (ProductVectorizerEmbedder) não foi ajustado. Chame 'fit(texts, labels)' primeiro.")
        if self._embedding_dim == 0: # Caso onde o fit não resultou em classes/dimensões
            # print("Aviso: Dimensão do embedding é 0. Retornando array vazio com shape (n_samples, 0).")
            return np.empty((len(texts), 0))

        texts_hash = self._generate_data_hash(texts)
        # O cache do transform não precisa do labels_hash, pois o estado do pv_instance (que depende dos labels) já foi carregado/criado.
        cache_file = self._get_cache_filename("transform_output", texts_hash)

        if cache_file and os.path.exists(cache_file):
             try:
                # print(f"Transform: Carregando embeddings do cache: {cache_file}")
                with open(cache_file, 'rb') as f: embeddings = pickle.load(f)
                if isinstance(embeddings, np.ndarray) and embeddings.shape[1] == self._embedding_dim:
                    # print(f"Transform: Embeddings carregados. Shape: {embeddings.shape}")
                    return embeddings
                # else: print("Transform: Dados do cache inválidos. Recalculando...")
             except Exception as e:
                  print(f"Transform: Erro ao carregar cache {cache_file}: {e}. Recalculando...")

        # print(f"Transform: Gerando embeddings para {len(texts)} textos...")
        if not texts: return np.empty((0, self._embedding_dim))

        proba_raw = self.pv_instance.predict_proba(texts)
        embeddings = proba_raw.T
        # print(f"Transform: Concluído. Shape: {embeddings.shape}")

        if cache_file:
             try:
                # print(f"Transform: Salvando embeddings no cache: {cache_file}")
                with open(cache_file, 'wb') as f: pickle.dump(embeddings, f)
             except Exception as e:
                  print(f"Transform: Erro ao salvar embeddings no cache {cache_file}: {e}")
        return embeddings

    # O método encode da BaseEmbedder já chama self.transform,
    # então não precisamos sobrescrevê-lo aqui, contanto que
    # BaseEmbedder.encode seja suficiente (ele é).


# --- Função Fábrica ATUALIZADA (mantida como antes) ---
def get_embedder(config):
    embedder_type = config.get('type')
    params = config.get('params', {}).copy()

    # print(f"Embedder Factory: Criando tipo '{embedder_type}' com params: {params}")

    if embedder_type == 'PVProb':
        vectorizer_params = params.get('vectorizer_params', {}).copy()
        if 'ngram_range' in vectorizer_params and isinstance(vectorizer_params['ngram_range'], list):
            vectorizer_params['ngram_range'] = tuple(vectorizer_params['ngram_range'])
        return ProductVectorizerEmbedder(
            vectorizer_params=vectorizer_params,
            cache_dir=params.get('cache_dir', ".pv_embedder_cache") # Default cache dir
        )
    elif embedder_type == 'ST': # Exemplo para SentenceTransformer
        from sentence_transformers import SentenceTransformer
        model_name = params.get('model_name', 'paraphrase-multilingual-mpnet-base-v2')
        # Para ST, não há um 'fit' complexo, mas podemos ter uma classe wrapper se necessário.
        # Por simplicidade, se for só para `encode`, o próprio objeto SentenceTransformer pode ser usado.
        # Mas para aderir à interface BaseEmbedder, criaríamos um wrapper:
        class SentenceTransformerEmbedderWrapper(BaseEmbedder):
            def __init__(self, model_name_st):
                self.model = SentenceTransformer(model_name_st)
                self._embedding_dim = self.model.get_sentence_embedding_dimension()
            def fit(self, texts, labels=None): pass # ST é pré-treinado
            def transform(self, texts):
                return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            # encode da BaseEmbedder já serve
        return SentenceTransformerEmbedderWrapper(model_name)
    else:
        raise ValueError(f"Tipo de embedder desconhecido: {embedder_type}")