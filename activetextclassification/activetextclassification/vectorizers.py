# activetextclassification/vectorizers.py
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.special import softmax # Importar a função softmax diretamente é conveniente

class ProductVectorizer:
    """
    Esta classe realiza a vetorização de descrições de produtos utilizando técnicas de vetorização de texto,
    como TF-IDF, para transformar texto em um formato numérico que pode ser utilizado por modelos de machine learning.
    
    Attributes:
        method (str): Método de vetorização a ser utilizado. Opções incluem 'binary', 'termfrequency', 'tfidf', entre outros.
        ngram_range (tuple): O intervalo de n-grams para a vetorização.
        norm (str): A norma a ser aplicada à matriz de termos. Opções são 'l1', 'l2' ou None.
        query_method (str): Método de vetorização para a consulta.
        norm_query (str): A norma a ser aplicada à matriz de termos da consulta.
        category_index (dict): Dicionário mapeando categorias para índices numéricos.
        index_category (dict): Dicionário mapeando índices numéricos para categorias.
        model (Any): Modelo de machine learning utilizado após a vetorização.
        vectorizer (TfidfVectorizer): Instância do vetorizador utilizado para transformar os textos.
        query (TfidfVectorizer): Instância do vetorizador utilizado para a consulta.
        out[None,'predict']: Se não informado devolve o vetor na dimensão do vocabulário
    """

    def __init__(self, method='tfidf', ngram_range=(1, 1), norm=None, query='binary', query_norm=None, out = None):
        """
        Inicializa o vetorizador de produtos com um método específico de vetorização de texto.

        Args:
            method (str): Método de vetorização para os documentos.
            ngram_range (tuple): Intervalo de n-grams para a vetorização.
            norm (str, optional): Norma a ser aplicada na vetorização.
            query (str, optional): Método de vetorização para a consulta.
            query_norm (str, optional): Norma a ser aplicada na vetorização da consulta.
        """
        self.configure(method=method, ngram_range=ngram_range, norm=norm, query=query, query_norm=query_norm, out=out)
        self.vectorizer = self.initialize_vectorizer()
        self.query = self.initialize_query()   

    def configure(self, **params):
        """
        Configura os parâmetros do vetorizador.
        """
        for param, value in params.items():
            setattr(self, param, value)
        self.fixtfidf = (self.method in {'tfidf'})

    def initialize_vectorizer(self):
        """
        Inicializa o vetorizador TfidfVectorizer com base no método especificado.

        Returns:
            TfidfVectorizer: Uma instância de TfidfVectorizer configurada.
        """
        vectorizer_options = {
            'binary': {'binary': True, 'use_idf': False},
            'termfrequency': {'binary': False, 'use_idf': False},
            'tfsmooth': {'use_idf': False, 'sublinear_tf': True},
            'tfidf': {'use_idf': True, 'smooth_idf': False},
            'tfidfscikit': {}
        }
        options = vectorizer_options.get(self.method, {})
        return TfidfVectorizer(
            token_pattern=r'(?u)\b\w+\b',
            lowercase=True,
            strip_accents='ascii',
            ngram_range=self.ngram_range,
            norm=self.norm,
            **options
        )

    def initialize_query(self):
        """
        Inicializa o vetorizador para consulta com base no método especificado.

        Returns:
            TfidfVectorizer: Uma instância de TfidfVectorizer configurada para consultas.
        """
        query_options = {
            'binary': {'binary': True, 'use_idf': False},
            'termfrequency': {'binary': False, 'use_idf': False},
            'tf_smooth': {'use_idf': False, 'sublinear_tf': True},
            'tfidf': {'use_idf': True, 'smooth_idf': False},
            'tfidfscikit': {}
        }
        options = query_options.get(self.query, {})
        return TfidfVectorizer(
            token_pattern=r'(?u)\b\w+\b',
            lowercase=True,
            strip_accents='ascii',
            ngram_range=self.ngram_range,
            norm=self.query_norm,
            **options
        )

    def fit(self, X, y=None):
        """
        Ajusta o vetorizador aos dados fornecidos e constrói o mapeamento de categorias, se fornecido.

        Args:
            X (list of str): Lista de descrições de produtos para ajustar o vetorizador.
            y (list of str, optional): Lista de categorias para cada descrição de produto.
        """
        self.original_descriptions = np.array(X)
        self.documents = X.copy()
        if y is not None:
            self.original_categories = np.array(y)
            self.build_category_mapping(y)
            self.group_descriptions_by_category()

        self.vectorizer.fit(self.documents)
        if self.fixtfidf:
            self.vectorizer.idf_ -= 1

        self.query.fit(self.documents)

    def build_category_mapping(self, categories):
        """
        Constrói o mapeamento de categorias para índices e vice-versa.

        Args:
            categories (list of str): Lista de categorias para construir o mapeamento.
        """
        unique_categories = set(categories)
        self.category_index = {category: idx for idx, category in enumerate(unique_categories)}
        self.index_category = {idx: category for category, idx in self.category_index.items()}

    def group_descriptions_by_category(self):
        """
        Agrupa as descrições por suas categorias.
        """
        self.documents = [' '.join(self.original_descriptions[self.original_categories == category]) 
                          for category in self.category_index]

    def get_term_document_matrix(self):
        """
        Retorna a matriz de termos-documento após a vetorização.

        Returns:
            pandas.DataFrame: DataFrame representando a matriz de termos-documento.
        """
        X = self.vectorizer.transform(self.documents)
        if hasattr(self, 'original_categories'):
            df = pd.DataFrame(X.toarray(), columns=self.vectorizer.get_feature_names_out(), index=self.category_index.keys())
        else:
            df = pd.DataFrame(X.toarray(), columns=self.vectorizer.get_feature_names_out())
        return df.T

    def get_term_query_matrix(self, X):
        """
        Retorna a matriz de termos-consulta após a vetorização das consultas.

        Args:
            X (list of str): Lista de consultas para vetorização.

        Returns:
            pandas.DataFrame: DataFrame representando a matriz de termos-consulta.
        """
        X = self.query.transform(X)
        df = pd.DataFrame(X.toarray(), columns=self.query.get_feature_names_out())
        return df.T

    def transform(self, X):
        """
        Transforma as descrições fornecidas usando o vetorizador de consulta.

        Args:
            X (list of str): Lista de descrições para transformação.

        Returns:
            scipy.sparse.csr_matrix: Matriz esparsa resultante da transformação.
        """
        if self.out is None:
            X = np.array(X)
            return self.query.transform(X).T
        else:
            self.predict_score(X)

    def predict(self, X, out='number'):
        """
        Realiza a predição das categorias para as descrições fornecidas.

        Args:
            X (list of str): Lista de descrições para predição.
            out (str): Formato da saída, 'number' para índices numéricos ou qualquer outro valor para categorias.

        Returns:
            numpy.ndarray: Array com as categorias preditas ou seus índices correspondentes.
        """
        y = self.predict_score(X)
        y = np.argmax(y, axis=0)
        y = np.array(y).flatten() if out == 'number' else self.get_category_from_index(np.array(y).flatten())
        return y

    def predict_score(self, X):
        """
        Calcula os escores de similaridade entre documentos e consultas.

        Args:
            X (list of str): Lista de consultas.

        Returns:
            numpy.ndarray: Matriz de escores de similaridade.
        """
        A = self.vectorizer.transform(self.documents)
        q = self.query.transform(X)
        y = A @ q.T
        return y
    
    def predict_proba(self, X):
        """
        Calcula as probabilidades de pertencimento a cada categoria para as consultas fornecidas,
        aplicando a função softmax aos escores de similaridade.

        Esta função é mais significativa quando o método `fit` foi chamado com o argumento `y` (categorias).

        Args:
            X (list of str): Lista de consultas (descrições de produtos).

        Returns:
            numpy.ndarray: Matriz de probabilidades onde a coluna j representa as probabilidades
                           para a consulta X[j] pertencer a cada categoria i.
                           Shape: (num_categories, num_inputs). Retorna array vazio se X for vazio.
        """
        if not self.category_index:
             # Se fit foi chamado sem y, não faz sentido calcular prob por categoria
             raise RuntimeError("O método 'fit' deve ser chamado com 'X' e 'y' para usar predict_proba.")

        if not isinstance(X, list): X = list(X)
        if not X: # Se a lista de entrada X for vazia
             return np.empty((len(self.category_index), 0)) # Retorna array vazio com shape (num_cat, 0)

        # 1. Obter os scores brutos
        scores = self.predict_score(X).toarray() # Shape: (num_categories, num_inputs)

        # 2. Aplicar Softmax aos scores
        # Softmax é aplicado coluna por coluna (axis=0) para que as probs de cada consulta somem 1
        probabilities = softmax(scores, axis=0)

        return probabilities

    def get_index_from_category(self, categories):
        """
        Obtém os índices numéricos correspondentes às categorias fornecidas.

        Args:
            categories (list of str): Lista de categorias.

        Returns:
            numpy.ndarray: Array com os índices numéricos correspondentes.
        """
        return np.vectorize(lambda x: self.category_index.get(x, -1))(categories)

    def get_category_from_index(self, index):
        """
        Obtém as categorias correspondentes aos índices numéricos fornecidos.

        Args:
            index (list of int): Lista de índices numéricos.

        Returns:
            numpy.ndarray: Array com as categorias correspondentes.
        """
        return np.vectorize(lambda x: self.index_category.get(x, -1))(index)

    def fit_transform(self, X, y):
        self.fit(X,y)
        return self.transform(X)
    
    def set_params(self, **params):
        """
        Define os parâmetros do vetorizador.

        Args:
            **params: Dicionário de parâmetros e seus novos valores.
        """
        self.configure(**params)
        # for param, value in params.items():
        #     if hasattr(self, param):
        #         setattr(self, param, value)
        #     else:
        #         raise ValueError(f"Parameter {param} is not valid for ProductVectorizer.")
        
        # Re-inicializa o vetorizador com os novos parâmetros
        self.vectorizer = self.initialize_vectorizer()
        self.query = self.initialize_query()

