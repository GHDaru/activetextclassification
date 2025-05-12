# activetextclassification/models.py

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

# Importar classificadores sklearn que queremos suportar
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, SGDClassifier

# Importar nosso vetorizador base se PVClassifier for usá-lo
from .vectorizers import ProductVectorizer


# ==================================================
# Classe Base para Modelos baseados em FEATURES
# ==================================================
class BaseFeatureClassifier(ABC):
    """
    Classe base abstrata para classificadores que operam sobre
    matrizes de features numéricas pré-calculadas (embeddings).
    """
    def __init__(self, model_params=None):
        self.model_params = model_params if model_params is not None else {}
        self.model = None
        self._classes = None
        self.label_to_id = None
        self.id_to_label = None

    def _create_label_mapping(self, y_train_labels):
        self._classes = sorted(list(pd.unique(y_train_labels)))
        self.label_to_id = {label: i for i, label in enumerate(self._classes)}
        self.id_to_label = {i: label for label, i in self.label_to_id.items()}

    def _labels_to_ids(self, labels):
        if self.label_to_id is None: raise RuntimeError("Mapeamento não criado. Chame fit.")
        try: return np.array([self.label_to_id[lbl] for lbl in labels])
        except KeyError as e: raise ValueError(f"Label '{e}' não visto no fit.") from e

    def _ids_to_labels(self, ids):
        if self.id_to_label is None: raise RuntimeError("Mapeamento não criado. Chame fit.")
        try: return np.array([self.id_to_label[id_] for id_ in ids])
        except KeyError as e: raise ValueError(f"ID '{e}' inválido.") from e

    @abstractmethod
    def fit(self, X_features, y_labels):
        pass

    @abstractmethod
    def predict(self, X_features):
        pass

    @abstractmethod
    def predict_proba(self, X_features):
        pass

    def get_classes(self):
        return self._classes


# ==================================================
# Wrapper para Modelos Sklearn (baseados em Features)
# ==================================================
class SklearnFeatureClassifier(BaseFeatureClassifier):
    """ Wrapper genérico para classificadores sklearn. """
    def __init__(self, sklearn_model_class, model_params=None):
        super().__init__(model_params)
        try:
             self.model = sklearn_model_class(**self.model_params)
        except TypeError as e:
             raise ValueError(f"Erro ao inicializar {sklearn_model_class.__name__}: {e}") from e

    def fit(self, X_features, y_labels):
        print(f"Fitting {type(self.model).__name__} com {X_features.shape[0]} amostras...")
        self._create_label_mapping(y_labels)
        y_train_ids = self._labels_to_ids(y_labels)
        self.model.fit(X_features, y_train_ids)
        print("Fit concluído.")
        # Verificação opcional de classes
        if hasattr(self.model, 'classes_') and not np.array_equal(self.model.classes_, np.arange(len(self._classes))):
             print("AVISO: Classes internas do sklearn não batem com mapeamento!")
        return self

    def predict(self, X_features):
        if self.model is None or self._classes is None: raise RuntimeError("Modelo não treinado.")
        predicted_ids = self.model.predict(X_features)
        return self._ids_to_labels(predicted_ids)

    def predict_proba(self, X_features):
        if self.model is None or self._classes is None: raise RuntimeError("Modelo não treinado.")
        if not hasattr(self.model, "predict_proba"):
            # Tentar retornar NaN ou similar em vez de erro? Ou erro é melhor? Erro força fallback.
            # Vamos retornar array de NaNs com shape correto.
            print(f"AVISO: {type(self.model).__name__} não suporta predict_proba. Retornando NaNs.")
            return np.full((X_features.shape[0], len(self._classes)), np.nan)
            # raise NotImplementedError(f"{type(self.model).__name__} não suporta predict_proba.")

        probabilities = self.model.predict_proba(X_features)
        if probabilities.shape[1] != len(self._classes):
             print(f"AVISO: Shape predict_proba ({probabilities.shape}) != n_classes ({len(self._classes)}).")
             # Preencher com zeros para classes ausentes? Perigoso. Melhor retornar como está.
        return probabilities


# ==================================================
# Classe Base para Modelos baseados em TEXTO
# ==================================================
class BaseTextClassifier(ABC):
    """
    Classe base abstrata para classificadores que operam diretamente sobre texto.
    """
    def __init__(self, model_params=None): # Adicionado __init__ básico
        self.model_params = model_params if model_params is not None else {}
        self._classes = None
        self.label_to_id = None
        self.id_to_label = None

    @abstractmethod
    def fit(self, X_train, y_train):
        """ Treina com listas de texto e labels string. """
        pass

    @abstractmethod
    def predict(self, X):
        """ Prediz labels string a partir de lista de texto. """
        pass

    @abstractmethod
    def predict_proba(self, X):
        """ Prediz probabilidades (n_samples, n_classes) a partir de lista de texto. """
        pass

    def get_classes(self):
        """ Retorna a lista ordenada de nomes de classes. """
        return self._classes


# ==================================================
# Wrapper para ProductVectorizer (baseado em Texto)
# ==================================================
class ProductVectorizerClassifier(BaseTextClassifier):
    """ Wrapper para ProductVectorizer como um BaseTextClassifier. """
    def __init__(self, vectorizer_params=None): # Mantido simples
        # Chama __init__ da BaseTextClassifier
        super().__init__(model_params=vectorizer_params) # Passa params para BaseTextClassifier
        # Renomeado _vectorizer_params para _model_params herdado
        # self._vectorizer_params = vectorizer_params if vectorizer_params is not None else {}
        self.pv_instance = ProductVectorizer(**self.model_params) # Usa self.model_params

    def fit(self, X_train, y_train):
        print(f"Fitting ProductVectorizerClassifier com {len(X_train)} amostras...")
        # O fit do PV cria os mapeamentos internos dele
        self.pv_instance.fit(X_train, y_train)
        # Copiar mapeamentos e classes para a interface da Base
        self.label_to_id = self.pv_instance.category_index
        self.id_to_label = self.pv_instance.index_category
        if self.id_to_label:
            # Garantir ordem consistente (alfabética ou pela ordem do PV)
            self._classes = sorted(list(self.id_to_label.values()))
            # Recriar mapeamentos baseados na ordem alfabética para consistência?
            # Ou confiar na ordem do PV? Vamos confiar na ordem do PV por enquanto.
            # self._classes = self.pv_instance.get_category_from_index(np.arange(len(self.label_to_id)))
        print("Fit concluído.")
        return self

    def predict(self, X):
        if self.pv_instance is None or self._classes is None: raise RuntimeError("Modelo não treinado.")
        return self.pv_instance.predict(X, out='category') # PV já retorna string

    def predict_proba(self, X):
        if self.pv_instance is None or self._classes is None: raise RuntimeError("Modelo não treinado.")
        proba_raw = self.pv_instance.predict_proba(X) # (n_classes_pv, n_samples)

        # Garantir ordem das classes e shape (n_samples, n_classes)
        internal_classes = self.pv_instance.get_category_from_index(np.arange(len(self.pv_instance.category_index)))
        internal_class_map = {label: i for i, label in enumerate(internal_classes)}

        # Criar matriz de saída na ordem self._classes
        output_proba = np.zeros((len(X), len(self._classes)))

        for i, target_label in enumerate(self._classes):
             if target_label in internal_class_map:
                 internal_idx = internal_class_map[target_label]
                 output_proba[:, i] = proba_raw[internal_idx, :]
             # else: Deixa como zero se uma classe target não foi vista no fit interno do PV

        return output_proba # Shape (n_samples, n_classes)


# ==================================================
# Função Fábrica de Modelos
# ==================================================
def get_model(config):
    """ Instancia um modelo classificador baseado na configuração. """
    model_type = config.get('type')
    params = config.get('params', {}).copy() # Trabalhar com cópia

    print(f"Model Factory: Criando tipo '{model_type}' com params: {params}")

    if model_type == 'PVBin':
        # Converter ngram_range se necessário
        if 'ngram_range' in params and isinstance(params['ngram_range'], list):
            original_ngram = params['ngram_range']
            params['ngram_range'] = tuple(original_ngram)
            print(f"Model Factory (PVBin): Convertido ngram_range {original_ngram} para tupla {params['ngram_range']}")
        # PVBinClassifier usa os params diretamente como vectorizer_params
        return ProductVectorizerClassifier(vectorizer_params=params)

    elif model_type in ['GNB', 'LSVC', 'LR', 'SGD']:
        # Estes são BaseFeatureClassifier via SklearnFeatureClassifier
        model_map = {'GNB': GaussianNB, 'LSVC': LinearSVC, 'LR': LogisticRegression, 'SGD': SGDClassifier}
        sklearn_class = model_map[model_type]

        # Avisos sobre predict_proba
        if model_type == 'LSVC': print("AVISO: LinearSVC não suporta predict_proba diretamente.")
        if model_type == 'SGD' and params.get('loss','').endswith('_loss') is False : print("AVISO: SGDClassifier sem loss='*_loss' não suporta predict_proba.")

        # SklearnFeatureClassifier recebe a classe e os params
        return SklearnFeatureClassifier(sklearn_class, model_params=params)
    else:
        raise ValueError(f"Tipo de modelo desconhecido: {model_type}")