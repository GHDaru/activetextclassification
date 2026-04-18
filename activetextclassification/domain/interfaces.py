"""
activetextclassification.domain.interfaces
==========================================
Contratos (ABCs) que definem as interfaces de cada componente do ciclo de
aprendizado ativo.  Nenhuma camada pode importar de outra — apenas deste módulo.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# IClassifier
# ---------------------------------------------------------------------------

class IClassifier(ABC):
    """Interface para classificadores de texto ou de features numéricas."""

    @abstractmethod
    def fit(self, X, y_labels: List[str]) -> "IClassifier":
        """
        Treina o classificador.

        Args:
            X: Entradas — lista de textos *ou* np.ndarray de features.
            y_labels: Lista de rótulos string correspondentes.

        Returns:
            self
        """

    @abstractmethod
    def predict(self, X) -> np.ndarray:
        """
        Prediz rótulos string.

        Returns:
            np.ndarray de shape (n_samples,) com rótulos string.
        """

    @abstractmethod
    def predict_proba(self, X) -> np.ndarray:
        """
        Prediz probabilidades por classe.

        Returns:
            np.ndarray de shape (n_samples, n_classes).
        """

    def get_classes(self) -> Optional[List[str]]:
        """Retorna a lista ordenada de rótulos vistos no fit."""
        return getattr(self, "_classes", None)


# ---------------------------------------------------------------------------
# IEmbedder
# ---------------------------------------------------------------------------

class IEmbedder(ABC):
    """Interface para geradores de embeddings / features a partir de texto."""

    @abstractmethod
    def fit(self, texts: List[str], labels: Optional[List[str]] = None) -> "IEmbedder":
        """
        Ajusta o embedder (pode ser no-op para modelos pré-treinados).

        Args:
            texts:  Lista de textos para treino.
            labels: Rótulos opcionais (necessários para alguns embedders).

        Returns:
            self
        """

    @abstractmethod
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transforma textos em matriz numérica.

        Returns:
            np.ndarray de shape (n_samples, embedding_dim).
        """

    def fit_transform(
        self, texts: List[str], labels: Optional[List[str]] = None
    ) -> np.ndarray:
        """Combina fit e transform."""
        self.fit(texts, labels)
        return self.transform(texts)

    def encode(self, texts: List[str], **kwargs) -> np.ndarray:
        """Alias de transform para compatibilidade com SentenceTransformer."""
        return self.transform(texts)

    def get_embedding_dimension(self) -> Optional[int]:
        """Retorna a dimensionalidade do vetor gerado."""
        return getattr(self, "_embedding_dim", None)


# ---------------------------------------------------------------------------
# IOracle
# ---------------------------------------------------------------------------

class IOracle(ABC):
    """Interface para oráculos de rotulação."""

    @abstractmethod
    def query(self, data_to_label) -> List:
        """
        Obtém rótulos para os dados fornecidos.

        Args:
            data_to_label: pd.DataFrame ou lista de dicts com os itens a rotular.

        Returns:
            Lista de rótulos string (pode conter None em caso de falha por item).
        """


# ---------------------------------------------------------------------------
# IQueryStrategy
# ---------------------------------------------------------------------------

class IQueryStrategy(ABC):
    """Interface para estratégias de seleção do próximo lote de consulta."""

    @abstractmethod
    def select(
        self,
        pool_size: int,
        probabilities: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Seleciona índices do pool para consulta.

        Args:
            pool_size:     Tamanho total do pool não rotulado.
            probabilities: Array (n_pool, n_classes) de probabilidades preditas.
                           Pode ser None para estratégias puramente aleatórias.
            rng:           Gerador de números aleatórios (para reprodutibilidade).
                           Se None, usa np.random.default_rng().

        Returns:
            np.ndarray de índices selecionados (inteiros).
        """

    @property
    @abstractmethod
    def batch_size(self) -> int:
        """Quantidade de instâncias selecionadas por chamada."""


# ---------------------------------------------------------------------------
# IColdStart
# ---------------------------------------------------------------------------

class IColdStart(ABC):
    """Interface para estratégias de seleção do lote inicial L0."""

    @abstractmethod
    def select(
        self,
        U_df,
        n_initial: int,
        embeddings: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Seleciona os índices do lote inicial.

        Args:
            U_df:       DataFrame do pool não rotulado.
            n_initial:  Número de amostras a selecionar.
            embeddings: Embeddings de U_df (necessário para estratégias baseadas em clustering).
            rng:        Gerador de números aleatórios.

        Returns:
            np.ndarray de índices (posições em U_df) selecionados.
        """
