"""Oráculo simulado — retorna rótulos já existentes no DataFrame."""

from __future__ import annotations

import logging
from typing import List, Union

import pandas as pd

from ..domain.interfaces import IOracle

logger = logging.getLogger(__name__)


class SimulatedOracle(IOracle):
    """
    Oráculo simulado para experimentos com dados já rotulados.

    Retorna os rótulos da coluna ``label_column`` diretamente, sem
    interação humana ou chamada de API.  Ideal para avaliar estratégias
    de seleção em benchmarks.

    Args:
        label_column: Nome da coluna (ou chave de dict) que contém o rótulo verdadeiro.
    """

    def __init__(self, label_column: str):
        if not label_column:
            raise ValueError("SimulatedOracle requer label_column não vazio.")
        self.label_column = label_column
        logger.debug("SimulatedOracle inicializado com coluna '%s'.", label_column)

    def query(
        self,
        data_to_label: Union[pd.DataFrame, List[dict]],
    ) -> List[str]:
        """
        Retorna os rótulos verdadeiros para o lote fornecido.

        Args:
            data_to_label: ``pd.DataFrame`` ou lista de dicionários contendo
                           a coluna ``label_column``.

        Returns:
            Lista de rótulos string.
        """
        if isinstance(data_to_label, pd.DataFrame):
            if self.label_column not in data_to_label.columns:
                raise ValueError(
                    f"Coluna '{self.label_column}' não encontrada no DataFrame."
                )
            return data_to_label[self.label_column].tolist()

        if isinstance(data_to_label, list) and all(
            isinstance(item, dict) for item in data_to_label
        ):
            try:
                return [item[self.label_column] for item in data_to_label]
            except KeyError as exc:
                raise ValueError(
                    f"Chave '{self.label_column}' não encontrada em um dos itens."
                ) from exc

        raise TypeError(
            "SimulatedOracle espera pd.DataFrame ou lista de dicionários."
        )
