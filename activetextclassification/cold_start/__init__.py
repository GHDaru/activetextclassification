"""
activetextclassification.cold_start
=====================================
Estratégias de inicialização do conjunto rotulado (L0).
"""

# ── New DDD classes ──────────────────────────────────────────────────────
from .random_cold_start import RandomColdStart
from .kmedians_cold_start import KMediansColdStart
from .factory import get_cold_start, REGISTRY as COLD_START_REGISTRY

# ── Legacy functional API (backward compat) ──────────────────────────────
from ..cold_start_legacy import select_initial_batch, random_cold_start, kmedians_cold_start

__all__ = [
    # DDD classes
    "RandomColdStart",
    "KMediansColdStart",
    "get_cold_start",
    "COLD_START_REGISTRY",
    # Legacy
    "select_initial_batch",
    "random_cold_start",
    "kmedians_cold_start",
]
