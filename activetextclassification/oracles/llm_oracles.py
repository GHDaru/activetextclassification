"""Oráculos LLM (Ollama, Google Gemini, OpenAI, Anthropic)."""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Any, Dict, List, Optional, Tuple

from ..domain.interfaces import IOracle

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Importações opcionais — falha silenciosa
# ---------------------------------------------------------------------------
try:
    from google import genai as _genai
    from google.genai import types as _genai_types
except ImportError:
    _genai = None  # type: ignore
    _genai_types = None  # type: ignore

try:
    from ollama import Client as _OllamaClient  # type: ignore
except ImportError:
    _OllamaClient = None  # type: ignore

try:
    import openai as _openai  # type: ignore
except ImportError:
    _openai = None  # type: ignore

try:
    import anthropic as _anthropic  # type: ignore
except ImportError:
    _anthropic = None  # type: ignore

try:
    from unidecode import unidecode as _unidecode  # type: ignore
except ImportError:
    _unidecode = lambda x: x  # type: ignore


# ---------------------------------------------------------------------------
# BaseLLMOracle
# ---------------------------------------------------------------------------

class BaseLLMOracle(IOracle):
    """
    Classe base para oráculos que interagem com APIs de LLM.

    Subclasses implementam ``_perform_query()`` específico do provedor.
    """

    def __init__(
        self,
        model_name: str,
        temperature: float,
        prompt_template: str,
        labels_str: str,
        retries: int = 3,
        initial_timeout: int = 60,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.prompt_template = prompt_template
        self.labels_str = labels_str
        self.retries = retries
        self.initial_timeout = initial_timeout
        self.client: Any = None

    # ------------------------------------------------------------------ #
    #  Retry wrapper                                                       #
    # ------------------------------------------------------------------ #

    def _make_api_call_with_retry(
        self, api_func, kwargs: dict, desc_log: str
    ) -> Any:
        for attempt in range(self.retries):
            timeout = self.initial_timeout * (2 ** attempt)
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    return executor.submit(api_func, **kwargs).result(
                        timeout=timeout
                    )
            except TimeoutError:
                if attempt == self.retries - 1:
                    return "TIMEOUT_ERROR"
            except Exception as exc:
                logger.warning(
                    "Erro inesperado na API (tentativa %d/%d): %s",
                    attempt + 1,
                    self.retries,
                    exc,
                )
                time.sleep(2)
        return None

    # ------------------------------------------------------------------ #
    #  Response parsing                                                    #
    # ------------------------------------------------------------------ #

    def _parse_llm_response(
        self, text_response: Optional[str], batch_to_process: List[dict], call_id: str
    ) -> List[dict]:
        processed: List[dict] = []
        original_map = {i + 1: item for i, item in enumerate(batch_to_process)}
        try:
            data = json.loads(text_response or "{}")
            for res_item in data.get("resultados", []):
                try:
                    item_id = int(res_item.get("id"))
                except (ValueError, TypeError):
                    continue
                if item_id in original_map:
                    orig = original_map.pop(item_id)
                    parsed = {
                        "call_id": call_id,
                        "descrição_original": orig["text_sample"],
                        "descrição_retornada_pelo_llm": res_item.get("descrição"),
                        "descrição_expandida": res_item.get("descrição_expandida"),
                        "descricao_machine_learning": _unidecode(
                            res_item.get("descrição_expandida", "").lower()
                        ),
                        "categoria": res_item.get("categoria", "_RARE_"),
                        "racional": res_item.get("racional"),
                        "ground_truth_categoria": orig.get(
                            "true_label_sample", "_NOT_AVAILABLE_"
                        ),
                        "predicao_categoria": res_item.get("categoria", "_RARE_"),
                    }
                    processed.append(parsed)
        except (json.JSONDecodeError, AttributeError):
            pass

        # Itens ausentes na resposta
        for _, missing in original_map.items():
            processed.append(
                {
                    "call_id": call_id,
                    "descrição_original": missing["text_sample"],
                    "racional": "Item não retornado na resposta do lote.",
                    "predicao_categoria": "ERRO_PROCESSAMENTO",
                    "ground_truth_categoria": missing.get(
                        "true_label_sample", "_NOT_AVAILABLE_"
                    ),
                }
            )
        return processed

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def query(self, batch_to_process: List[dict]) -> Tuple[List[dict], dict]:  # type: ignore[override]
        if not batch_to_process:
            return [], {}
        batch_input = [
            {"id": i + 1, "descricao": item["text_sample"]}
            for i, item in enumerate(batch_to_process)
        ]
        descricoes_json = json.dumps(batch_input, indent=2, ensure_ascii=False)
        prompt = self.prompt_template.format(
            lista_categorias_str=self.labels_str,
            descricoes_lote_json=descricoes_json,
        )
        text_response, call_log = self._perform_query(prompt, len(batch_to_process))
        results = self._parse_llm_response(
            text_response, batch_to_process, call_log.get("call_id", "")
        )
        return results, call_log

    @abstractmethod
    def _perform_query(
        self, prompt: str, num_items: int
    ) -> Tuple[Optional[str], dict]:
        """Implementação específica do provedor."""


# ---------------------------------------------------------------------------
# OllamaOracle
# ---------------------------------------------------------------------------

class OllamaOracle(BaseLLMOracle):
    """Oráculo para modelos locais via Ollama."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, initial_timeout=120, **kwargs)
        if not _OllamaClient:
            raise ImportError("Biblioteca 'ollama' não instalada.  Execute: pip install ollama")
        try:
            self.client = _OllamaClient(host="http://localhost:11434")
            self.client.ps()
        except Exception as exc:
            raise RuntimeError(f"Falha ao conectar ao servidor Ollama: {exc}") from exc

    def _perform_query(self, prompt: str, num_items: int) -> Tuple[Optional[str], dict]:
        call_id = str(uuid.uuid4())
        start = time.time()
        api_response = self._make_api_call_with_retry(
            self.client.chat,
            {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "format": "json",
                "stream": False,
                "options": {"temperature": self.temperature},
            },
            f"Lote de {num_items} para {self.model_name}",
        )
        duration = time.time() - start
        call_log: dict = {
            "call_id": call_id,
            "timestamp": time.time(),
            "model_name": self.model_name,
            "provider": "ollama",
            "num_items_in_call": num_items,
            "duration_sec": duration,
        }
        text_response = None
        if isinstance(api_response, str):
            call_log["api_status"] = api_response
        elif api_response is None:
            call_log["api_status"] = "API_ERROR"
        else:
            call_log["api_status"] = "SUCCESS"
            try:
                text_response = api_response["message"]["content"]
                pt = api_response.get("prompt_eval_count", 0) or 0
                ct = api_response.get("eval_count", 0) or 0
                call_log.update({"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct})
            except (KeyError, TypeError):
                call_log["api_status"] = "PROCESSING_ERROR"
        return text_response, call_log


# ---------------------------------------------------------------------------
# GoogleOracle (Gemini)
# ---------------------------------------------------------------------------

class GoogleOracle(BaseLLMOracle):
    """Oráculo para modelos Google Gemini."""

    _client = None
    _client_configured = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not _genai:
            raise ImportError("Instale: pip install google-generativeai")
        if not GoogleOracle._client_configured:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("Variável de ambiente 'GEMINI_API_KEY' não encontrada.")
            try:
                GoogleOracle._client = _genai.Client(api_key=api_key)
                GoogleOracle._client_configured = True
            except Exception as exc:
                raise RuntimeError(f"Falha ao configurar cliente Gemini: {exc}") from exc

    def _perform_query(self, prompt: str, num_items: int) -> Tuple[Optional[str], dict]:
        call_id = str(uuid.uuid4())
        start = time.time()
        cfg = _genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=self.temperature,
            safety_settings=[
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ],
        )
        api_response = self._make_api_call_with_retry(
            GoogleOracle._client.models.generate_content,
            {"model": f"models/{self.model_name}", "contents": prompt, "config": cfg},
            f"Lote de {num_items} para {self.model_name}",
        )
        duration = time.time() - start
        call_log: dict = {
            "call_id": call_id,
            "timestamp": time.time(),
            "model_name": self.model_name,
            "provider": "google",
            "num_items_in_call": num_items,
            "duration_sec": duration,
        }
        text_response = None
        if isinstance(api_response, str):
            call_log["api_status"] = api_response
        elif api_response is None:
            call_log.update({"api_status": "API_ERROR", "error_message": "None retornado."})
        else:
            call_log["api_status"] = "SUCCESS"
            try:
                text_response = api_response.text
                meta = api_response.usage_metadata
                pt = (meta.prompt_token_count or 0) if meta else 0
                ct = (meta.candidates_token_count or 0) if meta else 0
                call_log.update({"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct})
            except (AttributeError, ValueError) as exc:
                call_log.update({"api_status": "PROCESSING_ERROR", "error_message": str(exc)})
        return text_response, call_log


# ---------------------------------------------------------------------------
# OpenaiOracle
# ---------------------------------------------------------------------------

class OpenaiOracle(BaseLLMOracle):
    """Oráculo para modelos da OpenAI (GPT)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not _openai:
            raise ImportError("Instale: pip install openai")
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("Variável de ambiente 'OPENAI_API_KEY' não encontrada.")
        self.client = _openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def _perform_query(self, prompt: str, num_items: int) -> Tuple[Optional[str], dict]:
        call_id = str(uuid.uuid4())
        start = time.time()
        api_response = self._make_api_call_with_retry(
            self.client.chat.completions.create,
            {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "response_format": {"type": "json_object"},
            },
            f"Lote de {num_items} para {self.model_name}",
        )
        duration = time.time() - start
        call_log: dict = {
            "call_id": call_id,
            "timestamp": time.time(),
            "model_name": self.model_name,
            "provider": "openai",
            "num_items_in_call": num_items,
            "duration_sec": duration,
        }
        text_response = None
        if isinstance(api_response, str):
            call_log["api_status"] = api_response
        elif api_response is None:
            call_log["api_status"] = "API_ERROR"
        else:
            call_log["api_status"] = "SUCCESS"
            try:
                text_response = api_response.choices[0].message.content
                pt = api_response.usage.prompt_tokens or 0
                ct = api_response.usage.completion_tokens or 0
                call_log.update({"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct})
            except (AttributeError, IndexError):
                call_log["api_status"] = "PROCESSING_ERROR"
        return text_response, call_log


# ---------------------------------------------------------------------------
# AnthropicOracle
# ---------------------------------------------------------------------------

class AnthropicOracle(BaseLLMOracle):
    """Oráculo para modelos da Anthropic (Claude)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not _anthropic:
            raise ImportError("Instale: pip install anthropic")
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise ValueError("Variável de ambiente 'ANTHROPIC_API_KEY' não encontrada.")
        self.client = _anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def _perform_query(self, prompt: str, num_items: int) -> Tuple[Optional[str], dict]:
        call_id = str(uuid.uuid4())
        start = time.time()
        api_response = self._make_api_call_with_retry(
            self.client.messages.create,
            {
                "model": self.model_name,
                "messages": [{"role": "user", "content": f"[\n{prompt}"}],
                "temperature": self.temperature,
                "max_tokens": 4096,
                "system": "Responda apenas com um único objeto JSON válido.",
            },
            f"Lote de {num_items} para {self.model_name}",
        )
        duration = time.time() - start
        call_log: dict = {
            "call_id": call_id,
            "timestamp": time.time(),
            "model_name": self.model_name,
            "provider": "anthropic",
            "num_items_in_call": num_items,
            "duration_sec": duration,
        }
        text_response = None
        if isinstance(api_response, str):
            call_log["api_status"] = api_response
        elif api_response is None:
            call_log["api_status"] = "API_ERROR"
        else:
            call_log["api_status"] = "SUCCESS"
            try:
                text_response = api_response.content[0].text
                pt = api_response.usage.input_tokens or 0
                ct = api_response.usage.output_tokens or 0
                call_log.update({"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct})
            except (AttributeError, IndexError):
                call_log["api_status"] = "PROCESSING_ERROR"
        return text_response, call_log
