# -*- coding: utf-8 -*-
"""Módulo contendo as classes de Oráculo para interação com diferentes APIs de LLM."""

import json
import time
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    def load_dotenv(*args, **kwargs): pass  # type: ignore
from unidecode import unidecode

# Importações com tratamento de erro
try: 
    from google import genai
    from google.genai import types
except ImportError: genai = None
try: from ollama import Client
except ImportError: Client = None
try: import openai
except ImportError: openai = None
try: import anthropic
except ImportError: anthropic = None

try:
    from .prompts import PROMPTS_ORACULO
except ImportError:
    PROMPTS_ORACULO = {}
    print("AVISO CRÍTICO: Não foi possível importar 'prompts.py'.")

class BaseOracle(ABC):
    """Interface abstrata para Oráculos de classificação."""
    @abstractmethod
    def query(self, batch_to_process):
        """Consulta o oráculo para obter rótulos e metadados para um lote de dados."""
        pass

class BaseLLMOracle(BaseOracle):
    """Classe base para oráculos que interagem com APIs de LLM."""
    def __init__(self, model_name, temperature, prompt_template, labels_str, retries=3, initial_timeout=60):
        self.model_name = model_name
        self.temperature = temperature
        self.prompt_template = prompt_template
        self.labels_str = labels_str
        self.retries = retries
        self.initial_timeout = initial_timeout
        self.client = None # Será definido pelas subclasses

    def _make_api_call_with_retry(self, api_func, kwargs, desc_log):
        """Wrapper genérico para chamadas de API com timeout e retentativas."""
        for attempt in range(self.retries):
            timeout = self.initial_timeout * (2 ** attempt)
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    return executor.submit(api_func, **kwargs).result(timeout=timeout)
            except TimeoutError:
                if attempt == self.retries - 1: return "TIMEOUT_ERROR"
            except Exception as e:
                print(f"ERRO INESPERADO NA API (tentativa {attempt + 1}): {e}")
                time.sleep(2)
        return None

    def _parse_llm_response(self, text_response, batch_to_process, call_id):
        """Parseia a resposta JSON do LLM e mapeia de volta para o lote original."""
        processed_records = []
        original_item_map = {i + 1: item for i, item in enumerate(batch_to_process)}
        try:
            data = json.loads(text_response or "{}")
            results_list = data.get("resultados", [])
            for result_item in results_list:
                try: item_id = int(result_item.get("id"))
                except (ValueError, TypeError): continue
                if item_id in original_item_map:
                    original_item = original_item_map.pop(item_id)
                    parsed = {"call_id": call_id, "descrição_original": original_item['text_sample'], "descrição_retornada_pelo_llm": result_item.get("descrição"), "descrição_expandida": result_item.get("descrição_expandida"), "descricao_machine_learning": unidecode(result_item.get("descrição_expandida", "").lower()), "categoria": result_item.get("categoria", "_RARE_"), "racional": result_item.get("racional"), "ground_truth_categoria": original_item.get('true_label_sample', '_NOT_AVAILABLE_'), "predicao_categoria": result_item.get("categoria", "_RARE_")}
                    processed_records.append(parsed)
        except (json.JSONDecodeError, AttributeError): pass 
        for _, missing_item in original_item_map.items():
            error_record = {"call_id": call_id, "descrição_original": missing_item['text_sample'], "racional": "Item não retornado na resposta do lote ou JSON inválido.", "predicao_categoria": "ERRO_PROCESSAMENTO", "ground_truth_categoria": missing_item.get('true_label_sample', '_NOT_AVAILABLE_')}
            processed_records.append(error_record)
        return processed_records

    @abstractmethod
    def _perform_query(self, prompt, num_items):
        """Método específico do provedor para realizar a chamada."""
        pass

    def query(self, batch_to_process):
        """Executa a consulta para um lote, orquestrando a formatação e o parsing."""
        if not batch_to_process: return [], {}
        batch_input_list = [{"id": i + 1, "descricao": item['text_sample']} for i, item in enumerate(batch_to_process)]
        descricoes_json = json.dumps(batch_input_list, indent=2, ensure_ascii=False)
        prompt = self.prompt_template.format(lista_categorias_str=self.labels_str, descricoes_lote_json=descricoes_json)
        text_response, call_log = self._perform_query(prompt, len(batch_to_process))
        results = self._parse_llm_response(text_response, batch_to_process, call_log.get("call_id"))
        return results, call_log

class OllamaOracle(BaseLLMOracle):
    """Oráculo para modelos locais via Ollama."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, initial_timeout=120)
        if not Client: raise ImportError("Biblioteca 'ollama' não instalada.")
        try:
            self.client = Client(host="http://localhost:11434"); self.client.ps()
        except Exception as e: raise RuntimeError(f"Falha ao conectar ao servidor Ollama: {e}")

    def _perform_query(self, prompt, num_items):
        call_id = str(uuid.uuid4()); start_time = time.time()
        api_func, kwargs = self.client.chat, {'model': self.model_name, 'messages': [{'role': 'user', 'content': prompt}], 'format': "json", 'stream': False, 'options': {'temperature': self.temperature}}
        
        print(f"\nAUDITORIA OLLAMA: Enviando prompt para {self.model_name}...")
        api_response = self._make_api_call_with_retry(api_func, kwargs, f"Lote de {num_items} para {self.model_name}")
        duration = time.time() - start_time
        print(f"AUDITORIA OLLAMA: Resposta bruta recebida: {api_response}")

        call_log = {"call_id": call_id, "timestamp": time.time(), "model_name": self.model_name, "provider": "ollama", "num_items_in_call": num_items, "duration_sec": duration}
        text_response = None
        if isinstance(api_response, str): call_log.update({"api_status": api_response})
        elif api_response is None: call_log.update({"api_status": "API_ERROR"})
        else:
            call_log["api_status"] = "SUCCESS"
            try:
                text_response = api_response['message']['content']; pt, ct = api_response.get('prompt_eval_count', 0) or 0, api_response.get('eval_count', 0) or 0
                call_log.update({"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct})
            except (KeyError, TypeError): call_log.update({"api_status": "PROCESSING_ERROR"})
        return text_response, call_log

class GoogleOracle(BaseLLMOracle):
    """Oráculo para modelos do Google (Gemini) usando a nova API baseada em cliente."""
    
    _client = None 
    _client_configured = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not genai:
            raise ImportError("Biblioteca 'google-generativeai' não instalada.")
        
        if not GoogleOracle._client_configured:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("Chave de API 'GEMINI_API_KEY' não encontrada no ambiente.")
            try:
                GoogleOracle._client = genai.Client(api_key=api_key)
                GoogleOracle._client_configured = True
                print("Cliente Google GenAI configurado com sucesso.")
            except Exception as e:
                raise RuntimeError(f"Falha ao configurar o cliente Gemini: {e}")

    def _perform_query(self, prompt, num_items):
        """Executa a chamada à API usando o método client.models.generate_content."""
        call_id = str(uuid.uuid4())
        start_time = time.time()
        
        api_func = GoogleOracle._client.models.generate_content
        
        # --- CORREÇÃO PRINCIPAL AQUI ---
        # Agora, tanto os parâmetros de geração quanto os de segurança
        # são definidos dentro do mesmo objeto de configuração.
        
        config_kwargs = genai.types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=self.temperature,
            safety_settings=[  # <--- safety_settings foi movido para cá
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]
        )

        kwargs = {
            'model': f'models/{self.model_name}',
            'contents': prompt,
            'config': config_kwargs, # Agora este é o único argumento de configuração
        }
        
        print(f"\nAUDITORIA GEMINI: Enviando prompt para {self.model_name}...")
        api_response = self._make_api_call_with_retry(api_func, kwargs, f"Lote de {num_items} para {self.model_name}")
        duration = time.time() - start_time
        print(f"AUDITORIA GEMINI: Resposta bruta recebida: {api_response}")

        # A lógica de log e parsing permanece a mesma
        call_log = {
            "call_id": call_id, "timestamp": time.time(), "model_name": self.model_name,
            "provider": "google", "num_items_in_call": num_items, "duration_sec": duration
        }
        text_response = None
        
        if isinstance(api_response, str):
            call_log.update({"api_status": api_response})
        elif api_response is None:
            call_log.update({"api_status": "API_ERROR", "error_message": "A chamada à API retornou None."})
        else:
            call_log["api_status"] = "SUCCESS"
            try:
                text_response = api_response.text
                meta = api_response.usage_metadata
                pt, ct = (meta.prompt_token_count or 0), (meta.candidates_token_count or 0)
                call_log.update({"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct})
            except (AttributeError, ValueError) as e:
                call_log.update({"api_status": "PROCESSING_ERROR", "error_message": f"Resposta inesperada: {e}"})
                if hasattr(api_response, 'text'):
                    text_response = api_response.text

        return text_response, call_log

class OpenaiOracle(BaseLLMOracle):
    """Oráculo para modelos da OpenAI (GPT)."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not openai: raise ImportError("Biblioteca 'openai' não instalada.")
        if not os.getenv("OPENAI_API_KEY"): raise ValueError("Chave de API 'OPENAI_API_KEY' não encontrada.")
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def _perform_query(self, prompt, num_items):
        call_id = str(uuid.uuid4()); start_time = time.time(); api_func, kwargs = self.client.chat.completions.create, {'model': self.model_name, 'messages': [{'role': 'user', 'content': prompt}], 'temperature': self.temperature, 'response_format': {'type': 'json_object'}}
        
        print(f"\nAUDITORIA OPENAI: Enviando prompt para {self.model_name}...")
        api_response = self._make_api_call_with_retry(api_func, kwargs, f"Lote de {num_items} para {self.model_name}")
        duration = time.time() - start_time
        print(f"AUDITORIA OPENAI: Resposta bruta recebida: {api_response}")

        call_log = {"call_id": call_id, "timestamp": time.time(), "model_name": self.model_name, "provider": "openai", "num_items_in_call": num_items, "duration_sec": duration}
        text_response = None
        if isinstance(api_response, str): call_log.update({"api_status": api_response})
        elif api_response is None: call_log.update({"api_status": "API_ERROR"})
        else:
            call_log["api_status"] = "SUCCESS"
            try:
                text_response = api_response.choices[0].message.content; pt, ct = api_response.usage.prompt_tokens or 0, api_response.usage.completion_tokens or 0
                call_log.update({"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct})
            except (AttributeError, IndexError): call_log.update({"api_status": "PROCESSING_ERROR"})
        return text_response, call_log

class AnthropicOracle(BaseLLMOracle):
    """Oráculo para modelos da Anthropic (Claude)."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not anthropic: raise ImportError("Biblioteca 'anthropic' não instalada.")
        if not os.getenv("ANTHROPIC_API_KEY"): raise ValueError("Chave de API 'ANTHROPIC_API_KEY' não encontrada.")
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def _perform_query(self, prompt, num_items):
        call_id = str(uuid.uuid4()); start_time = time.time()
        api_func, kwargs = self.client.messages.create, {'model': self.model_name, 'messages': [{'role': 'user', 'content': f"[\n{prompt}"}], 'temperature': self.temperature, 'max_tokens': 4096, 'system': "Responda apenas com um único objeto JSON válido que começa com '{' e termina com '}'."}
        
        print(f"\nAUDITORIA ANTHROPIC: Enviando prompt para {self.model_name}...")
        api_response = self._make_api_call_with_retry(api_func, kwargs, f"Lote de {num_items} para {self.model_name}")
        duration = time.time() - start_time
        print(f"AUDITORIA ANTHROPIC: Resposta bruta recebida: {api_response}")

        call_log = {"call_id": call_id, "timestamp": time.time(), "model_name": self.model_name, "provider": "anthropic", "num_items_in_call": num_items, "duration_sec": duration}
        text_response = None
        if isinstance(api_response, str): call_log.update({"api_status": api_response})
        elif api_response is None: call_log.update({"api_status": "API_ERROR"})
        else:
            call_log["api_status"] = "SUCCESS"
            try:
                text_response = api_response.content[0].text; pt, ct = api_response.usage.input_tokens or 0, api_response.usage.output_tokens or 0
                call_log.update({"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct})
            except (AttributeError, IndexError): call_log.update({"api_status": "PROCESSING_ERROR"})
        return text_response, call_log

def get_oracle(config, labels_str):
    """Fábrica que cria e retorna uma instância de um oráculo."""
    model_name = config.get("model_name")
    
    provider_map = {'openai': OpenaiOracle, 'anthropic': AnthropicOracle, 'google': GoogleOracle, 'ollama': OllamaOracle}
    
    provider = None
    # Lógica para encontrar provedor
    if model_name.startswith(('gpt-4', 'gpt-3.5')): provider = 'openai'
    elif model_name.startswith('claude-3'): provider = 'anthropic'
    elif model_name.startswith('gemini'): provider = 'google'
    elif model_name in ['gemma3', 'qwen2.5', 'deepseek-r1']: provider = 'ollama'
    
    OracleClass = provider_map.get(provider)
    if not OracleClass: raise ValueError(f"Provedor '{provider}' ou modelo '{model_name}' não suportado.")
        
    prompt_key = config.get("prompt_version_key", "v3_universal_batch")
    prompt_template = PROMPTS_ORACULO.get(prompt_key)
    if not prompt_template: raise ValueError(f"Chave de prompt '{prompt_key}' não encontrada. Chaves disponíveis: {list(PROMPTS_ORACULO.keys())}")

    return OracleClass(model_name=model_name, temperature=config.get("temperature", 0.2), prompt_template=prompt_template, labels_str=labels_str)