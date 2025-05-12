# activetextclassification/prompts.py

import string

# Template usando f-string (mais direto)
CLASSIFICATION_PROMPT_FSTRING = """
Você é um especialista em classificação de produtos. Analise a seguinte descrição de produto e determine a categoria mais apropriada a partir da lista fornecida.

Descrição do Produto:
"{product_description}"

Categorias Disponíveis:
{category_list_str}

Responda APENAS com a categoria exata da lista que melhor se encaixa. Não adicione nenhuma explicação extra.
"""

# Template alternativo usando string.Template (para substituição mais segura)
CLASSIFICATION_PROMPT_TEMPLATE = string.Template(
"""
    Você é um especialista em classificação de produtos. 
    Analise a seguinte descrição de produto e determine a categoria mais apropriada a partir da lista fornecida.
    Amplie as abreviações ampliando a descrição.
    Exemplo cv br lt 350 para Cerveja Brahma Lata 350ml.
    Na ampliação deve constar apenas uma descrição do produto, sem abreviações, explicações devem estar no racional. 
    Forneça uma breve explicação do porquê esta categoria foi escolhida.

Descrição do Produto:
"${product_description}"

Categorias Disponíveis:
${category_list_str}

<Exmeplo>
    {
        'predicted_category': 'cerveja',
        'augmented_description': 'Cerveja Brahma LT 350ml Malzbier',
        'rationale': "A descrição do produto menciona especificamente 'cerveja', que é uma categoria claramente definida na lista."
    }

""".strip()
)

# Schema para a resposta JSON da OpenAI
# Queremos apenas a categoria selecionada da lista
OPENAI_CLASSIFICATION_SCHEMA = {
    "name": "openai_classification",
    "schema": {
        "type": "object",
        "properties": {
            "augmented_description": { 
                "type": "string",
                "description": "Descrição ampliada, sem abrevições exemplo: cv br lt 350 para Cerveja Brahma Lata 350 ml"
            },
            "predicted_category": {
                "type": "string",
                "description": "A categoria exata selecionada da lista fornecida que melhor descreve o produto."                
            },
            "rationale": { 
                "type": "string",
                "description": "Breve explicação do porquê esta categoria foi escolhida."
            },
        },
        "required": ["predicted_category"],
        "additionalProperties": False # Evitar que o modelo adicione campos extras
    }
}

# Se você também quisesse o 'código da categoria' (assumindo que você tem um mapeamento):
OPENAI_CLASSIFICATION_WITH_CODE_SCHEMA = {
    "type": "object",
    "properties": {
        "predicted_category_name": {
            "type": "string",
            "description": "O nome exato da categoria selecionada da lista fornecida."
        },
        "predicted_category_code": {
            "type": "integer", 
            "description": "O código correspondente à categoria selecionada."
        },
        "rationale": { 
            "type": "string",
            "description": "Breve explicação do porquê esta categoria foi escolhida."
        }
    },
    "required": ["predicted_category_name", "predicted_category_code", "rationale"],
    "additionalProperties": False
}