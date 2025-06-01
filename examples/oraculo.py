# Código da Célula 1
lista_categorias_str = '["_RARE_", "abacate", "abacaxi", "abobora", "abobrinha", "absorvente", "acai", "acelga", "acendedor", "acessorio de audio", "acessorio de informatica", "acessorio de piscina", "acessorio de video", "achocolatado", "acucar", "adocante", "agriao", "agua", "agua de coco", "agua micelar", "agua oxigenada", "agua sanitaria", "agua tonica", "agulha descartavel", "alcool", "alecrim", "alface", "algodao", "alho", "alho poro", "alho processado", "alicate de unha", "alimento pet", "alimento pronto de bebe", "almeirao", "almofada", "almondega congelada", "alvejante", "amaciante", "ameixa", "amendoim", "amido de milho", "amora", "antisseptico bucal", "aparelho de barbear e depilatorio", "aperitivo", "apontador", "apresuntado", "ar condicionado", "aromatizador", "arroz", "artigo de natal", "aspargo", "aspirador", "assadeira", "assento sanitario", "automotivo", "aveia e seus derivados", "azeite de dende", "azeite de oliva", "azeitona", "bacalhau a peso", "bacalhau congelado", "bala", "balao de festa", "balde", "banana", "bananada", "banco", "barra alimentar", "batata", "batata congelada", "bebida de soja", "bebida lactea", "bebida mista", "berinjela", "bermuda", "beterraba", "bicarbonato de sodio", "biquini", "biscoito", "blusa", "body infantil", "boia", "bola", "bolsa termica", "bombom", "bone", "boneca", "boneco", "borracha", "borrifador/pulverizador", "brinquedo", "brinquedo pet", "brocolis", "cabide", "cachaca", "cadeira", "cadeira de escritorio", "cadeira de praia", "caderno", "cafe", "cafeteira", "caixa de som", "caixa organizadora", "caixa termica", "calca", "calcinha", "caldo industrializado", "camisa polo", "camisa termica", "camisa uv", "camiseta", "caneca", "canela", "caneta", "canjica", "canudo", "caqui", "cardigan/sueter", "carne bovina", "carne de ave", "carne de ave sazonal", "carne ovina", "carne suina", "carne suina sazonal", "carne vegetal", "carrinho", "cartucho de impressao", "carvao", "casquinha de sorvete", "cebola", "cebolinha", "celular", "cenoura", "cereal infantil ou farinha lactea", "cereal matinal", "cerveja", "cesta basica", "cesta de pascoa", "cha por infusao", "cha pronto", "chaleira eletrica", "charque e carne seca", "cheiro verde", "chicoria", "chinelo", "chocolate", "chocolate em po", "chuchu", "chupeta", "churrasqueira", "cigarro", "coador", "cobertor", "cobertura pronta", "coco seco", "coco verde", "coelho de chocolate", "coentro", "cogumelo", "cola", "colchao/colchonete", "colonia", "colonia infantil", "colorifico", "comedouro", "complemento alimentar", "condicionador", "condicionador infantil", "conhaque", "conserva doce", "copo", "copo descartavel", "corante alimentar", "corante de pintura", "corda de varal", "corretivo", "couve", "couve flor", "cravo da india", "creme corporal", "creme de avela", "creme de leite", "creme de tratamento", "creme dental", "creme dental infantil", "creme para assadura", "creme para cabelo", "creme para pentear", "cueca", "curativo adesivo", "curry", "demais funcionais e restritivos", "derivados do coco seco", "desengordurante", "desinfetante", "desodorante", "desumidificador", "detergente", "doce de amendoim", "ducha", "embalagem de presente", "empanado congelado", "energetico", "enxaguante bucal", "equipamento de limpeza piscina", "erva mate", "ervilha", "ervilha em conserva", "ervilha seca", "escova de dentes", "escova de limpeza", "escova e esponja para unha", "escova e pente para cabelo", "esmalte", "esparadrapo", "espelho decorativo", "espetinho congelado", "espeto", "espeto descartavel", "espiga de milho", "espinafre", "esponja", "esponja de banho", "espremedor de frutas", "espuma ou gel de barbear", "espumante", "essencia alimentar", "estante", "extensao eletrica", "extrato de tomate", "farinha de arroz", "farinha de linhaca", "farinha de mandioca", "farinha de milho", "farinha de trigo", "farinha para empanar", "farofa pronta", "feijao", "fermento", "ferro de passar", "finalizador leave in", "fio dental", "fisioterapia", "fita adesiva", "fita isolante", "fixador de dentadura", "floco de milho", "folha de aluminio", "fone de ouvido", "forma culinaria", "formula lactea", "forno eletrico", "forragem pet", "fosforo", "fralda", "fralda adulto", "fritadeira eletrica", "fronha", "fruta seca", "fruto do mar", "fuba", "garrafa termica", "gaze", "gel antisseptico para maos", "gelatina", "geleia", "geleia de mocoto", "gelo", "gengibre", "gin", "giz", "goiaba", "goiabada", "goma de mascar", "gordura vegetal", "gorro", "granola", "granular higienico", "grao de bico", "guarda-chuva", "guardanapo", "hamburguer congelado", "hamper", "haste flexivel", "higienizador de verduras", "hortela", "humus ou terra para jardinagem", "inativo", "inhame", "inseticida", "insumo", "interruptor", "iogurte", "isotonico", "isqueiro", "jaqueta", "jilo", "jogo americano", "jornal", "ketchup", "kit de molhos", "kit higiene bucal", "kit para cabelos", "kiwi", "lampada", "lapis", "laranja", "lasanha congelada", "lava louca", "lava roupas", "legging", "leite", "leite condensado", "leite de coco", "leite em po", "leite especial", "leite fermentado", "lenco de papel", "lenco umedecido", "lencol", "lentilha", "licor", "limao", "limpa vidros", "linguica", "liquidificador", "livro/revista", "lixa", "lixa profissional", "lixeira", "louro", "lustra moveis", "luva", "luvas de borracha", "maca", "macacao", "macarrao", "maionese", "mamao", "mandioca", "mandioquinha", "manga", "manjericao", "manta", "manta infantil", "manteiga", "maquina de lavar", "maracuja", "marcador de texto", "margarina", "mascara de protecao", "massa de lasanha", "massa de modelar", "massa de pao de queijo", "massa de pizza", "massa fresca", "maxixe", "meia", "meia calca", "mel", "melancia", "melao", "mexerica", "micro-ondas", "milho", "milho para pipoca", "milho verde e ervilha em conserva", "milho verde em conserva", "mix de folhas", "mix de frios", "modelador para cabelo", "modificadores de leite", "molho barbecue", "molho de pimenta", "molho de tomate", "molho ingles", "molho para salada", "molho shoyu", "morango", "mordedor", "mortadela", "mostarda", "mouse", "multiuso", "naftalina", "nectarina", "nozes, castanhas e outras oleaginosas", "oculos", "oleo corporal", "oleo de cabelo", "oleo vegetal", "oregano", "outra bebida", "outra bebida vegetal", "outra carne sazonal", "outra fruta", "outra verdura", "outro acessorio para cabelos", "outro bazar e departamentos", "outro condimento", "outro congelado", "outro cosmeticos e perfumaria", "outro decoracao e jardim", "outro descartavel de casa", "outro destilado", "outro doce", "outro embutido", "outro farma", "outro ferramentas e equipamentos", "outro grao", "outro home center", "outro lacteo", "outro laticinio refrigeirado", "outro legume", "outro molho", "outro papelaria", "outro pet shop", "outro produto de lavanderia", "outro produto de limpeza domestica", "outro resfriado", "outro telefonia", "outro utensilio de cozinha", "outro utensilio de limpeza", "outro vegetal processado", "ovo", "ovo de pascoa", "pa de lixo", "pacoca", "padaria e confeitaria fresca", "padaria e confeitaria industrializado", "palha de aco", "palito de dentes", "palito para unha", "palmito", "panela", "panela de pressao", "panificacao natal", "panificacao pascoa", "pano de prato", "pano para limpeza", "pantufa", "papel higienico", "papel manteiga", "papel para assar", "papel sulfite", "papel toalha", "passa facil", "pasta de amendoim", "pastilha para piso", "pate", "pedra sanitaria", "peito de peru", "peixe", "pelucia", "pepino", "pera", "perfume", "pessego", "petisco e salgadinho congelado", "picole", "pijama", "pilha", "pimenta", "pimentao", "pinca", "pinhao", "pipoca para microondas", "pipoca pronta", "pirulito", "piscina", "pizza congelada", "plantas e sementes", "plastico filme", "plug de tomada", "polenta", "polpa de fruta congelada", "polpa de tomate", "polvilho", "porta escova de dentes", "prato", "prato descartavel", "prato pronto", "prendedor de roupa", "preparo cappuccino", "preparo de alimento", "preparo de doce", "preparo para bolo", "preparo para maria mole", "preparo para pudim", "preparo para sopa", "preparo para sorvete", "preparo para suco", "preservativo", "presilha", "presunto", "produto de limpeza de piso", "produto facial", "produto labial", "produto para sapatos", "produto sazonal", "produtos de limpeza piscina", "proteina congelada", "proteina em conserva", "proteina fatiada", "proteina vegetal", "protetor de fogao", "protetor solar", "purificador", "queijo", "queijo cremoso", "quentao", "quiabo", "rabanete", "ralo", "raticida", "ratoeira", "recheio pronto", "recipientes", "refresco", "refrigerante", "regata", "regua", "removedor de esmalte", "removedor de maquiagem", "repelente", "repelente corporal", "repolho", "requeijao", "rodo", "rolo de pintura", "roupa intima descartavel", "roupao de banho", "rucula", "rum", "sabao em pasta", "sabao em pedra", "sabonete", "sabonete infantil", "sabonete liquido", "saboneteira", "saco de lixo", "saco para alimento", "saco para assar", "sacola descartavel", "sacola retornavel", "sagu", "sal", "salame", "salsa", "salsao", "salsicha", "sandalia", "sanduiche congelado", "sanduicheira", "saponaceo", "saque", "sementes", "seringa descartavel", "shampoo", "shampoo infantil", "shampoo pet", "short", "sidra", "silicone de vedacao", "sobremesa pronta resfriada", "sorvete", "suco pronto", "sunga", "suplemento alimentar", "suspiro", "sutia", "tabua de corte", "tabua de passar", "taca", "talco", "talco infantil", "talher", "talher descartavel", "tangerina", "tapete", "tapete higienico", "tapioca", "televisao", "tempero pronto", "termometro", "tesoura", "tinta", "tintura e descolorante", "toalha", "toalha de mesa", "tomada", "tomate", "top", "top fitness", "torrada", "touca para banho", "tratamento capilar", "travesseiro", "trigo para kibe", "uva", "vagem", "varal", "vaso para plantas", "vassoura", "veda rosca", "vegetal congelado", "vegetal em conserva", "vegetal em palha", "vegetal higienizado e processado", "vela", "vela comemorativa", "ventilador", "vestido", "vinagre", "vinho importado", "vinho nacional", "vitaminico", "vodca", "whisky", "xarope"]'


# Código da Célula 2
# Supondo que você tenha a lista de categorias em uma variável chamada 'lista_categorias_str'
# e a descrição do produto a ser processada em 'descricao_produto_input'

# Exemplo de como você usaria:
# lista_categorias_str = '["_RARE_", "abacate", ... , "xarope"]' # (string completa como definida acima)
# descricao_produto_input = "MEU PRODUTO XPTO 1LT"

prompt_template_str = """
Você é um assistente de IA especializado em processamento e catalogação de descrições de produtos. Sua tarefa é receber uma descrição de produto e retornar um objeto JSON com a seguinte estrutura:

{{
  "descrição": "string",
  "descrição_expandida": "string",
  "descricao_machine_learning": "string",
  "categoria": "string",
  "racional": "string"
}}

Detalhes dos campos:
1.  `descrição`: A descrição original do produto, como recebida.
2.  `descrição_expandida`: A descrição do produto com todas as abreviações comuns (como "LT", "KG", "G", "ML", "UND", "CX", "SAB", "C/", "S/", "P/", "M", "G", "GG", "INT", "DESC", "SEMIDESC", "CONC", etc.) expandidas para suas formas completas (por exemplo, "Litro", "Quilograma", "Grama", "Mililitro", "Unidade", "Caixa", "Sabor", "Com", "Sem", "Para", "Médio", "Grande", "Extra Grande", "Integral", "Desnatado", "Semidesnatado", "Concentrado"). A descrição expandida deve ser inequívoca, clara e gramaticalmente correta.
3.  `descricao_machine_learning`: Uma versão da `descrição_expandida` totalmente em letras minúsculas e sem acentuação gráfica (ex: "açúcar" se torna "acucar", "maçã" se torna "maca").
4.  `categoria`: A categoria do produto, escolhida EXATAMENTE de uma das opções da lista fornecida abaixo. A categoria deve ser a mais específica e precisa possível, sem ambiguidades. Se nenhuma categoria parecer adequada, use "_RARE_".
5.  `racional`: Uma breve explicação concisa (1-2 frases) do motivo pelo qual a categoria foi escolhida e como a descrição foi expandida (mencione as principais abreviações que foram expandidas).

Regras para `descricao_machine_learning`:
*   Converter todos os caracteres da `descrição_expandida` para minúsculas.
*   Remover todos os acentos gráficos. Exemplos de substituição:
    *   á, à, â, ä, ã -> a
    *   é, è, ê, ë -> e
    *   í, ì, î, ï -> i
    *   ó, ò, ô, ö, õ -> o
    *   ú, ù, û, ü -> u
    *   ç -> c

Use EXCLUSIVAMENTE as seguintes categorias para o campo `categoria`:
{lista_categorias_str}

Exemplos:

Entrada: "Leite Cond. Mococa TP 395G"
Saída:
```json
{{
  "descrição": "Leite Cond. Mococa TP 395G",
  "descrição_expandida": "Leite Condensado Mococa Tetra Pak 395 Gramas",
  "descricao_machine_learning": "leite condensado mococa tetra pak 395 gramas",
  "categoria": "leite condensado",
  "racional": "O produto é leite condensado da marca Mococa em embalagem Tetra Pak de 395 gramas. 'Cond.' foi expandido para 'Condensado', 'TP' para 'Tetra Pak' e 'G' para 'Gramas'. A categoria 'leite condensado' é a mais adequada."
}}
Entrada: "SABAO EM PO OMO LAV PERF CX 1.6KG"
Saída:
{{
  "descrição": "SABAO EM PO OMO LAV PERF CX 1.6KG",
  "descrição_expandida": "Sabão em Pó Omo Lavagem Perfeita Caixa 1.6 Quilograma",
  "descricao_machine_learning": "sabao em po omo lavagem perfeita caixa 1.6 quilograma",
  "categoria": "lava roupas",
  "racional": "O produto é um sabão em pó para lavar roupas da marca Omo, versão Lavagem Perfeita, em caixa de 1.6 quilograma. 'PO' foi expandido para 'Pó', 'LAV PERF' para 'Lavagem Perfeita', 'CX' para 'Caixa' e 'KG' para 'Quilograma'. A categoria 'lava roupas' é a mais apropriada."
}}
Entrada: "ACUCAR REFINADO UNIAO PCT 1KG"
Saída:
{{
  "descrição": "ACUCAR REFINADO UNIAO PCT 1KG",
  "descrição_expandida": "Açúcar Refinado União Pacote 1 Quilograma",
  "descricao_machine_learning": "acucar refinado uniao pacote 1 quilograma",
  "categoria": "acucar",
  "racional": "O produto é açúcar refinado da marca União, em pacote de 1 quilograma. 'PCT' foi expandido para 'Pacote' e 'KG' para 'Quilograma'. A categoria 'acucar' é a mais adequada."
}}
Entrada: "BISC CREAM CRACKER MARILAN TRAD 350G"
Saída:
{{
  "descrição": "BISC CREAM CRACKER MARILAN TRAD 350G",
  "descrição_expandida": "Biscoito Cream Cracker Marilan Tradicional 350 Gramas",
  "descricao_machine_learning": "biscoito cream cracker marilan tradicional 350 gramas",
  "categoria": "biscoito",
  "racional": "O produto é um biscoito do tipo Cream Cracker da marca Marilan, sabor tradicional, com 350 gramas. 'BISC' foi expandido para 'Biscoito' e 'TRAD' para 'Tradicional', 'G' para 'Gramas'. A categoria 'biscoito' é a correta."
}}
Agora, processe a seguinte descrição de produto:
{descricao_produto}
"""


# Código da Célula 3
# Instalando bibliotecas necessárias (execute uma vez se não tiver)
# !pip install google-generativeai scikit-learn tqdm unidecode ipython pandas python-dotenv

import os
import json
import re
import pandas as pd
from dotenv import load_dotenv

import google.generativeai as genai
from google.generativeai import types # Mantido caso precise de tipos específicos no futuro

from unidecode import unidecode
from sklearn.metrics import accuracy_score, f1_score
from tqdm.notebook import tqdm # Específico para Jupyter Notebook
from IPython.display import display, JSON # Para exibir JSON de forma elegante

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

# Flag para controlar se usamos a API real ou a simulação (mock)
USE_MOCK_API = False # Defina como True para testar sem chamadas reais à API

if not GOOGLE_API_KEY:
    print("AVISO: GEMINI_API_KEY não encontrada no arquivo .env. A chamada à API real falhará. Forçando USE_MOCK_API = True.")
    USE_MOCK_API = True
else:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        print("API Key do Google Gemini configurada com sucesso.")
    except Exception as e:
        print(f"Erro ao configurar a API Key do Google Gemini: {e}. Forçando USE_MOCK_API = True.")
        USE_MOCK_API = True

# Estrutura para respostas mockadas em caso de erro ou quando USE_MOCK_API = True
DEFAULT_MOCK_RARE_RESPONSE_STRUCTURE = {
    "descrição": "",
    "descrição_expandida": "",
    "descricao_machine_learning": "",
    "categoria": "_RARE_",
    "racional": "Simulação ou Erro API: Não foi possível processar."
}

# Modelo Gemini a ser usado (ajuste conforme sua necessidade e disponibilidade)
# O usuário especificou "gemini-2.5-pro-preview-05-06".
# Se este modelo exato não estiver disponível, tente 'gemini-1.5-pro-latest' ou 'gemini-1.5-flash-latest'.
# GEMINI_MODEL_NAME = "gemini-1.5-pro-latest" # Ajustado para um modelo mais comum, o usuário pode alterar
GEMINI_MODEL_NAME = "gemini-2.5-flash-preview-05-20" # Como especificado pelo usuário. Pode dar erro se não tiver acesso.

# Código da Célula 4
# Caminho para o arquivo CSV
csv_file_path = r"D:\Nuvem\ghdaru\OneDrive\030_DOUTORADO\120_TESE\130_TESEGIT\activetextclassification\data_splits_cache\dri_coldstart_selection_details_log.csv"
target_run_id = "DRI_L0size1000_Nc173_Seed1042"

try:
    df_full = pd.read_csv(csv_file_path)
    print(f"Arquivo CSV '{csv_file_path}' carregado com sucesso. Total de linhas: {len(df_full)}")
except FileNotFoundError:
    print(f"ERRO: Arquivo CSV não encontrado em: {csv_file_path}")
    print("Por favor, verifique o caminho do arquivo.")
    df_full = pd.DataFrame() # DataFrame vazio para evitar erros subsequentes

if not df_full.empty:
    # Filtrar pelo run_id especificado
    df_filtered = df_full[df_full['run_id'] == target_run_id].copy() # .copy() para evitar SettingWithCopyWarning
    print(f"Linhas encontradas para run_id '{target_run_id}': {len(df_filtered)}")

    if len(df_filtered) == 0:
        print(f"AVISO: Nenhuma linha encontrada para o run_id '{target_run_id}'. Verifique o run_id ou o conteúdo do CSV.")
        # Criar dataset de avaliação vazio e lista de categorias padrão para evitar erros
        dataset_avaliacao = []
        lista_categorias_py = ["_RARE_"]
    else:
        # Preparar o dataset_avaliacao
        # Garantir que as colunas existem
        required_cols = ['text_sample', 'true_label_sample']
        if not all(col in df_filtered.columns for col in required_cols):
            print(f"ERRO: O CSV filtrado não contém as colunas necessárias: {required_cols}. Colunas encontradas: {df_filtered.columns.tolist()}")
            dataset_avaliacao = []
            lista_categorias_py = ["_RARE_"]
        else:
            # Remover linhas onde 'text_sample' ou 'true_label_sample' são NaN
            df_filtered.dropna(subset=['text_sample', 'true_label_sample'], inplace=True)
            print(f"Linhas após remover NaNs em text_sample/true_label_sample: {len(df_filtered)}")

            dataset_avaliacao = [
                {
                    "descricao_produto": row['text_sample'],
                    "ground_truth_categoria": str(row['true_label_sample']).strip() # Garantir que é string e sem espaços extras
                }
                for index, row in df_filtered.iterrows()
            ]
            print(f"Dataset de avaliação criado com {len(dataset_avaliacao)} itens.")

            # Derivar lista de categorias dos dados filtrados
            if not df_filtered['true_label_sample'].empty:
                 # Garantir que todas as categorias sejam strings e sem espaços extras
                unique_labels = [str(label).strip() for label in df_filtered['true_label_sample'].unique()]
                lista_categorias_py = sorted(list(set(unique_labels + ["_RARE_"])))
            else:
                lista_categorias_py = ["_RARE_"]
            print(f"Lista de categorias derivada dos dados ({len(lista_categorias_py)} categorias). Primeira: {lista_categorias_py[0] if lista_categorias_py else 'N/A'}")

    # Converter lista de categorias para string JSON para o prompt
    lista_categorias_str = json.dumps(lista_categorias_py, ensure_ascii=False)
else:
    print("AVISO: DataFrame está vazio. Não foi possível processar categorias ou dataset de avaliação.")
    dataset_avaliacao = []
    lista_categorias_py = ["_RARE_"]
    lista_categorias_str = json.dumps(lista_categorias_py, ensure_ascii=False)

# Exibir uma amostra do dataset de avaliação e categorias (se existirem)
if dataset_avaliacao:
    print("\nAmostra do dataset de avaliação (primeiros 3 itens):")
    for item in dataset_avaliacao[:3]:
        print(item)
else:
    print("\nDataset de avaliação está vazio.")

# print("\nLista de Categorias para o Prompt (JSON string):")
# print(lista_categorias_str[:500] + "..." if len(lista_categorias_str) > 500 else lista_categorias_str)

# Código da Célula 5
def chamar_google_gemini(prompt: str, descricao_original_para_log: str) -> str:
    """
    Envia o prompt para a API do Google Gemini e retorna a resposta em string (espera-se JSON).
    """
    if USE_MOCK_API:
        # print(f"INFO: Usando MOCK API para '{descricao_original_para_log}'") # Descomente para debug
        # Simulação básica para testes offline
        mock_response = DEFAULT_MOCK_RARE_RESPONSE_STRUCTURE.copy()
        mock_response["descrição"] = descricao_original_para_log
        # Tenta encontrar um exemplo para mock, senão usa o default RARE
        if descricao_original_para_log == "Leite Cond. Mococa TP 395G": # Exemplo do prompt
             mock_response = {
                "descrição": "Leite Cond. Mococa TP 395G",
                "descrição_expandida": "Leite Condensado Mococa Tetra Pak 395 Gramas (MOCK)",
                "descricao_machine_learning": "leite condensado mococa tetra pak 395 gramas (mock)",
                "categoria": "leite condensado",
                "racional": "Mock response: O produto é leite condensado..."
            }
        return json.dumps(mock_response, ensure_ascii=False)

    # Usar API Real
    try:
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        
        generation_config = genai.types.GenerationConfig(
            response_mime_type="application/json", # Solicita saída JSON diretamente
            temperature=0.1, # Baixa temperatura para respostas mais determinísticas
            # max_output_tokens=2048 # Ajuste se necessário, padrão geralmente é suficiente
        )
        
        # Configurações de segurança (ajuste para seu caso de uso)
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]

        response = model.generate_content(
            prompt,
            generation_config=generation_config,
            safety_settings=safety_settings
            # stream=False # Garantir que não está em modo stream para esta função
        )
        return response.text
    except Exception as e:
        print(f"Erro ao chamar a API do Gemini para '{descricao_original_para_log}': {e}")
        # Retornar um JSON de erro estruturado
        error_response = DEFAULT_MOCK_RARE_RESPONSE_STRUCTURE.copy()
        error_response["descrição"] = descricao_original_para_log
        error_response["descrição_expandida"] = "ERRO API"
        error_response["descricao_machine_learning"] = "erro api"
        error_response["racional"] = f"Erro ao processar via API: {str(e)}"
        return json.dumps(error_response, ensure_ascii=False)

# Código da Célula 6
def parse_e_validar_resposta_gemini(response_text: str, descricao_original: str) -> dict | None:
    """
    Parseia a string JSON da resposta do Gemini e realiza validações básicas.
    Retorna um dicionário ou None em caso de falha crítica.
    """
    parsed_json = None
    try:
        # Gemini com response_mime_type="application/json" deve retornar JSON puro.
        parsed_json = json.loads(response_text)
        if not isinstance(parsed_json, dict):
            print(f"Erro: Resposta do LLM não é um dicionário JSON. Desc: {descricao_original}. Resposta: {str(response_text)[:200]}")
            # Preencher com uma estrutura de erro para não quebrar o fluxo principal
            parsed_json = DEFAULT_MOCK_RARE_RESPONSE_STRUCTURE.copy()
            parsed_json["descrição"] = descricao_original
            parsed_json["racional"] += " Resposta do LLM não foi um dicionário JSON."
            return parsed_json # Retorna dict de erro
            
    except json.JSONDecodeError as e:
        print(f"Erro ao decodificar JSON da resposta do LLM para '{descricao_original}': {e}")
        print(f"Resposta recebida (primeiros 200 chars): {str(response_text)[:200]}...")
        # Tenta extrair JSON de blocos de código se o mime_type falhar (backup improvável, mas seguro)
        match = re.search(r"```json\s*([\s\S]*?)\s*```", response_text)
        if match:
            json_str_from_block = match.group(1)
            try:
                parsed_json = json.loads(json_str_from_block)
                print("INFO: JSON extraído de bloco de código com sucesso após falha inicial.")
                if not isinstance(parsed_json, dict):
                     print(f"Erro: JSON extraído de bloco de código não é um dicionário. Desc: {descricao_original}.")
                     parsed_json = None # Indica falha para o default
            except json.JSONDecodeError:
                print(f"Erro: Falha ao decodificar JSON mesmo após extração de bloco de código. Desc: {descricao_original}")
                parsed_json = None # Indica falha para o default
        else:
            parsed_json = None # Indica falha para o default

        if parsed_json is None: # Se ainda falhou
            parsed_json = DEFAULT_MOCK_RARE_RESPONSE_STRUCTURE.copy()
            parsed_json["descrição"] = descricao_original
            parsed_json["racional"] += f" Falha ao decodificar JSON: {e}."
            return parsed_json # Retorna dict de erro

    # Validação básica da estrutura esperada
    expected_keys = {"descrição", "descrição_expandida", "descricao_machine_learning", "categoria", "racional"}
    
    # Preenche chaves faltantes com valores padrão para evitar quebrar o downstream
    # E garante que a descrição original seja preservada
    if parsed_json.get("descrição") != descricao_original and descricao_original : # Se a descrição foi alterada ou está vazia no JSON
        if USE_MOCK_API and "(MOCK)" in parsed_json.get("descrição_expandida",""): # Não sobrescrever se for mock e tiver desc
             pass # Mock já tem descrição
        else:
            # print(f"INFO: Descrição no JSON ('{parsed_json.get('descrição')}') difere da original ('{descricao_original}'). Usando original.")
            parsed_json["descrição"] = descricao_original


    missing_keys = expected_keys - set(parsed_json.keys())
    if missing_keys:
        # print(f"Alerta: Resposta do LLM para '{descricao_original}' não contém todas as chaves esperadas. Faltando: {missing_keys}. Preenchendo com defaults.")
        for key in missing_keys:
            if key == "categoria": parsed_json[key] = "_RARE_"
            elif key == "descrição_expandida": parsed_json[key] = parsed_json.get("descrição", descricao_original) + " (Expansão padrão por chave ausente)"
            elif key == "descricao_machine_learning": parsed_json[key] = unidecode(parsed_json.get("descrição_expandida", "").lower())
            elif key == "racional": parsed_json[key] = "Racional padrão por chave ausente."
            else: parsed_json[key] = "VALOR AUSENTE DO LLM"
        if "racional" not in parsed_json or "VALOR AUSENTE DO LLM" in parsed_json["racional"] or "Racional padrão" in parsed_json["racional"]:
            parsed_json["racional"] = (parsed_json.get("racional","") + " ALERTA: Algumas chaves estavam ausentes na resposta do LLM.").strip()

    # Validação se a categoria está na lista_categorias_py (que foi carregada na Célula 2)
    # Certifique-se que lista_categorias_py está acessível globalmente ou passada como parâmetro
    if parsed_json.get("categoria") not in lista_categorias_py: # lista_categorias_py deve estar no escopo global
        original_cat_llm = parsed_json.get("categoria")
        print(f"Alerta: Categoria '{original_cat_llm}' retornada pelo LLM para '{descricao_original}' não está na lista permitida. Forçando para '_RARE_'.")
        parsed_json["categoria"] = "_RARE_"
        parsed_json["racional"] = (parsed_json.get("racional", "") + f" ALERTA: Categoria original do LLM '{original_cat_llm}' inválida, alterada para _RARE_.").strip()
        
    return parsed_json

# Código da Célula 7
def processar_descricao_produto(descricao_produto: str, lista_categorias_str: str) -> dict | None:
    """
    Processa uma única descrição de produto usando o LLM.
    'categorias_json_param_str' é a string JSON da lista de categorias. """
    prompt = prompt_template_str.format(
        lista_categorias_str=lista_categorias_str,
        descricao_produto=descricao_produto.strip()
    )
    
    gemini_response_text = chamar_google_gemini(prompt, descricao_produto) 
    
    resultado_final = parse_e_validar_resposta_gemini(gemini_response_text, descricao_produto)
    
    return resultado_final

# Código da Célula 8
predicoes_categorias = []
verdadeiros_categorias = []
resultados_completos = [] # Para inspecionar todas as saídas

if not dataset_avaliacao:
    print("ERRO: dataset_avaliacao está vazio. Não é possível prosseguir com a avaliação.")
    print("Verifique a Célula 2 para erros no carregamento ou filtragem dos dados.")
else:
    print(f"Iniciando processamento de {len(dataset_avaliacao)} itens. Usando API Mock: {USE_MOCK_API}\n")

    # Usar tqdm.notebook.tqdm para a barra de progresso no Jupyter
    for item in tqdm(dataset_avaliacao, desc="Processando produtos"):
        descricao_produto = item["descricao_produto"]
        verdadeiro_cat = item["ground_truth_categoria"]

        # 'lista_categorias_str' foi definida na Célula 2 com base nos dados filtrados
        resultado_processado = processar_descricao_produto(descricao_produto, lista_categorias_str)
        
        # Garantir que resultado_processado é sempre um dict
        if resultado_processado is None: # Segurança, embora parse_e_validar agora deva sempre retornar dict
            print(f"AVISO: resultado_processado foi None para '{descricao_produto}'. Usando default error structure.")
            resultado_processado = DEFAULT_MOCK_RARE_RESPONSE_STRUCTURE.copy()
            resultado_processado["descrição"] = descricao_produto
            resultado_processado["racional"] += " Falha crítica inesperada na função de processamento."

        resultados_completos.append(resultado_processado) 

        predito_cat = resultado_processado.get("categoria", "_RARE_") 
        
        predicoes_categorias.append(predito_cat)
        verdadeiros_categorias.append(verdadeiro_cat)

    print("\n--- Amostra de Resultados Detalhados (primeiros 3) ---")
    for i in range(min(3, len(resultados_completos))): 
        if i < len(dataset_avaliacao): # Checa se o índice é válido para dataset_avaliacao
            print(f"\nItem Original: {dataset_avaliacao[i]['descricao_produto']}")
            print(f"Ground Truth Categoria: {dataset_avaliacao[i]['ground_truth_categoria']}")
            if resultados_completos[i]:
                display(JSON(resultados_completos[i])) 
            else: # Caso raro, mas para segurança
                print("LLM Output: Falha crítica no processamento (resultado None/Inválido)")
        else:
            print(f"Índice {i} fora dos limites para dataset_avaliacao.")


    # Calcular métricas
    if not verdadeiros_categorias or not predicoes_categorias:
        print("\nNão foi possível calcular métricas: listas de verdadeiros ou predições estão vazias.")
    else:
        # Garantir que todas as labels em ground_truth_categoria estejam em lista_categorias_py para o f1_score
        # Isso é importante se alguma categoria do CSV não foi incluída por algum motivo (ex: erro de digitação no CSV)
        # A lista_categorias_py já deve conter todas as 'true_label_sample' + '_RARE_'
        labels_para_f1 = sorted(list(set(lista_categorias_py + verdadeiros_categorias + predicoes_categorias)))


        accuracy = accuracy_score(verdadeiros_categorias, predicoes_categorias)
        f1_macro = f1_score(verdadeiros_categorias, predicoes_categorias, average='macro', labels=labels_para_f1, zero_division=0)

        print("\n--- Resultados da Avaliação (Categoria) ---")
        print(f"Total de itens avaliados: {len(verdadeiros_categorias)}")
        print(f"Acurácia: {accuracy:.4f}")
        print(f"F1-Score (Macro): {f1_macro:.4f}")

        print("\nLabels consideradas para F1-Score:", labels_para_f1)
        # Contagem para debug
        # print("\nDistribuição Ground Truth (primeiras 10 labels):")
        # for cat in labels_para_f1[:10]:
        #     print(f"  {cat}: {verdadeiros_categorias.count(cat)}")
        # print("\nDistribuição Predições (primeiras 10 labels):")
        # for cat in labels_para_f1[:10]:
        #     print(f"  {cat}: {predicoes_categorias.count(cat)}")

        # Opcional: Salvar resultados completos em um arquivo JSON
        # with open("resultados_processamento_gemini.json", "w", encoding="utf-8") as f_out:
        #     json.dump(resultados_completos, f_out, ensure_ascii=False, indent=2)
        # print("\nResultados completos salvos em 'resultados_processamento_gemini.json'")

# Código da Célula 9
resultados_completos_df = pd.DataFrame(resultados_completos)
resultados_completos_df['ground_truth_categoria'] = verdadeiros_categorias
resultados_completos_df['predicao_categoria'] = predicoes_categorias
# Exibir DataFrame completo com resultados
display(resultados_completos_df.head(10))  # Exibe as primeiras 10 linhas do DataFrame
resultados_completos_df.to_excel("resultados_processamento_gemini.xlsx", index=False)
print("\nResultados completos salvos em 'resultados_processamento_gemini.xlsx'")