# 🚀 Consolidador de Código de Projeto

Este utilitário Python foi projetado para varrer recursivamente um diretório de projeto e consolidar o conteúdo de vários arquivos de código-fonte em um único arquivo de texto.

É uma ferramenta extremamente útil para:
*   Criar um "snapshot" completo do código para análise.
*   Alimentar o contexto de grandes modelos de linguagem (LLMs) como GPT, Gemini, etc.
*   Gerar um único arquivo para revisão de código ou arquivamento.
*   Analisar a complexidade do projeto através de estatísticas de código.

---

## ✨ Funcionalidades

*   **Consolidação de Arquivos:** Agrega múltiplos arquivos de código em um só, com cabeçalhos claros indicando a origem de cada trecho.
*   **Conversão Inteligente de Notebooks:** Converte arquivos Jupyter (`.ipynb`) para o formato de script Python (`.py`), incluindo células de Markdown como comentários e descartando os outputs.
*   **Sistema de Exclusão Poderoso:** Permite ignorar arquivos e pastas indesejados (como `__pycache__`, `.venv`, `node_modules`) usando padrões com curingas (wildcards).
*   **Geração de Estatísticas:** Oferece a opção de criar um arquivo `.csv` complementar com o nome de cada arquivo processado e sua contagem de caracteres.
*   **Design Orientado a Objetos:** Estruturado em classes, permitindo fácil extensão para tipos de projetos específicos (ex: JavaScript, Java, etc.).

---

## 🔧 Pré-requisitos e Instalação

1.  **Python 3.x** instalado.
2.  A biblioteca `nbconvert` para o processamento de notebooks.

Para instalar as dependências necessárias, execute o seguinte comando no seu terminal:

```bash
pip install notebook
```
*(Este comando instala `nbconvert` e `nbformat`, que são as dependências necessárias.)*

---

## ⚙️ Como Utilizar

Existem duas maneiras principais de usar esta ferramenta:

### 1. Como um Script Autônomo

Esta é a forma mais simples. Basta configurar e executar o script diretamente.

1.  **Coloque o arquivo `consolidadores.py`** no diretório raiz do projeto que você deseja consolidar.
2.  **Abra o arquivo `consolidadores.py`** e edite a seção `if __name__ == "__main__":` no final do arquivo para ajustar as configurações.
3.  **Execute o script** a partir do seu terminal:

    ```bash
    python consolidadores.py
    ```

### 2. Como uma Biblioteca (em um Notebook ou Script)

Esta abordagem oferece mais flexibilidade.

1.  **Coloque o arquivo `consolidadores.py`** no mesmo diretório do seu script ou notebook.
2.  **Importe e instancie a classe** `PythonProjectConsolidator`.

    ```python
    from consolidadores import PythonProjectConsolidator
    import os

    meu_consolidador = PythonProjectConsolidator(
        root_dir=".",
        output_file="meu_snapshot.txt",
        stats=True, 
        exclude_list=["dados", "*.log", "temp*"]
    )
    meu_consolidador.consolidate()
    ```

---

## 🧠 Funcionamento

O script opera em três etapas principais:

1.  **Varredura de Diretórios (`os.walk`):** O coração do processo é a função `os.walk`. Ela percorre a árvore de diretórios de cima para baixo. Em cada pasta, ela nos informa quais subpastas e arquivos existem ali. É neste momento que aplicamos a **lista de exclusão**: se o nome de uma subpasta corresponde a um dos padrões de exclusão, instruímos o `os.walk` a **não entrar** nela, economizando tempo e evitando conteúdo indesejado.

2.  **Filtragem e Extração de Conteúdo:** Para cada arquivo encontrado na varredura (que não foi excluído), o script verifica se sua extensão (ex: `.py`, `.ipynb`) está na lista de inclusão. Se estiver, ele seleciona uma "função extratora" apropriada:
    *   **Para `.ipynb`:** Usa a biblioteca `nbconvert` para converter o notebook em um script Python limpo.
    *   **Para outros arquivos:** Usa um extrator padrão que simplesmente lê o conteúdo do arquivo como texto.

3.  **Consolidação e Geração de Saída:** O conteúdo extraído de cada arquivo é adicionado ao arquivo de saída principal, precedido por um cabeçalho que identifica o caminho original do arquivo. Se a opção `stats=True` estiver ativa, o nome e o tamanho de cada arquivo processado são armazenados e, ao final, escritos em um novo arquivo `.csv`.

---

## 🃏 Exemplos de Exclusão (`exclude_list`)

A `exclude_list` é a parte mais poderosa para customizar sua consolidação. Aqui estão alguns cenários práticos:

| O que você quer excluir? | Padrão a ser usado na lista | Exemplo Prático |
| :--- | :--- | :--- |
| **Uma pasta específica e todo o seu conteúdo** | `"nome_da_pasta"` | `"utils"`, `".venv"`, `"__pycache__"`, `"docs"` |
| **Todos os arquivos com uma certa extensão** | `"* .extensao"` | `"*.log"`, `"*.tmp"`, `"*.csv"`, `"*.md"` |
| **Arquivos ou pastas que começam com um prefixo** | `"prefixo*"` | `"temp*"` (exclui `temp_data` e `temp_file.txt`), `".git*"` |
| **Arquivos ou pastas que terminam com um sufixo** | `"*sufixo"` | `"*_backup"` (exclui `db_backup` e `script_backup.py`) |
| **Arquivos ou pastas que contêm uma palavra** | `"*palavra*"` | `"*test*"` (exclui `tests/`, `test_utils.py`, `utils_test.py`) |
| **Um arquivo específico** | `"nome_completo_do_arquivo.ext"` | `"config.secreto.ini"`, `"README.md"` |

**Exemplo de uma `exclude_list` robusta para um projeto Python:**
```python
exclude_list = [
    # Pastas de ambiente e cache
    "__pycache__",
    ".venv",
    ".git",
    ".vscode",
    "build",
    "dist",
    
    # Pastas com dados ou documentação
    "dados",
    "notebooks_antigos",
    
    # Padrões de arquivos a ignorar
    "*.log",
    "*.tmp",
    "*.csv",
    
    # Arquivos ou pastas específicos
    "config.local.py",
    "experimento_temp*" # Ignora pastas como 'experimento_temp_01'
]
```

---

## 📄 Arquivos de Saída

O script irá gerar um ou dois arquivos, dependendo da sua configuração:

1.  **`nome_do_arquivo.txt`**: O arquivo principal contendo todo o código consolidado.
2.  **`nome_do_arquivo_stats.csv`** (Opcional): Gerado se `stats=True`. É um arquivo CSV com duas colunas:
    *   `filename`: O caminho relativo do arquivo.
    *   `character_count`: O número de caracteres do conteúdo do arquivo.