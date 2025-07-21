import nbformat
from nbconvert import PythonExporter
import os

def converter_ipynb_para_py(caminho_ipynb, caminho_saida=None):
    """
    Converte um arquivo .ipynb para um arquivo .py.

    Args:
        caminho_ipynb (str): O caminho para o arquivo .ipynb de entrada.
        caminho_saida (str, optional): O caminho para o arquivo .py de saída.
                                      Se não for fornecido, o arquivo .py será
                                      salvo no mesmo diretório que o arquivo de entrada
                                      com o mesmo nome base.

    Returns:
        str: O caminho para o arquivo .py gerado.
    """
    if not os.path.exists(caminho_ipynb):
        raise FileNotFoundError(f"O arquivo de entrada não foi encontrado: {caminho_ipynb}")

    # Define o caminho de saída se não for fornecido
    if caminho_saida is None:
        base, _ = os.path.splitext(caminho_ipynb)
        caminho_saida = f"{base}.py"

    # Carrega o notebook
    with open(caminho_ipynb, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # Inicializa o exportador para Python
    python_exporter = PythonExporter()

    # Converte o notebook para um script Python
    script_python, _ = python_exporter.from_notebook_node(nb)

    # Salva o script Python no arquivo de saída
    with open(caminho_saida, 'w', encoding='utf-8') as f:
        f.write(script_python)

    return caminho_saida

if __name__ == '__main__':
    # Exemplo de uso
    # Substitua 'seu_notebook.ipynb' pelo caminho do seu arquivo .ipynb
    try:
        caminho_arquivo_py = converter_ipynb_para_py('seu_notebook.ipynb')
        print(f"Arquivo convertido com sucesso para: {caminho_arquivo_py}")
    except FileNotFoundError as e:
        print(f"Erro: {e}")
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")