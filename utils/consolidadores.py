import os
import fnmatch
import csv
from abc import ABC, abstractmethod

try:
    import nbformat
    from nbconvert import PythonExporter
    NB_CONVERT_AVAILABLE = True
except ImportError:
    NB_CONVERT_AVAILABLE = False

class ProjectConsolidator:
    # ... (O __init__ e os outros métodos permanecem os mesmos) ...
    def __init__(self, root_dir, output_file, include_extensions, exclude_list=None, stats=False):
        self.root_dir = root_dir
        self.output_file = output_file
        self.include_extensions = include_extensions
        self.exclude_list = exclude_list if exclude_list is not None else []
        self.stats = stats
        base_name, _ = os.path.splitext(self.output_file)
        self.stats_file = f"{base_name}_stats.csv"
        self.exclude_list.extend([os.path.basename(self.output_file), os.path.basename(self.stats_file)])
        self.exclude_list = list(set(self.exclude_list))
        self._extractors = {}
        self.register_extractor('default', self._extract_text_default)

    def register_extractor(self, extension, function):
        self._extractors[extension] = function

    def _get_extractor(self, file_extension):
        return self._extractors.get(file_extension, self._extractors['default'])

    def _extract_text_default(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            return f"# ERRO AO LER O ARQUIVO {os.path.basename(file_path)}: {e}\n"

    def consolidate(self):
        """
        Executa o processo de consolidação com lógica de exclusão aprimorada
        que suporta caminhos relativos nos padrões.
        """
        print(f"Iniciando consolidação do projeto em '{self.root_dir}'...")
        print(f"Incluindo extensões: {self.include_extensions}")
        print(f"Excluindo itens e padrões: {self.exclude_list}")
        print("-" * 50)

        stats_data = []

        with open(self.output_file, "w", encoding="utf-8") as out_file:
            for root, dirs, files in os.walk(self.root_dir, topdown=True):
                
                # --- LÓGICA DE EXCLUSÃO DE DIRETÓRIOS APRIMORADA ---
                excluded_dirs = set()
                for d in dirs:
                    # Calcula o caminho relativo e normaliza as barras
                    relative_dir_path = os.path.relpath(os.path.join(root, d), self.root_dir).replace(os.path.sep, '/')
                    for pattern in self.exclude_list:
                        # Verifica o padrão contra o nome da pasta E contra o caminho relativo
                        if fnmatch.fnmatch(d, pattern) or fnmatch.fnmatch(relative_dir_path, pattern):
                            excluded_dirs.add(d)
                            break
                dirs[:] = [d for d in dirs if d not in excluded_dirs]
                
                for file in sorted(files):
                    relative_file_path = os.path.relpath(os.path.join(root, file), self.root_dir).replace(os.path.sep, '/')
                    
                    # --- LÓGICA DE EXCLUSÃO DE ARQUIVOS APRIMORADA ---
                    is_excluded = False
                    for pattern in self.exclude_list:
                        # Verifica o padrão contra o nome do arquivo E contra o caminho relativo
                        if fnmatch.fnmatch(file, pattern) or fnmatch.fnmatch(relative_file_path, pattern):
                            is_excluded = True
                            break
                    if is_excluded:
                        continue

                    _, ext = os.path.splitext(file)
                    if ext not in self.include_extensions:
                        continue

                    file_path = os.path.join(root, file)
                    print(f"Processando: {relative_file_path}")
                    
                    header = f"\n# {'='*80}\n# Arquivo: {relative_file_path}\n# {'='*80}\n"
                    out_file.write(header)
                    extractor_func = self._get_extractor(ext)
                    content = extractor_func(file_path)
                    out_file.write(content)
                    out_file.write("\n\n")

                    if self.stats:
                        stats_data.append({'filename': relative_file_path, 'character_count': len(content)})
        
        print("-" * 50)
        print(f"✅ Consolidação principal concluída! Salvo em: {self.output_file}")
        if self.stats and stats_data:
            self._write_stats_file(stats_data)

    # ... (O resto da classe e do arquivo continua o mesmo) ...

    def _write_stats_file(self, stats_data):
        """Escreve a lista de dados de estatísticas em um arquivo CSV."""
        print(f"Gerando arquivo de estatísticas em: {self.stats_file}...")
        try:
            with open(self.stats_file, 'w', newline='', encoding='utf-8') as csv_file:
                fieldnames = ['filename', 'character_count']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                
                writer.writeheader()
                writer.writerows(stats_data)
            
            print(f"✅ Arquivo de estatísticas gerado com sucesso.")
        except IOError as e:
            print(f"❌ Erro ao escrever o arquivo de estatísticas: {e}")

class PythonProjectConsolidator(ProjectConsolidator):
    DEFAULT_PYTHON_EXTENSIONS = ['.py', '.ipynb', '.md', '.txt', '.toml', '.ini', '.sh']
    DEFAULT_PYTHON_EXCLUDES = ['__pycache__', '.venv', '.git', '.vscode', 'dist', 'build', 'docs']

    def __init__(self, root_dir, output_file, include_extensions=None, exclude_list=None, stats=False):
        final_includes = include_extensions if include_extensions is not None else self.DEFAULT_PYTHON_EXTENSIONS
        final_excludes = self.DEFAULT_PYTHON_EXCLUDES + (exclude_list or [])
        super().__init__(root_dir, output_file, final_includes, list(set(final_excludes)), stats=stats)
        if NB_CONVERT_AVAILABLE:
            self.register_extractor('.ipynb', self._extract_ipynb_as_py_script)
        else:
            print("Aviso: 'nbformat' e 'nbconvert' não estão disponíveis. Arquivos .ipynb serão lidos como texto bruto.")

    def _extract_ipynb_as_py_script(self, file_path):
        try:
            python_exporter = PythonExporter()
            with open(file_path, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)
            (body, _) = python_exporter.from_notebook_node(nb)
            return body
        except Exception as e:
            return f"# ERRO AO CONVERTER O NOTEBOOK {os.path.basename(file_path)}: {e}\n"