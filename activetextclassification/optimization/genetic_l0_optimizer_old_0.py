# activetextclassification/optimization/genetic_l0_optimizer.py

import numpy as np
import pandas as pd
import time
import random
import os
import hashlib # Para hash de índices no log detalhado
# import json # Não diretamente necessário aqui no log, mas pode ser útil
from tqdm.notebook import tqdm # Para barras de progresso

# Imports da própria biblioteca
# Supondo que get_model e as classes base (BaseTextClassifier, BaseFeatureClassifier) estão em models
from ..models import get_model, BaseTextClassifier, BaseFeatureClassifier
# Supondo que BaseEmbedder está em embeddings (necessário para type hinting)
from ..embeddings import BaseEmbedder

# Imports de ML
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter # Para métricas auxiliares de L0


class GeneticL0Optimizer:
    """
    Otimiza a seleção do conjunto inicial L0 de tamanho fixo (semente) usando Algoritmos Genéticos.
    O objetivo é encontrar um L0 que maximize ou minimize a performance
    de um classificador treinado apenas nele, avaliado no dataset completo.
    Registra detalhes da composição do L0 e métricas para análise.
    """
    def __init__(self,
                 df_full,
                 text_column,
                 label_column,
                 classifier_config,
                 initial_l0_size,
                 all_possible_labels, # Lista de todos os labels para F1 consistente
                 population_size=50,
                 n_generations=100,
                 crossover_rate=0.7,
                 mutation_rate=0.1, # Probabilidade de um indivíduo sofrer mutação
                 mutation_strength=1, # Quantos genes (índices) trocar na mutação
                 elitism_rate=0.1,
                 fitness_metric='accuracy_on_full', # ou 'f1_macro_on_full'
                 optimization_goal='maximize', # 'maximize' ou 'minimize'
                 tournament_size=3, # Para seleção por torneio
                 random_seed=None,
                 embedder=None, # Embedder AJUSTADO, se classificador for BaseFeatureClassifier
                 # --- PARÂMETROS PARA LOG DETALHADO ---
                 log_detailed_fitness=True,
                 detailed_log_file="ag_detailed_fitness_log.csv"
                 # --- FIM PARÂMETROS ---
                 ):
        self.df_full = df_full.reset_index(drop=True) # Garantir índice 0..N-1
        self.text_column = text_column
        self.label_column = label_column
        self.classifier_config = classifier_config
        self.initial_l0_size = initial_l0_size
        self.all_possible_labels = sorted(list(set(all_possible_labels))) # Garantir ordenação e unicidade
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.elitism_rate = elitism_rate
        self.fitness_metric = fitness_metric
        self.optimization_goal = optimization_goal
        self.tournament_size = tournament_size
        self.embedder = embedder # Armazenar o embedder global ajustado

        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)

        # Validações
        if optimization_goal not in ['maximize', 'minimize']:
            raise ValueError("optimization_goal deve ser 'maximize' ou 'minimize'.")
        if fitness_metric not in ['accuracy_on_full', 'f1_macro_on_full']:
            raise ValueError("fitness_metric deve ser 'accuracy_on_full' ou 'f1_macro_on_full'.")
        if not (0 <= self.elitism_rate < 1): raise ValueError("Elitism rate deve ser entre 0 e 1 (exclusivo de 1).")
        if self.initial_l0_size > len(self.df_full): raise ValueError("initial_l0_size não pode ser maior que o dataset.")
        if self.initial_l0_size <= 0: raise ValueError("initial_l0_size deve ser positivo.")

        self.n_elite = int(self.population_size * self.elitism_rate)
        self.dataset_indices = np.array(self.df_full.index) # Índices do dataset completo

        # Cache para fitness {tuple(sorted_indices): {'acc': val, 'f1': val}}
        # Armazena as métricas REAIS (não o fitness transformado para minimização)
        self._fitness_cache = {}

        # Configuração e inicialização do log detalhado
        self.log_detailed_fitness = log_detailed_fitness
        self.detailed_log_file = detailed_log_file
        if self.log_detailed_fitness:
            self._initialize_detailed_log()

        print(f"GeneticL0Optimizer inicializado.")
        print(f" - População: {self.population_size}, Gerações: {self.n_generations}, L0 Size: {self.initial_l0_size}")
        print(f" - Objetivo: {self.optimization_goal}, Métrica: {self.fitness_metric}")
        if self.log_detailed_fitness: print(f" - Log Detalhado ATIVADO em: {os.path.abspath(self.detailed_log_file)}")


    def _initialize_detailed_log(self):
        """ Cria ou verifica o cabeçalho do arquivo de log detalhado. """
        log_dir = os.path.dirname(self.detailed_log_file)
        if log_dir and not os.path.exists(log_dir):
            try:
                os.makedirs(log_dir, exist_ok=True)
                print(f"Diretório de log detalhado criado: {log_dir}")
            except Exception as e:
                print(f"AVISO: Não foi possível criar diretório de log detalhado {log_dir}: {e}")

        if not os.path.exists(self.detailed_log_file):
            header = "generation,individual_id,l0_size,accuracy_on_full,f1_macro_on_full,num_tokens,num_distinct_tokens,num_classes_in_l0,most_frequent_class_in_l0,l0_indices_hash\n"
            try:
                with open(self.detailed_log_file, 'w', encoding='utf-8') as f:
                    f.write(header)
                # print(f"Arquivo de log detalhado '{self.detailed_log_file}' criado com cabeçalho.")
            except Exception as e:
                print(f"AVISO: Não foi possível escrever cabeçalho no log detalhado '{self.detailed_log_file}': {e}")
        else:
            # print(f"Arquivo de log detalhado '{self.detailed_log_file}' já existe. Resultados serão anexados.")
            pass # Mensagem já é dada no __init__


    def _log_individual_details(self, generation, individual_id, individual_indices, acc, f1):
        """ Loga detalhes de um indivíduo (L0) e suas métricas em CSV. """
        if not self.log_detailed_fitness: return

        try:
            L0_df_current = self.df_full.iloc[individual_indices]
            texts = L0_df_current[self.text_column].tolist()
            labels = L0_df_current[self.label_column].tolist()

            # Análise de Conteúdo do L0
            all_tokens = [token for text in texts for token in str(text).lower().split() if token] # Ignorar tokens vazios
            num_tokens = len(all_tokens)
            num_distinct_tokens = len(set(all_tokens))

            if labels:
                class_counts = Counter(labels)
                num_classes_in_l0 = len(class_counts)
                most_frequent_class_in_l0 = class_counts.most_common(1)[0][0] if class_counts else "N/A"
            else:
                num_classes_in_l0 = 0; most_frequent_class_in_l0 = "N/A"

            # Hash dos índices para identificar unicamente o L0 (opcional)
            l0_indices_hash = hashlib.sha256(str(sorted(individual_indices)).encode('utf-8')).hexdigest()[:10]

            # Formatar valores (substituir NaN/inf por strings para CSV)
            acc_str = f"{acc:.6f}" if acc > -np.inf else "NaN"
            f1_str = f"{f1:.6f}" if f1 > -np.inf else "NaN"

            log_line = f"{generation},{individual_id},{len(individual_indices)},{acc_str},{f1_str},{num_tokens},{num_distinct_tokens},{num_classes_in_l0},{most_frequent_class_in_l0},{l0_indices_hash}\n"
            with open(self.detailed_log_file, 'a', encoding='utf-8') as f:
                f.write(log_line)
        except Exception as e:
            print(f"AVISO (Log Detalhado): Erro ao escrever linha para indivíduo (Gen {generation}, ID {individual_id}): {e}")


    def _create_individual(self):
        """ Cria um indivíduo (L0) aleatório: uma lista de initial_l0_size índices únicos do df_full. """
        return np.random.choice(self.dataset_indices, size=self.initial_l0_size, replace=False)

    def _initialize_population(self):
        """ Inicializa a população com indivíduos aleatórios. """
        print("Inicializando população...")
        # Garantir que initial_l0_size <= len(dataset_indices) já foi validado no __init__
        population = []
        for i in tqdm(range(self.population_size), desc="Criando População Inicial"):
            population.append(self._create_individual())
        return population


    def _calculate_fitness_and_metrics(self, individual_indices, generation, individual_id_in_pop):
        """
        Calcula o fitness (para o AG) e as métricas reais (acc, f1) para um indivíduo (L0).
        Usa cache e log detalhado (se ativado).

        Retorna: (fitness_value, {'acc': valor, 'f1': valor})
        """
        cache_key = tuple(sorted(individual_indices))
        if cache_key in self._fitness_cache:
            cached_metrics = self._fitness_cache[cache_key]
            # print(f"    Cache hit: Gen {generation}, Ind {individual_id_in_pop}, Acc={cached_metrics['acc']:.4f}, F1={cached_metrics['f1']:.4f}")
            if self.log_detailed_fitness:
                 # Logar mesmo se for do cache
                 self._log_individual_details(generation, individual_id_in_pop, individual_indices, cached_metrics['acc'], cached_metrics['f1'])

            metric_for_goal = cached_metrics.get(self.fitness_metric.split('_')[0], -np.inf)
            # Calcular o fitness_value que o AG usará para maximizar
            if self.optimization_goal == 'minimize':
                # Se performance real é P (> -inf), fitness para AG é 1-P
                # Se performance real é -inf (erro), fitness para AG deve ser -inf
                fitness_value = (1.0 - metric_for_goal) if metric_for_goal > -np.inf else -np.inf
            else: # Maximize
                fitness_value = metric_for_goal # Fitness é a própria métrica real

            return fitness_value, cached_metrics # Retornar o fitness para o AG e o dict de métricas reais


        # Se não está no cache, calcular do zero
        L0_df_current = self.df_full.iloc[individual_indices]
        X_train_l0_text = L0_df_current[self.text_column].tolist()
        y_train_l0_labels = L0_df_current[self.label_column].tolist()
        model = get_model(self.classifier_config)

        acc_score, f1_score_val = -np.inf, -np.inf # Default para erro

        try:
            X_train_input = None
            if isinstance(model, BaseFeatureClassifier):
                if self.embedder is None: raise ValueError("Embedder necessário para BaseFeatureClassifier.")
                X_train_input = self.embedder.transform(X_train_l0_text)
            elif isinstance(model, BaseTextClassifier):
                X_train_input = X_train_l0_text
            else: raise TypeError(f"Tipo de modelo {type(model)} não suportado.")

            # Verificar se o input de treino não está vazio antes de fitar
            if (isinstance(X_train_input, np.ndarray) and X_train_input.size == 0) or \
               (isinstance(X_train_input, list) and not X_train_input):
                 raise ValueError("Input de treino está vazio. Não é possível treinar.")

            model.fit(X_train_input, y_train_l0_labels)

            X_eval_full_text = self.df_full[self.text_column].tolist()
            y_true_full_labels = self.df_full[self.label_column].tolist()
            X_eval_input = None
            if isinstance(model, BaseFeatureClassifier): X_eval_input = self.embedder.transform(X_eval_full_text)
            elif isinstance(model, BaseTextClassifier): X_eval_input = X_eval_full_text
            else: raise TypeError(f"Tipo de modelo {type(model)} não suportado para avaliação.")

            y_pred_full_labels = model.predict(X_eval_input)
            acc_score = accuracy_score(y_true_full_labels, y_pred_full_labels)
            f1_score_val = f1_score(y_true_full_labels, y_pred_full_labels, average='macro', zero_division=0, labels=self.all_possible_labels)

        except Exception as e:
            print(f"    ERRO ao calcular fitness para indivíduo (Gen {generation}, ID {individual_id_in_pop}): {type(e).__name__} - {e}")
            # acc_score e f1_score_val permanecem -np.inf (padrão)
            # traceback.print_exc() # Descomentar para debug


        current_metrics = {'acc': acc_score, 'f1': f1_score_val}
        self._fitness_cache[cache_key] = current_metrics # Salvar métricas reais no cache
        if self.log_detailed_fitness:
            self._log_individual_details(generation, individual_id_in_pop, individual_indices, acc_score, f1_score_val)

        # Determinar qual métrica REAL usar para o objetivo de otimização
        if self.fitness_metric == 'accuracy_on_full':
            metric_for_goal = acc_score
        else: # f1_macro_on_full
            metric_for_goal = f1_score_val

        # Calcular o fitness_value que o AG usará para maximizar
        if self.optimization_goal == 'minimize':
            # Se performance real é P (> -inf), fitness para AG é 1-P
            # Se performance real é -inf (erro), fitness para AG deve ser -inf
            fitness_value = (1.0 - metric_for_goal) if metric_for_goal > -np.inf else -np.inf
        else: # Maximize
            fitness_value = metric_for_goal # Fitness é a própria métrica real

        return fitness_value, current_metrics # Retornar o fitness para o AG e o dict de métricas reais


    def _selection(self, population, fitness_values_for_selection):
        """ Seleção por Torneio. Usa fitness_values_for_selection. """
        selected_parents = []
        for _ in range(len(population)):
            tournament_indices = np.random.choice(len(population), size=self.tournament_size, replace=True)
            tournament_fitness = [fitness_values_for_selection[i] for i in tournament_indices]
            winner_index_in_tournament = np.argmax(tournament_fitness)
            selected_parents.append(population[tournament_indices[winner_index_in_tournament]])
        return selected_parents

    def _crossover(self, parent1_indices, parent2_indices):
        """
        Crossover de um ponto para indivíduos de tamanho fixo (L0).
        Garante que não haja duplicatas e mantém o tamanho do L0.
        Retorna np.array de índices.
        """
        parent1 = np.array(parent1_indices); parent2 = np.array(parent2_indices)
        if self.initial_l0_size <= 1 : return parent1.copy(), parent2.copy() # Não pode fazer crossover ou é trivial
        crossover_point = random.randint(1, self.initial_l0_size - 1)
        child1_proto = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        child2_proto = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))

        # Função de reparo para garantir unicidade e tamanho
        def repair_child(child_proto, p1, p2):
            unique_genes = list(np.unique(child_proto))
            needed = self.initial_l0_size - len(unique_genes)
            if needed > 0:
                # Candidatos: genes dos pais que não estão no filho, ou genes do dataset que não estão no filho
                candidates_p = np.setdiff1d(np.concatenate((p1,p2)), unique_genes)
                candidates_ds = np.setdiff1d(self.dataset_indices, unique_genes)
                # Combinar candidatos e embaralhar
                all_candidates = np.concatenate((candidates_p, candidates_ds))
                np.random.shuffle(all_candidates)

                if len(all_candidates) < needed:
                     # Muito improvável a menos que L0_size seja quase o dataset inteiro
                     print(f"AVISO Crossover: Não há genes suficientes no dataset para completar o filho. Retornando indivíduo aleatório.")
                     return self._create_individual() # Recriar aleatoriamente

                unique_genes.extend(all_candidates[:needed]) # Pegar o necessário dos candidatos

            # Se por algum motivo o tamanho ainda estiver errado (pós reparo), reamostra ou recria
            if len(unique_genes) != self.initial_l0_size:
                 # Se ficou maior (deveria ser evitado pelo np.unique)
                 if len(unique_genes) > self.initial_l0_size:
                      print(f"AVISO Crossover: Tamanho final > L0_size. Reamostrando.")
                      return np.random.choice(unique_genes, size=self.initial_l0_size, replace=False)
                 else: # Se menor (improvável após o reparo)
                      print(f"AVISO Crossover: Tamanho final < L0_size. Recriando aleatoriamente.")
                      return self._create_individual()

            return np.array(unique_genes)

        return repair_child(child1_proto, parent1, parent2), repair_child(child2_proto, parent2, parent1)


    def _mutation(self, individual_indices):
        """
        Mutação: troca 'mutation_strength' genes (índices) do indivíduo
        por outros aleatórios do dataset (que não estão no indivíduo).
        Retorna np.array de índices.
        """
        mutated_individual = np.array(individual_indices)
        if len(mutated_individual) == 0: return mutated_individual

        # Encontrar candidatos para substituição (todos os índices do dataset MENOS os já presentes no indivíduo)
        pool_for_mutation = np.setdiff1d(self.dataset_indices, mutated_individual)

        if len(pool_for_mutation) == 0:
             # Não há amostras fora do L0 para trocar. Mutação impossível.
             return mutated_individual # Retorna cópia original

        # Quantos genes realmente tentar mutar (não mais do que a força, nem mais do que os disponíveis no pool)
        actual_mutation_count = min(self.mutation_strength, self.initial_l0_size, len(pool_for_mutation))

        if actual_mutation_count <= 0: return mutated_individual

        # Escolher quais posições no indivíduo serão mutadas
        positions_to_mutate = np.random.choice(len(mutated_individual), size=actual_mutation_count, replace=False)

        # Escolher os novos genes do pool de mutação
        replacement_genes = np.random.choice(pool_for_mutation, size=actual_mutation_count, replace=False)

        # Aplicar a mutação
        mutated_individual[positions_to_mutate] = replacement_genes

        return mutated_individual


    def run_optimization(self):
        print(f"\n--- Iniciando Otimização Genética para L0 (Tam: {self.initial_l0_size}, Obj: {self.optimization_goal}, Métrica: {self.fitness_metric}) ---")
        start_opt_time = time.time()
        population = self._initialize_population()
        optimization_history = [] # Para fitness max/min/avg (performance real) por geração

        # Inicializar melhor geral (baseado na performance REAL da métrica alvo)
        if self.optimization_goal == 'maximize':
            best_actual_performance_overall = -np.inf
        else: # minimize
            best_actual_performance_overall = np.inf
        best_individual_overall = None


        for gen in tqdm(range(self.n_generations), desc="Gerações AG"):
            gen_start_time = time.time()
            print(f"\n  Geração {gen + 1}/{self.n_generations}")

            # Calcular Fitness e Métricas Reais para a População Atual
            fitness_values_for_selection = [] # O que o AG vai usar para selecionar (pode ser 1-perf)
            actual_metrics_population = [] # Lista de dicts {'acc': v_acc, 'f1': v_f1}

            for i, ind in enumerate(tqdm(population, desc="Fitness Calc", leave=False)):
                 # Passar generation e individual_id_in_pop para o log detalhado
                 fitness_val, metrics_dict = self._calculate_fitness_and_metrics(ind, gen + 1, i)
                 fitness_values_for_selection.append(fitness_val)
                 actual_metrics_population.append(metrics_dict)
                 print(f"    Indivíduo {i+1}/{self.population_size}: Fitness={fitness_val:.4f}, Acc={metrics_dict['acc']:.4f}, F1={metrics_dict['f1']:.4f}")

            # Melhor da Geração Atual (baseado na performance REAL da métrica alvo)
            ####Correção
            # Precisamos dos valores reais para encontrar o max/min real
            if self.fitness_metric == 'accuracy_on_full':
                current_gen_performances_real = [m.get('acc', -np.inf if self.optimization_goal == 'maximize' else np.inf) for m in actual_metrics_population]  
            else: # f1_macro_on_full
                current_gen_performances_real = [m.get('f1', -np.inf if self.optimization_goal == 'maximize' else np.inf) for m in actual_metrics_population]
            # current_gen_performances_real = [m.get(self.fitness_metric.split('_')[0], -np.inf if self.optimization_goal == 'maximize' else np.inf) for m in actual_metrics_population]
            print(f"    Performance Real Geração {gen+1}: {current_gen_performances_real}")
            if self.optimization_goal == 'maximize':
                # Encontrar o índice do indivíduo com a performance real MÁXIMA
                if any(p > -np.inf for p in current_gen_performances_real):
                     best_idx_this_gen = np.nanargmax(current_gen_performances_real)
                     current_gen_best_actual_perf = current_gen_performances_real[best_idx_this_gen]
                else: # Todos os resultados foram -np.inf
                     best_idx_this_gen = 0 # Pega o primeiro como placeholder
                     current_gen_best_actual_perf = -np.inf
            else: # Minimize (encontrar o índice do indivíduo com a performance real MÍNIMA)
                if any(p < np.inf for p in current_gen_performances_real):
                     best_idx_this_gen = np.nanargmin(current_gen_performances_real)
                     current_gen_best_actual_perf = current_gen_performances_real[best_idx_this_gen]
                else: # Todos os resultados foram +np.inf
                     best_idx_this_gen = 0 # Pega o primeiro como placeholder
                     current_gen_best_actual_perf = np.inf

            current_gen_best_individual = population[best_idx_this_gen]
            print(f"    Melhor da Geração {gen+1}: Fitness={fitness_values_for_selection[best_idx_this_gen]:.4f}, Acc={actual_metrics_population[best_idx_this_gen]['acc']:.4f}, F1={actual_metrics_population[best_idx_this_gen]['f1']:.4f}")

            # Atualizar Melhor Geral (usando performance REAL)
            if self.optimization_goal == 'maximize':
                if current_gen_best_actual_perf > best_actual_performance_overall:
                    best_actual_performance_overall = current_gen_best_actual_perf
                    best_individual_overall = current_gen_best_individual
                    print(f"    Novo melhor (max): Performance Real={best_actual_performance_overall:.4f}")
            else: # Minimize
                if current_gen_best_actual_perf < best_actual_performance_overall:
                    best_actual_performance_overall = current_gen_best_actual_perf
                    best_individual_overall = current_gen_best_individual
                    print(f"    Novo 'melhor' (min): Performance Real={best_actual_performance_overall:.4f}")


            # Registrar histórico da geração (com performances REAIS de acc E f1)
            gen_actual_acc_scores = [m.get('acc', -np.inf) for m in actual_metrics_population]
            gen_actual_f1_scores  = [m.get('f1', -np.inf) for m in actual_metrics_population]
            print(f"    Acc Real Geração {gen+1}: {gen_actual_acc_scores}")
            print(f"    F1 Real Geração  {gen+1}: {gen_actual_f1_scores}")
            # Remover -np.inf e np.inf antes de calcular média/min/max para evitar warnings/erros
            valid_acc = [s for s in gen_actual_acc_scores if s > -np.inf and s < np.inf]
            valid_f1 = [s for s in gen_actual_f1_scores if s > -np.inf and s < np.inf]

            optimization_history.append({
                'generation': gen + 1,
                'max_acc': np.max(valid_acc) if valid_acc else -np.inf,
                'avg_acc': np.mean(valid_acc) if valid_acc else -np.inf,
                'min_acc': np.min(valid_acc) if valid_acc else -np.inf,
                'max_f1': np.max(valid_f1) if valid_f1 else -np.inf,
                'avg_f1': np.mean(valid_f1) if valid_f1 else -np.inf,
                'min_f1': np.min(valid_f1) if valid_f1 else -np.inf,
            })
            print(f"    Acc Real Geração {gen+1}: Max={optimization_history[-1]['max_acc']:.4f}, Avg={optimization_history[-1]['avg_acc']:.4f}, Min={optimization_history[-1]['min_acc']:.4f}")
            print(f"    F1 Real Geração  {gen+1}: Max={optimization_history[-1]['max_f1']:.4f}, Avg={optimization_history[-1]['avg_f1']:.4f}, Min={optimization_history[-1]['min_f1']:.4f}")

            # Seleção e Elitismo (usa fitness_values_for_selection, que o AG maximiza)
            sorted_population_with_fitness = sorted(zip(population, fitness_values_for_selection), key=lambda x: x[1], reverse=True)
            elite = [ind for ind, fit_val in sorted_population_with_fitness[:self.n_elite]]
            parents = self._selection(population, fitness_values_for_selection)

            # Crossover e Mutação
            next_population = elite.copy()
            num_offspring_needed = self.population_size - self.n_elite
            current_offspring_count = 0; parent_idx = 0
            while current_offspring_count < num_offspring_needed:
                p1_idx = parent_idx % len(parents); parent_idx = (parent_idx + 1) % len(parents)
                p2_idx = parent_idx % len(parents); parent_idx = (parent_idx + 1) % len(parents)
                p1 = parents[p1_idx]
                p2 = parents[p2_idx]
                if p1 is p2 and len(parents) > 1 : # Tentar pegar pais diferentes
                    parent_idx = (parent_idx + 1) % len(parents)
                    p2 = parents[parent_idx % len(parents)]

                child1, child2 = (p1.copy(), p2.copy()) if random.random() >= self.crossover_rate else self._crossover(p1, p2)

                # Aplicar mutação individualmente
                if random.random() < self.mutation_rate: child1 = self._mutation(child1)
                if random.random() < self.mutation_rate: child2 = self._mutation(child2)

                next_population.append(child1); current_offspring_count +=1
                if current_offspring_count < num_offspring_needed:
                    next_population.append(child2); current_offspring_count += 1

            population = next_population[:self.population_size]
            gen_duration = time.time() - gen_start_time
            print(f"    Duração Geração {gen+1}: {gen_duration:.2f}s")

        opt_duration = time.time() - start_opt_time
        print(f"\n--- Otimização Genética Concluída ({opt_duration:.2f} seg) ---")
        print(f"Melhor Performance Real Encontrada ({self.fitness_metric}, Objetivo: {self.optimization_goal}): {best_actual_performance_overall:.4f}")

        return best_individual_overall, best_actual_performance_overall, pd.DataFrame(optimization_history)

# Adicionar import de hashlib no topo do arquivo
import hashlib