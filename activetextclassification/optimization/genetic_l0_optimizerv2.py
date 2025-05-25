# activetextclassification/optimization/genetic_l0_optimizer.py

import numpy as np
import pandas as pd
import time
import random
import os
import hashlib
from tqdm.notebook import tqdm
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score

# Imports da própria biblioteca
from ..models import get_model, BaseTextClassifier, BaseFeatureClassifier
from ..embeddings import BaseEmbedder

# NOVO: Import para multiprocessing
from multiprocessing import Pool, cpu_count
# NOVO: Para passar argumentos para a função worker de forma mais limpa
from functools import partial


# --- Helper function para ser usada com multiprocessing.Pool ---
# Deve ser definida no nível do módulo para ser "picklable"
def _calculate_fitness_worker(individual_indices_tuple, # Tuple para ser hasheable para cache
                              df_full_text_list, # Passar listas/arrays numpy básicos
                              df_full_label_list,
                              text_column, # Nome da coluna, não o objeto Series
                              label_column,
                              classifier_config,
                              all_possible_labels,
                              embedder_instance, # O embedder pré-ajustado, se houver
                              fitness_metric,
                              optimization_goal,
                              generation_id, # Para logging, se necessário
                              individual_id_in_gen # Para logging
                              ):
    # Recriar DataFrame L0 dentro do worker a partir dos índices e dados passados
    # Isso evita passar DataFrames grandes, que podem ser lentos para serializar/desserializar
    # No entanto, para df_full, passamos as listas de texto/label.
    # Para L0, podemos reconstruir um DataFrame temporário se necessário, ou apenas usar listas.
    
    individual_indices = list(individual_indices_tuple) # Converter de volta para lista

    # Preparar L0 a partir dos dados completos e índices individuais
    # df_full é acessado indiretamente via df_full_text_list e df_full_label_list
    L0_texts = [df_full_text_list[i] for i in individual_indices]
    L0_labels = [df_full_label_list[i] for i in individual_indices]

    fitness_calc_start_time = time.time()
    model = get_model(classifier_config)
    acc, f1 = -np.inf, -np.inf

    try:
        X_in_L0 = embedder_instance.transform(L0_texts) if isinstance(model, BaseFeatureClassifier) and embedder_instance else L0_texts
        if not ((isinstance(X_in_L0, np.ndarray) and X_in_L0.size > 0) or (isinstance(X_in_L0, list) and X_in_L0)):
            raise ValueError("Input de treino para L0 vazio.")
        
        model.fit(X_in_L0, L0_labels)

        # Avaliação no dataset completo
        X_eval_full_texts = df_full_text_list # Já é uma lista
        y_true_full = df_full_label_list    # Já é uma lista

        X_eval_full_transformed = embedder_instance.transform(X_eval_full_texts) if isinstance(model, BaseFeatureClassifier) and embedder_instance else X_eval_full_texts
        
        y_pred_full = model.predict(X_eval_full_transformed)
        
        acc = accuracy_score(y_true_full, y_pred_full)
        f1 = f1_score(y_true_full, y_pred_full, average='macro', zero_division=0, labels=all_possible_labels)

    except Exception as e:
        # É importante capturar exceções aqui e talvez retornar um valor de fitness ruim
        # para que o processo principal possa lidar com isso.
        # O print pode ser útil para debug, mas em produção pode ser melhor logar ou retornar a exceção.
        print(f"    ERRO no worker (Gen {generation_id}, ID {individual_id_in_gen}): {type(e).__name__} - {e}")
        # Retornar as métricas como -np.inf em caso de erro

    fitness_calc_time = time.time() - fitness_calc_start_time
    current_metrics = {'acc': acc, 'f1': f1}

    # Determinar o valor de fitness com base no objetivo e métrica
    target_metric_key = 'acc' if fitness_metric == 'accuracy_on_full' else 'f1'
    metric_for_goal = current_metrics.get(target_metric_key, -np.inf) # Default para -inf se erro
    
    fitness_value = (1.0 - metric_for_goal) if optimization_goal == 'minimize' and metric_for_goal > -np.inf else metric_for_goal
    # Lidar com erros: se métrica é -inf, fitness também deve ser o "pior" possível
    if optimization_goal == 'minimize' and metric_for_goal <= -np.inf:
        fitness_value = np.inf # Minimizar, erro é o pior (fitness alto)
    elif optimization_goal == 'maximize' and metric_for_goal <= -np.inf:
        fitness_value = -np.inf # Maximizar, erro é o pior (fitness baixo)

    # Retornar tudo que o processo principal precisa
    return individual_indices_tuple, fitness_value, current_metrics, fitness_calc_time, None # O último None é placeholder para erro, se quiser propagar


class GeneticL0Optimizer:
    def __init__(self,
                 df_full,
                 text_column,
                 label_column,
                 classifier_config,
                 initial_l0_size,
                 all_possible_labels,
                 population_size=50,
                 n_generations=100,
                 crossover_rate=0.7,
                 mutation_rate=0.1,
                 mutation_strength=1,
                 elitism_rate=0.1,
                 fitness_metric='accuracy_on_full',
                 optimization_goal='maximize',
                 tournament_size=3,
                 random_seed=None,
                 embedder=None,
                 log_detailed_fitness=True,
                 detailed_log_file="ag_detailed_fitness_log.csv",
                 n_jobs=-1 # NOVO: Número de processos paralelos, -1 para usar todos os cores
                 ):
        self.df_full = df_full.reset_index(drop=True)
        self.text_column = text_column
        self.label_column = label_column
        self.classifier_config = classifier_config
        self.initial_l0_size = initial_l0_size
        self.all_possible_labels = sorted(list(set(all_possible_labels)))
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.elitism_rate = elitism_rate
        self.fitness_metric = fitness_metric
        self.optimization_goal = optimization_goal
        self.tournament_size = tournament_size
        self.embedder = embedder # Este é o embedder AJUSTADO (se aplicável)
        self.log_detailed_fitness = log_detailed_fitness
        self.detailed_log_file = detailed_log_file
        
        if n_jobs == -1:
            self.n_jobs = cpu_count()
        elif n_jobs <= 0:
            self.n_jobs = 1 # Sem paralelismo
        else:
            self.n_jobs = n_jobs
        print(f"Usando {self.n_jobs} processos para cálculo de fitness.")

        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)

        # Validações (mantidas)
        if optimization_goal not in ['maximize', 'minimize']:
            raise ValueError("optimization_goal deve ser 'maximize' ou 'minimize'.")
        if fitness_metric not in ['accuracy_on_full', 'f1_macro_on_full']:
            raise ValueError("fitness_metric deve ser 'accuracy_on_full' ou 'f1_macro_on_full'.")
        if not (0 <= self.elitism_rate < 1): raise ValueError("Elitism rate deve ser entre 0 e 1 (exclusivo de 1).")
        if self.initial_l0_size > len(self.df_full): raise ValueError("initial_l0_size não pode ser maior que o dataset.")
        if self.initial_l0_size <= 0: raise ValueError("initial_l0_size deve ser positivo.")

        self.n_elite = int(self.population_size * self.elitism_rate)
        self.dataset_indices = np.array(self.df_full.index)
        self._fitness_cache = {}

        if self.log_detailed_fitness:
            self._initialize_detailed_log()

        # --- Pré-processar dados para os workers ---
        # Isso evita que cada worker acesse self.df_full diretamente,
        # o que pode ser problemático com multiprocessing em algumas plataformas (Windows).
        # Passar listas/arrays numpy é mais seguro.
        self._df_full_text_list_for_workers = self.df_full[self.text_column].tolist()
        self._df_full_label_list_for_workers = self.df_full[self.label_column].tolist()

        print(f"GeneticL0Optimizer inicializado.")
        print(f" - População: {self.population_size}, Gerações: {self.n_generations}, L0 Size: {self.initial_l0_size}")
        print(f" - Objetivo: {self.optimization_goal}, Métrica: {self.fitness_metric}")
        if self.log_detailed_fitness: print(f" - Log Detalhado ATIVADO em: {os.path.abspath(self.detailed_log_file)}")


    def _initialize_detailed_log(self): # Mantido como está
        log_dir = os.path.dirname(self.detailed_log_file)
        if log_dir and not os.path.exists(log_dir):
            try: os.makedirs(log_dir, exist_ok=True)
            except Exception as e: print(f"AVISO: Não criar dir log {log_dir}: {e}")

        if not os.path.exists(self.detailed_log_file) or os.path.getsize(self.detailed_log_file) == 0:
            header = ("generation,individual_id,l0_size,accuracy_on_full,f1_macro_on_full,"
                      "num_tokens,num_distinct_tokens,num_classes_in_l0,most_frequent_class_in_l0,"
                      "l0_indices_hash,fitness_calc_time_sec\n")
            try:
                with open(self.detailed_log_file, 'w', encoding='utf-8') as f: f.write(header)
            except Exception as e: print(f"AVISO: Não escrever header log '{self.detailed_log_file}': {e}")

    def _log_individual_details(self, generation, individual_id, individual_indices, acc, f1, fitness_calc_time): # Mantido como está
        if not self.log_detailed_fitness: return
        try:
            # Usar as listas pré-processadas para eficiência se df_full for muito grande
            L0_texts = [self._df_full_text_list_for_workers[i] for i in individual_indices]
            L0_labels = [self._df_full_label_list_for_workers[i] for i in individual_indices]
            
            tokens = [t for txt in L0_texts for t in str(txt).lower().split() if t]
            n_tok = len(tokens); n_dist_tok = len(set(tokens))
            counts = Counter(L0_labels); n_cls = len(counts); mfc = counts.most_common(1)[0][0] if counts else "N/A"
            h = hashlib.sha256(str(sorted(individual_indices)).encode('utf-8')).hexdigest()[:10]
            acc_s = f"{acc:.6f}" if pd.notna(acc) and acc > -np.inf and acc < np.inf else "Error"
            f1_s  = f"{f1:.6f}" if pd.notna(f1) and f1 > -np.inf and f1 < np.inf else "Error"
            time_s = f"{fitness_calc_time:.4f}" if pd.notna(fitness_calc_time) else "N/A"

            log_line = (f"{generation},{individual_id},{len(individual_indices)},{acc_s},{f1_s},"
                        f"{n_tok},{n_dist_tok},{n_cls},{mfc},{h},{time_s}\n")
            with open(self.detailed_log_file, 'a', encoding='utf-8') as f: f.write(log_line)
        except Exception as e: print(f"AVISO (Log Detalhado) Erro Gen {generation}, ID {individual_id}: {e}")


    def _create_individual(self): # Mantido como está
        return np.random.choice(self.dataset_indices, size=self.initial_l0_size, replace=False)

    def _initialize_population(self): # Mantido como está
        print("Inicializando população...");
        return [self._create_individual() for _ in tqdm(range(self.population_size), desc="Criando População Inicial")]

    # _calculate_fitness_and_metrics será um wrapper para o cache e para chamar o pool
    def _calculate_fitness_for_population(self, population, generation):
        tasks_for_pool = []
        cached_results = {} # Resultados desta geração que vieram do cache

        for i, ind_indices_array in enumerate(population):
            # Cache key deve ser um tipo hasheável, como tupla de inteiros ordenados
            cache_key = tuple(sorted(ind_indices_array.tolist())) # Convert array to list then tuple

            if cache_key in self._fitness_cache:
                cached_data = self._fitness_cache[cache_key]
                cached_metrics = cached_data['metrics']
                original_calc_time = cached_data.get('calc_time', 0.0)

                target_metric_key = 'acc' if self.fitness_metric == 'accuracy_on_full' else 'f1'
                metric_for_goal = cached_metrics.get(target_metric_key, -np.inf)
                
                fitness_value = (1.0 - metric_for_goal) if self.optimization_goal == 'minimize' and metric_for_goal > -np.inf else metric_for_goal
                if self.optimization_goal == 'minimize' and metric_for_goal <= -np.inf : fitness_value = np.inf
                if self.optimization_goal == 'maximize' and metric_for_goal <= -np.inf : fitness_value = -np.inf
                
                cached_results[cache_key] = (fitness_value, cached_metrics, original_calc_time)
                
                if self.log_detailed_fitness:
                    self._log_individual_details(generation, i, list(cache_key),
                                                 cached_metrics.get('acc', -np.inf),
                                                 cached_metrics.get('f1', -np.inf),
                                                 original_calc_time) # Log com tempo original
            else:
                # Adicionar à lista de tarefas para o pool de processos
                # Passar o cache_key (tupla de índices) para o worker
                tasks_for_pool.append(
                    (cache_key, # tupla de índices para o worker
                     self._df_full_text_list_for_workers,
                     self._df_full_label_list_for_workers,
                     self.text_column, # nome da coluna
                     self.label_column, # nome da coluna
                     self.classifier_config,
                     self.all_possible_labels,
                     self.embedder, # Passar o embedder (deve ser picklable)
                     self.fitness_metric,
                     self.optimization_goal,
                     generation, # Para logging no worker, se necessário
                     i # ID do indivíduo na geração, para logging
                    )
                )
        
        # Processar tarefas não cacheadas em paralelo
        processed_results_list = []
        if tasks_for_pool:
            if self.n_jobs > 1:
                with Pool(processes=self.n_jobs) as pool:
                    # Usar starmap para desempacotar os argumentos de tasks_for_pool
                    # A função _calculate_fitness_worker é definida no nível do módulo
                    results_from_pool = list(tqdm(pool.starmap(_calculate_fitness_worker, tasks_for_pool),
                                                  total=len(tasks_for_pool),
                                                  desc=f"Fitness Calc Gen {generation} (Parallel)",
                                                  leave=False))
            else: # Execução sequencial se n_jobs <= 1
                results_from_pool = []
                for task_args in tqdm(tasks_for_pool, desc=f"Fitness Calc Gen {generation} (Sequential)", leave=False):
                    results_from_pool.append(_calculate_fitness_worker(*task_args))


            for i, (individual_indices_tuple, fitness_val, metrics_dict, fitness_calc_time, error_info) in enumerate(results_from_pool):
                # O primeiro elemento de task_args é o cache_key (tupla de índices)
                # Se você precisar do índice original `i` da população, você pode parear
                # as `tasks_for_pool` com `results_from_pool` ou passá-lo e retorná-lo do worker.
                # Aqui, vamos assumir que a ordem é mantida ou que o worker retorna o ID.
                # O worker já retorna individual_indices_tuple, que é o cache_key.

                cache_key = individual_indices_tuple # Este é o tuple(sorted(indices))
                
                # Atualizar cache
                self._fitness_cache[cache_key] = {'metrics': metrics_dict, 'calc_time': fitness_calc_time}
                
                processed_results_list.append((cache_key, fitness_val, metrics_dict, fitness_calc_time))

                if self.log_detailed_fitness:
                    # O worker já retorna os índices como uma tupla ordenada (cache_key)
                    self._log_individual_details(generation,
                                                 tasks_for_pool[i][-1], # individual_id_in_gen original
                                                 list(cache_key), # Converter tupla de volta para lista para log
                                                 metrics_dict.get('acc', -np.inf),
                                                 metrics_dict.get('f1', -np.inf),
                                                 fitness_calc_time)
        
        # Montar a lista final de fitness e métricas na ordem da população original
        final_fitness_values = []
        final_actual_metrics = []

        for ind_indices_array in population:
            cache_key = tuple(sorted(ind_indices_array.tolist()))
            if cache_key in cached_results:
                fitness_val, metrics_dict, _ = cached_results[cache_key]
            else:
                # Encontrar o resultado processado correspondente
                # Isto assume que processed_results_list contém uma entrada para cada cache_key não cacheado
                found = False
                for pk, pv, pm, pt in processed_results_list:
                    if pk == cache_key:
                        fitness_val, metrics_dict = pv, pm
                        found = True
                        break
                if not found:
                    # Isso não deveria acontecer se a lógica estiver correta
                    print(f"AVISO: Resultado não encontrado para indivíduo {cache_key} após processamento.")
                    fitness_val = -np.inf if self.optimization_goal == 'maximize' else np.inf
                    metrics_dict = {'acc': -np.inf, 'f1': -np.inf}

            final_fitness_values.append(fitness_val)
            final_actual_metrics.append(metrics_dict)
            
        return final_fitness_values, final_actual_metrics

    # As funções _selection, _crossover, _mutation permanecem as mesmas
    def _selection(self, population, fitness_values):
        parents = []
        pop_indices = np.arange(len(population))
        for _ in range(len(population)):
            tour_idx_pop = np.random.choice(pop_indices, size=self.tournament_size, replace=True)
            tour_fit_values = [fitness_values[i] for i in tour_idx_pop]
            
            # Lidar com possíveis -np.inf ou np.inf para que argmax/argmin funcione
            # Se todos no torneio forem problemáticos, escolhe um aleatoriamente
            valid_fitness_in_tournament = [f for f in tour_fit_values if f not in [-np.inf, np.inf] and pd.notna(f)]

            if not valid_fitness_in_tournament: # Todos são -inf, inf ou NaN
                # Se o objetivo é maximizar e todos são -inf, ou minimizar e todos são +inf,
                # qualquer escolha é "igualmente ruim". Escolha aleatória.
                # Se há uma mistura de -inf e +inf, a lógica de argmax/argmin já não é ideal.
                # Uma abordagem mais robusta seria preferir não-infinitos.
                winner_local_idx = np.random.randint(0, self.tournament_size)
            else:
                # Para maximização, queremos o maior. Para minimização, queremos o menor.
                # np.argmax/argmin funciona como esperado se não houver NaNs e os infs forem consistentes.
                # Se o objetivo é 'maximize', usamos argmax. Se é 'minimize', usamos argmin (em -fitness).
                # No entanto, fitness_values já foram ajustados (1-metric para min). Então sempre maximizamos fitness.
                temp_tour_fit = np.array(tour_fit_values)
                
                # Substituir -np.inf por um valor muito pequeno e np.inf por um muito grande
                # para que argmax funcione como esperado se houver uma mistura.
                # Se todos são -np.inf, o if anterior trata.
                # Se todos são np.inf (para minimização, isso significa erro), o if anterior trata.
                # O que importa é que `fitness_values` já está configurado para que MAIOR seja MELHOR.
                
                # Se todos são -np.inf (erro para maximização), o `if` anterior deveria ter pego.
                # Se todos são np.inf (erro para minimização, mas fitness é 1-m, então é -np.inf),
                # o `if` anterior também deveria ter pego.
                # Esta lógica assume que fitness_values são sempre comparáveis (maior é melhor).
                if all(f == -np.inf for f in tour_fit_values): # Todos ruins para maximização
                    winner_local_idx = np.random.randint(0, self.tournament_size)
                elif all(f == np.inf for f in tour_fit_values): # Todos ruins para minimização (fitness seria -inf) - não deveria acontecer
                    winner_local_idx = np.random.randint(0, self.tournament_size)
                else:
                    # Substitui -np.inf para que argmax não os escolha se houver finitos
                    # E np.inf por valores que argmax escolheria se for o caso.
                    # A suposição aqui é que `fitness_values` são sempre para maximização.
                    processed_tour_fit = np.array([f if f != -np.inf else -1e18 for f in temp_tour_fit]) # -1e18 é um número muito pequeno
                    winner_local_idx = np.argmax(processed_tour_fit)

            parents.append(population[tour_idx_pop[winner_local_idx]])
        return parents

    def _crossover(self, p1_idx, p2_idx): # Mantido como está
        parent1 = np.array(p1_idx); parent2 = np.array(p2_idx)
        if self.initial_l0_size <= 1 : return parent1.copy(), parent2.copy()
        cp = random.randint(1, self.initial_l0_size - 1) if self.initial_l0_size >=2 else 1
        c1_p=np.concatenate((parent1[:cp],parent2[cp:])); c2_p=np.concatenate((parent2[:cp],parent1[cp:]))
        def repair(c_p,p1,p2):
            u_set = set(c_p)
            unique_genes = list(u_set)
            needed = self.initial_l0_size - len(unique_genes)
            if needed > 0:
                parent_pool = set(p1) | set(p2)
                candidates_p = list(parent_pool - u_set)
                random.shuffle(candidates_p)
                can_add_from_parents = min(needed, len(candidates_p))
                unique_genes.extend(candidates_p[:can_add_from_parents])
                needed -= can_add_from_parents
            if needed > 0:
                dataset_set = set(self.dataset_indices)
                # Garantir que unique_genes seja um set para a diferença
                pool_ds = list(dataset_set - set(unique_genes)) 
                random.shuffle(pool_ds)
                if len(pool_ds) < needed: return self._create_individual() # fallback
                unique_genes.extend(pool_ds[:needed])
            if len(unique_genes) > self.initial_l0_size:
                 return np.random.choice(unique_genes, size=self.initial_l0_size, replace=False)
            elif len(unique_genes) < self.initial_l0_size: # Should not happen if logic is correct
                 return self._create_individual() # fallback
            return np.array(unique_genes)
        return repair(c1_p,parent1,parent2), repair(c2_p,parent2,parent1)

    def _mutation(self, ind_idx): # Mantido como está
        mut=np.array(ind_idx)
        if len(mut)==0 or self.mutation_strength==0: return mut
        # Garantir que dataset_indices é 1D array para setdiff1d
        pool=np.setdiff1d(np.array(self.dataset_indices).flatten(), mut.flatten())
        if len(pool)==0: return mut
        act_mut_cnt=min(self.mutation_strength, len(mut), len(pool))
        if act_mut_cnt<=0: return mut
        pos_mut=np.random.choice(len(mut),size=act_mut_cnt,replace=False)
        repl_genes=np.random.choice(pool,size=act_mut_cnt,replace=False)
        mut[pos_mut]=repl_genes; return mut

    def run_optimization(self):
        print(f"\n--- Iniciando Otimização Genética (Tam: {self.initial_l0_size}, Obj: {self.optimization_goal}, Métrica: {self.fitness_metric}) ---")
        start_opt_time = time.time()
        population = self._initialize_population()
        optimization_history = []

        best_actual_performance_overall = -np.inf if self.optimization_goal == 'maximize' else np.inf
        best_individual_overall = None
        if not population: # population é uma lista de arrays
            print("ERRO: População não inicializada."); return np.array([],dtype=int),best_actual_performance_overall,pd.DataFrame(optimization_history)

        for gen in tqdm(range(self.n_generations), desc="Gerações AG"):
            gen_start_time = time.time()
            print(f"\n  Geração {gen + 1}/{self.n_generations}")

            # Chamada à nova função de cálculo de fitness para a população
            fitness_values_for_selection, actual_metrics_population = self._calculate_fitness_for_population(population, gen + 1)

            target_metric_key = 'acc' if self.fitness_metric == 'accuracy_on_full' else 'f1'
            default_val_target = -np.inf if self.optimization_goal == 'maximize' else np.inf #pior valor possível
            current_gen_performances_real = [m.get(target_metric_key, default_val_target) for m in actual_metrics_population]

            valid_perf_mask = pd.Series(current_gen_performances_real).notna() & \
                              (pd.Series(current_gen_performances_real) != -np.inf) & \
                              (pd.Series(current_gen_performances_real) != np.inf)
            
            best_idx_this_gen = -1 # Default se nenhum válido
            if not valid_perf_mask.any():
                print("    AVISO: Nenhuma performance válida na geração.")
                current_gen_best_actual_perf = default_val_target
            else:
                valid_subset_performances = np.array(current_gen_performances_real)[valid_perf_mask]
                original_indices_of_valid = np.where(valid_perf_mask)[0]

                if self.optimization_goal == 'maximize':
                    best_local_idx_in_valid = np.argmax(valid_subset_performances)
                else: # Minimize
                    best_local_idx_in_valid = np.argmin(valid_subset_performances)
                
                best_idx_this_gen = original_indices_of_valid[best_local_idx_in_valid]
                current_gen_best_actual_perf = current_gen_performances_real[best_idx_this_gen]

            current_gen_best_individual = population[best_idx_this_gen] if best_idx_this_gen != -1 and population else None

            new_best_found_this_gen = False
            if self.optimization_goal == 'maximize':
                if pd.notna(current_gen_best_actual_perf) and current_gen_best_actual_perf > -np.inf: # Válido e não erro
                    if current_gen_best_actual_perf > best_actual_performance_overall:
                        best_actual_performance_overall=current_gen_best_actual_perf; best_individual_overall=current_gen_best_individual; new_best_found_this_gen=True
            else: # Minimize
                if pd.notna(current_gen_best_actual_perf) and current_gen_best_actual_perf < np.inf: # Válido e não erro
                    if current_gen_best_actual_perf < best_actual_performance_overall: # Melhor que o anterior
                        best_actual_performance_overall=current_gen_best_actual_perf; best_individual_overall=current_gen_best_individual; new_best_found_this_gen=True
                    elif best_actual_performance_overall == np.inf and current_gen_best_actual_perf > -np.inf: # Primeiro valor real para min (não erro)
                        best_actual_performance_overall=current_gen_best_actual_perf; best_individual_overall=current_gen_best_individual; new_best_found_this_gen=True
            
            if new_best_found_this_gen: print(f"    Novo melhor (Obj: {self.optimization_goal}): Perf Real={best_actual_performance_overall:.4f}")

            # ... (restante da lógica de logging da geração, elitismo, seleção, crossover, mutação) ...
            gen_actual_acc = [m.get('acc', np.nan) for m in actual_metrics_population]
            gen_actual_f1  = [m.get('f1', np.nan) for m in actual_metrics_population]
            valid_acc = [s for s in gen_actual_acc if pd.notna(s) and np.isfinite(s)]
            valid_f1  = [s for s in gen_actual_f1  if pd.notna(s) and np.isfinite(s)]
            gen_duration = time.time() - gen_start_time

            optimization_history.append({
                'generation': gen + 1,
                'max_acc': np.max(valid_acc) if valid_acc else np.nan, 'avg_acc': np.mean(valid_acc) if valid_acc else np.nan, 'min_acc': np.min(valid_acc) if valid_acc else np.nan,
                'max_f1': np.max(valid_f1) if valid_f1 else np.nan,   'avg_f1': np.mean(valid_f1) if valid_f1 else np.nan,   'min_f1': np.min(valid_f1) if valid_f1 else np.nan,
                'generation_time_sec': gen_duration
            })
            print(f"    Acc Real Geração {gen+1}: Max={optimization_history[-1]['max_acc']:.4f}, Avg={optimization_history[-1]['avg_acc']:.4f}, Min={optimization_history[-1]['min_acc']:.4f}")
            print(f"    F1 Real Geração  {gen+1}: Max={optimization_history[-1]['max_f1']:.4f}, Avg={optimization_history[-1]['avg_f1']:.4f}, Min={optimization_history[-1]['min_f1']:.4f}")
            print(f"    Duração Geração {gen+1}: {gen_duration:.2f}s")

            # Elitismo e Seleção
            # Fitness_values_for_selection já são "maior é melhor"
            # Garantir que não haja NaN ou Inf problemáticos para sorted
            population_with_fitness = []
            for i, ind in enumerate(population):
                fit_val = fitness_values_for_selection[i]
                # Tratar NaN/inf para ordenação, substituindo por "pior" valor possível
                if pd.isna(fit_val) or fit_val == -np.inf: # Pior para maximização
                    processed_fit_val = -np.inf
                elif fit_val == np.inf: # Pior para minimização (fitness seria -inf)
                     # Se fitness_value já é ajustado (1-m), então np.inf aqui significaria um erro
                     # ou um valor extremamente ruim para maximizar.
                     processed_fit_val = -np.inf # Tratar como erro se o fitness deve ser maximizado
                else:
                    processed_fit_val = fit_val
                population_with_fitness.append((ind, processed_fit_val))

            # Ordenar pela fitness processada (maior é melhor)
            sorted_pop_fit = sorted(population_with_fitness, key=lambda x: x[1], reverse=True)
            
            elite = [ind for ind, fv in sorted_pop_fit[:self.n_elite]]
            
            # Seleção de Pais (usar os fitness_values_for_selection originais, pois _selection lida com eles)
            # No entanto, _selection também espera que "maior seja melhor".
            parents = self._selection(population, fitness_values_for_selection)

            next_pop = elite.copy(); offspring_needed = self.population_size - self.n_elite; offspring_count = 0; p_idx_counter = 0
            if not parents and offspring_needed > 0: # Caso _selection retorne vazio por algum motivo
                 print("    AVISO: Nenhum pai selecionado. Preenchendo com novos indivíduos.")
                 for _ in range(offspring_needed): next_pop.append(self._create_individual())
            elif parents:
                while offspring_count < offspring_needed:
                    if not parents : # Se a lista de pais se esgotar (não deveria com o loop de p_idx_counter)
                        p1, p2 = self._create_individual(), self._create_individual()
                    else:
                        idx1 = p_idx_counter % len(parents); p_idx_counter +=1
                        idx2 = p_idx_counter % len(parents); p_idx_counter +=1
                        p1=parents[idx1]; p2=parents[idx2]

                    c1,c2 = (p1.copy(),p2.copy()) if random.random() >= self.crossover_rate else self._crossover(p1,p2)
                    if random.random() < self.mutation_rate: c1=self._mutation(c1)
                    if random.random() < self.mutation_rate: c2=self._mutation(c2)
                    next_pop.append(c1); offspring_count+=1
                    if offspring_count < offspring_needed: next_pop.append(c2); offspring_count+=1
            
            population = next_pop[:self.population_size]
            # Garantir que a população tenha o tamanho correto, preenchendo se necessário (não deveria ser preciso)
            while len(population) < self.population_size:
                print(f"    AVISO: População menor que o esperado ({len(population)}/{self.population_size}). Preenchendo...")
                population.append(self._create_individual())


        opt_duration = time.time() - start_opt_time
        print(f"\n--- Otimização Genética Concluída ({opt_duration:.2f} seg) ---")
        if best_individual_overall is not None and len(best_individual_overall) > 0 :
             print(f"Melhor Performance Real Encontrada ({self.fitness_metric}, Objetivo: {self.optimization_goal}): {best_actual_performance_overall:.4f}")
        else:
             print("Nenhum indivíduo válido encontrado como 'melhor geral'.")
             best_individual_overall = np.array([], dtype=int) # Retornar array vazio
        return best_individual_overall, best_actual_performance_overall, pd.DataFrame(optimization_history)