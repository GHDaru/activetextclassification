# activetextclassification/optimization/genetic_l0_optimizer.py

import numpy as np
import pandas as pd
import time
import random
import os
import hashlib 
from tqdm.auto import tqdm 

from ..models import get_model, BaseFeatureClassifier # Removido BaseTextClassifier se não for usado
from ..embeddings import BaseEmbedder # Mantido caso use embedder
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter


class GeneticL0Optimizer:
    """
    Otimiza a seleção do conjunto inicial L0 de tamanho fixo (semente) usando Algoritmos Genéticos.
    O objetivo é encontrar um L0 (amostrado de df_l0_pool) que maximize ou minimize a performance
    de um classificador treinado apenas nele, quando avaliado em um df_evaluation_set separado.
    Mantém os nomes das métricas como '_on_full' por compatibilidade, mas o cálculo é no df_evaluation_set.
    """
    def __init__(self,
                 df_full, # Este será o df_train_pool do notebook (pool para amostragem de L0)
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
                 fitness_metric='accuracy_on_full', # NOMES MANTIDOS (e.g., 'accuracy_on_full')
                 optimization_goal='maximize', 
                 tournament_size=3, 
                 random_seed=None,
                 embedder=None, 
                 log_detailed_fitness=True, 
                 detailed_log_file="ag_detailed_fitness_log.csv",
                 df_evaluation_set=None # NOVO PARÂMETRO: DataFrame para avaliação do fitness
                 ):
        
        self.df_l0_pool = df_full.reset_index(drop=True) # Pool de onde os L0s são amostrados
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
        self.fitness_metric = fitness_metric # Nomes como 'accuracy_on_full' são mantidos
        self.optimization_goal = optimization_goal
        self.tournament_size = tournament_size
        self.embedder = embedder # Deve ser treinado no df_l0_pool (ou seu superset original)
        self.log_detailed_fitness = log_detailed_fitness
        self.detailed_log_file = detailed_log_file

        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)

        # Define qual DataFrame usar para avaliação do fitness
        if df_evaluation_set is not None and not df_evaluation_set.empty:
            self.df_eval = df_evaluation_set.reset_index(drop=True)
            print(f"INFO (GeneticL0Optimizer): Fitness será avaliado em um conjunto de avaliação separado de {len(self.df_eval)} amostras.")
        else:
            self.df_eval = self.df_l0_pool # Fallback: avalia no mesmo pool do L0
            print("AVISO (GeneticL0Optimizer): Nenhum df_evaluation_set fornecido. Fitness será avaliado no mesmo DataFrame do pool de L0s (df_full).")

        # Validações
        if optimization_goal not in ['maximize', 'minimize']:
            raise ValueError("optimization_goal deve ser 'maximize' ou 'minimize'.")
        # Os nomes das métricas são mantidos, mas seu significado agora é "no conjunto de avaliação"
        if fitness_metric not in ['accuracy_on_full', 'f1_macro_on_full']: 
            raise ValueError("fitness_metric deve ser 'accuracy_on_full' ou 'f1_macro_on_full'.")
        if not (0 <= self.elitism_rate < 1): raise ValueError("Elitism rate deve ser entre 0 e 1 (exclusivo de 1).")
        if self.initial_l0_size > len(self.df_l0_pool): # Validar contra o pool de L0
            raise ValueError("initial_l0_size não pode ser maior que o dataset de pool de L0s (df_full).")
        if self.initial_l0_size <= 0: raise ValueError("initial_l0_size deve ser positivo.")

        self.n_elite = int(self.population_size * self.elitism_rate)
        self.dataset_indices = np.array(self.df_l0_pool.index) # Índices do pool de L0s
        self._fitness_cache = {} 

        if self.log_detailed_fitness:
            self._initialize_detailed_log()

        print(f"GeneticL0Optimizer inicializado.")
        print(f" - Pool para L0s (df_full): {len(self.df_l0_pool)} amostras.")
        print(f" - População: {self.population_size}, Gerações: {self.n_generations}, L0 Size: {self.initial_l0_size}")
        print(f" - Objetivo: {self.optimization_goal}, Métrica: {self.fitness_metric} (avaliada em df_eval)")
        if self.log_detailed_fitness: print(f" - Log Detalhado ATIVADO em: {os.path.abspath(self.detailed_log_file)}")


    def _initialize_detailed_log(self):
        log_dir = os.path.dirname(self.detailed_log_file)
        if log_dir and not os.path.exists(log_dir):
            try: os.makedirs(log_dir, exist_ok=True)
            except Exception as e: print(f"AVISO: Não criar dir log {log_dir}: {e}")

        if not os.path.exists(self.detailed_log_file) or os.path.getsize(self.detailed_log_file) == 0:
            # Nomes das colunas no log são mantidos por compatibilidade,
            # mas 'accuracy_on_full' agora significa 'accuracy_on_df_eval'
            header = ("generation,individual_id,l0_size,accuracy_on_full,f1_macro_on_full,"
                      "num_tokens,num_distinct_tokens,num_classes_in_l0,most_frequent_class_in_l0,"
                      "l0_indices_hash,fitness_calc_time_sec\n") 
            try:
                with open(self.detailed_log_file, 'w', encoding='utf-8') as f: f.write(header)
            except Exception as e: print(f"AVISO: Não escrever header log '{self.detailed_log_file}': {e}")

    def _log_individual_details(self, generation, individual_id, individual_indices, acc_on_eval, f1_on_eval, fitness_calc_time):
        # acc_on_eval e f1_on_eval são as métricas calculadas em self.df_eval
        if not self.log_detailed_fitness: return
        try:
            # Características são do L0 amostrado de self.df_l0_pool
            L0 = self.df_l0_pool.iloc[individual_indices]
            texts = L0[self.text_column].tolist(); labels = L0[self.label_column].tolist()
            tokens = [t for txt in texts for t in str(txt).lower().split() if t]; n_tok = len(tokens); n_dist_tok = len(set(tokens))
            counts = Counter(labels); n_cls = len(counts); mfc = counts.most_common(1)[0][0] if counts else "N/A"
            h = hashlib.sha256(str(sorted(individual_indices)).encode('utf-8')).hexdigest()[:10]
            
            # Formata os valores de acc_on_eval e f1_on_eval para o log
            acc_s = f"{acc_on_eval:.6f}" if pd.notna(acc_on_eval) and np.isfinite(acc_on_eval) else "Error"
            f1_s  = f"{f1_on_eval:.6f}" if pd.notna(f1_on_eval) and np.isfinite(f1_on_eval) else "Error"
            time_s = f"{fitness_calc_time:.4f}" if pd.notna(fitness_calc_time) else "N/A"

            # Os nomes das colunas no log ('accuracy_on_full') são mantidos
            log_line = (f"{generation},{individual_id},{len(individual_indices)},{acc_s},{f1_s},"
                        f"{n_tok},{n_dist_tok},{n_cls},{mfc},{h},{time_s}\n")
            with open(self.detailed_log_file, 'a', encoding='utf-8') as f: f.write(log_line)
        except Exception as e: print(f"AVISO (Log Detalhado) Erro Gen {generation}, ID {individual_id}: {e}")

    def _create_individual(self):
        # Amostra do self.df_l0_pool
        return np.random.choice(self.dataset_indices, size=self.initial_l0_size, replace=False)

    def _initialize_population(self):
        print("Inicializando população...");
        return [self._create_individual() for _ in tqdm(range(self.population_size), desc="Criando População Inicial")]

    def _calculate_fitness_and_metrics(self, individual_indices, generation, individual_id_in_pop):
        cache_key = tuple(sorted(individual_indices))
        if cache_key in self._fitness_cache:
            cached_data = self._fitness_cache[cache_key]
            cached_metrics = cached_data['metrics'] # Contém {'acc': val_no_eval, 'f1': val_no_eval}
            original_calc_time = cached_data.get('calc_time', 0.0)

            if self.log_detailed_fitness:
                 self._log_individual_details(generation, individual_id_in_pop, individual_indices,
                                              cached_metrics.get('acc', -np.inf), # Passa os valores do cache
                                              cached_metrics.get('f1', -np.inf),
                                              original_calc_time)

            # Nomes 'accuracy_on_full'/'f1_macro_on_full' são usados para selecionar a métrica,
            # mas as chaves no cached_metrics são 'acc'/'f1' (referentes à avaliação em df_eval)
            target_metric_key_for_goal = 'acc' if self.fitness_metric == 'accuracy_on_full' else 'f1'
            metric_for_goal = cached_metrics.get(target_metric_key_for_goal, -np.inf)

            fitness_value = (1.0 - metric_for_goal) if self.optimization_goal == 'minimize' and np.isfinite(metric_for_goal) else metric_for_goal
            if not np.isfinite(metric_for_goal): fitness_value = -np.inf
            return fitness_value, cached_metrics # Retorna as métricas do cache (calculadas em df_eval)

        fitness_calc_start_time = time.time()
        # L0 é amostrado de self.df_l0_pool (o antigo df_full)
        L0 = self.df_l0_pool.iloc[individual_indices]
        X_txt_L0 = L0[self.text_column].tolist()
        y_lbl_L0 = L0[self.label_column].tolist()
        
        model = get_model(self.classifier_config)
        acc_on_eval, f1_on_eval = -np.inf, -np.inf # Métricas no self.df_eval

        try:
            X_in_L0 = X_txt_L0 # Default se não for BaseFeatureClassifier
            if self.embedder and isinstance(model, BaseFeatureClassifier):
                 X_in_L0 = self.embedder.transform(X_txt_L0)
            
            if not ((isinstance(X_in_L0, np.ndarray) and X_in_L0.size > 0) or \
                    (isinstance(X_in_L0, list) and X_in_L0) or \
                    (hasattr(X_in_L0, "shape") and X_in_L0.shape[0] > 0) ): # Check para sparse matrices
                raise ValueError("Input de treino (L0) vazio ou inválido.")
            
            model.fit(X_in_L0, y_lbl_L0) # Treina no L0

            # --- AVALIAÇÃO NO self.df_eval ---
            X_eval_txt_effective = self.df_eval[self.text_column].tolist()
            y_true_effective = self.df_eval[self.label_column].tolist()
            
            X_eval_in_effective = X_eval_txt_effective # Default
            if self.embedder and isinstance(model, BaseFeatureClassifier):
                X_eval_in_effective = self.embedder.transform(X_eval_txt_effective)
            
            y_pred_effective = model.predict(X_eval_in_effective)
            
            acc_on_eval = accuracy_score(y_true_effective, y_pred_effective)
            f1_on_eval = f1_score(y_true_effective, y_pred_effective, average='macro', zero_division=0, labels=self.all_possible_labels)
            # ------------------------------------

        except Exception as e: 
            print(f"    ERRO fitness (Gen {generation}, Indiv {individual_id_in_pop}, L0tam {len(individual_indices)}): {type(e).__name__} - {e}")
            # import traceback
            # traceback.print_exc() # Descomente para traceback completo do erro de fitness

        fitness_calc_time = time.time() - fitness_calc_start_time
        # As chaves no dicionário de métricas são mantidas como 'acc' e 'f1' para consistência interna
        current_metrics_dict = {'acc': acc_on_eval, 'f1': f1_on_eval} 
        
        self._fitness_cache[cache_key] = {'metrics': current_metrics_dict, 'calc_time': fitness_calc_time}
        
        if self.log_detailed_fitness:
            self._log_individual_details(generation, individual_id_in_pop, individual_indices, acc_on_eval, f1_on_eval, fitness_calc_time)

        # Nomes de fitness_metric ('accuracy_on_full', 'f1_macro_on_full') são usados para selecionar
        # a chave ('acc' ou 'f1') do current_metrics_dict
        target_metric_key_for_goal = 'acc' if self.fitness_metric == 'accuracy_on_full' else 'f1'
        metric_for_goal = current_metrics_dict.get(target_metric_key_for_goal, -np.inf)
        
        fitness_value = (1.0 - metric_for_goal) if self.optimization_goal == 'minimize' and np.isfinite(metric_for_goal) else metric_for_goal
        if not np.isfinite(metric_for_goal): fitness_value = -np.inf # Tratar NaN/Inf vindo da métrica
            
        return fitness_value, current_metrics_dict # Retorna métricas calculadas em df_eval

    def _selection(self, population, fitness_values):
        # ... (código como no seu original)
        parents = []
        pop_indices = np.arange(len(population))
        for _ in range(len(population)):
            tour_idx_pop = np.random.choice(pop_indices, size=self.tournament_size, replace=True)
            tour_fit_values = [fitness_values[i] for i in tour_idx_pop]
            if all(f == -np.inf for f in tour_fit_values): # Lidar com all -np.inf
                winner_local_idx = np.random.randint(0, self.tournament_size)
            else:
                # np.argmax ignora NaNs implicitamente se não forem todos NaNs/Infs
                # para garantir, podemos converter -np.inf para um valor muito baixo que não seja NaN
                safe_tour_fit_values = [f if np.isfinite(f) else -1e9 for f in tour_fit_values]
                winner_local_idx = np.argmax(safe_tour_fit_values)
            parents.append(population[tour_idx_pop[winner_local_idx]])
        return parents


    def _crossover(self, p1_idx, p2_idx):
        # ... (código como no seu original)
        parent1 = np.array(p1_idx); parent2 = np.array(p2_idx)
        if self.initial_l0_size <= 1 : return parent1.copy(), parent2.copy()
        cp = random.randint(1, self.initial_l0_size - 1) if self.initial_l0_size >=2 else 1 
        c1_p=np.concatenate((parent1[:cp],parent2[cp:])); c2_p=np.concatenate((parent2[:cp],parent1[cp:]))
        def repair(c_p,p1,p2):
            unique_genes = list(set(c_p)) 
            needed = self.initial_l0_size - len(unique_genes)
            if needed > 0:
                parent_pool = set(p1) | set(p2)
                candidates_p = list(parent_pool - set(unique_genes)) # Corrigido para usar set(unique_genes)
                random.shuffle(candidates_p)
                can_add_from_parents = min(needed, len(candidates_p))
                unique_genes.extend(candidates_p[:can_add_from_parents])
                needed -= can_add_from_parents
            if needed > 0:
                dataset_set = set(self.dataset_indices)
                pool_ds = list(dataset_set - set(unique_genes)) 
                random.shuffle(pool_ds)
                if len(pool_ds) < needed: return self._create_individual() # Recriar se não houver genes suficientes
                unique_genes.extend(pool_ds[:needed])
            
            final_genes = np.array(unique_genes)
            if len(final_genes) > self.initial_l0_size:
                 return np.random.choice(final_genes, size=self.initial_l0_size, replace=False)
            elif len(final_genes) < self.initial_l0_size: # Não deveria acontecer se a lógica estiver correta
                 return self._create_individual() 
            return final_genes
        return repair(c1_p,parent1,parent2), repair(c2_p,parent2,parent1)

    def _mutation(self, ind_idx):
        # ... (código como no seu original)
        mut=np.array(ind_idx)
        if len(mut)==0 or self.mutation_strength==0: return mut
        pool=np.setdiff1d(self.dataset_indices,mut, assume_unique=True) # assume_unique pode otimizar
        if len(pool)==0: return mut
        act_mut_cnt=min(self.mutation_strength, len(mut), len(pool)) 
        if act_mut_cnt<=0: return mut
        pos_mut=np.random.choice(len(mut),size=act_mut_cnt,replace=False)
        repl_genes=np.random.choice(pool,size=act_mut_cnt,replace=False)
        mut[pos_mut]=repl_genes; return mut

    def run_optimization(self):
        print(f"\n--- Iniciando Otimização Genética ---")
        print(f"  L0 Size: {self.initial_l0_size}, População: {self.population_size}, Gerações: {self.n_generations}")
        print(f"  Objetivo: {self.optimization_goal}, Métrica: {self.fitness_metric} (avaliada em df_eval de {len(self.df_eval)} amostras)")

        start_opt_time = time.time()
        population = self._initialize_population()
        optimization_history = []

        best_actual_performance_overall = -np.inf if self.optimization_goal == 'maximize' else np.inf
        best_individual_overall = None
        
        if not population or len(population) == 0 : # Checagem mais robusta
            print("ERRO: População não inicializada ou vazia."); 
            return np.array([],dtype=int), best_actual_performance_overall, pd.DataFrame(optimization_history)

        for gen in tqdm(range(self.n_generations), desc="Gerações AG"):
            gen_start_time = time.time()
            # print(f"\n  Geração {gen + 1}/{self.n_generations}") # tqdm já mostra isso
            fitness_values_for_selection = []
            actual_metrics_population = [] # Lista de dicts {'acc': val, 'f1': val}

            # Loop de avaliação do fitness para a população atual
            for i, ind in enumerate(tqdm(population, desc=f"Fitness Gen {gen+1}", leave=False, position=1)):
                 fitness_val, metrics_dict = self._calculate_fitness_and_metrics(ind, gen + 1, i)
                 fitness_values_for_selection.append(fitness_val)
                 actual_metrics_population.append(metrics_dict)

            # Determinar qual chave de métrica usar (com base em self.fitness_metric que usa '_on_full')
            # Mas as chaves em actual_metrics_population são 'acc' e 'f1' (referentes a df_eval)
            target_metric_key_in_dict = 'acc' if 'accuracy' in self.fitness_metric else 'f1'
            default_val_target = -np.inf if self.optimization_goal == 'maximize' else np.inf
            
            current_gen_performances_real = [m.get(target_metric_key_in_dict, default_val_target) for m in actual_metrics_population]

            # Lidar com NaN/Inf antes de argmax/argmin
            valid_perf_mask = pd.Series(current_gen_performances_real).apply(lambda x: pd.notna(x) and np.isfinite(x))
            
            current_gen_best_actual_perf = default_val_target
            best_idx_this_gen = -1 # Default se não houver válidos

            if valid_perf_mask.any():
                valid_subset_performances = np.array(current_gen_performances_real)[valid_perf_mask]
                original_indices_of_valid = np.where(valid_perf_mask)[0]

                if self.optimization_goal == 'maximize':
                    best_local_idx_in_valid = np.argmax(valid_subset_performances)
                else: # Minimize
                    best_local_idx_in_valid = np.argmin(valid_subset_performances)
                
                if original_indices_of_valid.size > 0 : # Garantir que há índices válidos
                    best_idx_this_gen = original_indices_of_valid[best_local_idx_in_valid]
                    current_gen_best_actual_perf = current_gen_performances_real[best_idx_this_gen]
                else: # Caso raro onde valid_perf_mask.any() é True mas original_indices_of_valid é vazio (não deveria acontecer)
                    print("    AVISO: Inconsistência na máscara de performance válida.")
            else:
                print(f"    AVISO: Nenhuma performance válida na geração {gen+1}.")


            current_gen_best_individual = population[best_idx_this_gen] if best_idx_this_gen != -1 and len(population) > best_idx_this_gen else None

            new_best_found_this_gen = False
            if self.optimization_goal == 'maximize':
                if np.isfinite(current_gen_best_actual_perf) and current_gen_best_actual_perf > best_actual_performance_overall:
                    best_actual_performance_overall=current_gen_best_actual_perf
                    best_individual_overall=current_gen_best_individual
                    new_best_found_this_gen=True
            else: # Minimize
                if np.isfinite(current_gen_best_actual_perf) and current_gen_best_actual_perf < best_actual_performance_overall:
                    # Adicionalmente, para minimização, não queremos -np.inf como o "melhor"
                    if current_gen_best_actual_perf > -np.inf: 
                        best_actual_performance_overall=current_gen_best_actual_perf
                        best_individual_overall=current_gen_best_individual
                        new_best_found_this_gen=True
                elif best_actual_performance_overall == np.inf and np.isfinite(current_gen_best_actual_perf) and current_gen_best_actual_perf > -np.inf: 
                    best_actual_performance_overall=current_gen_best_actual_perf
                    best_individual_overall=current_gen_best_individual
                    new_best_found_this_gen=True
            
            if new_best_found_this_gen: 
                tqdm.write(f"    Gen {gen+1}: Novo melhor ({self.optimization_goal} {self.fitness_metric}): {best_actual_performance_overall:.4f}")


            # Para o histórico, os nomes das colunas são mantidos (max_acc, avg_acc, etc.)
            # mas eles agora se referem à performance no self.df_eval
            gen_actual_acc_eval = [m.get('acc', np.nan) for m in actual_metrics_population] # Chave 'acc' do dict
            gen_actual_f1_eval  = [m.get('f1', np.nan) for m in actual_metrics_population]  # Chave 'f1' do dict
            
            valid_acc_eval = [s for s in gen_actual_acc_eval if pd.notna(s) and np.isfinite(s)]
            valid_f1_eval  = [s for s in gen_actual_f1_eval  if pd.notna(s) and np.isfinite(s)]
            gen_duration = time.time() - gen_start_time

            history_entry = {
                'generation': gen + 1,
                'max_acc': np.max(valid_acc_eval) if valid_acc_eval else np.nan, 
                'avg_acc': np.mean(valid_acc_eval) if valid_acc_eval else np.nan, 
                'min_acc': np.min(valid_acc_eval) if valid_acc_eval else np.nan,
                'max_f1': np.max(valid_f1_eval) if valid_f1_eval else np.nan,   
                'avg_f1': np.mean(valid_f1_eval) if valid_f1_eval else np.nan,   
                'min_f1': np.min(valid_f1_eval) if valid_f1_eval else np.nan,
                'generation_time_sec': gen_duration
            }
            optimization_history.append(history_entry)
            
            # Atualizar descrição da barra de progresso principal com o melhor atual
            desc_tqdm = f"AG Gen (Melhor: {best_actual_performance_overall:.4f})"
            # A barra de progresso externa (para gerações) pode ser atualizada assim:
            if gen == 0: # Para a primeira geração, a barra externa é criada
                pbar_generations = tqdm(range(self.n_generations), desc=desc_tqdm, position=0, initial=gen)
            if pbar_generations.n < gen : # Atualiza se a barra interna terminou
                 pbar_generations.update(gen - pbar_generations.n) # Avança para a geração atual
            pbar_generations.set_description(desc_tqdm) # Atualiza descrição com o melhor fitness
            if gen == self.n_generations - 1 and pbar_generations.n < self.n_generations:
                pbar_generations.update(self.n_generations - pbar_generations.n) # Garante que chegue ao final
                pbar_generations.close()


            # Seleção, Crossover, Mutação (como antes)
            # ... (lógica de seleção, elitismo, crossover, mutação como no seu original) ...
            valid_fitness_mask = pd.Series(fitness_values_for_selection).apply(lambda x: pd.notna(x) and np.isfinite(x))
            if not valid_fitness_mask.any():
                tqdm.write(f"    AVISO Gen {gen+1}: Todos os fitness são inválidos. Reinicializando população."); population = self._initialize_population(); continue

            population_vf = [population[i] for i, valid in enumerate(valid_fitness_mask) if valid]
            fitness_vf = [fitness_values_for_selection[i] for i, valid in enumerate(valid_fitness_mask) if valid]
            
            if not population_vf : 
                 tqdm.write(f"    AVISO Gen {gen+1}: População válida para fitness vazia. Reinicializando."); population = self._initialize_population(); continue

            # Ordenar pela fitness_value (que já considera o objetivo de minimização/maximização)
            sorted_pop_fit_valid = sorted(zip(population_vf, fitness_vf), key=lambda x: x[1], reverse=True) # True porque fitness_value é sempre para maximizar
            elite = [ind for ind, fv in sorted_pop_fit_valid[:self.n_elite]]
            
            parents = self._selection(population, fitness_values_for_selection) # Passa a população original e os fitness calculados
            next_pop = elite.copy(); offspring_needed = self.population_size - self.n_elite; offspring_count = 0; p_idx_counter = 0
            if not parents and offspring_needed > 0:
                 for _ in range(offspring_needed): next_pop.append(self._create_individual())
            elif parents: # Garantir que parents não seja None ou vazio
                parents_list = list(parents) # Converter para lista para evitar problemas com iteradores
                if not parents_list: # Se a seleção não retornou pais válidos (improvável com o fallback na seleção)
                    for _ in range(offspring_needed): next_pop.append(self._create_individual())
                else:
                    while offspring_count < offspring_needed:
                        idx1 = p_idx_counter % len(parents_list); p_idx_counter +=1
                        idx2 = p_idx_counter % len(parents_list); p_idx_counter +=1
                        p1=parents_list[idx1]; p2=parents_list[idx2]
                        
                        c1,c2 = (p1.copy(),p2.copy()) # Default: cópia dos pais
                        if random.random() < self.crossover_rate: # Probabilidade de crossover
                            c1,c2 = self._crossover(p1,p2)
                        
                        if random.random() < self.mutation_rate: c1=self._mutation(c1)
                        if random.random() < self.mutation_rate: c2=self._mutation(c2)
                        
                        next_pop.append(c1); offspring_count+=1
                        if offspring_count < offspring_needed: 
                            next_pop.append(c2); offspring_count+=1
            
            if len(next_pop) < self.population_size: # Se por algum motivo a população não foi preenchida
                needed_fill = self.population_size - len(next_pop)
                for _ in range(needed_fill): next_pop.append(self._create_individual())

            population = next_pop[:self.population_size]
        # Fechar a barra de progresso principal das gerações se não foi fechada
        if 'pbar_generations' in locals() and pbar_generations.n < self.n_generations:
             pbar_generations.update(self.n_generations - pbar_generations.n)
             pbar_generations.close()


        opt_duration = time.time() - start_opt_time
        print(f"\n--- Otimização Genética Concluída ({opt_duration:.2f} seg) ---")
        if best_individual_overall is not None:
             print(f"Melhor Performance ({self.fitness_metric} em df_eval) Encontrada: {best_actual_performance_overall:.4f}")
        else:
             print("Nenhum indivíduo válido encontrado como 'melhor geral'.")
             best_individual_overall = np.array([], dtype=int) # Retornar array vazio
        
        return best_individual_overall, best_actual_performance_overall, pd.DataFrame(optimization_history)