# activetextclassification/optimization/genetic_l0_optimizer.py

import numpy as np
import pandas as pd
import time
import random
import os
import hashlib # Para hash de índices no log detalhado
from tqdm.notebook import tqdm # Para barras de progresso

# Imports da própria biblioteca
from ..models import get_model, BaseTextClassifier, BaseFeatureClassifier
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
                 log_detailed_fitness=True, # Default para True para capturar tempos
                 detailed_log_file="ag_detailed_fitness_log.csv"
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
        self.embedder = embedder
        self.log_detailed_fitness = log_detailed_fitness
        self.detailed_log_file = detailed_log_file

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
        self.dataset_indices = np.array(self.df_full.index)
        self._fitness_cache = {} # Cache: {tuple(indices): {'metrics': {'acc':val, 'f1':val}, 'calc_time': time}}

        if self.log_detailed_fitness:
            self._initialize_detailed_log()

        print(f"GeneticL0Optimizer inicializado.")
        print(f" - População: {self.population_size}, Gerações: {self.n_generations}, L0 Size: {self.initial_l0_size}")
        print(f" - Objetivo: {self.optimization_goal}, Métrica: {self.fitness_metric}")
        if self.log_detailed_fitness: print(f" - Log Detalhado ATIVADO em: {os.path.abspath(self.detailed_log_file)}")


    def _initialize_detailed_log(self):
        log_dir = os.path.dirname(self.detailed_log_file)
        if log_dir and not os.path.exists(log_dir):
            try: os.makedirs(log_dir, exist_ok=True)
            except Exception as e: print(f"AVISO: Não criar dir log {log_dir}: {e}")

        if not os.path.exists(self.detailed_log_file) or os.path.getsize(self.detailed_log_file) == 0:
            header = ("generation,individual_id,l0_size,accuracy_on_full,f1_macro_on_full,"
                      "num_tokens,num_distinct_tokens,num_classes_in_l0,most_frequent_class_in_l0,"
                      "l0_indices_hash,fitness_calc_time_sec\n") # Cabeçalho atualizado
            try:
                with open(self.detailed_log_file, 'w', encoding='utf-8') as f: f.write(header)
            except Exception as e: print(f"AVISO: Não escrever header log '{self.detailed_log_file}': {e}")

    def _log_individual_details(self, generation, individual_id, individual_indices, acc, f1, fitness_calc_time):
        if not self.log_detailed_fitness: return
        try:
            L0 = self.df_full.iloc[individual_indices]; texts = L0[self.text_column].tolist(); labels = L0[self.label_column].tolist()
            tokens = [t for txt in texts for t in str(txt).lower().split() if t]; n_tok = len(tokens); n_dist_tok = len(set(tokens))
            counts = Counter(labels); n_cls = len(counts); mfc = counts.most_common(1)[0][0] if counts else "N/A"
            h = hashlib.sha256(str(sorted(individual_indices)).encode('utf-8')).hexdigest()[:10]
            acc_s = f"{acc:.6f}" if pd.notna(acc) and acc > -np.inf and acc < np.inf else "Error" # Lida com NaN/inf
            f1_s  = f"{f1:.6f}" if pd.notna(f1) and f1 > -np.inf and f1 < np.inf else "Error"  # Lida com NaN/inf
            time_s = f"{fitness_calc_time:.4f}" if pd.notna(fitness_calc_time) else "N/A"

            log_line = (f"{generation},{individual_id},{len(individual_indices)},{acc_s},{f1_s},"
                        f"{n_tok},{n_dist_tok},{n_cls},{mfc},{h},{time_s}\n")
            with open(self.detailed_log_file, 'a', encoding='utf-8') as f: f.write(log_line)
        except Exception as e: print(f"AVISO (Log Detalhado) Erro Gen {generation}, ID {individual_id}: {e}")

    def _create_individual(self):
        return np.random.choice(self.dataset_indices, size=self.initial_l0_size, replace=False)

    def _initialize_population(self):
        print("Inicializando população...");
        return [self._create_individual() for _ in tqdm(range(self.population_size), desc="Criando População Inicial")]

    def _calculate_fitness_and_metrics(self, individual_indices, generation, individual_id_in_pop):
        cache_key = tuple(sorted(individual_indices))
        if cache_key in self._fitness_cache:
            cached_data = self._fitness_cache[cache_key]
            cached_metrics = cached_data['metrics']
            original_calc_time = cached_data.get('calc_time', 0.0) # Pega tempo original

            if self.log_detailed_fitness:
                 self._log_individual_details(generation, individual_id_in_pop, individual_indices,
                                              cached_metrics.get('acc', -np.inf),
                                              cached_metrics.get('f1', -np.inf),
                                              original_calc_time) # Logar com o tempo do cálculo original

            target_metric_key = 'acc' if self.fitness_metric == 'accuracy_on_full' else 'f1'
            metric_for_goal = cached_metrics.get(target_metric_key, -np.inf)

            fitness_value = (1.0 - metric_for_goal) if self.optimization_goal == 'minimize' and metric_for_goal > -np.inf else metric_for_goal
            if self.optimization_goal == 'minimize' and metric_for_goal <= -np.inf : fitness_value = -np.inf # Erro é sempre ruim
            if self.optimization_goal == 'maximize' and metric_for_goal <= -np.inf : fitness_value = -np.inf

            return fitness_value, cached_metrics

        fitness_calc_start_time = time.time()
        L0=self.df_full.iloc[individual_indices]; X_txt=L0[self.text_column].tolist(); y_lbl=L0[self.label_column].tolist()
        model = get_model(self.classifier_config); acc, f1 = -np.inf, -np.inf
        try:
            X_in = self.embedder.transform(X_txt) if isinstance(model,BaseFeatureClassifier) and self.embedder else X_txt
            if not ((isinstance(X_in, np.ndarray) and X_in.size > 0) or (isinstance(X_in, list) and X_in)): raise ValueError("Input treino vazio.")
            model.fit(X_in, y_lbl)
            X_eval_txt = self.df_full[self.text_column].tolist(); y_true = self.df_full[self.label_column].tolist()
            X_eval_in = self.embedder.transform(X_eval_txt) if isinstance(model,BaseFeatureClassifier) and self.embedder else X_eval_txt
            y_pred = model.predict(X_eval_in)
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='macro', zero_division=0, labels=self.all_possible_labels)
        except Exception as e: print(f"    ERRO fitness (Gen {generation}, ID {individual_id_in_pop}): {type(e).__name__} - {e}")

        fitness_calc_time = time.time() - fitness_calc_start_time
        current_metrics = {'acc': acc, 'f1': f1}
        self._fitness_cache[cache_key] = {'metrics': current_metrics, 'calc_time': fitness_calc_time} # Salvar tempo
        if self.log_detailed_fitness:
            self._log_individual_details(generation, individual_id_in_pop, individual_indices, acc, f1, fitness_calc_time)

        target_metric_key = 'acc' if self.fitness_metric == 'accuracy_on_full' else 'f1'
        metric_for_goal = current_metrics.get(target_metric_key, -np.inf)
        fitness_value = (1.0 - metric_for_goal) if self.optimization_goal == 'minimize' and metric_for_goal > -np.inf else metric_for_goal
        if self.optimization_goal == 'minimize' and metric_for_goal <= -np.inf : fitness_value = -np.inf
        if self.optimization_goal == 'maximize' and metric_for_goal <= -np.inf : fitness_value = -np.inf
        return fitness_value, current_metrics

    def _selection(self, population, fitness_values):
        parents = []
        pop_indices = np.arange(len(population))
        for _ in range(len(population)):
            tour_idx_pop = np.random.choice(pop_indices, size=self.tournament_size, replace=True)
            tour_fit_values = [fitness_values[i] for i in tour_idx_pop]
            # Lidar com possíveis -np.inf para que argmax funcione corretamente
            # Se todos no torneio forem -np.inf, escolhe um aleatoriamente
            if all(f == -np.inf for f in tour_fit_values):
                winner_local_idx = np.random.randint(0, self.tournament_size)
            else:
                # Substituir -np.inf por um valor muito pequeno para argmax
                # (mas não tão pequeno que cause problemas de float se outros forem pequenos)
                # A melhor abordagem é filtrar ou usar nanargmax se houver NaNs
                temp_tour_fit = np.array(tour_fit_values)
                # Se houver valores não -inf, argmax funciona. Se todos -inf, o if anterior trata.
                winner_local_idx = np.argmax(temp_tour_fit)
            parents.append(population[tour_idx_pop[winner_local_idx]])
        return parents

    def _crossover(self, p1_idx, p2_idx):
        parent1 = np.array(p1_idx); parent2 = np.array(p2_idx)
        if self.initial_l0_size <= 1 : return parent1.copy(), parent2.copy()
        cp = random.randint(1, self.initial_l0_size - 1) if self.initial_l0_size >=2 else 1 # Correção para size 2
        c1_p=np.concatenate((parent1[:cp],parent2[cp:])); c2_p=np.concatenate((parent2[:cp],parent1[cp:]))
        def repair(c_p,p1,p2):
            u_set = set(c_p) # Usar set para unicidade rápida
            unique_genes = list(u_set)
            needed = self.initial_l0_size - len(unique_genes)
            if needed > 0:
                # Usar sets para diferença é mais eficiente
                parent_pool = set(p1) | set(p2)
                candidates_p = list(parent_pool - u_set)
                random.shuffle(candidates_p)
                can_add_from_parents = min(needed, len(candidates_p))
                unique_genes.extend(candidates_p[:can_add_from_parents])
                needed -= can_add_from_parents
            if needed > 0:
                dataset_set = set(self.dataset_indices)
                pool_ds = list(dataset_set - set(unique_genes)) # Agora unique_genes é set
                random.shuffle(pool_ds)
                if len(pool_ds) < needed: return self._create_individual()
                unique_genes.extend(pool_ds[:needed])
            # Garantir tamanho final
            if len(unique_genes) > self.initial_l0_size:
                 return np.random.choice(unique_genes, size=self.initial_l0_size, replace=False)
            elif len(unique_genes) < self.initial_l0_size:
                 return self._create_individual()
            return np.array(unique_genes)
        return repair(c1_p,parent1,parent2), repair(c2_p,parent2,parent1)

    def _mutation(self, ind_idx):
        mut=np.array(ind_idx)
        if len(mut)==0 or self.mutation_strength==0: return mut
        pool=np.setdiff1d(self.dataset_indices,mut)
        if len(pool)==0: return mut
        act_mut_cnt=min(self.mutation_strength, len(mut), len(pool)) # Não mutar mais do que o tamanho do L0
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
        if not population:
            print("ERRO: População não inicializada."); return np.array([],dtype=int),best_actual_performance_overall,pd.DataFrame(optimization_history)

        for gen in tqdm(range(self.n_generations), desc="Gerações AG"):
            gen_start_time = time.time()
            print(f"\n  Geração {gen + 1}/{self.n_generations}")
            fitness_values_for_selection = []
            actual_metrics_population = []

            for i, ind in enumerate(tqdm(population, desc=f"Fitness Calc Gen {gen+1}", leave=False)):
                 fitness_val, metrics_dict = self._calculate_fitness_and_metrics(ind, gen + 1, i)
                 fitness_values_for_selection.append(fitness_val)
                 actual_metrics_population.append(metrics_dict)

            target_metric_key = 'acc' if self.fitness_metric == 'accuracy_on_full' else 'f1'
            default_val_target = -np.inf if self.optimization_goal == 'maximize' else np.inf
            current_gen_performances_real = [m.get(target_metric_key, default_val_target) for m in actual_metrics_population]

            # Lidar com NaN/Inf antes de argmax/argmin
            valid_perf_mask = pd.Series(current_gen_performances_real).notna() & \
                              (pd.Series(current_gen_performances_real) != -np.inf) & \
                              (pd.Series(current_gen_performances_real) != np.inf)
            
            if not valid_perf_mask.any(): # Nenhuma performance válida
                print("    AVISO: Nenhuma performance válida na geração.")
                current_gen_best_actual_perf = default_val_target
                best_idx_this_gen = 0 if population else -1 # Evitar erro
            else:
                valid_subset_performances = np.array(current_gen_performances_real)[valid_perf_mask]
                original_indices_of_valid = np.where(valid_perf_mask)[0]

                if self.optimization_goal == 'maximize':
                    best_local_idx_in_valid = np.argmax(valid_subset_performances)
                else: # Minimize
                    best_local_idx_in_valid = np.argmin(valid_subset_performances)
                
                best_idx_this_gen = original_indices_of_valid[best_local_idx_in_valid]
                current_gen_best_actual_perf = current_gen_performances_real[best_idx_this_gen]

            if best_idx_this_gen != -1 : current_gen_best_individual = population[best_idx_this_gen]
            else: current_gen_best_individual = None


            new_best_found_this_gen = False
            if self.optimization_goal == 'maximize':
                if pd.notna(current_gen_best_actual_perf) and current_gen_best_actual_perf > -np.inf:
                    if current_gen_best_actual_perf > best_actual_performance_overall:
                        best_actual_performance_overall=current_gen_best_actual_perf; best_individual_overall=current_gen_best_individual; new_best_found_this_gen=True
            else: # Minimize
                if pd.notna(current_gen_best_actual_perf) and current_gen_best_actual_perf < np.inf:
                    if current_gen_best_actual_perf < best_actual_performance_overall and current_gen_best_actual_perf > -np.inf:
                        best_actual_performance_overall=current_gen_best_actual_perf; best_individual_overall=current_gen_best_individual; new_best_found_this_gen=True
                    elif best_actual_performance_overall == np.inf and current_gen_best_actual_perf > -np.inf: # Primeiro valor real para min
                        best_actual_performance_overall=current_gen_best_actual_perf; best_individual_overall=current_gen_best_individual; new_best_found_this_gen=True
            if new_best_found_this_gen: print(f"    Novo melhor (Obj: {self.optimization_goal}): Perf Real={best_actual_performance_overall:.4f}")

            gen_actual_acc = [m.get('acc', np.nan) for m in actual_metrics_population]
            gen_actual_f1  = [m.get('f1', np.nan) for m in actual_metrics_population]
            valid_acc = [s for s in gen_actual_acc if pd.notna(s) and np.isfinite(s)] # Filtra NaN e Inf
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

            # Seleção e Elitismo
            valid_fitness_mask = pd.Series(fitness_values_for_selection).notna() & \
                                 (pd.Series(fitness_values_for_selection) != -np.inf) & \
                                 (pd.Series(fitness_values_for_selection) != np.inf)
            if not valid_fitness_mask.any():
                print("    AVISO: Fitness inválido. Reinicializando população."); population = self._initialize_population(); continue

            population_vf = [population[i] for i, valid in enumerate(valid_fitness_mask) if valid]
            fitness_vf = [fitness_values_for_selection[i] for i, valid in enumerate(valid_fitness_mask) if valid]
            
            if not population_vf : # Se depois de filtrar, ficou vazio
                 print("    AVISO: População válida para fitness vazia. Reinicializando.")
                 population = self._initialize_population(); continue

            sorted_pop_fit_valid = sorted(zip(population_vf, fitness_vf), key=lambda x: x[1], reverse=True)
            elite = [ind for ind, fv in sorted_pop_fit_valid[:self.n_elite]]
            
            # Para seleção de pais, usar a população original e os fitness_values_for_selection originais
            # pois _selection lida com -np.inf
            parents = self._selection(population, fitness_values_for_selection)

            next_pop = elite.copy(); offspring_needed = self.population_size - self.n_elite; offspring_count = 0; p_idx_counter = 0
            if not parents and offspring_needed > 0:
                 for _ in range(offspring_needed): next_pop.append(self._create_individual())
            elif parents:
                while offspring_count < offspring_needed:
                    idx1 = p_idx_counter % len(parents); p_idx_counter +=1
                    idx2 = p_idx_counter % len(parents); p_idx_counter +=1
                    p1=parents[idx1]; p2=parents[idx2]
                    c1,c2 = (p1.copy(),p2.copy()) if random.random() >= self.crossover_rate else self._crossover(p1,p2)
                    if random.random() < self.mutation_rate: c1=self._mutation(c1)
                    if random.random() < self.mutation_rate: c2=self._mutation(c2)
                    next_pop.append(c1); offspring_count+=1
                    if offspring_count < offspring_needed: next_pop.append(c2); offspring_count+=1
            population = next_pop[:self.population_size]

        opt_duration = time.time() - start_opt_time
        print(f"\n--- Otimização Genética Concluída ({opt_duration:.2f} seg) ---")
        if best_individual_overall is not None:
             print(f"Melhor Performance Real Encontrada ({self.fitness_metric}, Objetivo: {self.optimization_goal}): {best_actual_performance_overall:.4f}")
        else:
             print("Nenhum indivíduo válido encontrado como 'melhor geral'.")
             best_individual_overall = np.array([], dtype=int)
        return best_individual_overall, best_actual_performance_overall, pd.DataFrame(optimization_history)