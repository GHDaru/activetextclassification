# activetextclassification/optimization/genetic_l0_optimizer.py

import numpy as np
import pandas as pd
import time
import random
import os
import hashlib 
import pickle # Para salvar/carregar estado do AG
from tqdm.notebook import tqdm # Para barras de progresso

# Imports da própria biblioteca
from ..models import get_model, BaseFeatureClassifier # Assumindo que BaseTextClassifier não é diretamente necessário aqui
from ..embeddings import BaseEmbedder # Se for usar classificadores que precisam de embeddings pré-processados

# Imports de ML
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter # Para métricas auxiliares de L0


class GeneticL0Optimizer:
    """
    Otimiza a seleção do conjunto inicial L0 de tamanho fixo (semente) usando Algoritmos Genéticos.
    O L0 é amostrado de df_full (que é o pool de treino/otimização).
    A performance (fitness) de um classificador treinado apenas no L0 é avaliada em df_evaluation_set.
    Mantém os nomes das métricas como '_on_full' por compatibilidade, mas o cálculo é no df_evaluation_set.
    Inclui funcionalidade de checkpoint para resumir otimizações.
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
                 embedder=None, # Embedder pré-ajustado no df_l0_pool
                 log_detailed_fitness=True, 
                 # detailed_log_file é passado para run_optimization
                 df_evaluation_set=None, # NOVO PARÂMETRO: DataFrame para avaliação do fitness
                 checkpoint_dir="ag_checkpoints", 
                 checkpoint_prefix="ag_run"      
                 ):
        
        self.df_l0_pool = df_full.reset_index(drop=True) 
        self.text_column = text_column
        self.label_column = label_column
        self.classifier_config = classifier_config
        self.initial_l0_size = initial_l0_size
        # Garantir que all_possible_labels seja uma lista de strings únicas e ordenadas
        self.all_possible_labels = sorted(list(set(map(str, all_possible_labels))))
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
        
        if df_evaluation_set is not None and not df_evaluation_set.empty:
            self.df_eval = df_evaluation_set.reset_index(drop=True)
        else:
            self.df_eval = self.df_l0_pool 
            print("AVISO (GeneticL0Optimizer): Nenhum df_evaluation_set fornecido ou está vazio. Fitness será avaliado no mesmo DataFrame do pool de L0s (df_full).")

        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
        
        # Validações
        if optimization_goal not in ['maximize', 'minimize']:
            raise ValueError("optimization_goal deve ser 'maximize' ou 'minimize'.")
        if fitness_metric not in ['accuracy_on_full', 'f1_macro_on_full']: 
            raise ValueError("fitness_metric deve ser 'accuracy_on_full' ou 'f1_macro_on_full'.")
        if not (0 <= self.elitism_rate < 1): raise ValueError("Elitism rate deve ser entre 0 e 1 (exclusivo de 1).")
        if self.initial_l0_size > len(self.df_l0_pool): 
            raise ValueError(f"initial_l0_size ({self.initial_l0_size}) não pode ser maior que o dataset de pool de L0s ({len(self.df_l0_pool)}).")
        if self.initial_l0_size <= 0: raise ValueError("initial_l0_size deve ser positivo.")

        self.n_elite = int(self.population_size * self.elitism_rate)
        self.dataset_indices = np.array(self.df_l0_pool.index) 
        
        # Atributos para o estado do AG
        self._reset_state_to_initial() # Inicializa atributos de estado

        self.checkpoint_dir = checkpoint_dir
        # O nome do arquivo de checkpoint agora inclui mais detalhes para unicidade
        self.checkpoint_file = os.path.join(
            checkpoint_dir, 
            f"{checkpoint_prefix}_l0_{initial_l0_size}_pop_{population_size}_gen_{n_generations}_{fitness_metric.replace('_on_full','')}_{optimization_goal}_ckpt.pkl"
        )
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        print(f"GeneticL0Optimizer inicializado.")
        print(f" - Pool para L0s (df_full): {len(self.df_l0_pool)} amostras.")
        print(f" - Avaliação Fitness em (df_eval): {len(self.df_eval)} amostras.")
        print(f" - População: {self.population_size}, Gerações: {self.n_generations}, L0 Size: {self.initial_l0_size}")
        print(f" - Objetivo: {self.optimization_goal}, Métrica: {self.fitness_metric} (avaliada em df_eval)")
        if self.log_detailed_fitness: print(f" - Log Detalhado será ativado.")
        print(f" - Checkpoint será salvo em/carregado de: {self.checkpoint_file}")

    def _reset_state_to_initial(self):
        """Reseta os atributos de estado do AG para um início limpo."""
        self.current_generation = 0
        self.population = []
        self.optimization_history = []
        self.best_actual_performance_overall = -np.inf if self.optimization_goal == 'maximize' else np.inf
        self.best_individual_overall = None
        self._fitness_cache = {}
        # print("   Estado do AG resetado para inicial.")


    def _save_checkpoint(self, detailed_log_file_path_used):
        state = {
            'current_generation': self.current_generation,
            'population': [ind.tolist() for ind in self.population], # Salvar arrays como listas
            'optimization_history': self.optimization_history,
            'best_actual_performance_overall': self.best_actual_performance_overall,
            'best_individual_overall': self.best_individual_overall.tolist() if self.best_individual_overall is not None else None,
            '_fitness_cache': self._fitness_cache,
            'random_state_np': np.random.get_state(),
            'random_state_py': random.getstate(),
            'detailed_log_file_path_at_checkpoint': detailed_log_file_path_used
        }
        try:
            with open(self.checkpoint_file, 'wb') as f:
                pickle.dump(state, f)
            # tqdm.write(f"   Checkpoint salvo na geração {self.current_generation + 1}") # +1 porque é após a gen completar
        except Exception as e:
            tqdm.write(f"   ERRO ao salvar checkpoint: {e}")

    def _load_checkpoint(self):
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'rb') as f:
                    state = pickle.load(f)
                self.current_generation = state.get('current_generation', 0) 
                # No checkpoint, current_generation é a última completada, então a próxima é current_generation + 1
                # O loop for gen in range(self.current_generation, self.n_generations) fará com que comece da correta.
                # Mas se a última salva foi a gen 0 (após a primeira iteração), current_generation será 0.
                # O importante é que o loop de geração comece de self.current_generation (se > 0) ou 0.
                
                self.population = [np.array(ind_list) for ind_list in state.get('population', [])]
                self.optimization_history = state.get('optimization_history', [])
                self.best_actual_performance_overall = state.get('best_actual_performance_overall', -np.inf if self.optimization_goal == 'maximize' else np.inf)
                best_ind_list = state.get('best_individual_overall', None)
                self.best_individual_overall = np.array(best_ind_list) if best_ind_list is not None else None
                self._fitness_cache = state.get('_fitness_cache', {})
                
                if 'random_state_np' in state: np.random.set_state(state['random_state_np'])
                if 'random_state_py' in state: random.setstate(state['random_state_py'])
                
                detailed_log_path_from_ckpt = state.get('detailed_log_file_path_at_checkpoint')

                print(f"   Checkpoint carregado. Resumindo da geração {self.current_generation + 1} (próxima a ser executada).") # +1 para display
                print(f"   Melhor performance até agora: {self.best_actual_performance_overall:.4f}")
                return True, detailed_log_path_from_ckpt
            except Exception as e:
                print(f"   ERRO ao carregar checkpoint ({self.checkpoint_file}): {e}. Iniciando do zero.")
                self._reset_state_to_initial()
                return False, None
        return False, None

    def _initialize_detailed_log(self, detailed_log_file_path, is_resuming_from_checkpoint):
        # Só cria header se o arquivo não existir OU (se estiver resumindo E o arquivo estiver vazio)
        should_write_header = not os.path.exists(detailed_log_file_path) or \
                              (is_resuming_from_checkpoint and os.path.getsize(detailed_log_file_path) == 0) or \
                              (not is_resuming_from_checkpoint) # Se não está resumindo, sempre (re)escreve se arquivo estiver vazio ou for novo
        
        if not os.path.exists(detailed_log_file_path) or os.path.getsize(detailed_log_file_path) == 0:
             actual_write_mode = 'w' # Novo arquivo ou vazio, sobrescrever/criar
        elif is_resuming_from_checkpoint :
             actual_write_mode = 'a' # Resumindo, arquivo existe e tem conteúdo, append
             should_write_header = False # Não escrever header se está resumindo e arquivo tem conteúdo
        else: # Não resumindo, arquivo existe e tem conteúdo -> sobrescrever
             actual_write_mode = 'w'
             should_write_header = True


        if should_write_header:
            log_dir = os.path.dirname(detailed_log_file_path)
            if log_dir and not os.path.exists(log_dir):
                try: os.makedirs(log_dir, exist_ok=True)
                except Exception as e: print(f"AVISO: Não criar dir log {log_dir}: {e}")
            
            header = ("generation,individual_id,l0_size,accuracy_on_full,f1_macro_on_full,"
                      "num_tokens,num_distinct_tokens,num_classes_in_l0,most_frequent_class_in_l0,"
                      "l0_indices_hash,fitness_calc_time_sec\n") 
            try:
                with open(detailed_log_file_path, actual_write_mode, encoding='utf-8') as f: 
                    if actual_write_mode == 'w': # Só escreve header se for 'w' (novo ou sobrescrita)
                        f.write(header)
                print(f"   Log detalhado {'inicializado (novo/sobrescrito)' if actual_write_mode == 'w' else 'pronto para append'} em: {detailed_log_file_path}")
            except Exception as e: print(f"AVISO: Não escrever/abrir log '{detailed_log_file_path}': {e}")
        elif is_resuming_from_checkpoint:
             print(f"   Continuando log detalhado em: {detailed_log_file_path}")


    def _log_individual_details(self, generation, individual_id, individual_indices, acc_on_eval, f1_on_eval, fitness_calc_time, detailed_log_file_path):
        if not self.log_detailed_fitness or not detailed_log_file_path: return
        try:
            L0 = self.df_l0_pool.iloc[individual_indices]
            texts = L0[self.text_column].tolist(); labels = L0[self.label_column].tolist()
            tokens = [t for txt in texts for t in str(txt).lower().split() if t]; n_tok = len(tokens); n_dist_tok = len(set(tokens))
            counts = Counter(labels); n_cls = len(counts); mfc = counts.most_common(1)[0][0] if counts else "N/A"
            h = hashlib.sha256(str(sorted(individual_indices)).encode('utf-8')).hexdigest()[:10]
            acc_s = f"{acc_on_eval:.6f}" if pd.notna(acc_on_eval) and np.isfinite(acc_on_eval) else "Error"
            f1_s  = f"{f1_on_eval:.6f}" if pd.notna(f1_on_eval) and np.isfinite(f1_on_eval) else "Error"
            time_s = f"{fitness_calc_time:.4f}" if pd.notna(fitness_calc_time) else "N/A"
            log_line = (f"{generation},{individual_id},{len(individual_indices)},{acc_s},{f1_s},"
                        f"{n_tok},{n_dist_tok},{n_cls},{mfc},{h},{time_s}\n")
            with open(detailed_log_file_path, 'a', encoding='utf-8') as f: f.write(log_line)
        except Exception as e: print(f"AVISO (Log Detalhado) Erro Gen {generation}, ID {individual_id}: {e}")

    def _create_individual(self):
        return np.random.choice(self.dataset_indices, size=self.initial_l0_size, replace=False)

    def _initialize_population(self):
        # print("Inicializando população..."); # tqdm já informa
        return [self._create_individual() for _ in tqdm(range(self.population_size), desc="Criando População Inicial", leave=False, position=1)]

    def _calculate_fitness_and_metrics(self, individual_indices, generation, individual_id_in_pop, detailed_log_file_path):
        cache_key = tuple(sorted(map(int, individual_indices))) # Garantir que índices sejam int para cache
        if cache_key in self._fitness_cache:
            cached_data = self._fitness_cache[cache_key]
            cached_metrics = cached_data['metrics'] 
            original_calc_time = cached_data.get('calc_time', 0.0)
            if self.log_detailed_fitness:
                 self._log_individual_details(generation, individual_id_in_pop, individual_indices,
                                              cached_metrics.get('acc', -np.inf), cached_metrics.get('f1', -np.inf),
                                              original_calc_time, detailed_log_file_path)
            target_metric_key_for_goal = 'acc' if self.fitness_metric == 'accuracy_on_full' else 'f1'
            metric_for_goal = cached_metrics.get(target_metric_key_for_goal, -np.inf)
            fitness_value = (1.0 - metric_for_goal) if self.optimization_goal == 'minimize' and np.isfinite(metric_for_goal) else metric_for_goal
            if not np.isfinite(metric_for_goal): fitness_value = -np.inf
            return fitness_value, cached_metrics

        fitness_calc_start_time = time.time()
        L0 = self.df_l0_pool.iloc[individual_indices]; X_txt_L0 = L0[self.text_column].tolist(); y_lbl_L0 = L0[self.label_column].tolist()
        model = get_model(self.classifier_config); acc_on_eval, f1_on_eval = -np.inf, -np.inf
        try:
            X_in_L0 = X_txt_L0 
            if self.embedder and isinstance(model, BaseFeatureClassifier): X_in_L0 = self.embedder.transform(X_txt_L0)
            if not ((isinstance(X_in_L0, np.ndarray) and X_in_L0.size > 0) or \
                    (isinstance(X_in_L0, list) and X_in_L0) or \
                    (hasattr(X_in_L0, "shape") and X_in_L0.shape[0] > 0) ):
                raise ValueError("Input de treino (L0) vazio ou inválido.")
            model.fit(X_in_L0, y_lbl_L0)
            X_eval_txt_effective = self.df_eval[self.text_column].tolist(); y_true_effective = self.df_eval[self.label_column].tolist()
            X_eval_in_effective = X_eval_txt_effective 
            if self.embedder and isinstance(model, BaseFeatureClassifier): X_eval_in_effective = self.embedder.transform(X_eval_txt_effective)
            y_pred_effective = model.predict(X_eval_in_effective)
            acc_on_eval = accuracy_score(y_true_effective, y_pred_effective)
            f1_on_eval = f1_score(y_true_effective, y_pred_effective, average='macro', zero_division=0, labels=self.all_possible_labels)
        except Exception as e: 
            tqdm.write(f"    ERRO fitness (Gen {generation}, Indiv {individual_id_in_pop}, L0tam {len(individual_indices)}): {type(e).__name__} - {e}")
        fitness_calc_time = time.time() - fitness_calc_start_time
        current_metrics_dict = {'acc': acc_on_eval, 'f1': f1_on_eval} 
        self._fitness_cache[cache_key] = {'metrics': current_metrics_dict, 'calc_time': fitness_calc_time}
        if self.log_detailed_fitness:
            self._log_individual_details(generation, individual_id_in_pop, individual_indices, acc_on_eval, f1_on_eval, fitness_calc_time, detailed_log_file_path)
        target_metric_key_for_goal = 'acc' if self.fitness_metric == 'accuracy_on_full' else 'f1'
        metric_for_goal = current_metrics_dict.get(target_metric_key_for_goal, -np.inf)
        fitness_value = (1.0 - metric_for_goal) if self.optimization_goal == 'minimize' and np.isfinite(metric_for_goal) else metric_for_goal
        if not np.isfinite(metric_for_goal): fitness_value = -np.inf
        return fitness_value, current_metrics_dict

    def _selection(self, population, fitness_values):
        # ... (código como na sua última versão)
        parents = []
        pop_indices = np.arange(len(population))
        for _ in range(len(population)):
            tour_idx_pop = np.random.choice(pop_indices, size=self.tournament_size, replace=True) # Amostra com reposição
            tour_fit_values = [fitness_values[i] for i in tour_idx_pop]
            # Lidar com todos -np.inf ou NaNs
            valid_tournament_fits = [f for f in tour_fit_values if pd.notna(f) and np.isfinite(f)]
            if not valid_tournament_fits: # Se todos inválidos, escolhe um aleatoriamente
                winner_local_idx = np.random.randint(0, self.tournament_size)
            else:
                # Encontrar o melhor entre os válidos (np.argmax não lida bem com NaNs misturados, precisa de máscara)
                # Alternativa: substituir NaNs/infs por um valor muito baixo antes de argmax
                temp_tour_fit_selection = [-np.inf if pd.isna(f) or not np.isfinite(f) else f for f in tour_fit_values]
                winner_local_idx = np.argmax(temp_tour_fit_selection)
            parents.append(population[tour_idx_pop[winner_local_idx]])
        return parents

    def _crossover(self, p1_idx, p2_idx):
        # ... (código como na sua última versão)
        parent1 = np.array(p1_idx); parent2 = np.array(p2_idx)
        if self.initial_l0_size <= 1 : return parent1.copy(), parent2.copy()
        cp = random.randint(1, self.initial_l0_size - 1) if self.initial_l0_size >=2 else 1 
        c1_p=np.concatenate((parent1[:cp],parent2[cp:])); c2_p=np.concatenate((parent2[:cp],parent1[cp:]))
        def repair(c_p,p1,p2):
            unique_genes = list(set(c_p)) 
            needed = self.initial_l0_size - len(unique_genes)
            if needed > 0:
                parent_pool = set(p1) | set(p2)
                candidates_p = list(parent_pool - set(unique_genes)) 
                random.shuffle(candidates_p)
                can_add_from_parents = min(needed, len(candidates_p))
                unique_genes.extend(candidates_p[:can_add_from_parents])
                needed -= can_add_from_parents
            if needed > 0:
                dataset_set = set(self.dataset_indices)
                pool_ds = list(dataset_set - set(unique_genes)) 
                random.shuffle(pool_ds)
                if len(pool_ds) < needed: return self._create_individual()
                unique_genes.extend(pool_ds[:needed])
            final_genes = np.array(unique_genes)
            if len(final_genes) > self.initial_l0_size:
                 return np.random.choice(final_genes, size=self.initial_l0_size, replace=False)
            elif len(final_genes) < self.initial_l0_size:
                 return self._create_individual() 
            return final_genes
        return repair(c1_p,parent1,parent2), repair(c2_p,parent2,parent1)

    def _mutation(self, ind_idx):
        # ... (código como na sua última versão)
        mut=np.array(ind_idx)
        if len(mut)==0 or self.mutation_strength==0: return mut
        pool=np.setdiff1d(self.dataset_indices,mut, assume_unique=True)
        if len(pool)==0: return mut
        act_mut_cnt=min(self.mutation_strength, len(mut), len(pool)) 
        if act_mut_cnt<=0: return mut
        pos_mut=np.random.choice(len(mut),size=act_mut_cnt,replace=False)
        repl_genes=np.random.choice(pool,size=act_mut_cnt,replace=False)
        mut[pos_mut]=repl_genes; return mut

    def run_optimization(self, detailed_log_file_path_from_notebook):
        # Tentar carregar checkpoint
        resumed_from_checkpoint, log_path_from_ckpt = self._load_checkpoint()
        
        current_detailed_log_file = detailed_log_file_path_from_notebook
        if resumed_from_checkpoint and log_path_from_ckpt:
            if log_path_from_ckpt != detailed_log_file_path_from_notebook:
                tqdm.write(f"   AVISO: Log detalhado do checkpoint ({log_path_from_ckpt}) difere do fornecido ({detailed_log_file_path_from_notebook}). Usando o do checkpoint.")
            current_detailed_log_file = log_path_from_ckpt
        
        if self.log_detailed_fitness:
            self._initialize_detailed_log(current_detailed_log_file, is_resuming_from_checkpoint=resumed_from_checkpoint)

        if not resumed_from_checkpoint or not self.population : # Se não resumido OU se população do checkpoint estiver vazia
            if not resumed_from_checkpoint: # Apenas resetar se não foi do checkpoint, pois _load_checkpoint já lida com estado.
                self._reset_state_to_initial() # Limpa atributos de estado
            self.population = self._initialize_population()
        
        if not self.population or len(self.population) == 0 :
            print("ERRO: População não pôde ser carregada ou inicializada."); 
            return np.array([],dtype=int), self.best_actual_performance_overall, pd.DataFrame(self.optimization_history)

        start_opt_time_this_run = time.time()
        
        # Inicia o loop da geração que o checkpoint parou (ou da 0 se novo)
        # self.current_generation é a ÚLTIMA geração COMPLETADA. O loop deve começar da próxima.
        # Se current_generation = 0 e é uma nova run, gen_loop_start = 0.
        # Se current_generation = 0 após carregar checkpoint (significa que parou antes da primeira gen completar o save), gen_loop_start = 0.
        # Se current_generation = K (>0) após carregar, significa que K gerações completaram e foram salvas, então a próxima é K.
        gen_loop_start = self.current_generation 
        if resumed_from_checkpoint and self.current_generation > 0: # Se resumido e já completou alguma geração
             gen_loop_start = self.current_generation # Começa da próxima a ser executada, que é a current_generation salva
                                                    # pois o loop do python é `range(start, end)` onde `end` não é incluído.
                                                    # O checkpoint salva `self.current_generation` como a última *completada*.
                                                    # O loop `for gen in pbar_generations:` deve ir de `self.current_generation` a `self.n_generations -1`.
                                                    # Ex: se n_generations=100, gen vai de 0 a 99.
                                                    # Se salvou em gen=0 (após primeira iteração), current_generation=0. Loop deve começar de 0.
                                                    # Se salvou em gen=5 (após sexta iteração), current_generation=5. Loop deve começar de 5.

        # O loop de progresso principal deve ir de 0 até n_generations-1
        # `initial` deve ser a `self.current_generation` se resumindo.
        # Se `self.current_generation` é a última completada, o loop deve começar dela.
        # Ex: se 10 gerações, n_generations = 10. Loop é range(0, 10) -> 0..9
        # Se parou na gen 4 (current_generation = 4), o próximo loop é range(4, 10) -> 4..9
        
        pbar_generations_outer = tqdm(range(gen_loop_start, self.n_generations), 
                                      desc=f"AG (Melhor: {self.best_actual_performance_overall:.4f})", 
                                      initial=gen_loop_start, total=self.n_generations, position=0, leave=True)

        for gen_idx in pbar_generations_outer:
            self.current_generation = gen_idx # gen_idx é o índice da geração atual (0 a n_generations-1)
            gen_display_num = self.current_generation + 1 # Para logs e display (1 a n_generations)
            
            gen_start_time_loop = time.time()
            
            fitness_values_for_selection = []
            actual_metrics_population = []

            for i, ind in enumerate(tqdm(self.population, desc=f"Fitness Gen {gen_display_num}", leave=False, position=1)):
                 fitness_val, metrics_dict = self._calculate_fitness_and_metrics(ind, gen_display_num, i, current_detailed_log_file)
                 fitness_values_for_selection.append(fitness_val)
                 actual_metrics_population.append(metrics_dict)
            
            target_metric_key_in_dict = 'acc' if 'accuracy' in self.fitness_metric else 'f1'
            default_val_target = -np.inf if self.optimization_goal == 'maximize' else np.inf
            current_gen_performances_real = [m.get(target_metric_key_in_dict, default_val_target) for m in actual_metrics_population]
            valid_perf_mask = pd.Series(current_gen_performances_real).apply(lambda x: pd.notna(x) and np.isfinite(x))
            current_gen_best_actual_perf = default_val_target
            best_idx_this_gen = -1

            if valid_perf_mask.any():
                valid_subset_performances = np.array(current_gen_performances_real)[valid_perf_mask]
                original_indices_of_valid = np.where(valid_perf_mask)[0]
                if self.optimization_goal == 'maximize': best_local_idx_in_valid = np.argmax(valid_subset_performances)
                else: best_local_idx_in_valid = np.argmin(valid_subset_performances)
                if original_indices_of_valid.size > 0 :
                    best_idx_this_gen = original_indices_of_valid[best_local_idx_in_valid]
                    current_gen_best_actual_perf = current_gen_performances_real[best_idx_this_gen]
            
            current_gen_best_individual = self.population[best_idx_this_gen] if best_idx_this_gen != -1 and len(self.population) > best_idx_this_gen else None
            new_best_found_this_gen = False
            if self.optimization_goal == 'maximize':
                if np.isfinite(current_gen_best_actual_perf) and current_gen_best_actual_perf > self.best_actual_performance_overall:
                    self.best_actual_performance_overall=current_gen_best_actual_perf; self.best_individual_overall=current_gen_best_individual; new_best_found_this_gen=True
            else: 
                if np.isfinite(current_gen_best_actual_perf) and current_gen_best_actual_perf < self.best_actual_performance_overall:
                    if current_gen_best_actual_perf > -np.inf: self.best_actual_performance_overall=current_gen_best_actual_perf; self.best_individual_overall=current_gen_best_individual; new_best_found_this_gen=True
                elif self.best_actual_performance_overall == np.inf and np.isfinite(current_gen_best_actual_perf) and current_gen_best_actual_perf > -np.inf: 
                    self.best_actual_performance_overall=current_gen_best_actual_perf; self.best_individual_overall=current_gen_best_individual; new_best_found_this_gen=True
            
            if new_best_found_this_gen: 
                 pbar_generations_outer.set_description(f"AG (Melhor: {self.best_actual_performance_overall:.4f}) Gen {gen_display_num}")
                 tqdm.write(f"    Gen {gen_display_num}: Novo melhor ({self.fitness_metric} em df_eval): {self.best_actual_performance_overall:.4f}")


            gen_actual_acc_eval = [m.get('acc', np.nan) for m in actual_metrics_population]
            gen_actual_f1_eval  = [m.get('f1', np.nan) for m in actual_metrics_population]
            valid_acc_eval = [s for s in gen_actual_acc_eval if pd.notna(s) and np.isfinite(s)]
            valid_f1_eval  = [s for s in gen_actual_f1_eval  if pd.notna(s) and np.isfinite(s)]
            gen_duration_loop = time.time() - gen_start_time_loop
            history_entry = {'generation': gen_display_num, 'max_acc': np.max(valid_acc_eval) if valid_acc_eval else np.nan, 'avg_acc': np.mean(valid_acc_eval) if valid_acc_eval else np.nan, 'min_acc': np.min(valid_acc_eval) if valid_acc_eval else np.nan,'max_f1': np.max(valid_f1_eval) if valid_f1_eval else np.nan, 'avg_f1': np.mean(valid_f1_eval) if valid_f1_eval else np.nan, 'min_f1': np.min(valid_f1_eval) if valid_f1_eval else np.nan,'generation_time_sec': gen_duration_loop}
            self.optimization_history.append(history_entry)
            
            # Seleção, Crossover, Mutação
            # ... (lógica como na sua última versão, usando self.population) ...
            # --- Seleção e Elitismo ---
            valid_fitness_mask = pd.Series(fitness_values_for_selection).apply(lambda x: pd.notna(x) and np.isfinite(x))
            if not valid_fitness_mask.any():
                tqdm.write(f"    AVISO Gen {gen_display_num}: Todos os fitness são inválidos. Reinicializando população."); self.population = self._initialize_population(); continue
            population_vf = [self.population[i] for i, valid in enumerate(valid_fitness_mask) if valid]
            fitness_vf = [fitness_values_for_selection[i] for i, valid in enumerate(valid_fitness_mask) if valid]
            if not population_vf : 
                 tqdm.write(f"    AVISO Gen {gen_display_num}: População válida para fitness vazia. Reinicializando."); self.population = self._initialize_population(); continue
            sorted_pop_fit_valid = sorted(zip(population_vf, fitness_vf), key=lambda x: x[1], reverse=True)
            elite = [ind for ind, fv in sorted_pop_fit_valid[:self.n_elite]]
            parents = self._selection(self.population, fitness_values_for_selection)
            next_pop = elite.copy(); offspring_needed = self.population_size - self.n_elite; offspring_count = 0; p_idx_counter = 0
            if not parents and offspring_needed > 0:
                 for _ in range(offspring_needed): next_pop.append(self._create_individual())
            elif parents:
                parents_list = list(parents) 
                if not parents_list: 
                    for _ in range(offspring_needed): next_pop.append(self._create_individual())
                else:
                    while offspring_count < offspring_needed:
                        idx1 = p_idx_counter % len(parents_list); p_idx_counter +=1
                        idx2 = p_idx_counter % len(parents_list); p_idx_counter +=1
                        p1=parents_list[idx1]; p2=parents_list[idx2]
                        c1,c2 = (p1.copy(),p2.copy()) 
                        if random.random() < self.crossover_rate: c1,c2 = self._crossover(p1,p2)
                        if random.random() < self.mutation_rate: c1=self._mutation(c1)
                        if random.random() < self.mutation_rate: c2=self._mutation(c2)
                        next_pop.append(c1); offspring_count+=1
                        if offspring_count < offspring_needed: next_pop.append(c2); offspring_count+=1
            if len(next_pop) < self.population_size:
                needed_fill = self.population_size - len(next_pop)
                for _ in range(needed_fill): next_pop.append(self._create_individual())
            self.population = next_pop[:self.population_size]
            
            # Salvar checkpoint (self.current_generation é o índice da geração que acabou de completar)
            if (self.current_generation + 1) % 1 == 0: # Salva a cada geração
                self._save_checkpoint(current_detailed_log_file)
        
        if 'pbar_generations_outer' in locals() and pbar_generations_outer is not None:
            pbar_generations_outer.close()

        opt_duration_total = time.time() - start_opt_time_this_run 
        print(f"\n--- Otimização Genética Concluída/Resumida ({opt_duration_total:.2f} seg) ---")
        if self.best_individual_overall is not None:
             print(f"Melhor Performance ({self.fitness_metric} em df_eval) Encontrada: {self.best_actual_performance_overall:.4f}")
        else:
             print("Nenhum indivíduo válido encontrado como 'melhor geral'.")
             self.best_individual_overall = np.array([], dtype=int)
        
        # if self.current_generation + 1 >= self.n_generations and os.path.exists(self.checkpoint_file):
        #     try:
        #         os.remove(self.checkpoint_file)
        #         print(f"   Checkpoint {self.checkpoint_file} removido após conclusão.")
        #     except Exception as e_rm:
        #         print(f"   AVISO: Não foi possível remover o arquivo de checkpoint {self.checkpoint_file}: {e_rm}")
        
        return self.best_individual_overall, self.best_actual_performance_overall, pd.DataFrame(self.optimization_history)