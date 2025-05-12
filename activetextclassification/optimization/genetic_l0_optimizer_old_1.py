# activetextclassification/optimization/genetic_l0_optimizer.py

import numpy as np
import pandas as pd
import time
import random
from tqdm.notebook import tqdm # Para barras de progresso

# Imports da própria biblioteca
# Supondo que get_model e BaseTextClassifier/BaseFeatureClassifier estão em models
from ..models import get_model, BaseTextClassifier, BaseFeatureClassifier
# Supondo que você pode precisar de um embedder se o modelo for baseado em features
from ..embeddings import BaseEmbedder

# Imports de ML
from sklearn.metrics import accuracy_score, f1_score


class GeneticL0Optimizer:
    """
    Otimiza a seleção do conjunto inicial L0 usando Algoritmos Genéticos.
    O objetivo é encontrar um L0 de tamanho fixo que maximize a performance
    de um classificador treinado apenas nele, avaliado no dataset completo.
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
                 mutation_rate=0.1, # Probabilidade de mutação POR INDIVÍDUO
                 mutation_strength=1, # Quantos genes (índices) trocar na mutação
                 elitism_rate=0.1,
                 fitness_metric='accuracy_on_full', # ou 'f1_macro_on_full'
                 tournament_size=3, # Para seleção por torneio
                 random_seed=None,
                 embedder=None # Embedder AJUSTADO, se classificador for BaseFeatureClassifier
                 ):
        """
        Args:
            df_full (pd.DataFrame): O dataset completo disponível.
            text_column (str): Nome da coluna de texto.
            label_column (str): Nome da coluna de rótulos.
            classifier_config (dict): Configuração para get_model.
            initial_l0_size (int): Tamanho fixo do L0 a ser otimizado.
            all_possible_labels (list): Lista de todos os labels únicos para cálculo de F1.
            population_size (int): Número de indivíduos (L0s) na população do AG.
            n_generations (int): Número de gerações para rodar o AG.
            crossover_rate (float): Probabilidade de realizar crossover.
            mutation_rate (float): Probabilidade de um indivíduo sofrer mutação.
            mutation_strength (int): Número de genes (índices) a serem alterados durante a mutação.
            elitism_rate (float): Proporção da população (melhores indivíduos) a ser
                                  diretamente transferida para a próxima geração.
            fitness_metric (str): Métrica a ser otimizada ('accuracy_on_full' ou 'f1_macro_on_full').
            tournament_size (int): Número de indivíduos no torneio para seleção.
            random_seed (int, optional): Semente para reprodutibilidade.
            embedder (BaseEmbedder, optional): Instância ajustada de um embedder,
                                               necessário se o classificador for BaseFeatureClassifier.
        """
        self.df_full = df_full.reset_index(drop=True) # Garantir índice 0..N-1
        self.text_column = text_column
        self.label_column = label_column
        self.classifier_config = classifier_config
        self.initial_l0_size = initial_l0_size
        self.all_possible_labels = all_possible_labels
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.elitism_rate = elitism_rate
        self.fitness_metric = fitness_metric
        self.tournament_size = tournament_size
        self.embedder = embedder # Armazenar o embedder global ajustado

        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)

        # Validações
        if not (0 <= self.crossover_rate <= 1): raise ValueError("Crossover rate must be between 0 and 1.")
        if not (0 <= self.mutation_rate <= 1): raise ValueError("Mutation rate must be between 0 and 1.")
        if not (0 <= self.elitism_rate < 1): raise ValueError("Elitism rate must be between 0 and 1 (exclusive of 1).")
        if self.initial_l0_size > len(self.df_full): raise ValueError("initial_l0_size cannot be larger than the dataset.")
        if self.fitness_metric not in ['accuracy_on_full', 'f1_macro_on_full']:
            raise ValueError("fitness_metric must be 'accuracy_on_full' or 'f1_macro_on_full'.")

        self.n_elite = int(self.population_size * self.elitism_rate)
        self.dataset_indices = np.array(self.df_full.index) # Índices do dataset completo

        # Cache para fitness (evitar recalcular para o mesmo L0 se ele reaparecer)
        self._fitness_cache = {}
        self._model_cache = {} # Opcional: cache de modelos treinados (pode consumir muita memória)

        print(f"GeneticL0Optimizer inicializado. Pop: {self.population_size}, Gerações: {self.n_generations}, L0 Size: {self.initial_l0_size}")

    def _create_individual(self):
        """ Cria um indivíduo (L0) aleatório: uma lista de initial_l0_size índices únicos. """
        return np.random.choice(self.dataset_indices, size=self.initial_l0_size, replace=False)

    def _initialize_population(self):
        """ Inicializa a população com indivíduos aleatórios. """
        print("Inicializando população...")
        population = []
        for _ in range(self.population_size):
            population.append(self._create_individual())
        return population

    def _calculate_fitness(self, individual_indices):
        """
        Calcula o fitness de um indivíduo (L0).
        Fitness é a performance do classificador treinado neste L0, avaliado no df_full.
        """
        # Usar tupla de índices ordenados como chave do cache para consistência
        cache_key = tuple(sorted(individual_indices))
        if cache_key in self._fitness_cache:
            # print(f"    Fitness cache hit for L0 size {len(individual_indices)}")
            return self._fitness_cache[cache_key]

        # print(f"    Calculando fitness para L0 de tamanho {len(individual_indices)}...")
        L0_df_current = self.df_full.iloc[individual_indices]

        X_train_l0_text = L0_df_current[self.text_column].tolist()
        y_train_l0_labels = L0_df_current[self.label_column].tolist()

        # Instanciar modelo (fábrica lida com conversão de ngram_range)
        model = get_model(self.classifier_config)
        train_time_sec = np.nan
        fitness_score = -np.inf # Default para pior fitness

        try:
            # Preparar input para o modelo
            X_train_input = None
            if isinstance(model, BaseFeatureClassifier):
                if self.embedder is None: raise ValueError("Embedder necessário para BaseFeatureClassifier.")
                X_train_input = self.embedder.transform(X_train_l0_text)
            elif isinstance(model, BaseTextClassifier):
                X_train_input = X_train_l0_text
            else:
                raise TypeError(f"Tipo de modelo {type(model)} não suportado.")

            # Treinar
            start_train = time.time()
            model.fit(X_train_input, y_train_l0_labels)
            train_time_sec = time.time() - start_train

            # Avaliar no dataset completo
            X_eval_full_text = self.df_full[self.text_column].tolist()
            y_true_full_labels = self.df_full[self.label_column].tolist()
            X_eval_input = None

            if isinstance(model, BaseFeatureClassifier):
                 X_eval_input = self.embedder.transform(X_eval_full_text)
            elif isinstance(model, BaseTextClassifier):
                 X_eval_input = X_eval_full_text

            y_pred_full_labels = model.predict(X_eval_input)

            if self.fitness_metric == 'accuracy_on_full':
                fitness_score = accuracy_score(y_true_full_labels, y_pred_full_labels)
            elif self.fitness_metric == 'f1_macro_on_full':
                fitness_score = f1_score(y_true_full_labels, y_pred_full_labels, average='macro', zero_division=0, labels=self.all_possible_labels)

        except Exception as e:
            print(f"    ERRO ao calcular fitness para um indivíduo: {e}")
            # Fitness baixo para indivíduos problemáticos
            fitness_score = -np.inf # Ou 0 se fitness não puder ser negativo

        self._fitness_cache[cache_key] = fitness_score
        return fitness_score


    def _selection(self, population, fitness_scores):
        """ Seleção por Torneio. """
        selected_parents = []
        for _ in range(len(population)): # Selecionar N pais para N filhos
            tournament_indices = np.random.choice(len(population), size=self.tournament_size, replace=True)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_index_in_tournament = np.argmax(tournament_fitness)
            selected_parents.append(population[tournament_indices[winner_index_in_tournament]])
        return selected_parents

    def _crossover(self, parent1_indices, parent2_indices):
        """
        Crossover de um ponto para indivíduos de tamanho fixo (L0).
        Garante que não haja duplicatas e mantém o tamanho do L0.
        """
        # Garantir que são arrays numpy para facilitar operações
        parent1 = np.array(parent1_indices)
        parent2 = np.array(parent2_indices)

        # Escolher um ponto de corte
        # (Não pode ser 0 nem len, pois senão um filho seria cópia de um pai)
        crossover_point = random.randint(1, self.initial_l0_size - 1)

        # Criar filhos
        child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))

        # --- Lidar com duplicatas e tamanho ---
        # Para cada filho, remover duplicatas e completar com genes do outro pai
        # ou aleatoriamente do dataset_indices se necessário.

        def repair_child(child, p1, p2):
            # Garantir unicidade e tamanho
            unique_genes, counts = np.unique(child, return_counts=True)
            new_child = list(unique_genes) # Começa com os únicos

            # Se menor que o tamanho L0, precisa adicionar
            needed = self.initial_l0_size - len(new_child)
            if needed > 0:
                # Tentar adicionar do outro pai (que não está no filho ainda)
                candidates_from_other_parent = np.setdiff1d(np.concatenate((p1,p2)), new_child) # Genes dos pais que não estão no filho
                np.random.shuffle(candidates_from_other_parent) # Aleatorizar
                take_from_parents = min(needed, len(candidates_from_other_parent))
                new_child.extend(candidates_from_other_parent[:take_from_parents])
                needed -= take_from_parents

            # Se ainda precisar, pegar do dataset geral (excluindo os já presentes)
            if needed > 0:
                pool_for_completion = np.setdiff1d(self.dataset_indices, new_child)
                np.random.shuffle(pool_for_completion)
                take_from_pool = min(needed, len(pool_for_completion))
                new_child.extend(pool_for_completion[:take_from_pool])

            # Se ficou maior (improvável com este método mas possível com outros crossovers)
            # ou se não conseguiu completar (pool esgotado, L0_size == |dataset|)
            if len(new_child) > self.initial_l0_size:
                 np.random.shuffle(new_child)
                 new_child = new_child[:self.initial_l0_size]
            elif len(new_child) < self.initial_l0_size:
                 # Isso não deveria acontecer se initial_l0_size <= len(dataset_indices)
                 print(f"AVISO Crossover: Não foi possível completar o filho para o tamanho {self.initial_l0_size}. Obtido: {len(new_child)}")
                 # Preencher com o que tem, ou duplicar aleatoriamente do próprio filho (não ideal)
                 # ou repetir o processo com pais diferentes? Por ora, pode retornar menor.
                 # Ou melhor: se não conseguir completar, recriar aleatoriamente
                 if len(pool_for_completion) < needed:
                      print("    Recriando filho aleatoriamente pois não há genes suficientes para completar.")
                      return self._create_individual()


            return np.array(new_child)

        child1_repaired = repair_child(child1, parent1, parent2)
        child2_repaired = repair_child(child2, parent2, parent1)

        return child1_repaired, child2_repaired


    def _mutation(self, individual_indices):
        """
        Mutação: troca 'mutation_strength' genes (índices) do indivíduo
        por outros aleatórios do dataset (que não estão no indivíduo).
        """
        mutated_individual = np.array(individual_indices) # Trabalhar com cópia
        for _ in range(self.mutation_strength):
            # Escolher um gene (índice) aleatório do indivíduo para remover
            idx_to_remove_pos = random.randint(0, len(mutated_individual) - 1)
            # gene_removed = mutated_individual[idx_to_remove_pos] # Não precisamos do valor

            # Encontrar candidatos para substituição (todos os índices do dataset MENOS os já presentes)
            possible_replacements = np.setdiff1d(self.dataset_indices, mutated_individual)

            if len(possible_replacements) > 0:
                replacement_gene = np.random.choice(possible_replacements)
                mutated_individual[idx_to_remove_pos] = replacement_gene # Substitui
            # Se não houver substitutos (improvável, a menos que L0_size == |dataset|), não faz nada
        return mutated_individual


    def run_optimization(self):
        """ Executa o algoritmo genético. """
        print(f"\n--- Iniciando Otimização Genética para L0 (Tamanho: {self.initial_l0_size}) ---")
        start_opt_time = time.time()

        population = self._initialize_population()
        optimization_history = [] # Para guardar fitness max/min/avg por geração

        best_fitness_overall = -np.inf
        best_individual_overall = None

        for gen in tqdm(range(self.n_generations), desc="Gerações AG"):
            gen_start_time = time.time()
            print(f"\n  Geração {gen + 1}/{self.n_generations}")

            # 1. Calcular Fitness da População Atual
            # print("    Calculando fitness da população...")
            fitness_scores = [self._calculate_fitness(ind) for ind in tqdm(population, desc="Fitness Calc", leave=False)]

            # Guardar melhor da geração atual
            current_gen_best_fitness = np.max(fitness_scores)
            current_gen_best_individual_idx = np.argmax(fitness_scores)
            current_gen_best_individual = population[current_gen_best_individual_idx]

            if current_gen_best_fitness > best_fitness_overall:
                best_fitness_overall = current_gen_best_fitness
                best_individual_overall = current_gen_best_individual
                print(f"    Novo melhor fitness geral encontrado: {best_fitness_overall:.4f}")

            # Registrar histórico da geração
            optimization_history.append({
                'generation': gen + 1,
                'max_fitness': current_gen_best_fitness,
                'avg_fitness': np.mean(fitness_scores),
                'min_fitness': np.min(fitness_scores)
            })
            print(f"    Fitness Geração {gen+1}: Max={current_gen_best_fitness:.4f}, Avg={np.mean(fitness_scores):.4f}, Min={np.min(fitness_scores):.4f}")


            # 2. Seleção e Elitismo
            # print("    Selecionando pais e aplicando elitismo...")
            # Ordenar população por fitness (decrescente)
            sorted_population_with_fitness = sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)
            elite = [ind for ind, fit in sorted_population_with_fitness[:self.n_elite]]

            # Selecionar pais da população total para crossover/mutação
            parents = self._selection(population, fitness_scores)


            # 3. Crossover e Mutação
            # print("    Realizando crossover e mutação...")
            next_population = elite.copy() # Começa com a elite
            num_offspring_needed = self.population_size - self.n_elite

            current_offspring_count = 0
            parent_idx = 0
            while current_offspring_count < num_offspring_needed:
                parent1 = parents[parent_idx % len(parents)] # Cicla pelos pais
                parent_idx += 1
                parent2 = parents[parent_idx % len(parents)] # Pega o próximo (pode ser o mesmo se len < 2)
                parent_idx += 1

                if random.random() < self.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy() # Clonagem

                if random.random() < self.mutation_rate:
                    child1 = self._mutation(child1)
                if random.random() < self.mutation_rate:
                    child2 = self._mutation(child2)

                next_population.append(child1)
                current_offspring_count += 1
                if current_offspring_count < num_offspring_needed:
                    next_population.append(child2)
                    current_offspring_count += 1

            population = next_population[:self.population_size] # Garante tamanho da população
            gen_duration = time.time() - gen_start_time
            print(f"    Duração Geração {gen+1}: {gen_duration:.2f}s")


        # --- Fim das Gerações ---
        opt_duration = time.time() - start_opt_time
        print(f"\n--- Otimização Genética Concluída ({opt_duration:.2f} seg) ---")
        print(f"Melhor Fitness Geral Encontrado ({self.fitness_metric}): {best_fitness_overall:.4f}")
        # print(f"Melhor Indivíduo (L0 Índices): {best_individual_overall}") # Pode ser grande

        return best_individual_overall, best_fitness_overall, pd.DataFrame(optimization_history)