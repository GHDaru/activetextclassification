# activetextclassification/cold_start/dri_cluster.py

import numpy as np
import time
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances 

# Importar seu ProductVectorizer (ajuste o caminho se necessário)
from activetextclassification.vectorizers import ProductVectorizer 

class DRIClusterColdStart:
    """
    Implementa o algoritmo DRI-Cluster para construir um conjunto inicial L0.
    - Usa KMeans para clustering semântico inicial.
    - Usa um ProductVectorizer interno para:
        1. Calcular a relevância TF-IDF de termos para cada cluster semântico (matriz A).
        2. Obter representações binárias de termos para amostras individuais (vetores q_x).
    - Seleção intra-cluster:
        - Primeira amostra: maior relevância q_x^T * A_j.
        - Subsequentes: maximizando (q_x AND (NOT template))^T * A_j (novidade ponderada pela relevância).
    - Preenchimento final aleatório.
    """

    def __init__(self,
                 i_target: int,
                 semantic_embedder, 
                 n_clusters_semantic: int, 
                 random_seed: int = 42,
                 min_samples_per_cluster: int = 1,
                 verbose: bool = False):
        
        self.i_target = i_target
        self.semantic_embedder = semantic_embedder
        self.verbose = verbose
        self.random_seed = random_seed

        self.semantic_clustering_algo = KMeans(
            n_clusters=n_clusters_semantic, 
            random_state=self.random_seed,
            n_init='auto'
        )
        self.n_clusters_semantic_requested = n_clusters_semantic
        
        # Configuração para o ProductVectorizer interno
        # Removidos token_pattern, lowercase, strip_accents para usar os defaults do ProductVectorizer
        self._internal_pv_config = {
            'method': 'tfidf', 
            'norm': 'l2', 
            'query': 'binary', 
            'query_norm': None, 
            'ngram_range': (1,1)
        }
        
        self.min_samples_per_cluster = min_samples_per_cluster
        self.relevance_pv_instance_ = None 
        self.A_term_x_cluster_tfidf_ = None 
        self.U0_binary_term_vectors_all_sparse_ = None 

        if self.verbose: print(f"DRIClusterColdStart (PV Interno com Defaults) inicializado. Alvo L0: {i_target}, Clusters Semânticos Solicitados: {n_clusters_semantic}")

    def _perform_semantic_clustering(self, U0_texts: list[str], N_U: int) -> tuple:
        time_start = time.time()
        if self.verbose: print("  DRI Sub-Process: 1. Clustering Semântico (K-Means)...")
        try: embeddings = self.semantic_embedder.encode(U0_texts, show_progress_bar=self.verbose)
        except TypeError: embeddings = self.semantic_embedder.encode(U0_texts)
        if N_U == 0: return np.array([]), np.array([]), np.array([]), 0, None
        current_n_clusters = self.n_clusters_semantic_requested
        if N_U < current_n_clusters: current_n_clusters = N_U
        if N_U > 0 and current_n_clusters == 0 : current_n_clusters = 1
        if current_n_clusters == 0: return np.array([]), np.array([]), np.array([]), 0, embeddings
        self.semantic_clustering_algo.set_params(n_clusters=current_n_clusters)
        labels = self.semantic_clustering_algo.fit_predict(embeddings)
        ids, counts = np.unique(labels, return_counts=True)
        n_effective = len(ids)
        if self.verbose: print(f"    K-Means concluído em {time.time() - time_start:.2f}s. {n_effective} clusters semânticos.")
        return labels, ids, counts, n_effective, embeddings

    def _prepare_internal_product_vectorizer(self, U0_texts: list[str], 
                                             semantic_cluster_labels: np.ndarray, 
                                             effective_semantic_cluster_ids: np.ndarray) -> bool:
        time_start = time.time()
        if self.verbose: print("  DRI Sub-Process: 2. Preparando ProductVectorizer interno...")
        semantic_cluster_documents_texts = []
        map_eff_sem_id_to_pv_idx = {}
        for i, eff_sem_clust_id_val in enumerate(effective_semantic_cluster_ids):
            texts = [U0_texts[pos_idx] for pos_idx, label in enumerate(semantic_cluster_labels) if label == eff_sem_clust_id_val]
            semantic_cluster_documents_texts.append(" ".join(texts) if texts else "")
            map_eff_sem_id_to_pv_idx[eff_sem_clust_id_val] = i 
        if not any(semantic_cluster_documents_texts): 
            if self.verbose: print("    ERRO: Nenhum documento de cluster não vazio para PV interno.")
            return False
        
        self.relevance_pv_instance_ = ProductVectorizer(**self._internal_pv_config)
        docs_for_fit = [doc for doc in semantic_cluster_documents_texts if doc]
        pv_labels_for_fit = [map_eff_sem_id_to_pv_idx[eff_id] for eff_id in effective_semantic_cluster_ids if semantic_cluster_documents_texts[map_eff_sem_id_to_pv_idx[eff_id]]]
        if not docs_for_fit: 
            if self.verbose: print("    ERRO: Nenhum documento de cluster para FIT do PV interno.")
            return False
        
        self.relevance_pv_instance_.fit(X=docs_for_fit, y=np.array(pv_labels_for_fit).astype(str))
        self.A_term_x_cluster_tfidf_ = self.relevance_pv_instance_.vectorizer.transform(semantic_cluster_documents_texts).T.toarray()
        self.U0_binary_term_vectors_all_sparse_ = self.relevance_pv_instance_.query.transform(U0_texts)
        
        if self.verbose: print(f"    PV Interno pronto em {time.time() - time_start:.2f}s. Matriz A: {self.A_term_x_cluster_tfidf_.shape}, Matriz q_x: {self.U0_binary_term_vectors_all_sparse_.shape}")
        if self.A_term_x_cluster_tfidf_.shape[0] != self.U0_binary_term_vectors_all_sparse_.shape[1]:
            print("ERRO CRÍTICO DRI: Incompatibilidade de vocabulário PV (A vs q_x).")
            print(f"  Vocabulário do PV (vectorizer): {len(self.relevance_pv_instance_.vectorizer.vocabulary_)}")
            print(f"  Vocabulário do PV (query): {len(self.relevance_pv_instance_.query.vocabulary_)}")
            return False
        return True

    def _calculate_sample_allocation(self, N_U: int, actual_i_target: int, 
                                     n_eff_clusters: int, cluster_counts: np.ndarray) -> np.ndarray:
        num_samples = np.zeros(n_eff_clusters, dtype=int)
        for i in range(n_eff_clusters):
            s_j = cluster_counts[i]; n_jf = (s_j/N_U)*actual_i_target if N_U > 0 else 0
            alloc = max(min(self.min_samples_per_cluster,s_j), int(round(n_jf))); num_samples[i]=min(alloc,s_j)
        curr_sum = np.sum(num_samples); diff = actual_i_target-curr_sum
        if N_U > 0 and len(cluster_counts)>0:
            s_idxs = np.argsort(cluster_counts)[::-1] if len(cluster_counts) > 0 else []
            if diff > 0:
                for _ in range(diff):
                    added=False
                    for ci in s_idxs: 
                        if num_samples[ci]<cluster_counts[ci]: num_samples[ci]+=1;added=True;break
                    if not added: break
            elif diff < 0:
                for _ in range(abs(diff)):
                    rmvd=False
                    elig=[j for j in range(n_eff_clusters) if num_samples[j]>self.min_samples_per_cluster]
                    if not elig: break
                    idx_dec=max(elig,key=lambda j:num_samples[j],default=-1)
                    if idx_dec!=-1: num_samples[idx_dec]-=1;rmvd=True
                    if not rmvd: break
        return num_samples
        
    def _select_samples_one_sem_cluster(self, sem_labels_all, sem_id, pv_clust_idx, n_select, 
                                        cand_pos_idxs, U0_orig_idxs, L0_sel_orig_idxs, 
                                        U_sel_mask_pos, actual_i_target):
        if self.verbose: print(f"    Cluster Sem. {sem_id} (PV idx {pv_clust_idx}): Sel. {n_select} de {len(cand_pos_idxs)}.")
        A_col_tfidf = self.A_term_x_cluster_tfidf_[:, pv_clust_idx]
        q_x_cand_sparse = self.U0_binary_term_vectors_all_sparse_[cand_pos_idxs]
        relevance_scores = q_x_cand_sparse.dot(A_col_tfidf)
        if len(relevance_scores) == 0: return

        first_local_idx = np.argmax(relevance_scores)
        first_pos_idx = cand_pos_idxs[first_local_idx]
        L0_sel_orig_idxs.append(U0_orig_idxs[first_pos_idx])
        U_sel_mask_pos[first_pos_idx] = False
        template_bin = self.U0_binary_term_vectors_all_sparse_[first_pos_idx].toarray().flatten().astype(bool)

        for k_num in range(1, n_select):
            if len(L0_sel_orig_idxs) >= actual_i_target: break
            curr_pos_indices = np.where((sem_labels_all == sem_id) & U_sel_mask_pos)[0]
            if len(curr_pos_indices) == 0: break
            q_x_rem_sparse = self.U0_binary_term_vectors_all_sparse_[curr_pos_indices]
            best_cand_pos = -1; max_score_nov_pond = -1.0; template_inv = ~template_bin
            for loc_idx, cand_pos in enumerate(curr_pos_indices):
                q_x_bool = q_x_rem_sparse[loc_idx].toarray().flatten().astype(bool)
                novos_mask = q_x_bool & template_inv
                score_nov_pond = np.sum(A_col_tfidf[novos_mask])
                if score_nov_pond > max_score_nov_pond: max_score_nov_pond = score_nov_pond; best_cand_pos = cand_pos
            if best_cand_pos != -1:
                L0_sel_orig_idxs.append(U0_orig_idxs[best_cand_pos])
                U_sel_mask_pos[best_cand_pos] = False
                template_bin = template_bin | self.U0_binary_term_vectors_all_sparse_[best_cand_pos].toarray().flatten().astype(bool)
            else:
                if len(curr_pos_indices) > 0:
                    fall_pos = np.random.choice(curr_pos_indices) # Corrigido: usar curr_pos_indices
                    L0_sel_orig_idxs.append(U0_orig_idxs[fall_pos]); U_sel_mask_pos[fall_pos] = False
                else: break
    
    def _fill_remaining_samples_randomly(self, actual_i_target: int,
                                L0_sel_orig_idxs: list, U_sel_mask_pos: np.ndarray, 
                                U0_orig_idxs: np.ndarray):
        num_sel = len(L0_sel_orig_idxs)
        if num_sel < actual_i_target:
            if self.verbose: print(f"  DRI: 4. Preenchimento Final (Aleatório) para {actual_i_target - num_sel} vagas...")
            num_fill = actual_i_target - num_sel
            cand_pos = np.where(U_sel_mask_pos)[0]
            if len(cand_pos) > 0:
                actual_fill = min(num_fill, len(cand_pos))
                if actual_fill > 0 :
                    # Garantir que a semente aleatória aqui seja consistente se desejado, ou variar
                    np.random.seed(self.random_seed + len(L0_sel_orig_idxs)) # Exemplo de variação de seed
                    sel_pos = np.random.choice(cand_pos, size=actual_fill, replace=False)
                    for p_idx in sel_pos: L0_sel_orig_idxs.append(U0_orig_idxs[p_idx])

    def select_initial_samples(self, U0_texts: list[str], U0_indices: np.ndarray = None) -> tuple[list[str], np.ndarray]:
        overall_start_time = time.time()
        N_U = len(U0_texts); actual_i_target = self.i_target
        if U0_indices is None: U0_indices_internal = np.arange(N_U)
        else: U0_indices_internal = np.array(U0_indices)
        if N_U == 0 or actual_i_target == 0: return [], np.array([])
        if actual_i_target > N_U: 
            if self.verbose: print(f"DRI: Alvo L0 ({actual_i_target}) > |U0| ({N_U}). Retornando todo U0.")
            return U0_texts, U0_indices_internal
        if self.verbose: print(f"DRI: Iniciando. Alvo: {actual_i_target}, U0: {N_U} textos.")

        sem_labels, eff_sem_ids, sem_counts, n_sem_eff, sem_embeds = self._perform_semantic_clustering(U0_texts, N_U)
        if (n_sem_eff == 0 and N_U > 0) or (sem_embeds is None and N_U > 0) : 
            if self.verbose: print("DRI: Falha no clustering semântico inicial ou U0 tornou-se problemático.")
            return [], np.array([])
        if N_U == 0 and n_sem_eff == 0 : # Se U0 era vazio desde o início
             return [], np.array([])


        if not self._prepare_internal_product_vectorizer(U0_texts, sem_labels, eff_sem_ids):
            if self.verbose: print("DRI: Falha ao preparar ProductVectorizer interno. Retornando seleção vazia.")
            return [], np.array([])

        num_samples_per_cluster = self._calculate_sample_allocation(N_U, actual_i_target, n_sem_eff, sem_counts)
        if self.verbose: print(f"  DRI: Alocação de amostras por cluster semântico: {num_samples_per_cluster}")

        L0_selected_original_indices = [] 
        U_selectable_by_pos_mask = np.ones(N_U, dtype=bool)
        map_original_idx_to_pos = {val: pos for pos, val in enumerate(U0_indices_internal)}

        time_sel_intra = time.time()
        if self.verbose: print("  DRI: 3. Seleção Intra-Cluster Principal...")
        
        for i, sem_id_val in enumerate(eff_sem_ids): # i é o índice para A_term_x_cluster_tfidf_
            n_sel = num_samples_per_cluster[i]
            if len(L0_selected_original_indices) >= actual_i_target or n_sel == 0 : continue
            
            mask_in_sem_cluster = (sem_labels == sem_id_val)
            candidate_pos_indices = np.where(mask_in_sem_cluster & U_selectable_by_pos_mask)[0]
            if len(candidate_pos_indices) == 0: continue
            
            self._select_samples_one_sem_cluster(
                sem_labels, sem_id_val, i, 
                min(n_sel, len(candidate_pos_indices)), 
                candidate_pos_indices, U0_indices_internal,
                L0_selected_original_indices, U_selectable_by_pos_mask, 
                actual_i_target
            )
            if len(L0_selected_original_indices) >= actual_i_target: break
        if self.verbose: print(f"    Seleção intra-cluster principal concluída em {time.time() - time_sel_intra:.2f}s.")

        self._fill_remaining_samples_randomly(actual_i_target, L0_selected_original_indices, 
                                             U_selectable_by_pos_mask, U0_indices_internal)
        
        if len(L0_selected_original_indices) > actual_i_target:
            L0_selected_original_indices = L0_selected_original_indices[:actual_i_target]

        final_selected_positions = []
        for orig_idx in L0_selected_original_indices:
            pos = map_original_idx_to_pos.get(orig_idx)
            if pos is not None:
                final_selected_positions.append(pos)
            else:
                if self.verbose: print(f"AVISO DRI: Índice original {orig_idx} não encontrado no mapeamento para posição.")
        
        L0_final_texts = [U0_texts[pos_idx] for pos_idx in final_selected_positions]
        
        if self.verbose: print(f"DRI: Seleção FINAL concluída. Total: {len(L0_selected_original_indices)}. Tempo: {time.time() - overall_start_time:.2f}s.")
        return L0_final_texts, np.array(L0_selected_original_indices, dtype=object)