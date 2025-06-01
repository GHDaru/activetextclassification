# activetextclassification/cold_start/dri_cluster.py

import numpy as np
import time
import csv # Para o log detalhado
import os  # Para checar existência do arquivo de log

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
                 semantic_embedder, # Objeto com método .encode() para embeddings semânticos
                 n_clusters_semantic: int, # Para o KMeans inicial
                 random_seed: int = 42,
                 min_samples_per_cluster: int = 1,
                 verbose: bool = False,
                 detailed_log_path: str = None, 
                 run_id: str = "run_0"):
        
        self.i_target = i_target
        self.semantic_embedder = semantic_embedder # Usado para o K-Means inicial
        self.verbose = verbose
        self.random_seed = random_seed
        self.detailed_log_path = detailed_log_path
        self.run_id = run_id

        self.semantic_clustering_algo = KMeans(
            n_clusters=n_clusters_semantic, 
            random_state=self.random_seed,
            n_init='auto'
        )
        self.n_clusters_semantic_requested = n_clusters_semantic
        
        # Configuração para o ProductVectorizer que será instanciado e treinado internamente
        self._internal_pv_config = {
            'method': 'tfidf', 
            'norm': 'l2', 
            'query': 'binary', 
            'query_norm': None, 
            'ngram_range': (1,1)
        }
        
        self.min_samples_per_cluster = min_samples_per_cluster
        
        # Atributos que serão populados durante a execução
        self.relevance_pv_instance_ = None 
        self.A_term_x_cluster_tfidf_ = None 
        self.U0_binary_term_vectors_all_sparse_ = None
        self._detailed_log_data_accumulator = [] 

        if self.verbose: print(f"DRIClusterColdStart inicializado. Alvo L0: {i_target}, Clusters Semânticos Solicitados: {n_clusters_semantic}, Log: {detailed_log_path}")

    def _init_detailed_log_file_if_needed(self):
        """Cria o arquivo de log com cabeçalho se não existir ou estiver vazio."""
        if self.detailed_log_path:
            # Verificar se o arquivo existe e está vazio para decidir sobre o header
            write_header = not os.path.exists(self.detailed_log_path) or os.path.getsize(self.detailed_log_path) == 0
            if write_header:
                try:
                    # Garantir que o diretório existe
                    log_dir = os.path.dirname(self.detailed_log_path)
                    if log_dir and not os.path.exists(log_dir):
                        os.makedirs(log_dir, exist_ok=True)
                    
                    with open(self.detailed_log_path, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            "run_id", "l0_target_size", "selection_order_in_l0", 
                            "original_index", "text_sample", "true_label_sample", 
                            "assigned_semantic_cluster_id", "selection_reason_code", "selection_score_value"
                        ])
                    if self.verbose: print(f"  DRI: Arquivo de log detalhado '{self.detailed_log_path}' inicializado com cabeçalho.")
                except Exception as e:
                    if self.verbose: print(f"  DRI: ERRO ao inicializar arquivo de log detalhado: {e}")
    
    def _log_selected_sample_detail(self, order_in_l0, orig_idx, text_val, label_val, sem_clust_id, reason_code, score_val):
        if self.detailed_log_path:
            self._detailed_log_data_accumulator.append({
                "run_id": self.run_id,
                "l0_target_size": self.i_target,
                "selection_order_in_l0": order_in_l0,
                "original_index": orig_idx,
                "text_sample": text_val,
                "true_label_sample": label_val,
                "assigned_semantic_cluster_id": sem_clust_id,
                "selection_reason_code": reason_code,
                "selection_score_value": score_val if score_val is not None and np.isfinite(score_val) else "" # Deixar em branco se NaN
            })

    def _flush_detailed_log_to_file(self):
        if self.detailed_log_path and self._detailed_log_data_accumulator:
            # Checar se o header é necessário novamente pode ser redundante se _init_detailed_log_file_if_needed já rodou,
            # mas é uma segurança se múltiplas execuções do select_initial_samples usarem o mesmo objeto DRI sem reinstanciar
            # e o arquivo for deletado entre elas.
            file_exists_and_has_content = os.path.exists(self.detailed_log_path) and os.path.getsize(self.detailed_log_path) > 0
            
            with open(self.detailed_log_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self._detailed_log_data_accumulator[0].keys())
                if not file_exists_and_has_content:
                    writer.writeheader()
                writer.writerows(self._detailed_log_data_accumulator)
            if self.verbose: print(f"  DRI: {len(self._detailed_log_data_accumulator)} registros de seleção detalhados salvos em {self.detailed_log_path}")
            self._detailed_log_data_accumulator = [] # Limpar acumulador

    def _perform_semantic_clustering(self, U0_texts: list[str], N_U: int) -> tuple:
        time_start = time.time()
        if self.verbose: print("  DRI: 1. Clustering Semântico (K-Means)...")
        try: embeddings = self.semantic_embedder.encode(U0_texts, show_progress_bar=self.verbose)
        except TypeError: embeddings = self.semantic_embedder.encode(U0_texts)
        if N_U == 0 or embeddings is None or embeddings.shape[0] != N_U : 
            if self.verbose: print(f"    KMeans: Problema com embeddings (shape {embeddings.shape if embeddings is not None else 'None'}) ou U0_texts vazio. N_U={N_U}.")
            return np.array([]), np.array([]), np.array([]), 0, None
        current_n_clusters = self.n_clusters_semantic_requested
        if N_U < current_n_clusters: current_n_clusters = N_U
        if N_U > 0 and current_n_clusters == 0 : current_n_clusters = 1
        if current_n_clusters == 0: return np.array([]), np.array([]), np.array([]), 0, embeddings
        
        self.semantic_clustering_algo.set_params(n_clusters=current_n_clusters) # Definir n_clusters no objeto KMeans
        labels = self.semantic_clustering_algo.fit_predict(embeddings)
        ids, counts = np.unique(labels, return_counts=True)
        n_effective = len(ids)
        if self.verbose: print(f"    K-Means concluído em {time.time() - time_start:.2f}s. {n_effective} clusters semânticos.")
        return labels, ids, counts, n_effective, embeddings

    def _prepare_internal_product_vectorizer(self, U0_texts: list[str], 
                                             semantic_cluster_labels: np.ndarray, 
                                             effective_semantic_cluster_ids: np.ndarray) -> bool:
        time_start = time.time()
        if self.verbose: print("  DRI: 2. Preparando ProductVectorizer interno...")
        semantic_cluster_documents_texts = []
        map_eff_sem_id_to_pv_idx = {}
        for i, eff_sem_clust_id_val in enumerate(effective_semantic_cluster_ids):
            texts = [U0_texts[pos_idx] for pos_idx, label in enumerate(semantic_cluster_labels) if label == eff_sem_clust_id_val]
            semantic_cluster_documents_texts.append(" ".join(texts) if texts else "")
            map_eff_sem_id_to_pv_idx[eff_sem_clust_id_val] = i 
        if not any(semantic_cluster_documents_texts): 
            if self.verbose: print("    ERRO: Nenhum documento de cluster não vazio para PV interno."); return False
        
        self.relevance_pv_instance_ = ProductVectorizer(**self._internal_pv_config)
        docs_for_fit = [doc for doc in semantic_cluster_documents_texts if doc]
        pv_labels_for_fit_str = [str(map_eff_sem_id_to_pv_idx[eff_id]) for eff_id in effective_semantic_cluster_ids if semantic_cluster_documents_texts[map_eff_sem_id_to_pv_idx[eff_id]]]
        if not docs_for_fit: 
            if self.verbose: print("    ERRO: Nenhum documento de cluster para FIT do PV interno."); return False
        
        self.relevance_pv_instance_.fit(X=docs_for_fit, y=pv_labels_for_fit_str) # y precisa ser lista de strings ou num. consistentes
        self.A_term_x_cluster_tfidf_ = self.relevance_pv_instance_.vectorizer.transform(semantic_cluster_documents_texts).T.toarray()
        self.U0_binary_term_vectors_all_sparse_ = self.relevance_pv_instance_.query.transform(U0_texts)
        
        if self.verbose: print(f"    PV Interno pronto em {time.time() - time_start:.2f}s. Matriz A: {self.A_term_x_cluster_tfidf_.shape}, Matriz q_x: {self.U0_binary_term_vectors_all_sparse_.shape}")
        if self.A_term_x_cluster_tfidf_.shape[0] != self.U0_binary_term_vectors_all_sparse_.shape[1]:
            print("ERRO CRÍTICO DRI: Incompatibilidade de vocabulário PV (A vs q_x)."); return False
        return True

    def _calculate_sample_allocation(self, N_U: int, actual_i_target: int, 
                                     n_eff_clusters: int, cluster_counts: np.ndarray) -> np.ndarray:
        num_samples = np.zeros(n_eff_clusters, dtype=int)
        for i in range(n_eff_clusters):
            s_j = cluster_counts[i]; n_jf = (s_j/N_U)*actual_i_target if N_U > 0 else 0
            alloc = max(min(self.min_samples_per_cluster,s_j), int(round(n_jf))); num_samples[i]=min(alloc,s_j)
        curr_sum = np.sum(num_samples); diff = actual_i_target-curr_sum
        if N_U > 0 and len(cluster_counts)>0 and n_eff_clusters > 0: # Adicionado n_eff_clusters > 0
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
                    idx_to_dec=max(elig,key=lambda j:num_samples[j],default=-1)
                    if idx_to_dec!=-1: num_samples[idx_to_dec]-=1;rmvd=True
                    if not rmvd: break
        return num_samples
        
    def _select_samples_one_sem_cluster(self, U0_texts, U0_labels, sem_labels_all, sem_id, pv_clust_idx, n_select, 
                                        cand_pos_idxs, U0_orig_idxs, L0_sel_orig_idxs, 
                                        U_sel_mask_pos, actual_i_target, selection_count_global_ref):
        if self.verbose: print(f"    Cluster Sem. {sem_id} (PV idx {pv_clust_idx}): Sel. {n_select} de {len(cand_pos_idxs)}.")
        A_col_tfidf = self.A_term_x_cluster_tfidf_[:, pv_clust_idx]
        q_x_cand_sparse = self.U0_binary_term_vectors_all_sparse_[cand_pos_idxs]
        relevance_scores = q_x_cand_sparse.dot(A_col_tfidf)
        if len(relevance_scores) == 0: return

        first_local_idx = np.argmax(relevance_scores)
        first_pos_idx = cand_pos_idxs[first_local_idx]
        
        L0_sel_orig_idxs.append(U0_orig_idxs[first_pos_idx])
        U_sel_mask_pos[first_pos_idx] = False
        selection_count_global_ref[0] += 1
        self._log_selected_sample_detail(selection_count_global_ref[0], U0_orig_idxs[first_pos_idx], U0_texts[first_pos_idx], U0_labels[first_pos_idx], sem_id, "first_relevance", relevance_scores[first_local_idx])
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
            
            reason_sel = "intra_novelty_relevance"; score_sel = max_score_nov_pond
            if best_cand_pos == -1: 
                if len(curr_pos_indices) > 0: best_cand_pos = np.random.choice(curr_pos_indices); reason_sel = "intra_fallback_random"; score_sel = np.nan
                else: break
            
            L0_sel_orig_idxs.append(U0_orig_idxs[best_cand_pos])
            U_sel_mask_pos[best_cand_pos] = False
            selection_count_global_ref[0] += 1
            self._log_selected_sample_detail(selection_count_global_ref[0], U0_orig_idxs[best_cand_pos], U0_texts[best_cand_pos], U0_labels[best_cand_pos], sem_id, reason_sel, score_sel)
            if reason_sel != "intra_fallback_random": template_bin = template_bin | self.U0_binary_term_vectors_all_sparse_[best_cand_pos].toarray().flatten().astype(bool)
    
    def _fill_remaining_samples_randomly(self, actual_i_target: int, L0_sel_orig_idxs: list, 
                                        U_sel_mask_pos: np.ndarray, U0_orig_idxs: np.ndarray, 
                                        U0_texts, U0_labels, selection_count_global_ref):
        num_sel = len(L0_sel_orig_idxs)
        if num_sel < actual_i_target:
            if self.verbose: print(f"  DRI: 4. Preenchimento Final (Aleatório) para {actual_i_target - num_sel} vagas...")
            num_fill = actual_i_target - num_sel
            cand_pos = np.where(U_sel_mask_pos)[0]
            if len(cand_pos) > 0:
                actual_fill = min(num_fill, len(cand_pos))
                if actual_fill > 0 :
                    np.random.seed(self.random_seed + len(L0_sel_orig_idxs))
                    sel_pos_indices = np.random.choice(cand_pos, size=actual_fill, replace=False)
                    for p_idx in sel_pos_indices: 
                        L0_sel_orig_idxs.append(U0_orig_idxs[p_idx])
                        U_sel_mask_pos[p_idx] = False 
                        selection_count_global_ref[0] += 1
                        self._log_selected_sample_detail(selection_count_global_ref[0], U0_orig_idxs[p_idx], U0_texts[p_idx], U0_labels[p_idx], -1, "random_fill", np.nan)

    def select_initial_samples(self, U0_texts: list[str], U0_labels: list[str], 
                               U0_indices: np.ndarray = None) -> tuple[list[str], np.ndarray]:
        self._detailed_log_data_accumulator = [] 
        if self.detailed_log_path: self._init_detailed_log_file_if_needed()

        overall_start_time = time.time()
        N_U = len(U0_texts); actual_i_target = self.i_target
        if U0_indices is None: U0_indices_internal = np.arange(N_U)
        else: U0_indices_internal = np.array(U0_indices)
        if N_U == 0 or actual_i_target == 0: self._flush_detailed_log_to_file(); return [], np.array([])
        if len(U0_labels) != N_U: raise ValueError("U0_texts e U0_labels devem ter o mesmo tamanho.")
        if actual_i_target > N_U: 
            if self.verbose: print(f"DRI: Alvo L0 ({actual_i_target}) > |U0| ({N_U}). Retornando todo U0.")
            for i_all in range(N_U): self._log_selected_sample_detail(i_all+1, U0_indices_internal[i_all], U0_texts[i_all], U0_labels[i_all], -2, "all_samples_selected_due_to_target_gt_N_U", np.nan)
            self._flush_detailed_log_to_file(); return U0_texts, U0_indices_internal
        if self.verbose: print(f"DRI: Iniciando. Alvo: {actual_i_target}, U0: {N_U} textos.")

        sem_labels, eff_sem_ids, sem_counts, n_sem_eff, _ = self._perform_semantic_clustering(U0_texts, N_U)
        if (n_sem_eff == 0 and N_U > 0) : 
            if self.verbose: print("DRI: Falha no clustering semântico ou U0 problemático."); self._flush_detailed_log_to_file(); return [], np.array([])
        if N_U > 0 and n_sem_eff == 0: self._flush_detailed_log_to_file(); return [], np.array([])
        if N_U == 0 and n_sem_eff == 0 : self._flush_detailed_log_to_file(); return [], np.array([])

        if not self._prepare_internal_product_vectorizer(U0_texts, sem_labels, eff_sem_ids):
            self._flush_detailed_log_to_file(); return [], np.array([])

        num_samples_per_cluster = self._calculate_sample_allocation(N_U, actual_i_target, n_sem_eff, sem_counts)
        if self.verbose: print(f"  DRI: Alocação por cluster semântico: {num_samples_per_cluster}")

        L0_selected_original_indices = [] 
        U_selectable_by_pos_mask = np.ones(N_U, dtype=bool)
        map_original_idx_to_pos = {val: pos for pos, val in enumerate(U0_indices_internal)}
        selection_count_ref = [0] 

        time_sel_intra = time.time()
        if self.verbose: print("  DRI: 3. Seleção Intra-Cluster Principal...")
        for i, sem_id_val in enumerate(eff_sem_ids):
            n_sel = num_samples_per_cluster[i]
            if len(L0_selected_original_indices) >= actual_i_target or n_sel == 0 : continue
            mask_in_sem_cluster = (sem_labels == sem_id_val)
            candidate_pos_indices = np.where(mask_in_sem_cluster & U_selectable_by_pos_mask)[0]
            if len(candidate_pos_indices) == 0: continue
            self._select_samples_one_sem_cluster(U0_texts, U0_labels, sem_labels, sem_id_val, i, 
                                                 min(n_sel, len(candidate_pos_indices)), 
                                                 candidate_pos_indices, U0_indices_internal,
                                                 L0_selected_original_indices, U_selectable_by_pos_mask, 
                                                 actual_i_target, selection_count_ref)
            if len(L0_selected_original_indices) >= actual_i_target: break
        if self.verbose: print(f"    Seleção intra-cluster concluída em {time.time() - time_sel_intra:.2f}s.")

        self._fill_remaining_samples_randomly(actual_i_target, L0_selected_original_indices, 
                                             U_selectable_by_pos_mask, U0_indices_internal,
                                             U0_texts, U0_labels, selection_count_ref)
        
        if len(L0_selected_original_indices) > actual_i_target:
            L0_selected_original_indices = L0_selected_original_indices[:actual_i_target]
            # O log pode ter mais entradas se o truncamento ocorreu APÓS o log das extras.
            # Para ser preciso, o log deveria ser truncado também, ou não logar extras.
            # Por simplicidade, o log pode ter algumas entradas a mais que o L0 final se houver truncamento.

        final_selected_positions = []
        for orig_idx in L0_selected_original_indices:
            pos = map_original_idx_to_pos.get(orig_idx)
            if pos is not None: 
                final_selected_positions.append(pos)
            else: 
                if self.verbose: 
                    print(f"AVISO DRI: Índice original {orig_idx} não encontrado no mapeamento.")
        
        L0_final_texts = [U0_texts[pos_idx] for pos_idx in final_selected_positions]
        
        self._flush_detailed_log_to_file() 
        if self.verbose: 
            print(f"DRI: Seleção FINAL concluída. Total: {len(L0_selected_original_indices)}. Tempo: {time.time() - overall_start_time:.2f}s.")
        return L0_final_texts, np.array(L0_selected_original_indices, dtype=object)