import torch
import torch.nn.functional as F
import numpy as np
import hdbscan
from functools import reduce
from sklearn.cluster import KMeans

def fedavg(model_updates, data_sizes, device):
    total = sum(data_sizes)
    global_update = {
        k: torch.zeros_like(model_updates[0][k], dtype=torch.float32).to(device)
        for k in model_updates[0]
    }
    for i, update in enumerate(model_updates):
        weight = data_sizes[i] / total
        for k in global_update:
            global_update[k] += weight * update[k]
    return global_update


def trim_mean(local_updates, device, f):
    n = len(local_updates)

    updates_vectorized = torch.stack([
        torch.cat([local_updates[i][k].view(-1) for k in sorted(local_updates[i].keys())])
        for i in range(n)
    ]).to(device)

    sorted_values, _ = torch.sort(updates_vectorized, dim=0)
    global_update_vector = torch.mean(sorted_values[f:(n - f)], dim=0)

    reshaped_update = {}
    idx = 0
    for key in sorted(local_updates[0].keys()):
        num_elements = torch.numel(local_updates[0][key])
        reshaped_update[key] = global_update_vector[idx: idx + num_elements].reshape(local_updates[0][key].shape)
        idx += num_elements

    return reshaped_update


def median(local_updates, device):
    n = len(local_updates)

    updates_vectorized = torch.stack([
        torch.cat([local_updates[i][k].view(-1) for k in sorted(local_updates[i].keys())])
        for i in range(n)
    ]).to(device)

    global_update_vector, _ = torch.median(updates_vectorized, dim=0)

    reshaped_update = {}
    idx = 0
    for key in sorted(local_updates[0].keys()):
        num_elements = torch.numel(local_updates[0][key])
        reshaped_update[key] = global_update_vector[idx: idx + num_elements].reshape(local_updates[0][key].shape)
        idx += num_elements

    return reshaped_update

def krum(local_updates, device, f):
    n = len(local_updates)
    if n <= 2 * f:
        raise ValueError("Krum requires at least 2*f + 3 clients.")

    updates_vectorized = torch.stack([
        torch.cat([local_updates[i][k].view(-1) for k in sorted(local_updates[i].keys())])
        for i in range(n)
    ]).to(device)

    dist_matrix = torch.cdist(updates_vectorized, updates_vectorized, p=2)
    sorted_distances, _ = torch.sort(dist_matrix, dim=-1)
    scores = torch.sum(sorted_distances[:, : (n - f - 1)], dim=-1)
    selected_idx = torch.argmin(scores).item()

    selected_vector = updates_vectorized[selected_idx]
    reshaped_update = {}
    idx = 0
    for key in sorted(local_updates[0].keys()):
        num_elements = torch.numel(local_updates[0][key])
        reshaped_update[key] = selected_vector[idx: idx + num_elements].reshape(local_updates[0][key].shape)
        idx += num_elements

    print(f"[Krum] Selected Client Index: {selected_idx}")
    print(f"[Krum] Score: {scores[selected_idx].item():.4f}")
    print(f"[Krum] All Scores: {[round(s.item(), 4) for s in scores]}")

    return reshaped_update


def krum(local_updates, device, f):
    n = len(local_updates)
    if n <= 2 * f:
        raise ValueError("Krum requires at least 2*f + 3 clients.")

    updates_vectorized = torch.stack([
        torch.cat([local_updates[i][k].view(-1) for k in sorted(local_updates[i].keys())])
        for i in range(n)
    ]).to(device)

    dist = torch.cdist(updates_vectorized, updates_vectorized, p=2)
    sorted_dist, _ = torch.sort(dist, dim=-1)
    scores = torch.sum(sorted_dist[:, : (n - f - 1)], dim=-1)

    selected_idx = torch.argmin(scores)
    global_update_vector = updates_vectorized[selected_idx]

    reshaped_update = {}
    idx = 0
    for key in sorted(local_updates[0].keys()):
        num_elements = torch.numel(local_updates[0][key])
        reshaped_update[key] = global_update_vector[idx: idx + num_elements].reshape(local_updates[0][key].shape)
        idx += num_elements

    return reshaped_update


def flod(local_updates, device, threshold):
    n = len(local_updates) - 1

    updates_vectorized = torch.stack([
        torch.cat([local_updates[i][k].view(-1) for k in sorted(local_updates[i].keys())])
        for i in range(n + 1)
    ]).to(device)

    sgn_param_list = torch.sign(updates_vectorized)
    bool_param_list = (sgn_param_list == 1)

    hd = torch.tensor([torch.sum(torch.bitwise_xor(bool_param_list[i], bool_param_list[-1])) for i in range(n)]).to(device)
    weight = F.relu(threshold - hd)

    if torch.sum(weight) == 0:  # Avoid division by zero
        weight = torch.ones_like(weight)

    global_update_vector = torch.zeros_like(updates_vectorized[0]).to(device)
    for i in range(n):
        global_update_vector += weight[i] * updates_vectorized[i]

    weight_sum = torch.sum(weight)
    if weight_sum > 0:
        global_update_vector /= weight_sum

    reshaped_update = {}
    idx = 0
    for key in sorted(local_updates[0].keys()):
        num_elements = torch.numel(local_updates[0][key])
        reshaped_update[key] = global_update_vector[idx: idx + num_elements].reshape(local_updates[0][key].shape)
        idx += num_elements

    return reshaped_update


def signguard(local_updates, device, seed):
    n = len(local_updates)

    updates_vectorized = torch.stack([
        torch.cat([local_updates[i][k].view(-1) for k in sorted(local_updates[i].keys())])
        for i in range(n)
    ]).to(device)

    num_params = updates_vectorized.shape[1]
    selection_fraction = 0.1

    L, R = 0.1, 3.0
    S1, S2 = [], []

    l2_norm = torch.stack([torch.norm(updates_vectorized[i], p=2.0) for i in range(n)])

    num_selection = int(num_params * selection_fraction)
    perm = torch.randperm(num_params)
    idx = perm[:num_selection]

    sign_grads = torch.stack([torch.sign(updates_vectorized[i][idx]) for i in range(n)])

    M = torch.median(l2_norm)
    for i in range(n):
        if L <= l2_norm[i] / M <= R:
            S1.append(i)

    sign_pos = torch.stack([grad.eq(1.0).float().mean() for grad in sign_grads])
    sign_zero = torch.stack([grad.eq(0.0).float().mean() for grad in sign_grads])
    sign_neg = torch.stack([grad.eq(-1.0).float().mean() for grad in sign_grads])

    sign_feat = torch.stack([sign_pos, sign_zero, sign_neg], dim=1).cpu().numpy()
    cluster = KMeans(n_clusters=2, max_iter=20, random_state=seed)
    labels = cluster.fit_predict(sign_feat)

    labels_tensor = torch.from_numpy(labels).to(device)
    count = torch.bincount(labels_tensor)

    if len(count) > 1:
        largest_cluster = torch.argmax(count)
        for i, value in enumerate(labels_tensor):
            if value == largest_cluster:
                S2.append(i)

    S = [i for i in S1 if i in S2]

    aggregated_update = {k: torch.zeros_like(local_updates[0][k]) for k in local_updates[0].keys()}
    if len(S) > 0:
        for idx in S:
            for k in aggregated_update.keys():
                aggregated_update[k] += local_updates[idx][k] / len(S)

    return aggregated_update

# def bartfl(local_updates, device, num_clusters=2):
#     n = len(local_updates)
#
#     updates_vectorized = torch.stack([
#         torch.cat([update[k].view(-1) for k in update.keys()]) for update in local_updates
#     ]).to(device)
#
#     cosine_sim = F.cosine_similarity(updates_vectorized.unsqueeze(1), updates_vectorized.unsqueeze(0), dim=2)
#     cosine_sim_np = cosine_sim.cpu().numpy()
#
#     kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
#     clusters = kmeans.fit_predict(cosine_sim_np)
#
#     client_mads = np.mean(np.abs(cosine_sim_np - np.mean(cosine_sim_np, axis=1, keepdims=True)), axis=1)
#     client_means = np.mean(cosine_sim_np, axis=1)
#
#     cluster_stats = {}
#     cluster_sizes = {}
#     cluster_avg_scores = {}
#     for c in range(num_clusters):
#         indices = np.where(clusters == c)[0]
#         cluster_sizes[c] = len(indices)
#         avg_cosine_score = np.mean(cosine_sim_np[indices][:, indices])
#         cluster_avg_scores[c] = avg_cosine_score
#
#         cluster_stats[c] = {
#             'mad': np.mean(client_mads[indices]),
#             'mean': np.mean(client_means[indices]),
#             'avg_cosine_score': avg_cosine_score
#         }
#
#     votes = {c: 0 for c in range(num_clusters)}
#     for metric in ['mad', 'mean']:
#         min_cluster = min(cluster_stats, key=lambda c: cluster_stats[c][metric])
#         votes[min_cluster] += 1
#
#     min_avg_cosine_cluster = min(cluster_avg_scores, key=lambda c: cluster_avg_scores[c])
#     votes[min_avg_cosine_cluster] += 1
#
#     final_benign_cluster = max(votes, key=votes.get)
#
#     benign_clients = [i for i in range(n) if clusters[i] == final_benign_cluster]
#     malicious_clients = [i for i in range(n) if i not in benign_clients]
#
#     # aggregated_update = {k: torch.zeros_like(local_updates[0][k]) for k in local_updates[0].keys()}
#     aggregated_update = {k: torch.zeros_like(local_updates[0][k], dtype=torch.float32) for k in local_updates[0].keys()}
#
#     for idx in benign_clients:
#         for k in aggregated_update.keys():
#             aggregated_update[k] += local_updates[idx][k] / len(benign_clients)
#
#     print(f"\nCluster Statistics:")
#     for c, stats in cluster_stats.items():
#         print(f"Cluster {c}: {stats}, Size: {cluster_sizes[c]}")
#     print(f"Votes per Cluster: {votes}")
#     print(f"Selected Benign Cluster: {final_benign_cluster}")
#     print(f"Benign Clients (Aggregated): {benign_clients}")
#     print(f"Malicious Clients (Excluded): {malicious_clients}")
#
#     return aggregated_update
#
# def bartfl(local_updates, device, num_clusters=2, ground_truth=None):
#
#     n = len(local_updates)
#
#     # Vectorize the local updates
#     updates_vectorized = torch.stack([
#         torch.cat([update[k].view(-1) for k in update.keys()]) for update in local_updates
#     ]).to(device)
#
#     # Compute cosine similarity matrix
#     cosine_sim = F.cosine_similarity(updates_vectorized.unsqueeze(1), updates_vectorized.unsqueeze(0), dim=2)
#     cosine_sim_np = cosine_sim.cpu().numpy()
#
#     # Apply KMeans clustering
#     kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
#     clusters = kmeans.fit_predict(cosine_sim_np)
#
#     # Calculate Mean Absolute Deviation (MAD) and mean cosine similarity
#     client_mads = np.mean(np.abs(cosine_sim_np - np.mean(cosine_sim_np, axis=1, keepdims=True)), axis=1)
#     client_means = np.mean(cosine_sim_np, axis=1)
#
#     cluster_stats = {}
#     cluster_sizes = {}
#     cluster_avg_scores = {}
#     for c in range(num_clusters):
#         indices = np.where(clusters == c)[0]
#         cluster_sizes[c] = len(indices)
#         avg_cosine_score = np.mean(cosine_sim_np[indices][:, indices])
#         cluster_avg_scores[c] = avg_cosine_score
#
#         cluster_stats[c] = {
#             'mad': np.mean(client_mads[indices]),
#             'mean': np.mean(client_means[indices]),
#             'avg_cosine_score': avg_cosine_score
#         }
#
#     # Voting mechanism to select the benign cluster
#     votes = {c: 0 for c in range(num_clusters)}
#     for metric in ['mad', 'mean']:
#         min_cluster = min(cluster_stats, key=lambda c: cluster_stats[c][metric])
#         votes[min_cluster] += 1
#
#     min_avg_cosine_cluster = min(cluster_avg_scores, key=lambda c: cluster_avg_scores[c])
#     votes[min_avg_cosine_cluster] += 1
#
#     final_benign_cluster = max(votes, key=votes.get)
#
#     # Identify benign and malicious clients
#     benign_clients = [i for i in range(n) if clusters[i] == final_benign_cluster]
#     malicious_clients = [i for i in range(n) if i not in benign_clients]
#
#     # Aggregated update calculation
#     aggregated_update = {k: torch.zeros_like(local_updates[0][k], dtype=torch.float32) for k in local_updates[0].keys()}
#     for idx in benign_clients:
#         for k in aggregated_update.keys():
#             aggregated_update[k] += local_updates[idx][k] / len(benign_clients)
#
#     # Calculate Malicious Detection Accuracy if Ground Truth is Provided
#     accuracy = None
#     if ground_truth is not None:
#         true_malicious = [i for i, label in enumerate(ground_truth) if label == 1]
#         true_benign = [i for i, label in enumerate(ground_truth) if label == 0]
#
#         tp = len(set(malicious_clients).intersection(true_malicious))  # True Positive
#         tn = len(set(benign_clients).intersection(true_benign))        # True Negative
#         fp = len(set(malicious_clients).intersection(true_benign))     # False Positive
#         fn = len(set(benign_clients).intersection(true_malicious))     # False Negative
#
#         accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else None
#
#     return aggregated_update, accuracy, malicious_clients

###########################Final result generated using following#####################



from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch

def bartfl(local_updates, device, num_clusters=2, ground_truth=None, pca_dim=0.95):  # now default: retain 95% variance
    n = len(local_updates)

    # Step 1: Flatten and vectorize updates
    updates_vectorized = torch.stack([
        torch.cat([update[k].view(-1) for k in sorted(update.keys())])
        for update in local_updates
    ]).cpu().float()

    updates_np = updates_vectorized.numpy()
    n_samples, n_features = updates_np.shape
    max_dim = min(n_samples, n_features)

    # Step 2: Memory optimization using PCA (retain high variance)
    use_pca = pca_dim is not None and n_samples > 5
    try:
        if use_pca:
            if isinstance(pca_dim, float) and 0 < pca_dim < 1:
                pca = PCA(n_components=pca_dim, random_state=42)
                updates_np = pca.fit_transform(updates_np)
                print(f"[BART-FL] PCA applied (variance={pca_dim}) → {pca.n_components_} components")
            elif isinstance(pca_dim, int) and pca_dim < max_dim:
                pca = PCA(n_components=pca_dim, random_state=42)
                updates_np = pca.fit_transform(updates_np)
                print(f"[BART-FL] PCA applied with n_components={pca_dim}")
            elif isinstance(pca_dim, int) and pca_dim >= max_dim:
                pca = PCA(n_components=max_dim, random_state=42)
                updates_np = pca.fit_transform(updates_np)
                print(f"[BART-FL] PCA applied with n_components={max_dim}")
        else:
            print("[BART-FL] PCA skipped (disabled or insufficient samples)")
    except Exception as e:
        print(f"[BART-FL] PCA failed: {e}")
        use_pca = False

    # Step 3: Cosine similarity
    cosine_sim_np = cosine_similarity(updates_np.astype(np.float32))
    mean_sim = np.mean(cosine_sim_np)
    std_sim = np.std(cosine_sim_np)

    # Optional: detect PCA overcompression
    if use_pca and std_sim < 0.01:
        print("[BART-FL] Warning: PCA collapsed similarity structure. Reverting to no PCA.")
        updates_np = updates_vectorized.numpy()  # restore original
        cosine_sim_np = cosine_similarity(updates_np.astype(np.float32))

    # Step 4: KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=5, max_iter=10)
    clusters = kmeans.fit_predict(cosine_sim_np)

    # Step 5: Cluster statistics
    client_mads = np.mean(np.abs(cosine_sim_np - cosine_sim_np.mean(axis=1, keepdims=True)), axis=1)
    client_means = np.mean(cosine_sim_np, axis=1)

    cluster_stats, cluster_sizes, cluster_avg_scores = {}, {}, {}
    for c in range(num_clusters):
        indices = np.where(clusters == c)[0]
        sim_block = cosine_sim_np[np.ix_(indices, indices)]
        cluster_sizes[c] = len(indices)
        cluster_avg_scores[c] = sim_block.mean()
        cluster_stats[c] = {
            'mad': np.mean(client_mads[indices]),
            'mean': np.mean(client_means[indices]),
            'avg_cosine_score': cluster_avg_scores[c]
        }

    # Step 6: Voting mechanism
    votes = {c: 0 for c in range(num_clusters)}
    for metric in ['mad', 'mean']:
        min_cluster = min(cluster_stats, key=lambda c: cluster_stats[c][metric])
        votes[min_cluster] += 1
    min_cosine_cluster = min(cluster_avg_scores, key=cluster_avg_scores.get)
    votes[min_cosine_cluster] += 1

    final_benign_cluster = max(votes, key=votes.get)
    benign_clients = [i for i in range(n) if clusters[i] == final_benign_cluster]
    malicious_clients = [i for i in range(n) if i not in benign_clients]

    # Step 7: Aggregate benign client updates
    aggregated_update = {
        k: torch.zeros_like(local_updates[0][k], dtype=torch.float32)
        for k in local_updates[0].keys()
    }
    for idx in benign_clients:
        for k in aggregated_update:
            aggregated_update[k] += local_updates[idx][k] / len(benign_clients)

    # Step 8: Evaluate clustering accuracy
    accuracy = None
    if ground_truth is not None:
        true_malicious = [i for i, label in enumerate(ground_truth) if label == 1]
        true_benign = [i for i, label in enumerate(ground_truth) if label == 0]
        tp = len(set(malicious_clients) & set(true_malicious))
        tn = len(set(benign_clients) & set(true_benign))
        fp = len(set(malicious_clients) & set(true_benign))
        fn = len(set(benign_clients) & set(true_malicious))
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else None

    # Print cluster info
    print(f"\n[BART-FL] Cluster Statistics:")
    for c, stats in cluster_stats.items():
        print(f"Cluster {c}: {stats}, Size: {cluster_sizes[c]}")
    print(f"[BART-FL] Votes: {votes}, Benign Cluster: {final_benign_cluster}")
    print(f"[BART-FL] Benign Clients: {benign_clients}")
    print(f"[BART-FL] Malicious Clients: {malicious_clients}")

    return aggregated_update, accuracy, malicious_clients

