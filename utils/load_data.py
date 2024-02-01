import numpy as np
from collections import defaultdict
import scipy.sparse as sp
import torch as th
from sklearn.preprocessing import OneHotEncoder
from torch_geometric.data import HeteroData
import random

data_folder = "data/"


def encode_onehot(labels):
    labels = labels.reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(labels)
    labels_onehot = enc.transform(labels).toarray()
    return labels_onehot


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    if isinstance(features, np.ndarray):
        return features
    else:
        return features.todense()


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = th.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = th.from_numpy(sparse_mx.data)
    shape = th.Size(sparse_mx.shape)
    return th.sparse.FloatTensor(indices, values, shape)


def process_data_in_pyg(neigs):
    d = defaultdict(dict)
    metapaths = []
    for mp_i, nei1 in enumerate(neigs):
        dst_array_concat = np.concatenate(nei1)
        src_array_concat = []
        for src_id, dst_array in enumerate(nei1):
            src_array_concat.extend([src_id] * len(dst_array))
        src_array_concat = np.array(src_array_concat)
        src_name = f"target"
        dst_name = f"dst_{mp_i}"
        relation = f"relation_{mp_i}"
        d[(src_name, relation + "-->", dst_name)]["edge_index"] = th.LongTensor(np.vstack([src_array_concat, dst_array_concat]))
        metapaths.append((src_name, relation + "-->", dst_name))
        d[(dst_name, "<--" + relation, src_name)]["edge_index"] = th.LongTensor(np.vstack([dst_array_concat, src_array_concat]))
        metapaths.append((dst_name, "<--" + relation, src_name))
    g = HeteroData(d)
    return g, metapaths



def load_data(type_num, num_pos_sample, walk_len, num_walks, num_pos_sample_ulu, walk_len_ulu,num_walks_ulu):
    # Change it with different dataset.
    path = "data/austin"
    feat_u = sp.eye(type_num[0])
    uu = sp.load_npz(path+"/uu.npz")
    uu_f = sp.load_npz(path+"/uu_f.npz")
    ulu = sp.load_npz(path+"/ulu_top4.npz")
    ullu = sp.load_npz(path+"/ullu_top10.npz")
    train_f = sample_semi_supervised_context(uu_f, num_pos_sample)
    train = sample_context(
        uu,
        type_num[0],
        walk_len,
        num_walks,
        num_pos_sample
    )
    train_ulu = sample_context(
        ulu,
        type_num[0],
        walk_len_ulu,
        num_walks_ulu,
        num_pos_sample_ulu
    )
    train_ullu = sample_context(
        ullu,
        type_num[0],
        walk_len_ulu,
        num_walks_ulu,
        num_pos_sample_ulu
    )
    feat_u = th.FloatTensor(preprocess_features(feat_u))
    uu = sparse_mx_to_torch_sparse_tensor(normalize_adj(uu))
    uu_f = sparse_mx_to_torch_sparse_tensor(normalize_adj(uu_f))
    ulu = sparse_mx_to_torch_sparse_tensor(normalize_adj(ulu))
    ullu = sparse_mx_to_torch_sparse_tensor(normalize_adj(ullu))

    val = np.load(path + "/vaL_edge.npy")
    test = np.load(path + "/test_edge.npy")
    val_false = np.load(path + "/vaL_edge_false.npy")
    test_false = np.load(path + "/test_edge_false.npy")

    return [feat_u], [uu,uu_f,ulu, ullu], val, val_false, test, test_false, train_f, train, train_ulu, train_ullu

def convert_to_adj_list(graph):
    adj_list = {}
    rows = graph.row
    cols = graph.col
    for row, col in zip(rows, cols):
        adj_list.setdefault(row, [])
        adj_list[row].append(col)
    print('number of isolated nodes:', graph.shape[0] - len(adj_list)) #the shape of sparse matrix(like csc) is a normal matrix
    return adj_list

def sample_semi_supervised_context(partial_social_graph, num_pos_sample):
    num_users = partial_social_graph.shape[0]
    partial_social_graph = convert_to_adj_list(partial_social_graph)
    samples = np.zeros((num_users, num_pos_sample), dtype=int)
    for u in range(num_users):
        if u not in partial_social_graph:
            samples[u] = u * np.ones(num_pos_sample, dtype=int)
            continue
        context = []
        for v in partial_social_graph[u]:
            context.append(v)
        samples[u] = np.random.choice(context, num_pos_sample, replace=True)
    return samples

def neg_sample(neg_sample_size, num_users):
    neg_samples = np.random.randint(0, num_users,
                 (num_users, neg_sample_size), dtype=int)
    #neg_samples = np.vstack((neg_samples,(num_users) * np.ones((1, neg_sample_size), dtype=int)))
    return neg_samples


def sample_context(graph, num_users,walk_len, num_walks, num_pos_sample):
    graph = convert_to_adj_list(graph)
    num_nodes = num_users
    homo_samples = dict()

    for node in range(num_nodes):
        if node not in graph:
            continue
        for v in graph[node]:
            homo_samples.setdefault(node, [])
            homo_samples[node].append(v)

    homo_samples_walk = dict()
    for node in range(num_nodes):
        if node not in graph:
            continue
        for _ in range(num_walks):
            curr_node = node
            for _ in range(walk_len):
                next_node = random.choice(graph[curr_node])
                if curr_node != node:
                    homo_samples_walk.setdefault(node, [])
                    homo_samples_walk[node].append(curr_node)
                curr_node = next_node
        if node % 1000 == 0:
            print("finish random walk for", node, "nodes")

    # fill blanks in positive sample:

    homo_samples_matrix = np.zeros((num_nodes, num_pos_sample))
    for node in range(num_nodes):
        if node in homo_samples and node in homo_samples_walk:
            num_samples = len(homo_samples_walk[node])
            samples = np.random.choice(homo_samples[node], num_pos_sample // 2, replace=True)
            if num_samples >= num_pos_sample:
                samples = np.concatenate((
                    samples,
                    np.random.choice(homo_samples_walk[node], num_pos_sample // 2, replace=False)
                ))
                homo_samples_matrix[node] = samples
            else:
                samples = np.concatenate((samples,
                          np.random.choice(homo_samples_walk[node], num_pos_sample // 2, replace=True)))
                homo_samples_matrix[node] = samples

    del homo_samples
    return homo_samples_matrix