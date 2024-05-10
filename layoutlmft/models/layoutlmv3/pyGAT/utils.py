import numpy as np
import scipy.sparse as sp
import torch
import os
import pickle

def centroid(bbox):
    return ((bbox[0] + (bbox[2]-bbox[0])/2), (bbox[1] + (bbox[3] - bbox[1])/2))

def get_centroid_map(bboxes):
    centroid_map = {}
    for bb in bboxes:
        centroid_map[bb] = centroid(bb)
    return centroid_map
    
def euclid_dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def get_idx_pair_maps(bb_tok_pairs):
    idx_pair_map, pair_idx_map = {},{}
    idx_pair_maps, pair_idx_maps = [{} for _ in range(len(bb_tok_pairs))], [{} for _ in range(len(bb_tok_pairs))]
    for k, bb_tok_pair in enumerate(bb_tok_pairs):
        for i, bt_pair in enumerate(bb_tok_pair):
            if (bt_pair[0], bt_pair[1]) in pair_idx_maps[k]:
                continue
            pair_idx_maps[k][(bt_pair[0], bt_pair[1])] = i
            # idx_pair_map[i] = (bb_tok_map[bb], bb)
        idx_pair_maps[k] = {j:i for i,j in pair_idx_maps[k].items()}
    return idx_pair_maps, pair_idx_maps
    
def valid_dist(bb1, bb2, type, thresh = None):
    cen_bb1, cen_bb2 = centroid(bb1), centroid(bb2)
    if type == 'vd':
        height = abs(bb1[3] - bb1[1])
        if cen_bb1[1] < cen_bb2[1] and abs(cen_bb1[1] - cen_bb2[1]) > height:
            return True
        return False   
    elif type == 'vu':
        height = abs(bb1[3] - bb1[1])
        if cen_bb1[1] > cen_bb2[1] and abs(cen_bb1[1] - cen_bb2[1]) > height:
            return True
        return False   
    elif type == 'hr':
        width = abs(bb1[2] - bb1[0])
        if cen_bb1[0] < cen_bb2[0] and abs(cen_bb1[0] - cen_bb2[0]) > width:
            return True
        return False   
    elif type == 'hl':
        width = abs(bb1[2] - bb1[0])
        if cen_bb1[0] > cen_bb2[0] and abs(cen_bb1[0] - cen_bb2[0]) > width:
            return True
        return False


def make_edges(bb_tok_pairs, types = ['vd','vu','hr','hl']):
    # bb_cent_map = get_centroid_map(bboxes)
    bb_edge_maps = []
    for bb_tok_pair in bb_tok_pairs:
        bb_edge_map = {t:{} for t in types}
        for type in types:
            for i, b_t1 in enumerate(bb_tok_pair):
                bb1, tok1 = b_t1[0], b_t1[1]
                min_dist = 100000
                if bb1 == (0,0,0,0):
                    continue
                # multiple_vals = []
                for j, b_t2 in enumerate(bb_tok_pair):
                    bb2, tok2 = b_t2[0], b_t2[1]
                    if i!=j:
                        if not valid_dist(bb1, bb2, type) or bb2 == (0,0,0,0):
                            continue
                        dist = euclid_dist(bb1, bb2)
                        if dist < min_dist:
                            min_dist = dist
                            # tok1, tok2 = bb_tok_map[bb1], bb_tok_map[bb2]
                            bb_edge_map[type][(bb1, tok1)] = (bb2, tok2)
                        # bb_edge_map[(tok1, bb_cent_map[(tok1, bb1)])] = (tok2, bb_cent_map[(tok2, bb2)])
        bb_edge_maps.append(bb_edge_map)
    return bb_edge_maps

def get_edges_idxs(bb_tok_pairs, types = ['vd','vu','hr','hl'] ):
    bb_edge_maps = make_edges(bb_tok_pairs)
    idx_pair_maps, pair_idx_maps = get_idx_pair_maps(bb_tok_pairs)
    batch_edges_idxs = []
    for bb_edge_map, pair_idx_map in zip(bb_edge_maps, pair_idx_maps):
        edges_idxs = []
        for type in types:
            for x, y in bb_edge_map[type].items():
                edges_idxs.append(tuple(sorted([pair_idx_map[x], pair_idx_map[y]])))
        batch_edges_idxs.append(np.array(edges_idxs))
    return batch_edges_idxs

def get_adjs(data_path, device, feats_shape = 512):
    edges_idxs = pickle.load(open(os.path.join(data_path, 'edges_idxs.pkl'),'rb'))
    adjs = []
    for edges in edges_idxs:
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(feats_shape,feats_shape), dtype=np.float32)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = normalize_adj(adj + sp.eye(adj.shape[0]))
        adj = torch.FloatTensor(np.array(adj.todense())).to(device)
        adjs.append(adj)
    return torch.stack(adjs)
        
def encode_onehot(labels):
    # The classes must be sorted before encoding to enable static class encoding.
    # In other words, make sure the first class always maps to index 0.
    classes = sorted(list(set(labels)))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot
    

def load_data(path="./data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize_features(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

