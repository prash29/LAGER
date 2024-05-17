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

