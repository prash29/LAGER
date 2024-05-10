import imp
import logging
import numpy as np
import scipy.sparse as sp
import torch
import os
import pickle
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont
from pdb import set_trace as bp
from math import atan2, degrees
import math

'''
Math functions for calculating shortest distance between a point and a line segment
'''

## Try 2D version

def dot(v,w):
    """
    Compute the dot product of two 2D vectors.

    Parameters:
        v (tuple): The first vector represented as a tuple (x, y).
        w (tuple): The second vector represented as a tuple (X, Y).

    Returns:
        float: The dot product of the two vectors.
    """

    x,y = v
    X,Y = w
    return x*X + y*Y

def length(v):
    """
    Compute the length (magnitude) of a 2D vector.

    Parameters:
        v (tuple): The vector represented as a tuple (x, y).

    Returns:
        float: The length of the vector.
    """
    x,y = v
    return math.sqrt(x*x + y*y)

def vector(b, e):
    """
    Compute the vector from the beginning point to the end point.

    Parameters:
        b (tuple): The beginning point represented as a tuple (x, y).
        e (tuple): The end point represented as a tuple (X, Y).

    Returns:
        tuple: The vector from b to e represented as a tuple (X-x, Y-y).
    """
    x, y = b
    X, Y = e
    return (X-x, Y-y)

def unit(v):
    """
    Compute the unit vector (normalized vector) of a 2D vector.

    Parameters:
        v (tuple): The vector represented as a tuple (x, y).

    Returns:
        tuple: The unit vector of v represented as a tuple (x/mag, y/mag),
               where mag is the magnitude of v.
    """
    x, y = v
    mag = length(v)
    return (x/mag, y/mag)

def distance(p0, p1):
    """
    Compute the Euclidean distance between two points in 2D space.

    Parameters:
        p0 (tuple): The first point represented as a tuple (x0, y0).
        p1 (tuple): The second point represented as a tuple (x1, y1).

    Returns:
        float: The Euclidean distance between the two points.
    """
    return length(vector(p0, p1))

def scale(v, sc):
    """
    Scale (multiply) a 2D vector by a scalar value.

    Parameters:
        v (tuple): The vector represented as a tuple (x, y).
        sc (float): The scalar value by which to scale the vector.

    Returns:
        tuple: The scaled vector represented as a tuple (x * sc, y * sc).
    """
    x, y = v
    return (x * sc, y * sc)
  
def add(v,w):
    """
    Add two 2D vectors element-wise.

    Parameters:
        v (tuple): The first vector represented as a tuple (x, y).
        w (tuple): The second vector represented as a tuple (X, Y).

    Returns:
        tuple: The sum of the two vectors represented as a tuple (x+X, y+Y).
    """
    x,y = v
    X,Y = w
    return (x+X, y+Y)

def pnt2line(pnt, start, end):
    """
    Compute the shortest distance between a point and a line segment defined by two endpoints.

    Parameters:
        pnt (tuple): The point for which the distance to the line segment is calculated, represented as a tuple (x, y).
        start (tuple): The starting point of the line segment, represented as a tuple (x_start, y_start).
        end (tuple): The ending point of the line segment, represented as a tuple (x_end, y_end).

    Returns:
        tuple: A tuple containing the distance between the point and the line segment, and the nearest point on the line segment.
               - The distance (float) represents the shortest distance between the point and the line segment.
               - The nearest point on the line segment represented as a tuple (x, y).
    """
    line_vec = vector(start, end)
    pnt_vec = vector(start, pnt)
    line_len = length(line_vec)
    line_unitvec = unit(line_vec)
    pnt_vec_scaled = scale(pnt_vec, 1.0/line_len)
    t = dot(line_unitvec, pnt_vec_scaled)    
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    nearest = scale(line_vec, t)
    dist = distance(nearest, pnt_vec)
    nearest = add(nearest, start)
    return (dist, nearest)


'''
Functions for finding min. distance between two bounding boxes and some helper functions
'''


def get_corner_coords(bbox):
    return [(bbox[0], bbox[1]), (bbox[2], bbox[1]), (bbox[2], bbox[3]), (bbox[0], bbox[3])]

def get_segments(coords):
    return [[coords[0], coords[1]], [coords[1], coords[2]], [coords[2], coords[3]], [coords[3], coords[1]]]

def find_min_distance(coords_a, coords_b):
    ''' Returns the minimum distance between bounding box A to bounding box B along with the point (from BB of A)
    and the corner/point (from BB of B)
    '''
    segments_b = get_segments(coords_b)
    distances = []
    for pt in coords_a:
        for seg in segments_b:
            try:
                dist, corner = pnt2line(pt, seg[0], seg[1])
                distances.append((dist, pt, corner))
            except:
                distances.append((10000000,pt,pt))
    distances.sort(key=lambda x:x[0])
    return distances[0]

def unnormalize_box(bbox, width, height):
     return [
         width * (bbox[0] / 1000),
         height * (bbox[1] / 1000),
         width * (bbox[2] / 1000),
         height * (bbox[3] / 1000),
     ]
     
def normalize_box(bbox, width, height):
    return [ int((1000*bbox[0])/width),
             int((1000*bbox[1])/height),
             int((1000*bbox[2])/width),
             int((1000*bbox[3])/height)
        
    ]

def centroid(bbox):
    return ((bbox[0] + (bbox[2]-bbox[0])/2), (bbox[1] + (bbox[3] - bbox[1])/2))

def get_centroid_map(bboxes):
    centroid_map = {}
    for bb in bboxes:
        centroid_map[bb] = centroid(bb)
    return centroid_map
    
def euclid_dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def get_rotated_point(coord, center, angle):
    x = (coord[0] - center[0])*np.cos(math.radians(angle)) - ((coord[1] - center[1])*np.sin(math.radians(angle))) + center[0]
    y = (coord[0] - center[0])*np.sin(math.radians(angle)) + ((coord[1] - center[1])*np.cos(math.radians(angle))) + center[1]
    if x<0:
        x = 0
    if y<0:
        y = 0
    if x> 1023:
        x = 1023
    if y>1023:
        y = 1023
    return (x,y)

def get_bbox_from_coords(coords):
    return [coords[0][0], coords[1][1], coords[2][0], coords[3][1]]

def get_bbox_from_rotated_coords(rotated_coords):
    return [rotated_coords[3][0], rotated_coords[0][1], rotated_coords[2][0], rotated_coords[2][1]]

def get_rotated_bboxes(bboxes, width_height_list = [], angle = 3):
    bboxes_rotated = []
    for bbox, width_height in zip(bboxes, width_height_list):
        bbox_rotated = []
        w, h = width_height[0], width_height[1]
        for bb in bbox:
            if bb == [0,0,0,0]:
                bbox_rotated.append(bb)
                continue
            coords = get_corner_coords(unnormalize_box(bb, w, h))
            rotated_coords = [get_rotated_point(coords[i], coords[3], -angle) for i in range(4)]
            new_bb = normalize_box(get_bbox_from_rotated_coords(rotated_coords), w, h)
            if new_bb[0] > 1000 or new_bb[1] > 1000 or new_bb[2] > 1000 or new_bb[3] > 1000:
                print(new_bb)
            bbox_rotated.append(new_bb)
        bboxes_rotated.append(bbox_rotated)
    return bboxes_rotated

def get_scaled_bboxes(bboxes, scale_factor = 2, width_height_list = []):
    bboxes_scaled = []
    for bbox, width_height in zip(bboxes, width_height_list):
        bbox_scaled = []
        w, h = width_height[0], width_height[1]
        for bb in bbox:
            if scale_factor==2:
                bb_new = [bb[0]/2, bb[1], bb[2]/2, bb[3]]
            else:
                # coords = get_corner_coords(unnormalize_box(bb, w, h))
                bb_new = [x/(scale_factor) for x in bb]
            bbox_scaled.append(bb_new)
        bboxes_scaled.append(bbox_scaled)
    return bboxes_scaled

def get_shifted_bboxes(bboxes, shift, width_height_list = []):
    bboxes_shifted = []
    for bbox, width_height in zip(bboxes, width_height_list):
        bbox_shifted = []
        w, h = width_height[0], width_height[1]
        for bb in bbox:
            if bb == [0,0,0,0]:
                bbox_shifted.append(bb)
                continue
            bb_shifted = [min(b+shift, h) if i%2 else min(b+shift, w) for i,b in enumerate(bb)]
            if bb_shifted[0]==bb_shifted[2]:
                bb_shifted[0]-=1
            if bb_shifted[1]==bb_shifted[3]:
                bb_shifted[1]-=1
            bbox_shifted.append(bb_shifted)
        bboxes_shifted.append(bbox_shifted)
    return bboxes_shifted         
    
def get_widths_heights(id_to_image_train_json, id_to_image_test_json, path = '/home/prashant/DocDatasets/FUNSD/raw'):
    train_path = os.path.join(path, 'training_data/images')
    test_path = os.path.join(path, 'testing_data/images')
    id_to_width_height_train, id_to_width_height_test = [], []
    
    for id1 in id_to_image_train_json.keys():
        image = Image.open(os.path.join(train_path, f"{id_to_image_train_json[id1]}.png"))
        image = image.convert("RGB")
        width, height = image.size
        id_to_width_height_train.append((width, height))
    
    for id1 in id_to_image_test_json.keys():
        image = Image.open(os.path.join(test_path, f"{id_to_image_test_json[id1]}.png"))
        image = image.convert("RGB")
        width, height = image.size
        id_to_width_height_test.append((width, height))
    
    return id_to_width_height_train, id_to_width_height_test

def get_widths_heights_cord(id_to_image_train_json, id_to_image_eval_json, id_to_image_test_json, path = '/home/prashant/DocDatasets/CORD/raw'):
    train_path = os.path.join(path, 'train/image')
    eval_path = os.path.join(path, 'dev/image')
    test_path = os.path.join(path, 'test/image')
    id_to_width_height_train, id_to_width_height_eval ,id_to_width_height_test = [], [], []
    
    for id1 in id_to_image_train_json.keys():
        image = Image.open(os.path.join(train_path, f"{id_to_image_train_json[id1]}.png"))
        image = image.convert("RGB")
        width, height = image.size
        id_to_width_height_train.append((width, height))
        
    for id1 in id_to_image_eval_json.keys():
        image = Image.open(os.path.join(eval_path, f"{id_to_image_eval_json[id1]}.png"))
        image = image.convert("RGB")
        width, height = image.size
        id_to_width_height_eval.append((width, height))
    
    for id1 in id_to_image_test_json.keys():
        image = Image.open(os.path.join(test_path, f"{id_to_image_test_json[id1]}.png"))
        image = image.convert("RGB")
        width, height = image.size
        id_to_width_height_test.append((width, height))
    
    return id_to_width_height_train, id_to_width_height_eval, id_to_width_height_test

def angle_between(p1, p2, p3):
    '''Get angle between the two lines formed by (x_1, y_1) & (x_2, y_2) : 
        y = y_1 and y = ((y_2 - y_1)/(x_2 - x_1)) x'''
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    deg1 = (360 + degrees(atan2(x1 - x2, y1 - y2))) % 360
    deg2 = (360 + degrees(atan2(x3 - x2, y3 - y2))) % 360
    return deg2 - deg1 if deg1 <= deg2 else 360 - (deg1 - deg2)

def is_bbox_in_range(bb1, bb2, theta1, theta2, width):
    ''' Check if the line between the centroid of bb1 and any of the corners of bb2 
    are in the range [theta_1, theta_2]'''
    point_1 = centroid(bb1)
    coords_2 = get_corner_coords(bb2)
    point_0 = (width, point_1[1])
    for point_2 in coords_2:
        angle = angle_between(point_0, point_1, point_2)
        # angle = get_angle_from_coords(point_1, pnt)
        if theta1<= angle <=theta2:
            return True
    return False

def get_bboxes_in_angle_range(bb_tok_pair, theta1, theta2, width = 762):
    ''' Get a dictionry where key = (bb, tok) and values are a list where each element is
    (bb', tok') where we check if line we draw from centroid of bb to any corner in bb'
    are in [theta_1, theta_2]
    '''
    bb_tok_angle_range_filtered = defaultdict(list)
    for i, b_t1 in enumerate(bb_tok_pair):
        bb1, tok1 = b_t1[0], b_t1[1]
        if bb1==(0,0,0,0):
            continue
        for j, b_t2 in enumerate(bb_tok_pair):
            bb2, tok2 = b_t2[0], b_t2[1]
            if bb1==bb2 or bb2 == (0,0,0,0):
                continue
            if is_bbox_in_range(bb1, bb2, theta1, theta2, width):
                bb_tok_angle_range_filtered[b_t1].append(b_t2)
    return bb_tok_angle_range_filtered

def get_end_coord_from_angle(start, angle, width, height):
    x = min(width, start[0] + (width*np.cos(np.radians(-angle))))
    y = min(height, start[1] + (height*np.sin(np.radians(-angle))))
    return (x,y)

def ccw(a,b,c):
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

def intersect(a,b,c,d):
    return ccw(a,c,d) != ccw(b,c,d) and ccw(a,b,c) != ccw(a,b,d)

def is_bbox_valid(point_start, point_end, bb2):
    coords_2 = get_corner_coords(bb2)
    
    if intersect(point_start, point_end, coords_2[0], coords_2[1]) or \
        intersect(point_start, point_end, coords_2[1], coords_2[2]) or \
            intersect(point_start, point_end, coords_2[2], coords_2[3]) or \
                intersect(point_start, point_end, coords_2[3], coords_2[0]):
                    return True
    return False

def get_valid_bboxes(bb_tok_pair, theta, width = 762, height = 1000):
    bb_tok_angle_range_filtered = defaultdict(list)
    for i, b_t1 in enumerate(bb_tok_pair):
        bb1, tok1 = b_t1[0], b_t1[1]
        if bb1==(0,0,0,0):
            continue
        for j, b_t2 in enumerate(bb_tok_pair):
            bb2, tok2 = b_t2[0], b_t2[1]
            if bb1==bb2 or bb2 == (0,0,0,0):
                continue
            point_start = centroid(bb1)
            point_end = get_end_coord_from_angle(point_start, theta, width, height)
            if is_bbox_valid(point_start, point_end, bb2):
                bb_tok_angle_range_filtered[b_t1].append(b_t2)
    return bb_tok_angle_range_filtered

def make_edges_new_angles_v2(bb_tok_pairs, theta, width_height_list, num_edges = 4, threshold = 300 ,plot_flag = False):
    # bb_cent_map = get_centroid_map(bboxes)
    edge_maps, edges_pts_coords_info_maps = [], []
    for k, bb_tok_pair_norm in enumerate(bb_tok_pairs):
        # bb_tok_pair = [(tuple(unnormalize_box(bb, width_height_list[k][0], width_height_list[k][1])), tok) for bb, tok in bb_tok_pair_norm]
        bb_tok_pair = [(bb, tok) for bb, tok in bb_tok_pair_norm]
        bb_tok_norm_map = {x[0]:y[0] for x,y in zip(bb_tok_pair, bb_tok_pair_norm)}
        filtered_angle_bb_tok_pair_map = get_valid_bboxes(bb_tok_pair, theta, width_height_list[k][0], width_height_list[k][1])
        # threshold = width_height_list[k][1]/3
        edge_map, edges_pts_coords_info_map = defaultdict(list), defaultdict(list) 
        for i, b_t1 in enumerate(bb_tok_pair):
            dist_pts_info = []
            curr_dists, uniq_edges = [], []
            bb1, tok1 = b_t1[0], b_t1[1]
            if tok1==0 or tok1==101 or tok1==102 or bb_tok_norm_map[bb1] == (0,0,0,0):
                continue
            coords_bb1 = get_corner_coords(bb1)
    
            for j, b_t2 in enumerate(bb_tok_pair):
                bb2, tok2 = b_t2[0], b_t2[1]
                if (bb_tok_norm_map[bb1]==bb_tok_norm_map[bb2]) or tok2==0 or tok2==101 or tok2==102 or (bb_tok_norm_map[bb2] == (0,0,0,0)):
                        continue
                if b_t2 not in filtered_angle_bb_tok_pair_map[b_t1]:
                    continue
                coords_bb2 = get_corner_coords(bb2)
                tmp = find_min_distance(coords_bb1, coords_bb2)
                dist_pts_info.append([(bb2, tok2), tmp])
            dist_pts_info.sort(key = lambda x:x[1][0])
            for curr in dist_pts_info[:num_edges]:
                if curr[1][0] > threshold:
                    continue
                if curr[0][0] not in uniq_edges:
                    uniq_edges.append(curr[0][0])
                if len(uniq_edges) > num_edges:
                    break
                curr_ = (bb_tok_norm_map[curr[0][0]], curr[0][1])
                edge_map[(bb_tok_norm_map[bb1], tok1)].append(curr_)
                edges_pts_coords_info_map[(bb1, tok1)].append(curr[1])
        edge_maps.append(edge_map)
        edges_pts_coords_info_maps.append(edges_pts_coords_info_map)
    if plot_flag:
        return edge_maps, edges_pts_coords_info_maps
    return edge_maps

def get_idx_pair_maps(bb_tok_pairs):
    # idx_pair_map, pair_idx_map = {},{}
    idx_pair_maps, pair_idx_maps = [{} for _ in range(len(bb_tok_pairs))], [{} for _ in range(len(bb_tok_pairs))]
    for k, bb_tok_pair in enumerate(bb_tok_pairs):
        for i, bt_pair in enumerate(bb_tok_pair):
            if (bt_pair[0], bt_pair[1]) in pair_idx_maps[k]:
                continue
            idx_pair_maps[k][i] = (bt_pair[0], bt_pair[1])
            pair_idx_maps[k][(bt_pair[0], bt_pair[1])] = i
            # idx_pair_map[i] = (bb_tok_map[bb], bb)
        # idx_pair_maps[k] = {j:i for i,j in pair_idx_maps[k].items()}
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

def make_edges_v3(bb_tok_pairs, types = ['vd','vu','hr','hl'], num_edges = 4):
    # bb_cent_map = get_centroid_map(bboxes)
    edge_maps = []
    for bb_tok_pair in bb_tok_pairs:
        edge_map = defaultdict(list)
        for i, b_t1 in enumerate(bb_tok_pair):
            curr_dists, uniq_edges = [], []
            bb1, tok1 = b_t1[0], b_t1[1]
            if tok1==0 or tok1==101 or tok1==102 or bb1 == (0,0,0,0):
                continue
            for j, b_t2 in enumerate(bb_tok_pair):
                bb2, tok2 = b_t2[0], b_t2[1]
                if (bb1==bb2) or tok2==0 or tok2==101 or tok2==102 or (bb2 == (0,0,0,0)):
                        continue
                # if not valid_dist(bb1, bb2, type):
                #     continue
                dist = euclid_dist(bb1, bb2)
                curr_dists.append([(bb2, tok2), dist])
            curr_dists.sort(key = lambda x:x[1])
            for curr in curr_dists[:num_edges]:
                if curr[0][0] not in uniq_edges:
                    uniq_edges.append(curr[0][0])
                if len(uniq_edges) > num_edges:
                    break
                edge_map[(bb1, tok1)].append(curr[0])
        edge_maps.append(edge_map)
    return edge_maps

def make_edges_from_label(bb_label_pairs):
    # bb_cent_map = get_centroid_map(bboxes)
    edge_maps = []
    for bb_lab_pair in bb_label_pairs:
        # bb_tok_num_edge_count = {x:0 for x in bb_tok_pair}
        edge_map = defaultdict(list)
        for i, b_l1 in enumerate(bb_lab_pair):
            curr_dists, uniq_edges = [], []
            bb1, lab1 = b_l1[0], b_l1[1]
            if lab1==-100 or bb1 == (0,0,0,0):
                continue
            for j, b_l2 in enumerate(bb_lab_pair):
                bb2, lab2 = b_l2[0], b_l2[1]
                if (bb1==bb2) or lab2==-100 or (bb2 == (0,0,0,0)):
                        continue
                if lab1==lab2:
                    edge_map[(bb1, lab1)].append((bb2, lab2))
                # if not valid_dist(bb1, bb2, type):
        edge_maps.append(edge_map)
    return edge_maps

def get_edges_idxs_v3(bb_tok_pairs, types = ['vd','vu','hr','hl']):
    edge_maps = make_edges_v3(bb_tok_pairs)
    idx_pair_maps, pair_idx_maps = get_idx_pair_maps(bb_tok_pairs)
    batch_edges_idxs = []
    for edge_map, pair_idx_map in zip(edge_maps, pair_idx_maps):
        edges_idxs = []
        for x, y in edge_map.items():
            for k in y:
                if tuple([pair_idx_map[x], pair_idx_map[k]]) not in edges_idxs and tuple([pair_idx_map[k], pair_idx_map[x]]) not in edges_idxs:
                    edges_idxs.append(tuple([pair_idx_map[x], pair_idx_map[k]]))
        # for pair, id in pair_idx_map.items():
        #     _, tok = pair[0], pair[1]
        #     if tok==101 or tok==102 or tok==0:
        #         continue
        #     edges_idxs.append((pair_idx_map[((0,0,0,0),101)], id)) # map[101]
        #     edges_idxs.append((pair_idx_map[((0,0,0,0),102)], id)) # map[102]
        batch_edges_idxs.append(np.array(edges_idxs))
    return batch_edges_idxs

def get_edges_idxs_new(bb_tok_pairs, types = ['vd','vu','hr','hl']):
    edge_maps = make_edges_new(bb_tok_pairs)
    idx_pair_maps, pair_idx_maps = get_idx_pair_maps(bb_tok_pairs)
    batch_edges_idxs = []
    for edge_map, pair_idx_map in zip(edge_maps, pair_idx_maps):
        edges_idxs = []
        for x, y in edge_map.items():
            for k in y:
                if tuple([pair_idx_map[x], pair_idx_map[k]]) not in edges_idxs and tuple([pair_idx_map[k], pair_idx_map[x]]) not in edges_idxs:
                    edges_idxs.append(tuple([pair_idx_map[x], pair_idx_map[k]]))
        # for pair, id in pair_idx_map.items():
        #     _, tok = pair[0], pair[1]
        #     if tok==101 or tok==102 or tok==0:
        #         continue
        #     edges_idxs.append((pair_idx_map[((0,0,0,0),101)], id)) # map[101]
        #     edges_idxs.append((pair_idx_map[((0,0,0,0),102)], id)) # map[102]
        batch_edges_idxs.append(np.array(edges_idxs))
    return batch_edges_idxs

def get_edges_idxs_new_angles(bb_tok_pairs, theta1, theta2, width_height_list, types = ['vd','vu','hr','hl']):
    edge_maps = make_edges_new_angles(bb_tok_pairs,theta1, theta2, width_height_list)
    idx_pair_maps, pair_idx_maps = get_idx_pair_maps(bb_tok_pairs)
    batch_edges_idxs = []
    for edge_map, pair_idx_map in zip(edge_maps, pair_idx_maps):
        edges_idxs = []
        for x, y in edge_map.items():
            for k in y:
                if tuple([pair_idx_map[x], pair_idx_map[k]]) not in edges_idxs and tuple([pair_idx_map[k], pair_idx_map[x]]) not in edges_idxs:
                    edges_idxs.append(tuple([pair_idx_map[x], pair_idx_map[k]]))
        # for pair, id in pair_idx_map.items():
        #     _, tok = pair[0], pair[1]
        #     if tok==101 or tok==102 or tok==0:
        #         continue
        #     edges_idxs.append((pair_idx_map[((0,0,0,0),101)], id)) # map[101]
        #     edges_idxs.append((pair_idx_map[((0,0,0,0),102)], id)) # map[102]
        batch_edges_idxs.append(np.array(edges_idxs))
    return batch_edges_idxs

def get_edges_idxs_new_angles_v2(bb_tok_pairs, theta, width_height_list):
    edge_maps = make_edges_new_angles_v2(bb_tok_pairs, theta, width_height_list)
    idx_pair_maps, pair_idx_maps = get_idx_pair_maps(bb_tok_pairs)
    batch_edges_idxs = []
    for edge_map, pair_idx_map in zip(edge_maps, pair_idx_maps):
        edges_idxs = []
        for x, y in edge_map.items():
            for k in y:
                if tuple([pair_idx_map[x], pair_idx_map[k]]) not in edges_idxs and tuple([pair_idx_map[k], pair_idx_map[x]]) not in edges_idxs:
                    edges_idxs.append(tuple([pair_idx_map[x], pair_idx_map[k]]))
        # for pair, id in pair_idx_map.items():
        #     _, tok = pair[0], pair[1]
        #     if tok==101 or tok==102 or tok==0:
        #         continue
        #     edges_idxs.append((pair_idx_map[((0,0,0,0),101)], id)) # map[101]
        #     edges_idxs.append((pair_idx_map[((0,0,0,0),102)], id)) # map[102]
        batch_edges_idxs.append(np.array(edges_idxs))
    return batch_edges_idxs

def make_edges_v2(bb_tok_pairs, types = ['vd','vu','hr','hl'], num_edges = 4):
    # bb_cent_map = get_centroid_map(bboxes)
    bb_edge_maps, edge_maps = [], []
    for bb_tok_pair in bb_tok_pairs:
        bb_edge_map = {t:{} for t in types}
        edge_map = defaultdict(list)
        for i, b_t1 in enumerate(bb_tok_pair):
            curr_dists, uniq_edges = [], []
            bb1, tok1 = b_t1[0], b_t1[1]
            for type in types:
                bb_edge_map[type][(bb1, tok1)] = []
                # min_dist = 100000
                if tok1==0 or tok1==101 or tok1==102:
                    continue
                for j, b_t2 in enumerate(bb_tok_pair):
                    bb2, tok2 = b_t2[0], b_t2[1]
                    if i!=j:
                        if tok2==0 or tok2==101 or tok2==102:
                            continue
                        if not valid_dist(bb1, bb2, type):
                            continue
                        dist = euclid_dist(bb1, bb2)
                        curr_dists.append([(bb2, tok2), dist])
                        # if dist < min_dist:
                        #     min_dist = dist
                        #     bb_edge_map[type][(bb1, tok1)] = (bb2, tok2)
            curr_dists.sort(key = lambda x:x[1])
            for curr in curr_dists:
                if curr[0][0] not in uniq_edges:
                    uniq_edges.append(curr[0][0])
                if len(uniq_edges) > num_edges:
                    break
                # if curr[0][0] in uniq_edges:
                #     continue
                edge_map[(bb1, tok1)].append(curr[0])
                # bb_edge_map[type][(bb1, tok1)].append(curr[0])
        # bb_edge_maps.append(bb_edge_map)
        edge_maps.append(edge_map)
    return edge_maps

def get_edges_idxs_v2(bb_tok_pairs, types = ['vd','vu','hr','hl'] ):
    edge_maps = make_edges_v2(bb_tok_pairs)
    idx_pair_maps, pair_idx_maps = get_idx_pair_maps(bb_tok_pairs)
    batch_edges_idxs = []
    for edge_map, pair_idx_map in zip(edge_maps, pair_idx_maps):
        edges_idxs = []
        for x,y in edge_map.items():
            for k in y:
                # if tuple(sorted([pair_idx_map[x], pair_idx_map[k]])) not in edges_idxs:
                #     edges_idxs.append(tuple(sorted([pair_idx_map[x], pair_idx_map[k]])))
                if tuple([pair_idx_map[x], pair_idx_map[k]]) not in edges_idxs or tuple([pair_idx_map[k], pair_idx_map[x]]) not in edges_idxs:
                    # edges_idxs.append(sorted(tuple([pair_idx_map[x], pair_idx_map[k]])))
                    edges_idxs.append(tuple([pair_idx_map[x], pair_idx_map[k]]))
                    # edges_idxs.append(tuple([pair_idx_map[k], pair_idx_map[x]]))
        for pair, id in pair_idx_map.items():
            _, tok = pair[0], pair[1]
            if tok==101 or tok==102 or tok==0:
                continue
            edges_idxs.append((pair_idx_map[((0,0,0,0),101)], id)) # map[101]
            edges_idxs.append((pair_idx_map[((0,0,0,0),102)], id)) # map[102]
        batch_edges_idxs.append(np.array(edges_idxs))
    return batch_edges_idxs

def make_edges_new(bb_tok_pairs, num_edges = 4, plot_flag = False):
    # bb_cent_map = get_centroid_map(bboxes)
    edge_maps, edges_pts_coords_info_maps = [], []
    for bb_tok_pair in bb_tok_pairs:
        edge_map, edges_pts_coords_info_map = defaultdict(list), defaultdict(list) 
        for i, b_t1 in enumerate(bb_tok_pair):
            dist_pts_info = []
            curr_dists, uniq_edges = [], []
            bb1, tok1 = b_t1[0], b_t1[1]
            if tok1==0 or tok1==101 or tok1==102 or bb1 == (0,0,0,0):
                continue
            coords_bb1 = get_corner_coords(bb1)
    
            for j, b_t2 in enumerate(bb_tok_pair):
                bb2, tok2 = b_t2[0], b_t2[1]
                if (bb1==bb2) or tok2==0 or tok2==101 or tok2==102 or (bb2 == (0,0,0,0)):
                        continue

                coords_bb2 = get_corner_coords(bb2)
                tmp = find_min_distance(coords_bb1, coords_bb2)
                dist_pts_info.append([(bb2, tok2), tmp])
            dist_pts_info.sort(key = lambda x:x[1][0])
            for curr in dist_pts_info[:num_edges]:
                if curr[0][0] not in uniq_edges:
                    uniq_edges.append(curr[0][0])
                if len(uniq_edges) > num_edges:
                    break
                edge_map[(bb1, tok1)].append(curr[0])
                edges_pts_coords_info_map[(bb1, tok1)].append(curr[1])
        edge_maps.append(edge_map)
        edges_pts_coords_info_maps.append(edges_pts_coords_info_map)
    if plot_flag:
        return edge_maps, edges_pts_coords_info_maps
    return edge_maps

def make_edges_new_angles(bb_tok_pairs, theta1, theta2, width_height_list, num_edges = 1,  types = ['vd','vu','hr','hl'], plot_flag = False):
    # bb_cent_map = get_centroid_map(bboxes)
    edge_maps, edges_pts_coords_info_maps = [], []
    for k, bb_tok_pair_norm in enumerate(bb_tok_pairs):
        bb_tok_pair = [(tuple(unnormalize_box(bb, width_height_list[k][0], width_height_list[k][1])), tok) for bb, tok in bb_tok_pair_norm]
        bb_tok_norm_map = {x[0]:y[0] for x,y in zip(bb_tok_pair, bb_tok_pair_norm)}
        filtered_angle_bb_tok_pair_map = get_bboxes_in_angle_range(bb_tok_pair, theta1, theta2, width_height_list[k][0])
        edge_map, edges_pts_coords_info_map = defaultdict(list), defaultdict(list) 
        for i, b_t1 in enumerate(bb_tok_pair):
            dist_pts_info = []
            curr_dists, uniq_edges = [], []
            bb1, tok1 = b_t1[0], b_t1[1]
            if tok1==0 or tok1==101 or tok1==102 or bb_tok_norm_map[bb1] == (0,0,0,0):
                continue
            coords_bb1 = get_corner_coords(bb1)
    
            for j, b_t2 in enumerate(bb_tok_pair):
                bb2, tok2 = b_t2[0], b_t2[1]
                if (bb_tok_norm_map[bb1]==bb_tok_norm_map[bb2]) or tok2==0 or tok2==101 or tok2==102 or (bb_tok_norm_map[bb2] == (0,0,0,0)):
                        continue
                if b_t2 not in filtered_angle_bb_tok_pair_map[b_t1]:
                    continue
                coords_bb2 = get_corner_coords(bb2)
                tmp = find_min_distance(coords_bb1, coords_bb2)
                dist_pts_info.append([(bb2, tok2), tmp])
            dist_pts_info.sort(key = lambda x:x[1][0])
            for curr in dist_pts_info[:num_edges]:
                if curr[0][0] not in uniq_edges:
                    uniq_edges.append(curr[0][0])
                if len(uniq_edges) > num_edges:
                    break
                curr_ = (bb_tok_norm_map[curr[0][0]], curr[0][1])
                edge_map[(bb_tok_norm_map[bb1], tok1)].append(curr_)
                edges_pts_coords_info_map[(bb1, tok1)].append(curr[1])
        edge_maps.append(edge_map)
        edges_pts_coords_info_maps.append(edges_pts_coords_info_map)
    if plot_flag:
        return edge_maps, edges_pts_coords_info_maps
    return edge_maps

def get_edges_idxs_from_label(bb_label_pairs, types = ['vd','vu','hr','hl'] ):
    edge_maps = make_edges_from_label(bb_label_pairs)
    idx_pair_maps, pair_idx_maps = get_idx_pair_maps(bb_label_pairs)
    batch_edges_idxs = []
    for edge_map, pair_idx_map in zip(edge_maps, pair_idx_maps):
        edges_idxs = []
        for x, y in edge_map.items():
            for k in y:
                if tuple([pair_idx_map[x], pair_idx_map[k]]) not in edges_idxs and tuple([pair_idx_map[k], pair_idx_map[x]]) not in edges_idxs:
                    edges_idxs.append(tuple([pair_idx_map[x], pair_idx_map[k]]))
        # for pair, id in pair_idx_map.items():
        #     _, tok = pair[0], pair[1]
        #     if tok==101 or tok==102 or tok==0:
        #         continue
        #     edges_idxs.append((pair_idx_map[((0,0,0,0),101)], id)) # map[101]
        #     edges_idxs.append((pair_idx_map[((0,0,0,0),102)], id)) # map[102]
        batch_edges_idxs.append(np.array(edges_idxs))
    return batch_edges_idxs

def pickle_gat_stuff_from_label(dataset):
    labels = dataset['labels']
    bboxes = dataset['bbox']
    bboxes = [[tuple(x) for x in bb]for bb in bboxes]
    bb_label_pairs = [[(bb, lab) for bb, lab in zip(bbox, label)] for bbox, label in zip(bboxes, labels)]
    edges_idxs = get_edges_idxs_from_label(bb_label_pairs)
    # if type=='test':
    #     pickle.dump(bb_tok_pairs, open(os.path.join(pickle_path,'bb_tok_pairs_{}.pkl'.format(type)),'wb'))
    #     pickle.dump(edges_idxs, open(os.path.join(pickle_path, 'edges_idxs_{}.pkl'.format(type)),'wb'))
    return edges_idxs

def pickle_gat_stuff(dataset, pickle_path, type, num_edges = 1):
    input_ids = dataset['input_ids']
    bboxes = dataset['bbox']
    bboxes = [[tuple(x) for x in bb]for bb in bboxes]
    bb_tok_pairs = [[(bb, tok) for bb, tok in zip(bbox, in_id)] for bbox, in_id in zip(bboxes, input_ids)]
    edges_idxs = get_edges_idxs_v3(bb_tok_pairs)
    # if type=='test':
    #     pickle.dump(bb_tok_pairs, open(os.path.join(pickle_path,'bb_tok_pairs_{}.pkl'.format(type)),'wb'))
    #     pickle.dump(edges_idxs, open(os.path.join(pickle_path, 'edges_idxs_{}.pkl'.format(type)),'wb'))
    return edges_idxs

def get_adjs(dataset, pickle_path, type, num_edges = 1, feats_shape = 512):
    if type=='test':
        pickle_path = '/'.join(pickle_path.split('/')[:-1])
    # if not os.path.exists(os.path.join(pickle_path,'bb_tok_pairs_{}.pkl'.format(type))) and not os.path.exists(os.path.join(pickle_path,'edges_idxs_{}.pkl'.format(type))):
    if True:
        edges_idxs = pickle_gat_stuff(dataset, pickle_path, type, num_edges)
    else:
        edges_idxs = pickle.load(open(os.path.join(pickle_path, 'edges_idxs_{}.pkl'.format(type)),'rb'))
    adjs = []
    for edges in edges_idxs:
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(feats_shape,feats_shape), dtype=np.float32)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = normalize_adj(adj + sp.eye(adj.shape[0]))
        # adj = adj + sp.eye(adj.shape[0])
        
        # adj = torch.FloatTensor(np.array(adj.todense()))
        adj = np.array(adj.todense())
        adjs.append(adj)
    return adjs
     
def get_adjs_from_labels(dataset, pickle_path, type, feats_shape = 512):
    if type=='test':
        # edges_idxs = pickle_gat_stuff(dataset, pickle_path, type)
        edges_idxs = pickle_gat_stuff_from_label(dataset)
    else:
        edges_idxs = pickle_gat_stuff_from_label(dataset)
        # pickle_path = '/'.join(pickle_path.split('/')[:-1])

    adjs = []
    for edges in edges_idxs:
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(feats_shape,feats_shape), dtype=np.float32)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = normalize_adj(adj + sp.eye(adj.shape[0]))
        # adj = adj + sp.eye(adj.shape[0])
        
        # adj = torch.FloatTensor(np.array(adj.todense()))
        adj = np.array(adj.todense())
        adjs.append(adj)
    return adjs

def get_adjs_new(dataset, feats_shape = 512):
    '''Finding closest edge not only based on centroid but using line segments of the bounding boxes'''
    
    input_ids = dataset['input_ids']
    bboxes = dataset['bbox']
    bboxes = [[tuple(x) for x in bb]for bb in bboxes]
    bb_tok_pairs = [[(bb, tok) for bb, tok in zip(bbox, in_id)] for bbox, in_id in zip(bboxes, input_ids)]
    edges_idxs = get_edges_idxs_new(bb_tok_pairs)
    adjs = []
    for edges in edges_idxs:
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(feats_shape,feats_shape), dtype=np.float32)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = normalize_adj(adj + sp.eye(adj.shape[0]))
        # adj = adj + sp.eye(adj.shape[0])
        
        # adj = torch.FloatTensor(np.array(adj.todense()))
        adj = np.array(adj.todense())
        adjs.append(adj)
    return adjs

def get_adjs_new_angles(dataset, pickle_path, type, theta1 = 0, theta2 = 30, width_height_list =[], feats_shape = 512):
    '''Finding closest edges based on a region which is between theta1 and theta2 from the centroid of a bounding box'''
    if type=='test':
        pickle_path = '/'.join(pickle_path.split('/')[:-1])
    # if not os.path.exists(os.path.join(pickle_path,'bb_tok_pairs_{}.pkl'.format(type))) and not os.path.exists(os.path.join(pickle_path,'edges_idxs_{}.pkl'.format(type))):
    
    input_ids = dataset['input_ids']
    bboxes = dataset['bbox']
    bboxes = [[tuple(x) for x in bb]for bb in bboxes]
    bb_tok_pairs = [[(bb, tok) for bb, tok in zip(bbox, in_id)] for bbox, in_id in zip(bboxes, input_ids)]
    edges_idxs = get_edges_idxs_new_angles(bb_tok_pairs, theta1, theta2, width_height_list)
    adjs = []
    for edges in edges_idxs:
        try:
            adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(feats_shape,feats_shape), dtype=np.float32)
            adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
            adj = normalize_adj(adj + sp.eye(adj.shape[0]))
            # adj = adj + sp.eye(adj.shape[0])
            
            # adj = torch.FloatTensor(np.array(adj.todense()))
            adj = np.array(adj.todense())
        except:
            adj = np.identity(512)
        adjs.append(adj)
    return adjs

def get_adjs_new_angles_v2(dataset, theta, width_height_list, feats_shape = 512):
    '''Find closest edges by drawing a ray/line segment from the centroid of a bounding box'''
    input_ids = dataset['input_ids']
    bboxes = dataset['bbox']
    bboxes = [[tuple(x) for x in bb]for bb in bboxes]
    bb_tok_pairs = [[(bb, tok) for bb, tok in zip(bbox, in_id)] for bbox, in_id in zip(bboxes, input_ids)]
    edges_idxs = get_edges_idxs_new_angles_v2(bb_tok_pairs, theta, width_height_list)
    adjs = []
    for edges in edges_idxs:
        try:
            adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(feats_shape,feats_shape), dtype=np.float32)
            adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
            adj = normalize_adj(adj + sp.eye(adj.shape[0]))
            # adj = adj + sp.eye(adj.shape[0])
            
            # adj = torch.FloatTensor(np.array(adj.todense()))
            adj = np.array(adj.todense())
        except:
            adj = np.identity(512)
        adjs.append(adj)
    return adjs
 
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

