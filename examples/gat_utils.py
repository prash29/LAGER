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
    """
    Get the corner coordinates of a bounding box.
    Parameters:
        bbox (list): Bounding box coordinates [x0, y0, x1, y1].
    Returns:
        list: List of tuples containing the corner coordinates of the bounding box.
    """
    return [(bbox[0], bbox[1]), (bbox[2], bbox[1]), (bbox[2], bbox[3]), (bbox[0], bbox[3])]

def get_segments(coords):
    """
    Get the segments from a list of coordinates.
    Parameters:
        coords (list): List of coordinates.
    Returns:
        list: List of segments defined by pairs of coordinates.
    """
    return [[coords[0], coords[1]], [coords[1], coords[2]], [coords[2], coords[3]], [coords[3], coords[1]]]

def find_min_distance(coords_a, coords_b):
    """
    Find the minimum distance between bounding box A to bounding box B along with the point (from BB of A)
    and the corner/point (from BB of B).

    Parameters:
        coords_a (list): List of corner coordinates of bounding box A.
        coords_b (list): List of corner coordinates of bounding box B.

    Returns:
        tuple: A tuple containing the minimum distance, the point from BB of A, and the corner/point from BB of B.
    """
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
    """
    Convert bounding box coordinates from normalized values (0-1000 range) to absolute coordinates based on image dimensions.

    Parameters:
        bbox (list): The normalized bounding box coordinates [x0, y0, x1, y1].
        width (int): The width of the image in pixels.
        height (int): The height of the image in pixels.

    Returns:
        list: The unnormalized bounding box coordinates [x0', y0', x1', y1'].
              Coordinates are in absolute pixel values.
    """
    return [
         width * (bbox[0] / 1000),
         height * (bbox[1] / 1000),
         width * (bbox[2] / 1000),
         height * (bbox[3] / 1000),
     ]
     
def normalize_box(bbox, width, height):
    """
    Convert bounding box coordinates from absolute coordinates to normalized coordinates (0-1000 range) based on image dimensions.

    Parameters:
        bbox (list): The absolute bounding box coordinates [x0', y0', x1', y1'].
        width (int): The width of the image in pixels.
        height (int): The height of the image in pixels.

    Returns:
        list: The normalized bounding box coordinates [x0, y0, x1, y1].
              Coordinates are in the range of 0 to 1000.
    """
    return [ int((1000*bbox[0])/width),
             int((1000*bbox[1])/height),
             int((1000*bbox[2])/width),
             int((1000*bbox[3])/height)
        
    ]

def centroid(bbox):
    return ((bbox[0] + (bbox[2]-bbox[0])/2), (bbox[1] + (bbox[3] - bbox[1])/2))

def euclid_dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def get_rotated_point(coord, center, angle):
    """
    Get the rotated coordinates of a point based on the specified angle.
    Parameters:
        coord (tuple): The point coordinates (x, y).
        center (tuple): The center coordinates (x, y) of the rotation.
        angle (int): The angle of rotation in degrees.
    Returns:
        tuple: The rotated coordinates of the point (x_rot, y_rot)."""
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

# def get_bbox_from_coords(coords):
#     return [coords[0][0], coords[1][1], coords[2][0], coords[3][1]]

def get_bbox_from_rotated_coords(rotated_coords):
    """
    Get the bounding box coordinates from a list of rotated corner coordinates.

    Parameters:
        rotated_coords (list): List of rotated corner coordinates [(x0, y0), (x1, y1), (x2, y2), (x3, y3)].

    Returns:
        list: Bounding box coordinates [x3, y0, x2, y2].
    """
    return [rotated_coords[3][0], rotated_coords[0][1], rotated_coords[2][0], rotated_coords[2][1]]

def get_rotated_bboxes(bboxes, width_height_list = [], angle = 3):
    """
    Get rotated bounding boxes based on the specified angle.

    Parameters:
        bboxes (list): List of bounding boxes, each represented as a list of coordinates [x0, y0, x1, y1].
        width_height_list (list): List of tuples containing width and height of each image corresponding to the bounding boxes.
        angle (int): The angle of rotation in degrees.

    Returns:
        list: List of rotated bounding boxes, each represented as a list of coordinates [x0_rot, y0_rot, x1_rot, y1_rot].
    """
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
    """
    Get scaled bounding boxes based on the specified scale factor.

    Parameters:
        bboxes (list): List of bounding boxes, each represented as a list of coordinates [x0, y0, x1, y1].
        scale_factor (int): The factor by which to scale the bounding boxes.
        width_height_list (list): List of tuples containing width and height of each image corresponding to the bounding boxes.

    Returns:
        list: List of scaled bounding boxes, each represented as a list of coordinates [x0_scaled, y0_scaled, x1_scaled, y1_scaled].
    """
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
    """
    Get shifted bounding boxes based on the specified shift (translation vector (a,a)).

    Parameters:
        bboxes (list): List of bounding boxes, each represented as a list of coordinates [x0, y0, x1, y1].
        shift (int): The amount by which to shift the bounding boxes.
        width_height_list (list): List of tuples containing width and height of each image corresponding to the bounding boxes.

    Returns:
        list: List of shifted bounding boxes, each represented as a list of coordinates [x0_shifted, y0_shifted, x1_shifted, y1_shifted].
    """
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
    """
    Get the widths and heights of images in the training and testing datasets.

    Parameters:
        id_to_image_train_json (dict): A dictionary mapping image IDs to their corresponding filenames in the training dataset.
        id_to_image_test_json (dict): A dictionary mapping image IDs to their corresponding filenames in the testing dataset.
        path (str): The path to the root directory of the dataset.

    Returns:
        id_to_width_height_train (list): contains tuples of (width, height) for images in the training dataset.
        id_to_width_height_test (list): contains tuples of (width, height) for images in the testing dataset.
    """
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
    """
    Get the widths and heights of images in the training, evaluation, and testing datasets for CORD dataset.

    Parameters:
        id_to_image_train_json (dict): A dictionary mapping image IDs to their corresponding filenames in the training dataset.
        id_to_image_eval_json (dict): A dictionary mapping image IDs to their corresponding filenames in the evaluation dataset.
        id_to_image_test_json (dict): A dictionary mapping image IDs to their corresponding filenames in the testing dataset.
        path (str): The path to the root directory of the dataset.

    Returns:
        id_to_width_height_train (list): contains tuples of (width, height) for images in the training dataset.
        id_to_width_height_eval (list): contains tuples of (width, height) for images in the evaluation dataset.
        id_to_width_height_test (list): contains tuples of (width, height) for images in the testing dataset.
    """
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
    """
    Get the angle between two lines formed by three points.
    Parameters:
        p1 (tuple): The first point represented as a tuple (x1, y1).
        p2 (tuple): The second point represented as a tuple (x2, y2).
        p3 (tuple): The third point represented as a tuple (x3, y3).
    Returns:
        float: The angle between the two lines formed by the three points."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    deg1 = (360 + degrees(atan2(x1 - x2, y1 - y2))) % 360
    deg2 = (360 + degrees(atan2(x3 - x2, y3 - y2))) % 360
    return deg2 - deg1 if deg1 <= deg2 else 360 - (deg1 - deg2)

def is_bbox_in_range(bb1, bb2, theta1, theta2, width):
    """
    Check if the line between the centroid of bb1 and any of the corners of bb2 are in the range [theta1, theta2].
    Parameters:
        bb1 (list): Bounding box coordinates [x0, y0, x1, y1].
        bb2 (list): Bounding box coordinates [x0, y0, x1, y1].
        theta1 (float): The lower angle in radians used for filtering bounding boxes.
        theta2 (float): The upper angle in radians used for filtering bounding boxes.
        width (int): The width of the image.
    Returns:
        bool: True if the line between the centroid of bb1 and any of the corners of bb2 are in the range [theta1, theta2], False otherwise."""
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
    """
    Get bounding boxes within the given angle range from a reference bounding box.
    Parameters:
        bb_tok_pair (list): List of bounding box-token pairs for each sample.
        theta1 (float): The lower angle in radians used for filtering bounding boxes.
        theta2 (float): The upper angle in radians used for filtering bounding boxes.
        width (int): The width of the image. Default is 762.
    Returns:
        defaultdict: A dictionary mapping each bounding box-token pair to a list of bounding box-token pairs within the angle range."""
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
    ''' Get the end coordinate of a line segment from the start coordinate at a given angle'''
    x = min(width, start[0] + (width*np.cos(np.radians(-angle))))
    y = min(height, start[1] + (height*np.sin(np.radians(-angle))))
    return (x,y)

def ccw(a,b,c):
    '''Check if three points are in counter-clockwise order.'''
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

def intersect(a,b,c,d):
    '''Check if two line segments intersect.'''
    return ccw(a,c,d) != ccw(b,c,d) and ccw(a,b,c) != ccw(a,b,d)

def is_bbox_valid(point_start, point_end, bb2):
    """
    Check if a bounding box is valid based on the intersection of the line segment with the bounding box.
    """
    coords_2 = get_corner_coords(bb2)
    
    if intersect(point_start, point_end, coords_2[0], coords_2[1]) or \
        intersect(point_start, point_end, coords_2[1], coords_2[2]) or \
            intersect(point_start, point_end, coords_2[2], coords_2[3]) or \
                intersect(point_start, point_end, coords_2[3], coords_2[0]):
                    return True
    return False

def get_valid_bboxes(bb_tok_pair, theta, width = 762, height = 1000):
    """
    Get valid bounding boxes within the given angle range from a reference bounding box.

    Parameters:
        bb_tok_pair (list): List of bounding box-token pairs for each sample.
        theta (float): The angle in radians used for filtering bounding boxes.
        width (int): The width of the image. 
        height (int): The height of the image.

    Returns:
        defaultdict: A dictionary mapping each bounding box-token pair to a list of valid bounding box-token pairs within the angle range.
    """
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
    """
    Create edges based on the closest edges found by drawing a ray/line segment from the centroid of a bounding box.
    (K-nearest neigbors at multiple angles heuristic)

    Parameters:
        bb_tok_pairs (list): List of lists containing bounding box-token pairs for each sample.
        theta (float): The angle in degrees used for finding closest edges.
        width_height_list (list): List of tuples containing width and height of each image corresponding to the bounding boxes.
        num_edges (int): The maximum number of edges to consider.
        threshold (int): The maximum distance threshold for considering an edge.
        plot_flag (bool): Flag to indicate whether to plot the edges. (used when visualizing test images)

    Returns:
        list: A list of dictionaries containing edge mappings for each sample in the dataset.
              Each dictionary maps a bounding box-token pair to a list of connected bounding box-token pairs.
    """

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
    """
    Get index-mapping dictionaries for bounding box-token pairs.

    Parameters:
        bb_tok_pairs (list): List of lists containing bounding box-token pairs for each sample.

    Returns:
        tuple: A tuple containing two dictionaries.
               - The first dictionary maps indices to bounding box-token pairs.
               - The second dictionary maps bounding box-token pairs to indices.
    """
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

def get_edges_idxs_new(bb_tok_pairs, num_edges = 4):
    """
    Get indices of edges between bounding box-token pairs.

    Parameters:
        bb_tok_pairs (list): List of lists containing bounding box-token pairs for each sample.
        types (list): List of edge types to consider. Default is ['vd', 'vu', 'hr', 'hl'].

    Returns:
        list: A list of arrays containing edge indices for each sample in the dataset.
    """
    edge_maps = make_edges_new(bb_tok_pairs, num_edges)
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


def get_edges_idxs_new_angles_v2(bb_tok_pairs, theta, width_height_list, num_edges = 4):
    """
    Get indices of edges based on the closest edges found by drawing a ray/line segment from the centroid of a bounding box.
    (K-nearest neigbors at multiple angles heuristic)

    Parameters:
        bb_tok_pairs (list): List of lists containing bounding box-token pairs for each sample.
        theta (float): The angle in degrees used for finding closest edges.
        width_height_list (list): List of tuples containing width and height of each image corresponding to the bounding boxes.

    Returns:
        list: A list of arrays containing edge indices for each sample in the dataset.
    """
    edge_maps = make_edges_new_angles_v2(bb_tok_pairs, theta, width_height_list, num_edges)
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

def make_edges_new(bb_tok_pairs, num_edges = 4, plot_flag = False):
    """
    Create edges based on bounding box-token pairs.

    Parameters:
        bb_tok_pairs (list): List of lists containing bounding box-token pairs for each sample.
        num_edges (int): Number of edges to consider for each bounding box-token pair. Default is 4.
        plot_flag (bool): Flag to indicate whether to plot the edges. (used when visualizing test images)

    Returns:
        list: A list of dictionaries containing edge mappings for each sample in the dataset.
        Each dictionary maps a bounding box-token pair to a list of connected bounding box-token pairs.
    """
   
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

def get_adjs_new(dataset, num_edges = 4, feats_shape = 512):
    """
    Get adjacency matrices based on the closest edge using line segments of the bounding boxes.
    (K-nearest neighbors in space heuristic)

    Parameters:
        dataset (dict): The dataset containing input IDs and bounding boxes.
        feats_shape (int): The shape of the feature vectors.

    Returns:
        list: A list of adjacency matrices for each sample in the dataset.
              Each adjacency matrix is represented as a numpy array.
    """
    input_ids = dataset['input_ids']
    bboxes = dataset['bbox']
    bboxes = [[tuple(x) for x in bb]for bb in bboxes]
    bb_tok_pairs = [[(bb, tok) for bb, tok in zip(bbox, in_id)] for bbox, in_id in zip(bboxes, input_ids)]
    edges_idxs = get_edges_idxs_new(bb_tok_pairs, num_edges)
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

def get_adjs_new_angles_v2(dataset, theta, width_height_list, num_edges = 4, feats_shape = 512):
    """
    Get adjacency matrices based on the closest edges found by drawing a ray/line segment from the centroid of a bounding box.
    (K-nearest neigbors at multiple angles heuristic)

    Parameters:
        dataset (dict): The dataset containing input IDs, bounding boxes, etc.
        theta (float): The angle in degrees used for finding closest edges.
        width_height_list (list): List of tuples containing width and height of each image in the dataset.
        feats_shape (int): The shape of the feature vectors.

    Returns:
        list: A list of adjacency matrices for each sample in the dataset.
              Each adjacency matrix is represented as a numpy array.
    """
    input_ids = dataset['input_ids']
    bboxes = dataset['bbox']
    bboxes = [[tuple(x) for x in bb]for bb in bboxes]
    bb_tok_pairs = [[(bb, tok) for bb, tok in zip(bbox, in_id)] for bbox, in_id in zip(bboxes, input_ids)]
    edges_idxs = get_edges_idxs_new_angles_v2(bb_tok_pairs, theta, width_height_list, num_edges)
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

def normalize_adj(mx):
    """
    Row-normalize sparse matrix.
    Parameters:
        mx (scipy.sparse matrix): Sparse matrix to be normalized.
    Returns:
        scipy.sparse matrix: Row-normalized sparse matrix.
    """
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

def normalize_features(mx):
    """
    Row-normalize sparse matrix.

    Parameters:
        mx (scipy.sparse matrix): Sparse matrix to be normalized.
    Returns:
        scipy.sparse matrix: Row-normalized sparse matrix.
    """
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

