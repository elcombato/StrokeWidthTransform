# -*- encoding: utf-8 -*-
from __future__ import division
import math
import numpy as np
import cv2
from scipy.spatial import KDTree

img_write = True


def scrub(filepath):
    """
    Apply Stroke-Width Transform to image.

    :param filepath: relative or absolute filepath to source image
    :return: numpy array representing result of transform
    """
    canny, sobelx, sobely, theta = _create_derivative(filepath)
    swt = _swt(theta, canny, sobelx, sobely)
    comp = _connect_components(swt)
    swts, heights, widths, topleft_pts, images = _find_letters(swt, comp)
    word_images = _find_words(swts, heights, widths, topleft_pts, images)

    final_mask = np.zeros(swt.shape)
    for word in word_images:
        final_mask += word
    return final_mask


def _create_derivative(filepath):
    img = cv2.imread(filepath, 0)
    edges = cv2.Canny(img, 175, 320, apertureSize=3)
    # Create gradient map using Sobel
    sobelx64f = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=-1)
    sobely64f = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=-1)

    theta = np.arctan2(sobely64f, sobelx64f)
    if img_write:
        cv2.imwrite('edges.jpg', edges)
        cv2.imwrite('sobelx64f.jpg', np.absolute(sobelx64f))
        cv2.imwrite('sobely64f.jpg', np.absolute(sobely64f))
        # amplify theta for visual inspection
        theta_visible = (theta + np.pi)*255/(2*np.pi)
        cv2.imwrite('theta.jpg', theta_visible)
    return edges, sobelx64f, sobely64f, theta


def _swt(theta, edges, sobelx64f, sobely64f):

    max_ray_len = 100
    max_angl_diff = math.pi / 2

    # create empty image, initialized to infinity
    swt = np.empty(theta.shape)
    swt[:] = np.Infinity
    rays = []

    # Determine gradient-direction [d] for all edges
    step_x = -1 * sobelx64f
    step_y = -1 * sobely64f
    mag = np.sqrt(step_x * step_x + step_y * step_y)

    with np.errstate(divide='ignore', invalid='ignore'):
        d_all_x = step_x / mag
        d_all_y = step_y / mag

    # Scan edge-image for rays [p]====[q]
    for p_x in range(edges.shape[1]):
        for p_y in range(edges.shape[0]):

            # Start ray if [p] is on edge
            if edges[p_y, p_x] > 0:
                d_p_x = d_all_x[p_y, p_x]
                d_p_y = d_all_y[p_y, p_x]
                if math.isnan(d_p_x) or math.isnan(d_p_y):
                    continue
                ray = [(p_x, p_y)]
                prev_x, prev_y, i = p_x, p_y, 0

                # Moving in the gradient direction [d_p] to search for ray-terminating [q]
                while True:
                    i += 1
                    q_x = math.floor(p_x + d_p_x * i)
                    q_y = math.floor(p_y + d_p_y * i)
                    if q_x != prev_x or q_y != prev_y:
                        try:
                            # Terminate ray if [q] is on edge
                            if edges[q_y, q_x] > 0:
                                ray.append((q_x, q_y))
                                # Check if length of ray is above threshold
                                if len(ray) > max_ray_len:
                                    break
                                # Check if gradient direction is roughly opposite
                                # d_q_x = d_all_x[q_y, q_x]
                                # d_q_y = d_all_y[q_y, q_x]
                                delta = max(min(d_p_x * -d_all_x[q_y, q_x] + d_p_y * -d_all_y[q_y, q_x], 1.0), -1.0)
                                if not math.isnan(delta) and math.acos(max([-1.0, min([1.0, delta])])) < max_angl_diff:
                                    # Save the ray and set SWT-values of ray-pixel
                                    ray_len = math.sqrt((q_x - p_x) ** 2 + (q_y - p_y) ** 2)
                                    for (rp_x, rp_y) in ray:
                                        swt[rp_y, rp_x] = min(ray_len, swt[rp_y, rp_x])
                                    rays.append(np.asarray(ray))
                                break
                            # If [q] is neither on edge nor out of bounds, append to ray
                            ray.append((q_x, q_y))
                        # Reached image boundary
                        except IndexError:
                            break
                        prev_x = q_x
                        prev_y = q_y

    if img_write:
        cv2.imwrite('swt_.jpg', swt * 100)

    # Compute median SWT
    for ray in rays:
        median = np.median(swt[ray[:, 1], ray[:, 0]])
        for (p_x, p_y) in ray:
            swt[p_y, p_x] = min(median, swt[p_y, p_x])

    if img_write:
        cv2.imwrite('swt.jpg', swt * 100)

    return swt


def _connect_components(swt):
    # Implementation of disjoint-set
    class Label(object):
        def __init__(self, value):
            self.value = value
            self.parent = self
            self.rank = 0

        def __eq__(self, other):
            if type(other) is type(self):
                return self.value == other.value
            else:
                return False

        def __ne__(self, other):
            return not self.__eq__(other)

    def make_set(x):
        try:
            return ld[x]
        except KeyError:
            ld[x] = Label(x)
            return ld[x]

    def find(item):
        if item.parent != item:
            item.parent = find(item.parent)
        return item.parent

    def union(x, y):
        """
        :param x:
        :param y:
        :return: root node of new union tree
        """
        x_root = find(x)
        y_root = find(y)
        if x_root == y_root:
            return x_root

        if x_root.rank < y_root.rank:
            x_root.parent = y_root
            return y_root
        elif x_root.rank > y_root.rank:
            y_root.parent = x_root
            return x_root
        else:
            y_root.parent = x_root
            x_root.rank += 1
            return x_root

    ld = {}

    # apply Connected Component algorithm, comparing SWT values.
    # components with a SWT ratio less extreme than 1:3 are assumed to be
    # connected. Apply twice, once for each ray direction/orientation, to
    # allow for dark-on-light and light-on-dark texts
    trees = {}
    # Assumption: we'll never have more than 65535-1 unique components
    label_map = np.zeros(shape=swt.shape, dtype=np.uint16)
    next_label = 1
    # First Pass, raster scan-style
    swt_ratio_thresh = 3.0
    for y in range(swt.shape[0]):
        for x in range(swt.shape[1]):
            sw_point = swt[y, x]
            if 0 < sw_point < np.Infinity:
                neighbors = [(y, x-1),    # west
                             (y-1, x-1),  # northwest
                             (y-1, x),    # north
                             (y-1, x+1)]  # northeast
                connected_neighbors = None
                neighborvals = []

                for neighbor in neighbors:
                    try:
                        sw_n = swt[neighbor]
                        label_n = label_map[neighbor]
                    # out of image boundary
                    except IndexError:
                        continue
                    # labeled neighbor pixel within SWT ratio threshold
                    if label_n > 0 and sw_n / sw_point < swt_ratio_thresh and sw_point / sw_n < swt_ratio_thresh:
                        neighborvals.append(label_n)
                        if connected_neighbors:
                            connected_neighbors = union(connected_neighbors, make_set(label_n))
                        else:
                            connected_neighbors = make_set(label_n)

                if not connected_neighbors:
                    # We don't see any connections to North/West
                    trees[next_label] = (make_set(next_label))
                    label_map[y, x] = next_label
                    next_label += 1
                else:
                    # We have at least one connection to North/West
                    label_map[y, x] = min(neighborvals)
                    # For each neighbor, make note that their respective connected_neighbors are connected
                    # for label in connected_neighbors.
                    # @TODO: do I need to loop at all neighbor trees?
                    trees[connected_neighbors.value] = union(trees[connected_neighbors.value], connected_neighbors)

    # Second pass. re-base all labeling with representative label for each connected tree
    layers = {}
    for x in range(swt.shape[1]):
        for y in range(swt.shape[0]):
            if label_map[y, x] > 0:
                item = ld[label_map[y, x]]
                common_label = find(item).value
                label_map[y, x] = common_label
                try:
                    layer = layers[common_label]
                except KeyError:
                    layers[common_label] = {'x': [], 'y': []}
                    layer = layers[common_label]

                layer['x'].append(x)
                layer['y'].append(y)
    return layers


def _find_letters(swt, comp):
    img_w = swt.shape[0]
    img_h = swt.shape[1]
    # STEP: Discard shapes that are probably not letters
    swts = []
    heights = []
    widths = []
    topleft_pts = []
    images = []

    for _, c in comp.items():
        east, west, south, north = max(c['x']), min(c['x']), max(c['y']), min(c['y'])
        width, height = east - west, south - north

        if width < 8 or height < 8:
            continue

        if width / height > 10 or height / width > 10:
            continue

        diameter = math.sqrt(width**2 + height**2)
        median_swt = np.median(swt[(c['y'], c['x'])])
        if diameter / median_swt > 10:
            continue

        if width / img_w > 0.4 or height / img_h > 0.4:
            continue

        # we use log_base_2 so we can do linear distance comparison later using k-d tree
        # ie, if log2(x) - log2(y) > 1, we know that x > 2*y
        # Assumption: we've eliminated anything with median_swt == 1
        swts.append([math.log(median_swt, 2)])
        heights.append([math.log(height, 2)])
        topleft_pts.append(np.asarray([north, west]))
        widths.append(width)
        fulllayer = np.zeros((img_w, img_h), dtype=np.uint16)
        for i in range(len(c['y'])):
            fulllayer[c['y'][i], c['x'][i]] = 1
        images.append(fulllayer)

    return swts, heights, widths, topleft_pts, images


def _find_words(swts, heights, widths, topleft_pts, images):
    # Find all shape pairs that have similar median stroke widths
    swt_tree = KDTree(np.asarray(swts))
    stp = swt_tree.query_pairs(1)

    # Find all shape pairs that have similar heights
    height_tree = KDTree(np.asarray(heights))
    htp = height_tree.query_pairs(1)

    # Intersection of valid pairings
    isect = htp.intersection(stp)

    chains = []
    pairs = []
    pair_angles = []
    for pair in isect:
        left = pair[0]
        right = pair[1]
        widest = max(widths[left], widths[right])
        distance = np.linalg.norm(topleft_pts[left] - topleft_pts[right])
        if distance < widest * 3:
            delta_yx = topleft_pts[left] - topleft_pts[right]
            angle = np.arctan2(delta_yx[0], delta_yx[1])
            if angle < 0:
                angle += math.pi

            pairs.append(pair)
            pair_angles.append(np.asarray([angle]))

    angle_tree = KDTree(np.asarray(pair_angles))
    atp = angle_tree.query_pairs(np.pi/12)

    for pair_idx in atp:
        pair_a = pairs[pair_idx[0]]
        pair_b = pairs[pair_idx[1]]
        left_a = pair_a[0]
        right_a = pair_a[1]
        left_b = pair_b[0]
        right_b = pair_b[1]

        # @todo - this is O(n^2) or similar, extremely naive. Use a search tree.
        added = False
        for chain in chains:
            if left_a in chain:
                chain.add(right_a)
                added = True
            elif right_a in chain:
                chain.add(left_a)
                added = True
        if not added:
            chains.append(set([left_a, right_a]))
        added = False
        for chain in chains:
            if left_b in chain:
                chain.add(right_b)
                added = True
            elif right_b in chain:
                chain.add(left_b)
                added = True
        if not added:
            chains.append(set([left_b, right_b]))

    word_images = []
    for chain in [c for c in chains if len(c) > 3]:
        for idx in chain:
            word_images.append(images[idx])

    return word_images


if __name__ == '__main__':
    final_mask = scrub("test.jpg")
    cv2.imwrite('final.jpg', final_mask * 255)