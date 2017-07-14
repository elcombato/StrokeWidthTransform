import math
import numpy as np
import cv2
import os
from time import time
from scipy.spatial import KDTree


def scrub(filepath, verbose=0):
    """
    Apply Stroke-Width Transform to image.

    :param filepath: relative or absolute filepath to source image
    :return: numpy array representing result of transform
    """
    # find components with SWT
    img_gray = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    canny, sobelx, sobely, theta = _create_derivative(img_gray, verbose)
    swt = _swt(theta, canny, sobelx, sobely, verbose)
    comp = _connect_components(swt)

    # find letter-candidates and words
    letter, letter_imgs = _find_letters(swt, comp)
    words = _find_words(letter)

    # create bounding boxes around letter-candidates and words
    letter_pts = _bounding_box(letter_data=letter)
    words_pts = _bounding_box(letter_data=letter, word_lists=words)

    # draw bounding boxes on image and return image
    img_rgb = cv2.imread(filepath, cv2.IMREAD_COLOR)
    for i, [pt1y, pt1x, pt2y, pt2x] in enumerate(letter_pts):
        cv2.rectangle(img_rgb, (pt1x, pt1y), (pt2x, pt2y), (0, 0, 255), 1)
        cv2.putText(img_rgb, str(i), (pt1x, pt1y-2), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 255))
    for i, [pt1y, pt1x, pt2y, pt2x] in enumerate(words_pts):
        cv2.rectangle(img_rgb, (pt1x, pt1y), (pt2x, pt2y), (255, 0, 0), 2)

    return img_rgb


def _bounding_box(letter_data, word_lists=None):

    # north-west point [pt1]
    pt1 = letter_data[:, 3:5].astype(int)

    # south-east point [pt2] = (north + height, west + width)
    pt2_y = letter_data[:, 3] + letter_data[:, 1]
    pt2_x = letter_data[:, 4] + letter_data[:, 2]
    pt2 = np.vstack([pt2_y, pt2_x]).T.astype(int)

    if word_lists is None:
        return np.column_stack([pt1, pt2])

    return np.array([[pt1[list(w)][:, 0].min(),
                      pt1[list(w)][:, 1].min(),
                      pt2[list(w)][:, 0].max(),
                      pt2[list(w)][:, 1].max()] for w in word_lists])


def _create_derivative(img, verbose=0):
    edges = cv2.Canny(img, 250, 400, apertureSize=3) # 175, 320
    # Create gradient map using Sobel
    sobelx64f = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=-1)
    sobely64f = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=-1)

    theta = np.arctan2(sobely64f, sobelx64f)
    if verbose > 0:
        cv2.imwrite('../_data/edges.jpg', edges)
        cv2.imwrite('../_data/sobelx64f.jpg', np.absolute(sobelx64f))
        cv2.imwrite('../_data/sobely64f.jpg', np.absolute(sobely64f))
        # amplify theta for visual inspection
        theta_visible = (theta + np.pi)*255/(2*np.pi)
        cv2.imwrite('../_data/theta.jpg', theta_visible)
    return edges, sobelx64f, sobely64f, theta


def _swt(theta, edges, sobelx64f, sobely64f, verbose=0):

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

    # Compute median SWT
    for ray in rays:
        median = np.median(swt[ray[:, 1], ray[:, 0]])
        for (p_x, p_y) in ray:
            swt[p_y, p_x] = min(median, swt[p_y, p_x])

    if verbose > 0:
        cv2.imwrite('../_data/swt.jpg', swt * 100)

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
    letter_infs = []
    letter_imgs = []

    for _, c in comp.items():
        east, west, south, north = max(c['x']), min(c['x']), max(c['y']), min(c['y'])
        width, height = east - west, south - north

        if width < 8 and height < 8:
            continue
        if width < 4 or height < 4:
            continue
        if width / height > 10 or height / width > 10:
            continue

        diameter = math.sqrt(width**2 + height**2)
        median_swt = np.median(swt[(c['y'], c['x'])])
        if diameter / median_swt > 15:  # TODO: this threshold can be improved?
            continue

        if width / img_w > 0.4 or height / img_h > 0.4:
            continue

        letter_img = np.zeros((img_w, img_h))
        for i in range(len(c['y'])):
            letter_img[c['y'][i], c['x'][i]] = 1
        letter_inf = [
            median_swt,
            height,
            width,
            north,
            west
        ]
        letter_infs.append(letter_inf)
        letter_imgs.append(letter_img)

    return np.asarray(letter_infs), np.asarray(letter_imgs)


def _find_words(letr_inf): # swts, heights, widths, topleft_pts, images):
    # Index-pairs of letter with similar median stroke widths and similar heights
    # We use log2 for linear distance comparison in KDTree
    # (i.e. if log2(x) - log2(y) > 1, we know that x > 2*y)
    s_ix_letr_pairs = KDTree(np.log2(letr_inf[:, 0:1])).query_pairs(1)
    h_ix_letr_pairs = KDTree(np.log2(letr_inf[:, 1:2])).query_pairs(1)

    # Calc the angle (direction of text) for all letter-pairs which are
    # similar and close to each other
    pairs = []
    for ix_letr1, ix_letr2 in h_ix_letr_pairs.intersection(s_ix_letr_pairs):
        diff = letr_inf[ix_letr1, 3:5] - letr_inf[ix_letr2, 3:5]
        # Distance between letters smaller than
        # 3 times the width of the wider letter
        dist = np.linalg.norm(diff)
        if dist < max(letr_inf[ix_letr1, 2], letr_inf[ix_letr2, 2]) * 3:
            angle = math.atan2(diff[0], diff[1])
            angle += math.pi if angle < 0 else 0
            pairs.append([ix_letr1, ix_letr2, angle])
    pairs = np.asarray(pairs)

    # Pairs of letter-pairs with a similar angle (direction of text)
    a_ix_pair_pairs = KDTree(pairs[:, 2:3]).query_pairs(math.pi / 12)

    chains = []
    for ix_pair_a, ix_pair_b in a_ix_pair_pairs:
        # Letter pairs [a] & [b] have a similar angle and each pair consists of
        # letter [1] & [2] which meet the similarity-requirements.
        pair_a_letr1, pair_a_letr2 = int(pairs[ix_pair_a, 0]), int(pairs[ix_pair_a, 1])
        pair_b_letr1, pair_b_letr2 = int(pairs[ix_pair_b, 0]), int(pairs[ix_pair_b, 1])

        # TODO: not correct?
        added = False
        for c in chains:
            if pair_a_letr1 in c:
                c.add(pair_a_letr2)
                added = True
            elif pair_a_letr2 in c:
                c.add(pair_a_letr1)
                added = True
        if not added:
            chains.append({pair_a_letr1, pair_a_letr2})
        added = False
        for c in chains:
            if pair_b_letr1 in c:
                c.add(pair_b_letr2)
                added = True
            elif pair_b_letr2 in c:
                c.add(pair_b_letr1)
                added = True
        if not added:
            chains.append({pair_b_letr1, pair_b_letr2})
    chains = np.asarray(chains)

    # List of sets of letters with possibly many duplicates
    # return chains
    # Single list of unique letters
    # return np.unique([int(ix) for chain in chains if len(chain) >= 3 for ix in chain])

    vecfunc = np.vectorize(len)
    chains = chains[vecfunc(chains) > 3]
    _, uniq_ix = np.unique(chains.astype(str), return_index=True)
    return chains[uniq_ix]


if __name__ == '__main__':

    for img_f in os.scandir("../_data/stills/"):
        if img_f.name.endswith('.PNG'):
            print(f'start analysis for {img_f.name} ..')
            tick = time()
            result_img = scrub(img_f.path)
            print(f'.. finished ({time()-tick:.2f}s)')
            cv2.imwrite(f"../_data/out/{img_f.name}", result_img)
    print('DONE')
