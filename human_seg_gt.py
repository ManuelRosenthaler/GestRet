import cv2
import numpy as np
from scipy import ndimage, array
from pprint import pprint

# ======================================================================================================================
# All the changes i did to this file are in the function human_seg_combine_argmax
# ======================================================================================================================

#part_ids = [ 0,  2,  4,  5,  6,  8, 13, 14, 16, 17, 18, 19, 20, 21, 24]
# the order is: (left right flipped)
# background, head, torso, left upper arm ,right upper arm, left forearm, right forearm,
#  left hand, right hand, left thigh, right thigh, left shank, right shank, left foot, right foot
part_ids = [0, 13, 2, 5, 8, 19, 20, 4, 24, 18, 6, 21, 16, 14, 17]

r_chan = [0, 127, 255, 255, 255, 127, 255, 127, 0, 0, 0, 0, 127, 255, 255]
g_chan = [0, 127, 0, 127, 255, 0, 0, 127, 255, 0, 255, 127, 255, 127, 255]
b_chan = [0, 127, 0, 0, 0, 255, 255, 0, 255, 255, 0, 255, 127, 127, 127]

def human_seg_spread_channel(human_seg_map):
    x = human_seg_map // 127
    x = x * np.array([9, 3, 1])
    x = np.add.reduce(x, 2)
    res = []
    for i in part_ids:
        res.append((x == i))
    res = np.stack(res, axis=-1)
    return res.astype(np.float32)

def human_seg_combine_channel(human_seg_split_map):
    r_chan_seg = np.add.reduce(human_seg_split_map * np.array(r_chan), 2)
    g_chan_seg = np.add.reduce(human_seg_split_map * np.array(g_chan), 2)
    b_chan_seg = np.add.reduce(human_seg_split_map * np.array(b_chan), 2)
    return np.stack([b_chan_seg, g_chan_seg, r_chan_seg], axis=-1).astype(np.uint8)

def human_seg_combine_argmax(human_seg_argmax_map):
    print('================================================')

    kernel = ndimage.generate_binary_structure(3, 2)

    # Here all the body parts that i want ( e.i. left hand, right hand, left forearm, right forearm, left upper arm and
    # right upper arm) are selected to be segmented. The other body parts are ignored)
    onehot = np.stack([(human_seg_argmax_map == i).astype(np.uint8) for i in [8]], axis=-1)
    onehot2 = np.stack([(human_seg_argmax_map == i).astype(np.uint8) for i in [7]], axis=-1)
    onehot3 = np.stack([(human_seg_argmax_map == i).astype(np.uint8) for i in [6]], axis=-1)
    onehot4 = np.stack([(human_seg_argmax_map == i).astype(np.uint8) for i in [5]], axis=-1)
    onehot5 = np.stack([(human_seg_argmax_map == i).astype(np.uint8) for i in [4]], axis=-1)
    onehot6 = np.stack([(human_seg_argmax_map == i).astype(np.uint8) for i in [3]], axis=-1)

    # Since it does not matter, what colour the masks will have, we can simply add all masks together to end up with
    # one big mask
    onehot = onehot + onehot2 + onehot3 + onehot4 + onehot5 + onehot6
    # Here the dilation around the mask is done. This is necessary to have a higher chance to include the fingers in the
    # Mask
    onehot = ndimage.binary_dilation(onehot, structure=kernel, iterations=10).astype(onehot.dtype)

    return human_seg_combine_channel(onehot)
