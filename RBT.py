# import packages
import numpy as np

def rbt(img_in):
    img_class = img_in.dtype
    if img_class == np.uint8:
        format_factor = 2**8 - 1
    elif img_class == np.uint16:
        format_factor = 2**16 - 1
    elif img_class == np.float32:
        format_factor = 1
    elif img_class == np.float64:
        format_factor = 1

    unique_vals, rank = np.unique(img_in, return_inverse=True)
    u = len(unique_vals)
    rbt = format_factor * (rank-1) / (u-1)
    img_rbt = rbt.astype(img_class)
    return img_rbt