import numpy as np

def rbt(img_in):
    """
    Apply Rank-Biased Transformation (RBT) to the input image.

    Parameters:
    - img_in (numpy.ndarray): Input image to be transformed.

    Returns:
    - numpy.ndarray: Transformed image after Rank-Biased Transformation.
    """
    img_class = img_in.dtype

    # Determine format factor based on input image data type
    if img_class == np.uint8:
        format_factor = 2**8 - 1
    elif img_class == np.uint16:
        format_factor = 2**16 - 1
    elif img_class == np.float32:
        format_factor = 1
    elif img_class == np.float64:
        format_factor = 1

    # Compute unique values and ranks for the input image
    unique_vals, rank = np.unique(img_in, return_inverse=True)
    u = len(unique_vals)

    # Perform Rank-Biased Transformation
    rbt = format_factor * (rank - 1) / (u - 1)

    # Convert the result to the original data type of the input image
    img_rbt = rbt.astype(img_class)

    return img_rbt
