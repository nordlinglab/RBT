# -*- coding: utf-8 -*-

"""
Python 2.7 script to perform histogram equalization and RBT on the example image Unequalized_Hawkes_Bay_NZ.jpg at
https://en.wikipedia.org/wiki/Histogram_equalization

This script reads a grayscale image and displays three versions:
1. Original
2. Histogram Equalized
3. Rank-Based Transformation (RBT)
4. Adaptive Histogram Equalization (AHE) computes a local-scale CDF in neighborhood regions, mapping pixel values based on their local ranks
5. Contrast-Limited AHE (CLAHE) limits histogram amplification by clipping peaks before computing the CDF
6. Bi-Histogram Equalization (BHE) divides the histogram at image mean and equalizes each sub‑histogram separately to preserve brightness

This script processes an image using Histogram Equalization and Rank-Based
Transformation (RBT). It saves the resulting images as PNG files and
generates PDF plots for the histograms and CDFs.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.lines as mlines
import cv2  # pip install opencv-contrib-python
from skimage import exposure  # pip install scikit-image
from sklearn.mixture import GaussianMixture


def hist_equalize(img_in):
    """
    Performs Histogram Equalization on a grayscale image.
    
    :param img_in: A numpy array representing the input grayscale image.
    :return: A numpy array representing the equalized image.
    """
    # Get image histogram
    hist, bins = np.histogram(img_in.flatten(), 256, [0, 256])

    # Calculate the cumulative distribution function (CDF)
    cdf = hist.cumsum()

    # Mask zeros in the CDF and apply the equalization formula
    cdf_masked = np.ma.masked_equal(cdf, 0)
    cdf_masked = (cdf_masked - cdf_masked.min()) * 255 / (cdf_masked.max() - cdf_masked.min())

    # Fill back the masked values and cast to uint8
    cdf_final = np.ma.filled(cdf_masked, 0).astype('uint8')

    # Apply the new CDF to the image to create the equalized image
    img_out = cdf_final[img_in]
    
    return img_out

def rbt(img_in):
    """
    Performs a robust Rank-Based Transformation (RBT) on an image.
    
    :param img_in: A numpy array representing the input grayscale image.
    :return: A numpy array representing the RBT image.
    """
    img_class = img_in.dtype
    format_factor = 255.0  # Use float for division

    original_shape = img_in.shape
    unique_vals, rank = np.unique(img_in, return_inverse=True)
    u = len(unique_vals)

    if u <= 1:
        return img_in.copy()

    rbt_values = format_factor * rank / (u - 1.0)
    img_out = rbt_values.reshape(original_shape).astype(img_class)

    return img_out

def ahe(img_in):
    """Adaptive histogram equalization (AHE), global clip."""
    img = img_in.astype(np.uint8)
    # From skimage: equalize_adapthist returns float in [0,1]
    img_eq = exposure.equalize_adapthist(img, kernel_size=None, clip_limit=1.0)
    return (img_eq * 255).astype(np.uint8)

def clahe_open_cv(img_in, clip=2.0, tile=(8,8)):
    """Contrast-Limited Adaptive Histogram Equalization using OpenCV."""
    img = img_in.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    return clahe.apply(img)

def bhe(img_in):
    """Bi-Histogram Equalization (BBHE variant), brightness-preserving."""
    img = img_in.astype(np.uint8)
    # hist, bins = np.histogram(img.flatten(), 256, [0,256]) # This line is not used
    mean = img.mean() # This will be a float

    # Ensure mean is an integer for splitting, rounding to nearest int
    mean_int = int(round(mean))

    # Split histogram at mean intensity
    lower = img <= mean_int
    upper = img > mean_int

    out = img.copy()

    # Process lower sub-image
    if np.sum(lower) > 0: # Check if there are any pixels in the lower half
        hist_l, _ = np.histogram(img[lower], mean_int + 1, [0, mean_int + 1])
        cdf_l = np.cumsum(hist_l)
        # Avoid division by zero if cdf_l[-1] is 0
        cdf_l_min = cdf_l[cdf_l > 0].min() if np.sum(cdf_l > 0) > 0 else 0
        map_l = (cdf_l - cdf_l_min) * mean_int / (cdf_l[-1] - cdf_l_min + 1e-8) # Add a small epsilon to avoid division by zero
        map_l = np.round(map_l).astype(np.uint8)
        # Apply the mapping to the lower part of the image
        out[lower] = map_l[img[lower]] # img[lower] are already integer indices

    # Process upper sub-image
    if np.sum(upper) > 0: # Check if there are any pixels in the upper half
        hist_u, _ = np.histogram(img[upper], 255 - mean_int, [mean_int + 1, 256])
        cdf_u = np.cumsum(hist_u)
        # Avoid division by zero if cdf_u[-1] - cdf_u[0] is 0
        cdf_u_min = cdf_u[cdf_u > 0].min() if np.sum(cdf_u > 0) > 0 else 0
        map_u = ((cdf_u - cdf_u_min) * (255 - mean_int) / (cdf_u[-1] - cdf_u_min + 1e-8) + mean_int)
        map_u = np.round(map_u).astype(np.uint8)

        # To use img[upper] directly as an index, we need to shift its values
        # so they start from 0 for indexing map_u.
        # map_u is created for values from (mean_int + 1) to 255.
        # So, map_u[0] corresponds to original pixel value mean_int + 1,
        # map_u[1] corresponds to mean_int + 2, and so on.
        # Thus, the index for map_u should be (original_pixel_value - (mean_int + 1)).
        out[upper] = map_u[img[upper] - (mean_int + 1)]

    return out

def gmmce(img_in, n_components=3):
    """
    Gaussian Mixture Model-based Contrast Enhancement (GMMCE).
    
    :param img_in: Grayscale input image as numpy array.
    :param n_components: Number of Gaussian components to use.
    :return: GMM contrast enhanced image.
    """
    img_flat = img_in.flatten().reshape(-1, 1).astype(np.float64)

    # Fit a Gaussian Mixture Model
    gmm = GaussianMixture(n_components=n_components, covariance_type='tied', random_state=0)
    gmm.fit(img_flat)

    # Compute cumulative probabilities for ranking
    pdf = np.exp(gmm.score_samples(img_flat))
    cdf = np.argsort(np.argsort(pdf)).astype(np.float64)
    cdf /= cdf.max()  # Normalize to [0,1]

    # Scale to [0, 255]
    img_gmmce = (cdf * 255).astype(np.uint8).reshape(img_in.shape)

    return img_gmmce

def plot_and_save_histograms(img, img_equalized, img_rbt):
    """
    Creates and saves a 3x2 plot showing each image and its
    corresponding histogram/CDF plot.
    """
    fig, axes = plt.subplots(3, 2, figsize=(10, 13))
    plt.gray()

    def plot_single_hist_cdf(ax, img_data, image_label):
        hist, bins = np.histogram(img_data.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max() / cdf.max()

        ax.bar(bins[:-1], hist, width=1, color='red', label='Histogram')
        ax.plot(cdf_normalized, color='black', label='CDF')

        ax.set_xlabel('Pixel Intensity')
        ax.set_ylabel('Frequency')
        ax.set_xlim([0, 256])
        ax.legend(loc='center right')

    # Plot Original, Equalized, and RBT images and histograms
    images = [(img, 'Original'), (img_equalized, 'Equalized'), (img_rbt, 'RBT')]
    for i, (im, label) in enumerate(images):
        axes[i, 0].imshow(im)
        axes[i, 0].axis('off')
        # Use a text box instead of a title for the image
        axes[i, 0].text(0.05, 0.95, label, transform=axes[i, 0].transAxes,
                        fontsize=14, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        plot_single_hist_cdf(axes[i, 1], im, label)
    
    fig.tight_layout()
    plt.savefig('Unequalized_Hawkes_Bay_NZ_All_Histograms.pdf')
    plt.close(fig) # Close the figure to free memory

def plot_and_save_only_histograms_cdfs_separately(img, img_equalized, img_rbt):
    """
    Creates and saves separate 1x1 plots for each image's histogram/CDF plot,
    saved as individual PDF files.
    """
    def plot_single_hist_cdf(ax, img_data, image_label):
        hist, bins = np.histogram(img_data.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max() / cdf.max()

        ax.bar(bins[:-1], hist, width=1, color='red', label='Histogram')
        ax.plot(cdf_normalized, color='black', label='CDF')

        ax.set_xlabel('Pixel Intensity')
        ax.set_ylabel('Frequency')
        ax.set_xlim([0, 256])
        ax.legend(loc='center right')
        ax.set_title(image_label + ' Histogram and CDF') # Add a title for clarity

    images_to_plot = [
        (img, 'Original', 'Unequalized_Hawkes_Bay_NZ_Original_Hist_CDF_Only.pdf'),
        (img_equalized, 'Equalized', 'Unequalized_Hawkes_Bay_NZ_Equalized_Hist_CDF_Only.pdf'),
        (img_rbt, 'RBT', 'Unequalized_Hawkes_Bay_NZ_RBT_Hist_CDF_Only.pdf')
    ]

    for im_data, label, filename in images_to_plot:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5)) # Create a single subplot for hist/cdf
        plot_single_hist_cdf(ax, im_data, label)
        
        fig.tight_layout()
        plt.savefig(filename)
        plt.close(fig) # Close the figure to free memory

def plot_and_save_all_cdfs(img, img_equalized, img_rbt):
    """
    Calculates the CDF for each image version, plots them all on a
    single graph, and saves the plot as a PDF.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Define Brewer colors (from qualitative Set1 palette)
    colors = ['#e41a1c', '#377eb8', '#4daf4a']
    
    # Data to plot: (image_data, label, color)
    data_to_plot = [
        (img, 'Original', 'black'), # Keeping original black for reference
        (img_equalized, 'Equalized', colors[1]), # Blue
        (img_rbt, 'RBT', colors[2])  # Green
    ]

    for im, label, color in data_to_plot:
        hist, bins = np.histogram(im.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        # Normalize CDF to a probability scale [0, 1]
        cdf_normalized = cdf / float(cdf.max())
        ax.plot(cdf_normalized, color=color, label=label)

    ax.set_xlabel('Pixel Intensity')
    ax.set_ylabel('Cumulative Probability')
    ax.set_xlim([0, 256])
    ax.set_ylim([0, 1])
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    
    plt.savefig('Unequalized_Hawkes_Bay_NZ_All_CDFs.pdf')
    plt.close(fig)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from skimage import exposure
from sklearn.mixture import GaussianMixture
import matplotlib.lines as mlines # Make sure to add this import at the top of your script


def plot_and_save_all_cdfs_extended(images, labels, colors, filename):
    """
    Updates the combined CDF plot function to include all methods with
    different markers placed at specified x-axis positions, with
    slight offsets to prevent overlap, and includes markers in the legend.
    """
    fig, ax = plt.subplots(figsize=(10, 7)) # Increased figure size slightly for better visibility of markers

    # Define a list of markers to cycle through for each method
    markers = ['o', 's', '^', 'D', 'v', 'X', 'P'] # Circle, Square, Triangle-up, Diamond, Triangle-down, X (filled), Plus (filled)

    # Base x-axis positions for markers (roughly 25% and 75% of the range 0-255)
    base_marker_x_pos_1 = int(256 * 0.25)
    base_marker_x_pos_2 = int(256 * 0.75)

    # Offset to spread markers horizontally for each line
    # Adjust this value based on visual inspection.
    # A larger value means more spread.
    x_offset_per_line = 5 # Pixels to offset each subsequent line's markers

    # Calculate total spread width to center the group of markers
    total_offset_width = (len(images) - 1) * x_offset_per_line
    starting_offset = -total_offset_width / 2.0 # To center the group around the base position

    legend_handles = [] # List to store custom legend handles (proxy artists)

    for i, (im, label, color) in enumerate(zip(images, labels, colors)):
        hist, bins = np.histogram(im.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf / float(cdf.max())

        # 1. Plot the CDF line WITHOUT any markers or label.
        # This line is just for the visual representation on the plot.
        ax.plot(cdf_normalized, color=color, linewidth=1.5)

        # 2. Calculate adjusted x-positions for the current line's markers
        adjusted_marker_x_pos_1 = int(base_marker_x_pos_1 + starting_offset + i * x_offset_per_line)
        adjusted_marker_x_pos_2 = int(base_marker_x_pos_2 + starting_offset + i * x_offset_per_line)

        # Ensure adjusted x-positions are within the valid range [0, 255]
        adjusted_marker_x_pos_1 = max(0, min(255, adjusted_marker_x_pos_1))
        adjusted_marker_x_pos_2 = max(0, min(255, adjusted_marker_x_pos_2))

        # Get the y-values from the CDF at these *adjusted* x-positions
        y_at_x1 = cdf_normalized[adjusted_marker_x_pos_1]
        y_at_x2 = cdf_normalized[adjusted_marker_x_pos_2]

        current_marker = markers[i % len(markers)]

        # 3. Plot the markers separately, without a label, using linestyle='None'.
        # These are the markers that will appear on the actual plot, spread out.
        ax.plot([adjusted_marker_x_pos_1, adjusted_marker_x_pos_2],
                [y_at_x1, y_at_x2],
                marker=current_marker,
                linestyle='None',
                color=color, # Keep marker color consistent with line
                markersize=8,
                markeredgecolor='black',
                markeredgewidth=0.8)

        # 4. Create a custom legend handle (proxy artist) for this line/marker combination.
        # This `Line2D` object defines how this specific entry will look in the legend.
        # It has both the line style and the marker.
        legend_handle = mlines.Line2D([], [], color=color, marker=current_marker,
                                      linestyle='-', linewidth=1.5, # Ensure it shows a line in legend
                                      markersize=8, markeredgecolor='black', markeredgewidth=0.8)
        legend_handles.append(legend_handle)

    ax.set_xlabel('Pixel Intensity')
    ax.set_ylabel('Cumulative Probability')
    ax.set_xlim([0, 256])
    ax.set_ylim([0, 1])
    ax.grid(True, linestyle='--', alpha=0.6)

    # Use the custom legend handles and the original labels for the legend.
    ax.legend(handles=legend_handles, labels=labels, loc='lower right', frameon=True, fancybox=True, shadow=True)

    plt.savefig(filename)
    plt.close(fig)

def main():
    """
    Main function to load image, perform transformations, and save all outputs.
    """
    image_file = 'images/Unequalized_Hawkes_Bay_NZ.jpg'
    print "Processing image: {}".format(image_file)

    try:
        img = mpimg.imread(image_file)
    except IOError:
        print "Error: Image file not found."
        return

    # Ensure image is 2D uint8 grayscale
    if img.ndim == 3: img = img[:, :, 0]
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)

    # 1. Perform transformations
    img_equalized = hist_equalize(img)
    img_rbt = rbt(img)
    img_ahe = ahe(img)
    img_clahe = clahe_open_cv(img)
    img_bhe = bhe(img)
    img_gmmce = gmmce(img)
    print "Image transformations complete."

    # 2. Save processed bitmap images as PNG
    plt.imsave('Unequalized_Hawkes_Bay_NZ_Equalized_Image.png', img_equalized, cmap='gray')
    plt.imsave('Unequalized_Hawkes_Bay_NZ_RBT_Image.png', img_rbt, cmap='gray')
    plt.imsave('Unequalized_Hawkes_Bay_NZ_AHE_Image.png', img_ahe, cmap='gray')
    plt.imsave('Unequalized_Hawkes_Bay_NZ_CLAHE_Image.png', img_clahe, cmap='gray')
    plt.imsave('Unequalized_Hawkes_Bay_NZ_BHE_Image.png', img_bhe, cmap='gray')
    plt.imsave('Unequalized_Hawkes_Bay_NZ_GMMCE_Image.png', img_gmmce, cmap='gray')
    print "Saved transformed images as PNG."

    # 3. Create and save combined histogram plots as PDF
    plot_and_save_histograms(img, img_equalized, img_rbt)
    print "Saved combined histogram plots as PDF."
    
    # 4. Create and save individual histogram+CDF plots without images as PDFs
    plot_and_save_only_histograms_cdfs_separately(img, img_equalized, img_rbt)
    print "Saved individual histogram+CDF plots (without images) as PDF."
    
    # 5. Create and save the combined CDF plot as PDF
    #plot_and_save_all_cdfs(img, img_equalized, img_rbt)
    # Call the extended CDF plotting function
    images_all = [img, img_equalized, img_rbt, img_ahe, img_bhe, img_gmmce]
    labels_all = ['Original', 'Equalized', 'RBT', 'AHE', 'BHE', 'GMMCE']
    images_all = [img, img_equalized, img_rbt, img_ahe, img_clahe, img_bhe, img_gmmce]
    labels_all = ['Original', 'Equalized', 'RBT', 'AHE', 'CLAHE', 'BHE', 'GMMCE']
    colors_all = ['black', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628', '#e41a1c']
    plot_and_save_all_cdfs_extended(images_all, labels_all, colors_all, 'Unequalized_Hawkes_Bay_NZ_All_Methods_CDF.pdf')
    print "Saved combined CDF plot as PDF."
    
    print "\nAll tasks finished."

if __name__ == '__main__':
    main()