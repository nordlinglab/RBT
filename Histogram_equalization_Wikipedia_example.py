# -*- coding: utf-8 -*-

"""
Python 2.7 script to perform histogram equalization and RBT on the example image Unequalized_Hawkes_Bay_NZ.jpg at
https://en.wikipedia.org/wiki/Histogram_equalization

This script reads a grayscale image and displays three versions:
1. Original
2. Histogram Equalized
3. Rank-Based Transformation (RBT)

This script processes an image using Histogram Equalization and Rank-Based
Transformation (RBT). It saves the resulting images as PNG files and
generates PDF plots for the histograms and CDFs.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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
    print "Image transformations complete."

    # 2. Save processed bitmap images as PNG
    plt.imsave('Unequalized_Hawkes_Bay_NZ_Equalized_Image.png', img_equalized, cmap='gray')
    plt.imsave('Unequalized_Hawkes_Bay_NZ_RBT_Image.png', img_rbt, cmap='gray')
    print "Saved transformed images as PNG."

    # 3. Create and save combined histogram plots as PDF
    plot_and_save_histograms(img, img_equalized, img_rbt)
    print "Saved combined histogram plots as PDF."
    
    # 4. Create and save individual histogram+CDF plots without images as PDFs
    plot_and_save_only_histograms_cdfs_separately(img, img_equalized, img_rbt)
    print "Saved individual histogram+CDF plots (without images) as PDF."
    
    # 5. Create and save the combined CDF plot as PDF
    plot_and_save_all_cdfs(img, img_equalized, img_rbt)
    print "Saved combined CDF plot as PDF."
    
    print "\nAll tasks finished."

if __name__ == '__main__':
    main()