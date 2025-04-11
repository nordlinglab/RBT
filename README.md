# Rank-Based Transformation (RBT)
The Rank-Based Transformation (RBT) method combines histogram expansion and normalization to enhance image contrast.
It works by ranking each pixel's intensity in ascending order, and then redistributing these ranks evenly across a specified intensity range.
Pixels with identical intensities share the same rank, ensuring consistent mapping.
This process produces an output image with uniformly spaced intensity levels, improving contrast while preserving relative intensity relationships.

# How RBT Works
1. Assign ranks to each pixel based on ascending intensity.
2. Normalize ranks to a [0, 1] interval.
3. Scale normalized ranks to the full dynamic range of the image type. (e.g. the full dynamic range of a unit16 image is [0, 65535].)
4. Replace original pixel intensities with the scaled ranks.

# How to Apply RBT in Your Work

- **For MATLAB Users**  
  Call the `RBT.m` function from your main script.

- **For Python Users**  
  Call the `RBT.py` function from your main script.

Make sure to specify the path to your input image and the directory where you want to save the output images in your main code.

# Citations
If you use RBT in a project, please cite

- Chen, Cheng-Hui, and Torbjörn EM Nordling. "Rank-based Transformation Algorithm for Image Contrast Adjustment." Authorea Preprints (2023). (preprint available on DOI: 10.36227/techrxiv.22952354.v2)

# Licensing
This project is licensed under the Apache License 2.0.
