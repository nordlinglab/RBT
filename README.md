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

# Sources of Example Images
- Budding yeast: Kindly shared by Prof. Diego di Bernardo, Dr. Filippo Melonascina, and Dr. Gianfranco Fiore.
- E. coli: Open access data from ''[DeepBacs](https://github.com/HenriquesLab/DeepBacs): Bacterial image analysis using open-source deep learning approaches. bioRxiv, 2021. DOI: https://doi.org/10.1101/2021.11.03.467152'' under CC0 1.0 Licence.
- Drosophila kc167 cells: image set [BBBC002v1](https://bbbc.broadinstitute.org/BBBC002) [Carpenter et al., Genome Biology, 2006] from the Broad Bioimage Benchmark Collection [Ljosa et al., Nature Methods, 2012].
- Mouse microglia: image set [BBBC054](https://bbbc.broadinstitute.org/BBBC054), available from the Broad Bioimage Benchmark Collection [Ljosa et al., Nature Methods, 2012], licensed under a Creative Commons Attribution 3.0 Unported License.
