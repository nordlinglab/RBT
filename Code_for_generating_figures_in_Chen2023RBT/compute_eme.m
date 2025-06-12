function val = compute_eme(img)
%%%%%%%
% COMPUTE_EME Computes the Enhancement Measure Estimation (EME) of an image.
%
%   val = COMPUTE_EME(img) calculates the EME value of a grayscale image `img`
%   based on local contrast in non-overlapping 8??8 blocks. EME is commonly
%   used to evaluate the effectiveness of image contrast enhancement.
%   A higher EME indicates stronger local contrast.
%
%   Input:
%     img - Grayscale image (uint8, uint16, or double). Will be converted to double.
%
%   Output:
%     val - Scalar EME value (higher indicates better enhancement).
%
%   The image is divided into non-overlapping 8??8 blocks, and the EME for each
%   block is computed using the formula:
%
%       EME_block = 20 * log10(Imax / Imin)
%
%   where Imax and Imin are the maximum and minimum intensities within the block.
%   A small constant (1e-6) is added to avoid division by zero.
%
%   Reference:
%     J. Ang, K. S. Sim, S. C. Tan, and C. P. Lim,
%     "Adaptive Contrast Enhancement With Lesion Focusing (ACELF),"
%     IEEE Access, vol. 13, pp. 41785??41796, 2025.
%     https://doi.org/10.1109/ACCESS.2025.3546913
%
%   Example:
%     eme_value = compute_eme(enhanced_img);
%%%%%%%

    block_size = 8;
    img = double(img);
    [rows, cols] = size(img);
    eme = 0;
    for i = 1:block_size:rows-block_size+1
        for j = 1:block_size:cols-block_size+1
            block = img(i:i+block_size-1, j:j+block_size-1);
            Imax = max(block(:)) + 1e-6;
            Imin = min(block(:)) + 1e-6;
            eme = eme + 20*log10(Imax / Imin);
        end
    end
    val = eme / ((rows/block_size)*(cols/block_size));
end