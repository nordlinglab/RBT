function output = AICHE(input)
%%%%%%%
% AICHE - Automatically choose AICHE for uint8 or uint16 input
%
% Input:
%   input - grayscale image (uint8 or uint16)
%
% Output:
%   output - enhanced image (same format as input)
%
% % Note: this code is derived from the algorithm proposed in
%   B. Ye, S. Jin, B. Li, S. Yan, and D. Zhang,
%   "Dual histogram equalization algorithm based on adaptive image correction,"
%   *Applied Sciences*, vol. 13, no. 19, article 10649, 2023.
%   https://doi.org/10.3390/app131910649
%%%%%%%

    if isa(input, 'uint8')
        output = AICHE_uint8(input);
    elseif isa(input, 'uint16')
        output = AICHE_uint16(input);
    else
        error('AICHE only supports uint8 or uint16 grayscale images.');
    end
end


function output = AICHE_uint8(input)
    % AICHE: Dual Histogram Equalization with Adaptive Brightness Correction for uint8 images
    % input: uint8 grayscale image
    % output: uint8 enhanced image

    input = double(input);
    [rows, cols] = size(input);
    L = 256;

    % Step 1: Improved Otsu thresholding
    threshold = improved_otsu(input);

    % Step 2: Histogram segmentation and equalization
    hist_input = imhist(uint8(input), L);
    hist1 = hist_input(1:threshold+1);
    hist2 = hist_input(threshold+2:end);

    cdf1 = cumsum(hist1) / sum(hist1);
    cdf2 = cumsum(hist2) / sum(hist2);

    map1 = round(cdf1 * threshold);
    map2 = round(cdf2 * (L - threshold - 1)) + threshold + 1;

    % Apply equalization mapping
    equalized = zeros(size(input));
    for i = 1:rows
        for j = 1:cols
            pixel = input(i,j);
            if pixel <= threshold
                equalized(i,j) = map1(pixel + 1);
            else
                equalized(i,j) = map2(pixel - threshold);
            end
        end
    end

    % Step 3: Adaptive local brightness correction
    output = adaptive_local_correction(input, equalized);
    output = uint8(output);
end

function threshold = improved_otsu(img)
    % Improved Otsu method considering local intra-region variance (for uint8)
    [counts, ~] = imhist(uint8(img));
    total = sum(counts);
    max_sigma = 0;
    threshold = 0;

    for t = 1:255
        w0 = sum(counts(1:t)) / total;
        w1 = sum(counts(t+1:end)) / total;
        if w0 == 0 || w1 == 0
            continue;
        end
        mu0 = sum((0:t-1)'.*counts(1:t)) / sum(counts(1:t));
        mu1 = sum((t:255)'.*counts(t+1:end)) / sum(counts(t+1:end));
        sigma_b = w0 * w1 * (mu0 - mu1)^2;
        if sigma_b > max_sigma
            max_sigma = sigma_b;
            threshold = t - 1;
        end
    end
end

function corrected = adaptive_local_correction(original, equalized)
    % Adaptive local brightness correction for uint8 images
    [rows, cols] = size(original);
    corrected = zeros(rows, cols);
    window_size = 3;
    pad_size = floor(window_size / 2);
    padded_original = padarray(original, [pad_size, pad_size], 'symmetric');
    padded_equalized = padarray(equalized, [pad_size, pad_size], 'symmetric');

    for i = 1:rows
        for j = 1:cols
            local_orig = padded_original(i:i+window_size-1, j:j+window_size-1);
            local_eq = padded_equalized(i:i+window_size-1, j:j+window_size-1);
            mean_orig = mean(local_orig(:));
            mean_eq = mean(local_eq(:));
            if mean_eq == 0
                corrected(i,j) = equalized(i,j);
            else
                corrected(i,j) = equalized(i,j) * (mean_orig / mean_eq);
            end
        end
    end
end

function output = AICHE_uint16(input)
    % AICHE for uint16 grayscale images (intensity range: [0, 65535])
    % input: uint16 grayscale image
    % output: uint16 enhanced image

    input = double(input);
    [rows, cols] = size(input);
    L = 65536; % For uint16 image

    % Step 1: Compute improved Otsu threshold
    threshold = improved_otsu_uint16(input, L);

    % Step 2: Histogram segmentation and equalization
    hist_input = imhist(uint16(input), L);
    hist1 = hist_input(1:threshold+1);
    hist2 = hist_input(threshold+2:end);

    cdf1 = cumsum(hist1) / sum(hist1);
    cdf2 = cumsum(hist2) / sum(hist2);

    map1 = round(cdf1 * threshold);
    map2 = round(cdf2 * (L - threshold - 1)) + threshold + 1;

    % Apply equalization mapping
    equalized = zeros(size(input));
    for i = 1:rows
        for j = 1:cols
            pixel = input(i,j);
            if pixel <= threshold
                equalized(i,j) = map1(pixel + 1);
            else
                equalized(i,j) = map2(pixel - threshold);
            end
        end
    end

    % Step 3: Adaptive local brightness correction
    output = adaptive_local_correction_uint16(input, equalized);
    output = uint16(min(max(output, 0), L - 1));
end

function threshold = improved_otsu_uint16(img, L)
    % Improved Otsu method for uint16 grayscale images
    [counts, ~] = imhist(uint16(img), L);
    total = sum(counts);
    max_sigma = 0;
    threshold = 0;

    for t = 1:L-1
        w0 = sum(counts(1:t)) / total;
        w1 = sum(counts(t+1:end)) / total;
        if w0 == 0 || w1 == 0
            continue;
        end
        mu0 = sum((0:t-1)'.*counts(1:t)) / sum(counts(1:t));
        mu1 = sum((t:L-1)'.*counts(t+1:end)) / sum(counts(t+1:end));
        sigma_b = w0 * w1 * (mu0 - mu1)^2;
        if sigma_b > max_sigma
            max_sigma = sigma_b;
            threshold = t - 1;
        end
    end
end

function corrected = adaptive_local_correction_uint16(original, equalized)
    % Adaptive local brightness correction for uint16 images
    [rows, cols] = size(original);
    corrected = zeros(rows, cols);
    window_size = 3;
    pad_size = floor(window_size / 2);
    padded_original = padarray(original, [pad_size, pad_size], 'symmetric');
    padded_equalized = padarray(equalized, [pad_size, pad_size], 'symmetric');

    for i = 1:rows
        for j = 1:cols
            local_orig = padded_original(i:i+window_size-1, j:j+window_size-1);
            local_eq = padded_equalized(i:i+window_size-1, j:j+window_size-1);
            mean_orig = mean(local_orig(:));
            mean_eq = mean(local_eq(:));
            if mean_eq == 0
                corrected(i,j) = equalized(i,j);
            else
                corrected(i,j) = equalized(i,j) * (mean_orig / mean_eq);
            end
        end
    end
end