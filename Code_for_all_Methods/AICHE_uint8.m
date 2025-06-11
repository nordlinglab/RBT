
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