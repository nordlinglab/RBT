function metrics_tbl = compute_all_metrics(img, ref_img, method_name)
% compute_all_metrics - Compute common image quality metrics
%
% Syntax:
%   metrics_tbl = compute_all_metrics(img, ref_img, method_name)
%
% Inputs:
%   img         - Processed image (2D grayscale, uint8/uint16/double)
%   ref_img     - Reference image (e.g., original)
%   method_name - Name of the method as string for table row label
%
% Output:
%   metrics_tbl - Table with all metrics (1 row)

% Convert to double for computation
img = double(img);
ref_img = double(ref_img);
imgSize = numel(img);

% Cm and Crms
[Cm, Crms] = compute_contrast_metrics(img);

% Entropy
entropy_val = entropy(img);

% Std deviation & mean
std_val = std2(img);
mean_val = mean2(img);

% AMBE
ambe_val = abs(mean2(ref_img) - mean2(img));

% SSIM
try
    ssim_val = ssim(img, ref_img);
catch
    ssim_val = NaN;
end

% PSNR
try
    psnr_val = psnr(img, ref_img);
catch
    psnr_val = NaN;
end

% BRISQUE
try
    brisque_val = brisque(uint8(mat2gray(img)*255));
catch
    brisque_val = NaN;
end

% NIQE
try
    niqe_val = niqe(uint8(mat2gray(img)*255));
catch
    niqe_val = NaN;
end

% PIQE
try
    piqe_val = piqe(uint8(mat2gray(img)*255));
catch
    piqe_val = NaN;
end

% EME
eme_val = compute_eme(img);

% CII
cii_val = std2(img) / std2(ref_img);

% LOE
loe_val = compute_loe(ref_img, img);

% FeatureSIM
FSIM_val = FeatureSIM(double(ref_img), double(img));

% Construct table
metrics_tbl = table(Cm, Crms, entropy_val, std_val, mean_val, ...
    ambe_val, ssim_val, brisque_val, niqe_val, psnr_val, ...
    piqe_val, eme_val, cii_val, loe_val, FSIM_val, ...
    'VariableNames', {'Cm', 'Crms', 'Entropy', 'Std', 'Mean', ...
    'AMBE', 'SSIM', 'BRISQUE', 'NIQE', 'PSNR', ...
    'PIQE', 'EME', 'CII', 'LOE', 'FSIM'}, ...
    'RowNames', {method_name});
end