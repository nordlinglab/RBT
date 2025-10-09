function [Cm, Crms] = compute_contrast_metrics(image)
% compute_contrast_metrics - Compute Cm and Crms contrast metrics
%
% Syntax:
%   [Cm, Crms] = compute_contrast_metrics(image)
%
% Inputs:
%   image - Input grayscale image (2D matrix, any numeric type)
%
% Outputs:
%   Cm    - Michelson-like contrast metric: (max - min) / (max + min)
%   Crms  - Root mean square contrast

    image = double(image); % Ensure precision
    imgSize = numel(image);

    % Cm: Michelson-like global contrast
    max_val = max(image(:));
    min_val = min(image(:));
    Cm = (max_val - min_val) / (max_val + min_val + eps);  % avoid divide-by-zero

    % Crms: RMS contrast
    Crms = sqrt(sum((image(:) - mean(image(:))).^2) / imgSize);
end