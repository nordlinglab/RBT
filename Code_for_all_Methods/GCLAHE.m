function [LEI, final_clip_factor, iterations] = GCLAHE(I)
% GCLAHE - Gradient-based adaptive CLAHE using similarity optimization
%
% Inputs:
%   I - Original grayscale image
%
% Outputs:
%   LEI               - Locally Enhanced Image
%   final_clip_factor - Best clip factor found
%   iterations        - Number of iterations performed

% Initialization
TS = 8;                      % Tile size for CLAHE
N = 0;                       % Iteration counter
LEI = I;                     % Initialize Local Enhanced Image with original
Clipping_Factor = 3;         % Initial clip factor

% Compute global enhancement for reference
GEI = histeq(LEI);                      % Global Histogram Equalization
Prev_F = evaluate_similarity(GEI, LEI); % Initial similarity score

max_iter = TS * TS - 1;      % Max iterations

while N < max_iter
    GEI = histeq(LEI);

    % Normalize the Clipping Factor for ClipLimit
    clip_limit = min(Clipping_Factor / 10, 1); % Ensure ClipLimit is between 0 and 1
    
    % Apply CLAHE with updated clip limit and tile grid [8 8]
    New_LEI = adapthisteq(LEI, ...
        'ClipLimit', clip_limit, ...
        'NumTiles', [TS TS]);

    F = evaluate_similarity(GEI, New_LEI);

    if F > Prev_F
        Prev_F = F;
        LEI = New_LEI;
        Clipping_Factor = Clipping_Factor + 1;
    else
        final_clip_factor = Clipping_Factor - 1;
        iterations = N;
        return; % Exit loop early and return best result
    end
    N = N + 1;
end

% If loop completes, return final result
final_clip_factor = Clipping_Factor;
iterations = N;
end

% ---------------------------------------------
% Similarity evaluation function (can be replaced with PSNR, etc.)
function score = evaluate_similarity(img1, img2)
% Evaluate similarity using SSIM (Structural Similarity Index)
score = ssim(im2double(img1), im2double(img2));
end