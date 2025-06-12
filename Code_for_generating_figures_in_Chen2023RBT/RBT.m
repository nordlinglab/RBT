function [img_RBT] = RBT(img_input)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% RBT - Rank-Based Transformation for Contrast Enhancement
%
%   img_RBT = RBT(img_input)
%
%   This function performs a rank-based transformation (RBT) on a grayscale
%   image by globally ranking all pixel intensities and mapping them
%   evenly across the dynamic range. The transformation is parameter-free
%   and does not consider intensity frequency or spatial context.
%
%   Input:
%     img_input - Input grayscale image (uint8, uint16, single, or double)
%
%   Output:
%     img_RBT   - Contrast-enhanced image with the same data type as input
%
%   The RBT algorithm is described in:
%     Chen, Cheng-Hui, and Torbjörn EM Nordling.
%     "Rank-based Transformation Algorithm for Image Contrast Adjustment."
%     Authorea Preprints (2023).
%     https://doi.org/10.36227/techrxiv.22952354.v2
%
%   Notes:
%     - The output intensities are uniformly spaced between 0 and the
%       maximum value of the input data type (e.g., 255 for uint8).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

img_class = class(img_input);

if img_class == "uint8"
    FormatFactor = 2^8-1;
elseif img_class == "uint16"
    FormatFactor = 2^16-1;
elseif img_class == "single"
    FormatFactor = 1;
elseif img_class == "double"
    FormatFactor = 1;
end

[temp,Rank] = ismember(img_input,unique(img_input));
U = length(unique(img_input));
%(Rank-1)./(U-1) force the transformed range to be [0, FormatFactor]
RBT = FormatFactor*(Rank-1)./(U-1);
img_RBT = cast(RBT,img_class);

end