% Rank-based transform (RBT) for image contrast adjustment
% The biggest/largest integer that can be stored in a double without losing
% precision is the same as the largest possible value of a double.
% That is, DBL_MAX or approximately 1.8 กั 10308,
% followed the IEEE standard IEEE 754 64-bit floating-point arithmetic.

function [img_RBT] = RBT(img_IN)
% img_IN: input image
% FormatFactor: the largest possible value the image format could have,
% or the number of distinct values that the data type of image allow
% Rank: the unique values on the whole image.
% U: the number of unique values.

img_class = class(img_IN);

if img_class == "uint8"
    FormatFactor = 2^8-1;
elseif img_class == "uint16"
    FormatFactor = 2^16-1;
elseif img_class == "single"
    FormatFactor = 1;
elseif img_class == "double"
    FormatFactor = 1;
end

[temp,Rank] = ismember(img_IN,unique(img_IN));
U = length(unique(img_IN));
%(Rank-1)./(U-1) force the transformed range to be [0, FormatFactor]
RBT = FormatFactor*(Rank-1)./(U-1);
img_RBT = cast(RBT,img_class);
end