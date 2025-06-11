function [img_RBT] = RBT(img_IN)

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