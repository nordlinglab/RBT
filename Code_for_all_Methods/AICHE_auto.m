function output = AICHE_auto(input)
% AICHE_auto - Automatically choose AICHE for uint8 or uint16 input
%
% Input:
%   input - grayscale image (uint8 or uint16)
%
% Output:
%   output - enhanced image (same format as input)

    if isa(input, 'uint8')
        output = AICHE_uint8(input);
    elseif isa(input, 'uint16')
        output = AICHE_uint16(input);
    else
        error('AICHE only supports uint8 or uint16 grayscale images.');
    end
end
