function loe = compute_loe(orig, enhanced)
%%%%%%%
% Computes the lightness order error (LOE) between two images.
%
%   loe = compute_loe(orig, enhanced) calculates the Local Order Error (LOE)
%   between the original image `orig` and the enhanced image `enhanced`.
%   LOE measures how much the local pixel intensity order is disrupted
%   after image enhancement. It reflects the preservation of local
%   structure and contrast direction.
%
%   Input:
%     orig     - Original grayscale image (uint8, uint16, or double)
%     enhanced - Enhanced grayscale image of the same size and type
%
%   Output:
%     loe      - A scalar value between 0 and 1 indicating the proportion
%                of local order violations. Lower LOE means better local
%                structure preservation.
%
%   Reference:
%     M. Akai, Y. Ueda, T. Koga, and N. Suetake, "Low-artifact and fast backlit
%     image enhancement method based on suppression of lightness order error,"
%     IEEE Access, vol. 11, pp. 121231??121245, 2023.
%     https://doi.org/10.1109/ACCESS.2023.3321844
%
%   Example:
%     loe_value = compute_loe(original_img, enhanced_img);
%%%%%%%

    orig = double(orig);
    enhanced = double(enhanced);
    [rows, cols] = size(orig);
    count = 0;
    error = 0;
    for i = 2:rows
        for j = 2:cols
            o_diff = orig(i,j) - orig(i-1,j-1);
            e_diff = enhanced(i,j) - enhanced(i-1,j-1);
            if sign(o_diff) ~= sign(e_diff)
                error = error + 1;
            end
            count = count + 1;
        end
    end
    loe = error / count;
end