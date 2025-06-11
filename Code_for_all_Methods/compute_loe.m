function loe = compute_loe(orig, enhanced)
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