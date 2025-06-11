function val = compute_eme(img)
    block_size = 8;
    img = double(img);
    [rows, cols] = size(img);
    eme = 0;
    for i = 1:block_size:rows-block_size+1
        for j = 1:block_size:cols-block_size+1
            block = img(i:i+block_size-1, j:j+block_size-1);
            Imax = max(block(:)) + 1e-6;
            Imin = min(block(:)) + 1e-6;
            eme = eme + 20*log10(Imax / Imin);
        end
    end
    val = eme / ((rows/block_size)*(cols/block_size));
end