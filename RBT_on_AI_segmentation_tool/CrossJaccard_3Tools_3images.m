clear
close all
%%
Cellpose_original = imread('/Users/chen/Desktop/RBT_on_AI_segmentation_tool/cellpose_org.tif');
a1 = imbinarize(Cellpose_original);

Cellpose_rbt = imread('/Users/chen/Desktop/RBT_on_AI_segmentation_tool/cellpose_rbt.tif');
a2 = imbinarize(Cellpose_rbt);
Cellpose_clahe = imread('/Users/chen/Desktop/RBT_on_AI_segmentation_tool/cellpose_clahe.tif');
a3 = imbinarize(Cellpose_clahe);

Ilastik_original = imread('/Users/chen/Desktop/RBT_on_AI_segmentation_tool/ilastik_org.tif');
b1 = Ilastik_original;

Ilastik_rbt = imread('/Users/chen/Desktop/RBT_on_AI_segmentation_tool/ilastik_rbt.tif');
b2 = Ilastik_rbt;

Ilastik_clahe = imread('/Users/chen/Desktop/RBT_on_AI_segmentation_tool/ilastik_clahe.tif');
b3 = Ilastik_clahe;

PlantSeg_original = imread('/Users/chen/Desktop/RBT_on_AI_segmentation_tool/plantseg_org.tif');
c1 = PlantSeg_original;

PlantSeg_rbt = imread('/Users/chen/Desktop/RBT_on_AI_segmentation_tool/plantseg_rbt.tif');
c2 = PlantSeg_rbt;

PlantSeg_clahe = imread('/Users/chen/Desktop/RBT_on_AI_segmentation_tool/plantseg_clahe.tif');
c3 = PlantSeg_clahe;

%%
masks = {a1,a2,a3,b1,b2,b3,c1,c2,c3};
names = {'a1','a2','a3','b1','b2','b3','c1','c2','c3'};

n = numel(masks);
J = zeros(n);

%%
for i = 1:n
    for j = i+0:n
        
        A = logical(masks{i});
        B = logical(masks{j});
        
        % calculate Jaccard index
        intersection = sum(A(:) & B(:));
        union        = sum(A(:) | B(:));
        J(i,j) = round(intersection / union, 4);
        J(j,i) = J(i,j); % symmetric
    end
end

J_table = array2table(J, 'VariableNames', names, 'RowNames', names);
disp(J_table);

writetable(J_table, '/Users/chen/Desktop/J_table.csv');