clear
close all
%%
origin_mask = imread('/Users/chen/Desktop/RBT_on_AI_segmentation_tool/cellpose_org.tif');
RBT_mask = imread('/Users/chen/Desktop/RBT_on_AI_segmentation_tool/cellpose_rbt.tif');
CLAHE_mask = imread('/Users/chen/Desktop/RBT_on_AI_segmentation_tool/cellpose_clahe.tif');
%%
inputImage = origin_mask;
image2compare = CLAHE_mask;

opt_radius = [-3, -2, -1, 0, 1, 2, 3];
outputFolder = '/Users/chen/Desktop/DilateErode';
Jaccard = zeros(size(opt_radius));
Dice = zeros(size(opt_radius));

for i = 1:length(opt_radius)
    r = opt_radius(i);
    
    if r < 0
        se = strel('disk', abs(r));
        processedImage = imerode(inputImage, se);
    elseif r > 0
        se = strel('disk', r);
        processedImage = imdilate(inputImage, se);
    else
        processedImage = inputImage;
    end
    
    intersection = sum(processedImage(:) & image2compare(:));
    unionSet = sum(processedImage(:) | image2compare(:));
    Jaccard(i) = intersection / unionSet;
    
    intersection = sum((processedImage & image2compare), 'all');
    sum_A = sum(processedImage, 'all');
    sum_B = sum(image2compare, 'all');

    Dice(i) = 2 * intersection / (sum_A + sum_B);
    
    outputFileName = fullfile(outputFolder, sprintf('processed_r%d.tif', r));
    imwrite(processedImage, outputFileName);
    disp(['Saved: ', outputFileName]);
end

disp('Processing completed.');

%%
%Jaccard_origin_RBT = [0.797105950244301,0.856766529298412,0.926985611510791,0.983060216472602,0.925174918817582,0.880237004396871,0.853219162230965];
%Jaccard_RBT_origin = [0.778042071602562,0.838886762965056,0.914224026995339,0.983060216472602,0.936884525649184,0.894127219029681,0.865865946555963];
Jaccard_origin_CLAHE = [0.795345182440445,0.854113936583487,0.921491468567558,0.969967610046196,0.924040010965572,0.881956584733646,0.854984614529933];
Jaccard_CLAHE_origin = [0.781772093971105,0.841795269512012,0.912170859108980,0.969967610046196,0.932916425970955,0.892479377700730,0.864457626174127];

figure(99), hold on
%plot(opt_radius, Jaccard_origin_RBT, '-o', 'LineWidth', 2, 'MarkerSize', 8);
%plot(opt_radius, Jaccard_RBT_origin, '-o', 'LineWidth', 2, 'MarkerSize', 8);
plot(opt_radius, Jaccard_origin_CLAHE, '-o', 'LineWidth', 2, 'MarkerSize', 8);
plot(opt_radius, Jaccard_CLAHE_origin, '-o', 'LineWidth', 2, 'MarkerSize', 8);

ylim([0.75, 1])
xlabel('Radius (r)','FontSize', 12);
ylabel('Jaccard Index','FontSize', 12);
set(gca, 'FontSize', 12);
%title('Jaccard Similarity vs. Operation Radius');
%title('Operated Orig. mask vs. RBT mask')
legend('Morph. Orig. mask vs. CLAHE mask','Morph. CLAHE mask vs. Orig.mask', 'Location','southeast', 'FontSize', 12)
grid on
hold off