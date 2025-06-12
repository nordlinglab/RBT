%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% RBT_ContrastEnhancementDemo.m
%
% This script batch-processes grayscale biological images using various
% contrast enhancement methods, including our proposed Rank-Based
% Transformation (RBT), traditional histogram equalization (HE),
% adaptive histogram equalization (CLAHE), and recent hybrid methods
% global CLAHE (G-CLAHE) and 
% adaptive image correction-based histogram equalization (AICHE).
%
% The results include enhanced images, histogram with normalized 
% cumulative distribution function (CDF) plots,
% and quantitative quality metrics, including Michelson contrast (Cm),
% root-mean-square contrast (Crms), average mean brightness error (AMBE),
% local entropy (EME), contrast improvement index (CII), 
% and lightness order error (LOE).
%
% The purpose of this script is to reproduce the results presented in the
% preprint:
%   Chen, Cheng-Hui, and Torbjörn EM Nordling.
%   "Rank-based Transformation Algorithm for Image Contrast Adjustment."
%   Authorea Preprints (2023). 
%   https://doi.org/10.36227/techrxiv.22952354.v2
%
% USAGE:
% - Modify the 'exampleName' and 'HomePath' variables to select the image.
% - Run the script to generate results including enhancement images,
%   metrics tables, and CDF plots.
%
% OUTPUTS:
% - Enhanced images saved per method
% - CDF plots (Figure 3??6 in the paper)
% - Excel spreadsheets with all and selected evaluation metrics
%
% NOTE:
% Input images are automatically converted to uint16 if needed.
% Default enhancement parameters are pre-set for reproducibility.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; close all

% Image name list
exampleName = {'yeast','Dkc167','Ecoli','mouse'};
k = 1; % specify which image to process
name = exampleName{k};
HomePath = fullfile('path/to/the/example/image', name);

% Load the input grayscale image
input_path = fullfile(HomePath, [name, '.tif']);
input_img = imread(input_path);

% Convert uint8 image to uint16 for consistent processing
if isa(input_img, 'uint8')
    input_img = uint16(double(input_img) * 257);
end

% Assign reference image
img = input_img;
ref_img = img;
d_img = double(img);

%% Pre-compute constants for some contrast enhancement methods
imgSize = numel(ref_img);
Imax = max(d_img(:));
Imin = min(d_img(:));
ymax = 65535;
%ymin = 0

if k == 4
    m_s = 1.1;
else
    m_s = floor(655350 / Imax) / 10;
end

m_l = round(m_s * 2);
c_s = floor(65535 / log(1 + (Imax - Imin) / (Imax - Imin)));
c_l = 2 * c_s;

% Define processing methods and their corresponding function handles
methods = {
    'Original',  @() ref_img
    'RBT',       @() RBT(ref_img);
    'Linear',    @() uint16(ymax .* (d_img - Imin) ./ (Imax - Imin));
    'mSmall',    @() uint16(m_s * ref_img);
    'mLarge',    @() uint16(m_l * ref_img);
    'cSmall',    @() uint16(c_s * log(1 + (d_img - Imin) / (Imax - Imin)));
    'cLarge',    @() uint16(c_l * log(1 + (d_img - Imin) / (Imax - Imin)));
    'PLr07',     @() uint16(65535 * ((d_img - Imin) / (Imax - Imin)).^0.7);
    'PLr03',     @() uint16(65535 * ((d_img - Imin) / (Imax - Imin)).^0.3);
    'PLr01',     @() uint16(65535 * ((d_img - Imin) / (Imax - Imin)).^0.1);
    'HE',        @() histeq(ref_img);
    'CLAHE',     @() adapthisteq(ref_img);
    'GCLAHE',    @() GCLAHE(ref_img);
    'AICHE',     @() AICHE(ref_img);
    };
%% Process with different methods
T_all = []; % create a table for evaluation metrics

for i = 1:size(methods,1)
    method_name = methods{i,1};
    process_func = methods{i,2};
    
    fprintf('Processing %s...\n', method_name);
    enhanced_img = process_func();
    
    HistCDF_PlotSave(HomePath, enhanced_img, [name, '_', method_name]);
    
    % Calculate metrics for evaluating enhancement
    T = compute_all_metrics(enhanced_img, ref_img, method_name);
    T_all = [T_all; T];
end

disp(T_all)
writetable(T_all, fullfile(HomePath, [name, '_AllMetrics.xlsx']), 'WriteRowNames', true);
close all

%% Select prefered matrix
Cm = T_all.Cm;
Crms = T_all.Crms;
AMBE = T_all.AMBE;
EME = T_all.EME;
CII = T_all.CII;
LOE = T_all.LOE;

T_selected = T_all(:, {'Cm', 'Crms', 'CII', 'LOE', 'AMBE', 'EME'});

% Force excel save output format as specified digits
for i = 1:width(T_selected)
    if isnumeric(T_selected{:,i}) || islogical(T_selected{:,i})
        T_selected.(i) = arrayfun(@(x) sprintf('%.2f', x), T_selected{:,i}, 'UniformOutput', false);
    end
end

disp(T_selected);
writetable(T_selected, fullfile(HomePath, [name, '_SelectedMetrics.xlsx']), 'WriteRowNames', true);

toc