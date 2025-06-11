clear; close all;
tic
% Dataset name list
exampleName = {'yeast','Dkc167','Ecoli','mouse'};
k = 1;
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
    'AICHE',     @() AICHE_auto(ref_img);
    };
%% Process with different methods and plot all CDF on the same figure
% Define line colors for plotting
colors = [
    0.0 0.0 0.0;   % Original-black
    1.0 0.0 0.0;   % RBT-red
    0.75 0.0 0.75; % Linear-fuchsia
    1.0 0.6 0.2;   % mSmall-light orange
    0.85 0.4 0.0;  % mLarge-dark orange
    0.6 0.85 0.45; % cSmall-light green
    0.0 0.5 0.0;   % cLarge-dark green
    0.6 0.8 1.0;   % PLr07-light blue
    0.2 0.4 0.8;   % PLr03-blue
    0.0 0.2 0.4;   % PLr01-dark blue
    0.85 0.7 1.0;  % HE-light purple
    0.6 0.4 0.8;   % CLAHE-purple
    0.4 0.2 0.6;   % GCLAHE-darker purple
    0.25 0.1 0.45; % AICHE-dark purple
    ];

legend_entries = cell(size(methods,1), 1);
T_all = [];
h_lines = gobjects(size(methods,1), 1);

for i = 1:size(methods,1)
    method_name = methods{i,1};
    process_func = methods{i,2};
    
    fprintf('Processing %s...\n', method_name);
    enhanced_img = process_func();
    
    HistoPlotSave(HomePath, enhanced_img, [name, '_', method_name]);
    figure(77); hold on;
    h = histogram(enhanced_img(:), 65535, 'BinLimits', [0, 65535], 'Visible', 'off');
    counts = h.Values;
    binCenters = h.BinEdges(1:end-1) + diff(h.BinEdges)/2;
    cdf = cumsum(counts) / sum(counts);
    
    % specify line style
    if mod(i, 2) == 1
        line_style = ':';
    else
        line_style = '-';
    end
    
    h_lines(i) = plot(binCenters, cdf, 'Color', colors(i,:), 'LineWidth', 2, 'LineStyle', line_style);
    legend_entries{i} = method_name;
    
    % Calculate matrics for evaluating enhancement
    T = compute_all_metrics(enhanced_img, ref_img, method_name);
    T_all = [T_all; T];
end

xlim([0,65535])
ylim([0,1])
xlabel('Intensity value', 'FontSize',18);
%ylabel('Cumulative distribution function (CDF)', 'FontSize',18);
ylabel('CDF', 'FontSize',18);
%title(sprintf('CDF on %d image', name));
legend(h_lines, legend_entries, 'Location', 'eastoutside');
grid on;
ax = gca;
ax.XAxis.Exponent = 0;
ax = gca;
ax.FontSize = 18;
scale = 2;
paperunits = 'centimeters';
filewidth = 18; %cm
fileheight = 7; %cm
size = [filewidth fileheight]*scale;
set(gcf,'paperunits',paperunits,'paperposition',[-1.2 0.01 size]);
set(gcf, 'PaperSize', [36,14.1]);

set(gca, 'LooseInset', get(gca,'TightInset'));
hold off

Figcdf = fullfile(HomePath, ['CDF_all_', name, '.tif']);
print(gcf, Figcdf, '-dtiff', '-r300');


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