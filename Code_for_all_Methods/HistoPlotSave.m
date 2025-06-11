function [HistoSave] = HistoPlotSave(HomePath, Image, ImageDenote)
figure(), imshow(Image)
tif_file = fullfile(HomePath, ['Img_', ImageDenote, '.tif']);
imwrite(Image, tif_file)

fz = 14;
figure(), hold on
set(gcf,'Position',[1000 10 1000 500]) %[left bottom width height]
edges = 0:1:65535;
h = histogram(Image,'BinWidth',1,'BinEdges', edges);
ulim = max(h.BinCounts)*1.1;
ylim([0,ulim])
xlim([0,65535])
%xlim([h.BinLimits])
%xlabel('Intensity value','FontSize',fz)
%ylabel('Number of pixels','FontSize',fz)
ax = gca;
ax.XAxis.Exponent = 0;
ax.YAxis.FontSize = 24;
ax.XAxis.FontSize = 24;
axis tight
box off

scale = 2;
paperunits = 'centimeters';
filewidth = 15; %cm %10
fileheight = 5; %cm %5
size = [filewidth fileheight]*scale;
set(gcf,'paperunits',paperunits,'paperposition',[-2.3 0.01 size]); %[-1.3 0.05 size] [-2.3 0.01 size]
set(gcf, 'PaperSize', [25.5,9.8]); %[18.1,9.9] [25.5,9.8]

set(gca, 'LooseInset', get(gca,'TightInset'));
%axis tight

hold off

% Fig1 = fullfile(HomePath, ['Histogram_', ImageDenote, '.pdf']);
% print(gcf, Fig1, '-dpdf', '-r300');
Fig1 = fullfile(HomePath, ['Histogram_', ImageDenote, '.tif']);
print(gcf, Fig1, '-dtiff', '-r300');

figure(), hold on
counts = h.Values;
binCenters = h.BinEdges(1:end-1) + diff(h.BinEdges)/2;
cdf = cumsum(counts) / sum(counts);
plot(binCenters, cdf, 'LineWidth', 2);
%xlabel('Intensity value', 'FontSize',fz);
%ylabel('cumulative distribution function', 'FontSize',fz);
grid on;
ax = gca;
ax.XAxis.Exponent = 0;
ax.YAxis.FontSize = 24;
ax.XAxis.FontSize = 24;
axis tight
hold off

Fig2 = fullfile(HomePath, ['CDF_Histogram_', ImageDenote, '.tif']);
print(gcf, Fig2, '-dtiff', '-r300');
end