function [HistoSave] = HistoPlotSave(HomePath, Image, ImageDenote)
figure(), imshow(Image)
tif_file = fullfile(HomePath, ['Img_', ImageDenote, '.tif']);
imwrite(Image, tif_file)

fz = 24;
figure(), hold on
set(gcf,'Position',[1000 10 1000 500]) %[left bottom width height]
edges = 0:1:65535;
h = histogram(Image,'BinWidth',1,'BinEdges', edges, 'FaceColor','k');
ulim = max(h.BinCounts)*1.1;
ylim([0,ulim])
xlim([0,65535])
%xlim([h.BinLimits])
%xlabel('Intensity value','FontSize',fz)
%ylabel('Number of pixels','FontSize',fz) %frequency
ax = gca;
ax.XAxis.Exponent = 0;
ax.YAxis.FontSize = fz;
ax.XAxis.FontSize = fz;
axis tight

counts = h.Values;
binCenters = h.BinEdges(1:end-1) + diff(h.BinEdges)/2;
cdf = cumsum(counts) / sum(counts);
cdf_normalized = cdf * max(counts) / max(cdf);
plot(binCenters, cdf_normalized, '-r','LineWidth', 2);
%legend('Histogram','CDF', 'FontSize', fz, 'Location', 'northeast')
box off

scale = 2;
paperunits = 'centimeters';
filewidth = 15; %cm
fileheight = 5; %cm
size = [filewidth fileheight]*scale;
set(gcf,'paperunits',paperunits,'paperposition',[-2.3 0.01 size]);
set(gcf, 'PaperSize', [25.5,9.8]);
set(gca, 'LooseInset', get(gca,'TightInset'));
hold off

Fig1 = fullfile(HomePath, ['Histogram_', ImageDenote, '.tif']);
print(gcf, Fig1, '-dtiff', '-r300');
end