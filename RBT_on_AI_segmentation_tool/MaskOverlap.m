clear
close all
%% load masks to compare
origin_mask = imread('/Users/chen/Desktop/RBT_on_AI_segmentation_tool/cellpose_org.tif');
RBT_mask = imread('/Users/chen/Desktop/RBT_on_AI_segmentation_tool/cellpose_rbt.tif');
CLAHE_mask = imread('/Users/chen/Desktop/RBT_on_AI_segmentation_tool/cellpose_clahe.tif');

maskC = origin_mask; 
maskM = RBT_mask; 
maskG = CLAHE_mask; 

C = maskC; % Cyan = (0,1,1)
M = maskM; % Magenta = (1,0,1)
G = maskG; % Green = (0,1,0)

onlyC =  C & ~M & ~G;
onlyM = ~C &  M & ~G;
onlyG = ~C & ~M &  G;

CM   =  C &  M & ~G;
CG   =  C & ~M &  G;
MG   = ~C &  M &  G;

CMG  =  C &  M &  G;

R = false(size(C));
Gch = false(size(C));   % avoid mistaken with maskG
B = false(size(C));

% cyan
Gch(onlyC) = 1;
B(onlyC)   = 1;

% magenta
R(onlyM) = 1;
B(onlyM) = 1;

% green
Gch(onlyG) = 1;

% C + M ¡÷ yellow
R(CM)   = 1;
Gch(CM) = 1;

% C + G ¡÷ green
Gch(CG) = 1;

% M + G ¡÷ red (1,0,0)
R(MG) = 1;

% C + M + G ¡÷ yellow (1,1,0)
R(CMG)   = 1;
Gch(CMG) = 1;

rgb = double(cat(3, R, Gch, B));
imshow(rgb)