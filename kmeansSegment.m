close all
clear all 

img = imread('RealScene/03.jpg');


cform = makecform('srgb2lab');
lab_img = applycform(img,cform);

ab = double(lab_img(:,:,2:3));
nrows = size(ab,1);
ncols = size(ab,2);
ab = reshape(ab,nrows*ncols,2);

nColors = 3;
figure;imshow(img), title('select kmeans seeds');
% seeds = ginput(nColors);
% repeat the clustering 3 times to avoid local minima
[cluster_idx, cluster_center] = kmeans(ab,nColors,'distance','sqEuclidean', ...
                                      'replicates', 3, 'start', 'sample');
% [cluster_idx, cluster_center] = kmeans(ab,nColors,'distance','sqEuclidean', ...
%                                       'emptyaction','singleton',...
%                                       'start',seeds);

pixel_labels = reshape(cluster_idx,nrows,ncols);
figure; imshow(pixel_labels,[]), title('image labeled by cluster index');

%% morphological operation

segmented_images = cell(1,nColors);
se90 = strel('line', 3, 90);
se0 = strel('line', 3, 0);
for k = 1:nColors  
    BW = pixel_labels==k;
    BWdil = imdilate(BW, [se90, se0]);
%     figure, imshow(BWdil), title('dilated gradient mask');

    BWdfill = imfill(BWdil, 'holes');
%     figure, imshow(BWdfill), title('binary image with filled holes');

    BWnobord = imclearborder(BWdfill, 4);
    BWnobord_inv = BWdfill - BWnobord;
%     figure, imshow(BWnobord_inv), title('inverse cleared border image');
   
    seD = strel('diamond',1);
    BWfinal = imerode(BWnobord_inv,seD);
    BWfinal = imerode(BWfinal,seD);
%     figure, imshow(BWfinal), title('segmented image');
    segmented_images{k} = BWfinal;
    pixel_labels(BWfinal==1) = k;
end
figure; imshow(pixel_labels,[]), title('image labeled after morphology');
%%
img_gray = rgb2gray(img);
[~, threshold] = edge(img_gray, 'sobel');
fudgeFactor = .5;
BW_edge = edge(img_gray,'sobel', threshold * fudgeFactor);
figure, imshow(BW_edge), title('binary gradient mask');

BWsdil = imdilate(BW_edge, [se90 se0]);

BWdfill = imfill(BWsdil, 'holes');
figure, imshow(BWdfill), title('binary image with filled holes');

BWnobord = imclearborder(BWdfill, 4);
BWnobord_inv = BWdfill - BWnobord;
figure, imshow(BWnobord_inv), title('inverse cleared border edge');

seD = strel('diamond',1);
BWfinal = imerode(BWnobord_inv,seD);
BWfinal = imerode(BWfinal,seD);
figure, imshow(BWfinal), title('segmented image');

BWfinal_edge = BWfinal;

labels = repmat(uint8(255./pixel_labels), [1,1,3]);
labels(BWfinal_edge==1) = 0;
imshow(labels);
%% display
rgb_label = repmat(pixel_labels,[1 1 3]);

for k = 1:nColors
    color = img;
    color(rgb_label ~= k) = 0;
    segmented_images{k} = color;
end

for i = 1:length(segmented_images)
    figure; imshow(segmented_images{i}), title(['objects in cluster ', int2str(i)]);
end
            