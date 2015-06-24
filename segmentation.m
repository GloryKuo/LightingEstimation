img = imread('./LightingEstimation/input/synth/synth_23.bmp');
img = rgb2gray(img);
figure(1); imshow(img);
pts = ginput(2);
close all

m = (pts(1,2)-pts(2,2)) / (pts(1,1)-pts(2,1))

img_label = zeros(size(img));

%%
for i = 1:size(img, 1)
    for j = 1:size(img, 2)
        c = m.*(j-pts(1,1))-(i-pts(1,2));
        img_label(i,j) = c>=0;
    end
end
imshow(img_label);
imwrite(img_label, './LightingEstimation/input/synth/synth_23_label.bmp');