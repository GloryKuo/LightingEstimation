img = imread('./LightingEstimation/input/input39.jpg');
img = rgb2gray(img);
figure(1); imshow(img);
pts = ginput(4);
close all

m(1) = (pts(1,2)-pts(2,2)) / (pts(1,1)-pts(2,1));
m(2) = (pts(1,2)-pts(3,2)) / (pts(1,1)-pts(3,1));
m(3) = (pts(1,2)-pts(4,2)) / (pts(1,1)-pts(4,1))

img_label = cell(1, 3);
sum_label = zeros(size(img));
%%
for s = 1:3
    img_label{s} = zeros(size(img));
    for i = 1:size(img, 1)
        for j = 1:size(img, 2)
            c = m(s).*(j-pts(1,1))-(i-pts(1,2));
            img_label{s}(i,j) = c>=0;
        end
    end
%     figure; imshow(img_label{s});
end
sum_label = img_label{1} + img_label{2};
result = zeros(size(img));
result(sum_label>0) = 1;
for i = 1:size(result, 1)
    for j = 1:size(result, 2)
        if(result(i,j)>0)
            result(i,j) = result(i,j) + img_label{3}(i,j)+1;
        end
    end
end


result_show = uint8((255./3).*result);
figure; imshow(result_show);



imwrite(result_show, './LightingEstimation/input/input39_label.bmp');