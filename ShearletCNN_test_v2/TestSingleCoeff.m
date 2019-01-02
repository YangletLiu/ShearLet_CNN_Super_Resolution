close all;
clear all;
clc;clear;
%% set parameters
up_scale = 2;
wtype = 'bior1.1';

model_l = 'model/models_DST9_iter_48195.mat';


image = imread('Samples/s_ChosenDrone_test/s_238.jpg');
% image = imread('Samples/s_ChosenDrone_train/s_97.jpg');

%% work on illuminance only
if size(image,3)>1
    image = rgb2ycbcr(image);
    im_Y = image(:, :, 1);
end

im_gnd = im2double(modcrop(im_Y, up_scale));

%% bicubic interpolation
im_low = imresize(im_gnd, 1/up_scale, 'bicubic');
im_bic = imresize(im_low, up_scale, 'bicubic');

%% SRCNN--DST

c1 = SRCNN(model_l, im_bic);

[coeffs_DST,~] = DSTImgDec(im_gnd,1);
c1_gnd = coeffs_DST(:,:,9);

%% SRCNN--DWT

% c1 = SRCNN(model_l, im_low);
% 
% [c1_gnd,~,~,~] = dwt2(im_gnd,wtype);


% if up_scale == 4
%     cA = SRCNN(model_l, im_h);
%     cH = SRCNN(model_h, im_h);
%     cV = SRCNN(model_v, im_h);
%     cD = SRCNN(model_d, im_h);
%     im_h = idwt2(cA, cH, cV, cD, wtype);
% end

c1 = shave(uint8(c1 * 255), [up_scale, up_scale]);
c1_gnd = shave(uint8(c1_gnd * 255), [up_scale, up_scale]);

%% compute PSNR
psnr_srcnn = compute_psnr(c1, c1_gnd);

%% show results
fprintf('PSNR for SRCNN Reconstruction: %f dB\n', psnr_srcnn);

figure, imshow(im_h); title('WMCNN Reconstruction');

% image(:, :, 2) = imresize(imresize(image(:, :, 2), 1/up_scale, 'bicubic'), up_scale, 'bicubic');
% image(:, :, 3) = imresize(imresize(image(:, :, 3), 1/up_scale, 'bicubic'), up_scale, 'bicubic');
% image = shave(image, [up_scale, up_scale]);
% image(:, :, 1)=im_h;
% im_H = ycbcr2rgb(image);
% image(:, :, 1)=im_bic;
% im_B = ycbcr2rgb(image);
% imwrite(im_B, ['Bicubic Interpolation' '.bmp']);
% imwrite(im_H, ['reconstruced' '.bmp']);
% imwrite(im_gt, ['gt_image' '.bmp']);
