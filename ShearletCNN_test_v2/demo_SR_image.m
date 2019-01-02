close all;
clear all;
clc;clear;
%% set parameters
up_scale = 2;
model_DST = 'model/models_DST9_iter_48195.mat';
model_DWT = 'model/models_DWT1__iter_39795.mat';
wtype = 'bior1.1';

image = imread('Samples/s_ChosenDrone_test/s_238.jpg');


%% work on illuminance only
if size(image,3)>1
    image = rgb2ycbcr(image);
    im_Y = image(:, :, 1);
end

im_gnd = im2double(modcrop(im_Y, up_scale));

%% bicubic interpolation
im_low = imresize(im_gnd, 1/up_scale, 'bicubic');
im_bic = imresize(im_low, up_scale, 'bicubic');

%% SRCNN

%% DST
c_DST_9 = SRCNN(model_DST, im_bic);

[coeffs_DST,shearletSystem] = DSTImgDec(im_bic,1);
coeffs_DST(:,:,9) = c_DST_9;

im_h_DST = SLshearrec2D(coeffs_DST,shearletSystem);

%% DWT

c_DWT_1 = SRCNN(model_DWT, im_low);

[~,c_DWT_2,c_DWT_3,c_DWT_4] = dwt2(im_bic,wtype);

im_h_DWT = idwt2(c_DWT_1,c_DWT_2,c_DWT_3,c_DWT_4,wtype);



% if up_scale == 4
%     cA = SRCNN(model_l, im_h);
%     cH = SRCNN(model_h, im_h);
%     cV = SRCNN(model_v, im_h);
%     cD = SRCNN(model_d, im_h);
%     im_h = idwt2(cA, cH, cV, cD, wtype);
% end

im_h_DST = shave(uint8(im_h_DST * 255), [up_scale, up_scale]);
im_h_DWT = shave(uint8(im_h_DWT * 255), [up_scale, up_scale]);
im_gnd = shave(uint8(im_gnd * 255), [up_scale, up_scale]);
im_bic = shave(uint8(im_bic * 255), [up_scale, up_scale]);

%% compute PSNR
psnr_bic = compute_psnr(im_gnd, im_bic);
psnr_DST = compute_psnr(im_gnd, im_h_DST);
psnr_DWT = compute_psnr(im_gnd, im_h_DWT);

%% show results
fprintf('PSNR for Bicubic Interpolation: %f dB\n', psnr_bic);
fprintf('PSNR for ShearletCNN Reconstruction: %f dB\n', psnr_DST);
fprintf('PSNR for WaveletCNN Reconstruction: %f dB\n', psnr_DWT);

figure, imshow(im_bic); title('Bicubic Interpolation');
figure, imshow(im_h_DST); title('ShearletCNN Reconstruction');
figure, imshow(im_h_DWT); title('WaveletCNN Reconstruction');

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
