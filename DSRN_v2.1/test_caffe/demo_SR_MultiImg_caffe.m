% clc;
clear;
addpath(genpath(cd));
% add when using server "tensor"
% addpath(genpath('/home/tensor/Desktop/Gengtianyu/caffe'));
% addpath(genpath('/home/tensor/Desktop/Gengtianyu/ShearLab3Dv11'));

folder = 'Samples/BSDS500_test_6p/';
cnt_result = 1;
Result(cnt_result,:) = {'img_name','method','up_scale','PSNR','SSIM','reconstructed_image'};
up_scale = 3;

format = {'*.jpg','*.png','*.bmp','*.tif'};
filepaths   =  [];
for i = 1 : length(format)
    filepaths = cat(1,filepaths, dir(fullfile(folder,format{i})));
end


for i = 1 : length(filepaths)
    
    fprintf('Image No.: %d \n', i);
    image = imread(fullfile(folder,filepaths(i).name));

%% work on illuminance only
if size(image,3)>1
    image = rgb2ycbcr(image);
    im_Y = image(:, :, 1);
else
    im_Y = image;
end

im_gnd = im2double(modcrop(im_Y, up_scale));
im_gnd_shave = shave(uint8(im_gnd * 255), [up_scale, up_scale]);

%% traditional
im_low = imresize(im_gnd, 1/up_scale, 'bicubic');
im_bic = imresize(im_low, up_scale, 'bicubic');
im_bic_shave = shave(uint8(im_bic * 255), [up_scale, up_scale]);
psnr_bic = compute_psnr(im_gnd_shave, im_bic_shave);
ssim_bic = ssim_index(im_gnd_shave, im_bic_shave);

cnt_result = cnt_result + 1;
Result(cnt_result,:) = {filepaths(i).name,'bicubic',up_scale,psnr_bic,ssim_bic,im_bic};

%% Sheatlet Decomposition
[coeffs_DST_bic,shearletSystem] = DSTImgDec(im_bic,1); 
shearletRMS = reshape(repmat((shearletSystem.RMS./min(shearletSystem.RMS)),[size(im_bic,1)*size(im_bic,2) 1]),[size(im_bic,1),size(im_bic,2),length(shearletSystem.RMS)]);

%% DSRN
model_DSRN = ['SDRN_deploy.prototxt'];
weights_DSRN = ['SDRN_SR_v1_iter_354400.caffemodel'];

coeffs_DST = [];
coeffs_DST = do_cnn_caffe(model_DSRN,weights_DSRN,coeffs_DST_bic);

res = SLshearrec2D(coeffs_DST,shearletSystem);
im_h_DST = im_bic + res;

im_h_DST_shave = shave(uint8(im_h_DST * 255), [up_scale, up_scale]);
psnr_DSRN = compute_psnr(im_gnd_shave, im_h_DST_shave);
ssim_DSRN = ssim_index(im_gnd_shave, im_h_DST_shave);
fprintf('PSNR for DSRN: %f dB\n', psnr_DSRN);

cnt_result = cnt_result + 1;
Result(cnt_result,:) = {filepaths(i).name,model_DSRN,up_scale,psnr_DSRN,ssim_DSRN,im_h_DST};


%% show results

fprintf('PSNR for Bicubic Interpolation: %f dB\n', psnr_bic);
fprintf('PSNR for DSRN: %f dB\n', psnr_DSRN);

fprintf('SSIM for Bicubic Interpolation: %f dB\n', ssim_bic);
fprintf('SSIM for DSRN: %f dB\n', ssim_DSRN);


end

save result_BSDS500_MI_v431_45_46_47.mat


