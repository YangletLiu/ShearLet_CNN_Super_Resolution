clear;close all;
clc;clear;
addpath(genpath(cd));
%% settings
folder = 'Train\BSDS500_train';
Savepath_HF = 'train_HF.h5';
Savepath_LF = 'train_LF.h5';
chunksz = 64;

DST_scale = 1;  %shearlet分解的scale
Num_shrlt = 9;  %在以上scale下得到的shearlet个数

size_input = 48; %网络输入图像块的大小
size_label = size_input; %网络输出图像块的大小
stride = 48;
scale = 2;

%% initialization
data = zeros(size_input, size_input, Num_shrlt, 1);     %data和label一定要是4维的，不然会无法训练
label = zeros(size_label, size_label, Num_shrlt, 1);
padding = abs(size_input - size_label)/2;

count = 0;

%% generate data
ext = {'*.jpg','*.png','*.bmp','*.tif'};
filepaths   =  [];
for i = 1 : length(ext)
    filepaths = cat(1,filepaths, dir(fullfile(folder,ext{i})));
end

for i = 1 : length(filepaths)
    fprintf('Image No.: %d \n', i);
    image = imread(fullfile(folder,filepaths(i).name));
    
    if size(image,3)>1
        im_ycbcr = rgb2ycbcr(image);
        im_ycbcr = im_ycbcr(:, :, 1);
    end
    im_gnd = modcrop(im2double(im_ycbcr(:, :, 1)), scale);    %网络输出
    [hei,wid] = size(im_gnd);
    
    %     im_input = imresize(imresize(im_gnd, 1/scale, 'bicubic'),[hei,wid],'bicubic');    %网络输入
    
    im_LR = imresize(im_gnd, 1/scale, 'bicubic');       %图像变小
    
    im_bic = imresize(im_LR,scale,'bicubic');
    
    [coeffs_DST_bic,shearletSystem] = DSTImgDec(im_bic,DST_scale);     %对bicubic图像做DST，作为网络input
    shearletRMS = reshape(repmat((shearletSystem.RMS./min(shearletSystem.RMS)),[size(im_bic,1)*size(im_bic,2) 1]),[size(im_bic,1),size(im_bic,2),length(shearletSystem.RMS)]);
    coeffs_DST_bic = coeffs_DST_bic./shearletRMS;
    
    [coeffs_DST_gnd,~] = DSTImgDec(im_gnd,DST_scale);   %对gnd图像做DST，作为网络label
    coeffs_DST_gnd = coeffs_DST_gnd./shearletRMS;
    
    
    for x = 1 : stride : hei-size_input+1
        for y = 1 :stride : wid-size_input+1
            count=count+1;            
            data(:, :, :, count) = coeffs_DST_bic(x : x+size_input-1, y : y+size_input-1,:);
            label(:, :, :, count) = coeffs_DST_gnd(x+padding : x+padding+size_label-1, y+padding : y+padding+size_label-1,:);
        end
    end
end

%% shuffle the images
order = randperm(count);
data = data(:, :, :, order);
label = label(:,:,:, order);


%% writing to HDF5
created_flag = false;
totalct = 0;

for batchno = 1:max(1,floor(count/chunksz))
    fprintf('Batch No.: %d \n', batchno);
    last_read=(batchno-1)*chunksz;
    startloc = struct('dat',[1,1,1, totalct+1], 'lab', [1,1,1, totalct+1]);
    
    batchdata_HF = data(:,:,1:8, last_read+1:last_read+chunksz);
    batchlabs_HF = label(:,:,1:8, last_read+1:last_read+chunksz);
    
    batchdata_LF = data(:,:,9, last_read+1:last_read+chunksz);
    batchlabs_LF = label(:,:,9, last_read+1:last_read+chunksz);

    curr_dat_sz_HF = store2hdf5(Savepath_HF, batchdata_HF, batchlabs_HF, ~created_flag, startloc, chunksz);
    curr_dat_sz_LF = store2hdf5(Savepath_LF, batchdata_LF, batchlabs_LF, ~created_flag, startloc, chunksz);
      
    created_flag = true;
    totalct = curr_dat_sz_HF(end);
end

h5disp(Savepath_HF)
h5disp(Savepath_LF)

