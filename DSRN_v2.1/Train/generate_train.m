clear;close all;
addpath(genpath(cd));
addpath(genpath('/home/tensor/Desktop/Gengtianyu/ShearLab3Dv11'))

%% settings
folder = '291-aug';
size_input = 41;
size_label = 41;
stride = 41;
padding = abs(size_input - size_label)/2;
chunksz = 64;

%% Shearlet Parameters
DST_scale = 1;  %shearlet分解的scale
Num_shrlt = 9;  %在以上scale下得到的shearlet个数


%% initialization

%% generate data
ext = {'*.jpg','*.png','*.bmp','*.tif'};
filepaths   =  [];
for i = 1 : length(ext)
    filepaths = cat(1,filepaths, dir(fullfile(folder,ext{i})));
end


randIdx = randperm(length(filepaths));

for j = 1 : 10
    count = 0;
    data = zeros(size_input, size_input, Num_shrlt, 1);
    label = zeros(size_label, size_label, Num_shrlt, 1);
    savepath = ['train' num2str(j) '.h5'];

    for i = 1 + (j-1)*291*2 : 291*2 + (j-1)*291*2
        for scale = 2:4
            
            fprintf('scale:%d,%d\n',scale,i);

            image = imread(fullfile(folder,filepaths(randIdx(i)).name));
            if size(image,3)>1
                im_ycbcr = rgb2ycbcr(image);
                im_gnd = modcrop(im2double(im_ycbcr(:, :, 1)), scale);
            else
                im_gnd = modcrop(im2double(image), scale);
            end
            [hei,wid] = size(im_gnd);

            im_bic = imresize(imresize(im_gnd,1/scale,'bicubic'),[hei,wid],'bicubic');

            [coeffs_DST_bic,~] = DSTImgDec(im_bic,DST_scale);     %对bicubic图像做DST，作为网络input

            [coeffs_DST_res,~] = DSTImgDec((im_gnd - im_bic),DST_scale);   %对gnd图像做DST，作为网络label


            for x = 1 : stride : hei-size_input+1
                for y = 1 :stride : wid-size_input+1
                    count=count+1;            
                    data(:, :, :, count) = coeffs_DST_bic(x : x+size_input-1, y : y+size_input-1,:);
                    label(:, :, :, count) = coeffs_DST_res(x+padding : x+padding+size_label-1, y+padding : y+padding+size_label-1,:);
                end
            end

        end
    end

    order = randperm(count);
    data = data(:, :, :, order);
    label = label(:, :, :, order);

    %% first time writing to HDF5
    created_flag = false;
    totalct = 0;

    for batchno = 1:floor(count/chunksz)
        fprintf('Batch No.: %d \n', batchno);
        last_read=(batchno-1)*chunksz;
        batchdata = data(:,:,:,last_read+1:last_read+chunksz);
        batchlabs = label(:,:,:,last_read+1:last_read+chunksz);

        startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,1,1,totalct+1]);
        curr_dat_sz = store2hdf5(savepath, batchdata, batchlabs, ~created_flag, startloc, chunksz);
        created_flag = true;
        totalct = curr_dat_sz(end);
    end
    h5disp(savepath);
end


