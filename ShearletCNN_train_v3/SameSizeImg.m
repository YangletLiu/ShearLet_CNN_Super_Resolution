folder = 'Train';
filepaths = dir(fullfile(folder,'*.tif'));
count = 0;

for i = 1 : length(filepaths)
    fprintf('Image No.: %d \n', i);
    count=count+1;
    image = imread(fullfile(folder,filepaths(i).name));
    image = imresize(image,[64,64]);
    
    imwrite(image,fullfile(folder,strcat('s_',filepaths(i).name)));
    
    
    end
