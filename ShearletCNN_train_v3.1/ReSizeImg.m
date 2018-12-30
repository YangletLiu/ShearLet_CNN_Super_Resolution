% folder = 'Train\ChosenDrone_train';
folder = 'Test\ChosenDrone_test';
filepaths = dir(fullfile(folder,'*.jpg'));
count = 0;
scale = 2;

for i = 1 : length(filepaths)
    fprintf('Image No.: %d \n', i);
    count=count+1;
    image = imread(fullfile(folder,filepaths(i).name));
    image = modcrop(im2double(image), scale);    %ÍøÂçÊä³ö
    
    image = imresize(image, 1/scale, 'bicubic');    %ÍøÂçÊäÈë
    
    
    imwrite(image,fullfile(folder,strcat('s_',filepaths(i).name)));
    
    
end
