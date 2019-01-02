function [coeffs,shearletSystem] = DSTImgDec(X,scales_DST)


% create shearlets
shearletSystem = SLgetShearletSystem2D(0,size(X,1),size(X,2),scales_DST);

% % To get the i-th shearlet in the time domain
% shearlets_timedomain = zeros(size(shearletSystem.shearlets));
% for i = 1:49
% 
%    shearlets_timedomain(:,:,i) = fftshift(ifft2(ifftshift(shearletSystem.shearlets(:,:,i))));
% end
% %image the shearlets_timedomain 
% imshow([shearlets_timedomain(:,:,1),shearlets_timedomain(:,:,2),shearlets_timedomain(:,:,3),shearlets_timedomain(:,:,4);shearlets_timedomain(:,:,5),shearlets_timedomain(:,:,6),shearlets_timedomain(:,:,7),shearlets_timedomain(:,:,8);shearlets_timedomain(:,:,9),shearlets_timedomain(:,:,10),shearlets_timedomain(:,:,11),shearlets_timedomain(:,:,12);shearlets_timedomain(:,:,13),shearlets_timedomain(:,:,14),shearlets_timedomain(:,:,15),shearlets_timedomain(:,:,16)],[]);
% 
% figure(3)
% image([shearlets_timedomain(:,:,17),shearlets_timedomain(:,:,18),shearlets_timedomain(:,:,19),shearlets_timedomain(:,:,20);shearlets_timedomain(:,:,21),shearlets_timedomain(:,:,22),shearlets_timedomain(:,:,23),shearlets_timedomain(:,:,24);shearlets_timedomain(:,:,25),shearlets_timedomain(:,:,26),shearlets_timedomain(:,:,27),shearlets_timedomain(:,:,28);shearlets_timedomain(:,:,29),shearlets_timedomain(:,:,30),shearlets_timedomain(:,:,31),shearlets_timedomain(:,:,32)]);
% 
% figure(4)
% image([shearlets_timedomain(:,:,33),shearlets_timedomain(:,:,34),shearlets_timedomain(:,:,35),shearlets_timedomain(:,:,36);shearlets_timedomain(:,:,37),shearlets_timedomain(:,:,38),shearlets_timedomain(:,:,39),shearlets_timedomain(:,:,40);shearlets_timedomain(:,:,41),shearlets_timedomain(:,:,42),shearlets_timedomain(:,:,43),shearlets_timedomain(:,:,44);shearlets_timedomain(:,:,45),shearlets_timedomain(:,:,46),shearlets_timedomain(:,:,47),shearlets_timedomain(:,:,48)]);
% 
% figure(5)
% image(shearlets_timedomain(:,:,49))




% decomposition
% coeffs = SLsheardec2D(Xnoisy,shearletSystem);
coeffs = SLsheardec2D(X,shearletSystem);


% cod_coeffs = zeros(size(coeffs));
% for i = 1:length(size(coeffs,3))
%     cod_coeffs(:,:,i) = wcodemat(coeffs(:,:,i),nbcol);
% end
% 
% image([cod_coeffs(:,:,1),cod_coeffs(:,:,2);cod_coeffs(:,:,3),cod_coeffs(:,:,4)]);


% % plot CCDF
% All_coef = abs(coeffs(:));
% All_coef = sort(All_coef,'descend');
% l_All_coef = length(All_coef);
% 
% norm_X_dwt2 = norm(All_coef,1);
% 
% rate = [0.01:0.01:1];
% Energy = zeros(size(rate));
% for i = 1:length(rate)
%     Energy(i) = norm(All_coef(floor(rate(i)*l_All_coef):end),1)/norm_X_dwt2;
% end
% 
% rate = [0,rate];
% Energy = [1,Energy];
% figure(1)
% plot(rate,Energy);
% xlabel('number of coeffs');
% ylabel('Energy');
% 
% 
% % compute energy of different layers of shrl
% sum_coeffs = sum(sum(abs(coeffs),1),2);
% sum_coeffs = sum_coeffs(:);



% %summarize every 16 layers of coeffs, leave the 49th, image them
% [a, b] = size(coeffs(:,:,1));
% 
% ImgDec = zeros(a,b,4);
% for i = 1:3
%     for j = (1+16*(i-1)):(16+16*(i-1))
%         ImgDec(:,:,i) = ImgDec(:,:,i)+coeffs(:,:,j);
%     end
%     ImgDec(:,:,i) = wcodemat(ImgDec(:,:,i),nbcol);
% end
% ImgDec(:,:,4) = wcodemat(coeffs(:,:,49),nbcol);
% 
% figure(2);
% imshow(uint8([ImgDec(:,:,1),ImgDec(:,:,2);ImgDec(:,:,3),ImgDec(:,:,4)]));
    

% %image the coeffs 
% image([coeffs(:,:,1),coeffs(:,:,2),coeffs(:,:,3),coeffs(:,:,4);coeffs(:,:,5),coeffs(:,:,6),coeffs(:,:,7),coeffs(:,:,8);coeffs(:,:,9),coeffs(:,:,10),coeffs(:,:,11),coeffs(:,:,12);coeffs(:,:,13),coeffs(:,:,14),coeffs(:,:,15),coeffs(:,:,16)]);
% 
% figure(3)
% image([coeffs(:,:,17),coeffs(:,:,18),coeffs(:,:,19),coeffs(:,:,20);coeffs(:,:,21),coeffs(:,:,22),coeffs(:,:,23),coeffs(:,:,24);coeffs(:,:,25),coeffs(:,:,26),coeffs(:,:,27),coeffs(:,:,28);coeffs(:,:,29),coeffs(:,:,30),coeffs(:,:,31),coeffs(:,:,32)]);
% 
% figure(4)
% image([coeffs(:,:,33),coeffs(:,:,34),coeffs(:,:,35),coeffs(:,:,36);coeffs(:,:,37),coeffs(:,:,38),coeffs(:,:,39),coeffs(:,:,40);coeffs(:,:,41),coeffs(:,:,42),coeffs(:,:,43),coeffs(:,:,44);coeffs(:,:,45),coeffs(:,:,46),coeffs(:,:,47),coeffs(:,:,48)]);
% 
% figure(5)
% image(coeffs(:,:,49))



%%thresholding
% coeffs = coeffs.*(abs(coeffs) > thresholdingFactor*reshape(repmat(shearletSystem.RMS,[size(X,1)*size(X,2) 1]),[size(X,1),size(X,2),length(shearletSystem.RMS)])*sigma);
% 
% %%reconstruction
% Xrec = SLshearrec2D(coeffs,shearletSystem);
% 
% elapsedTime = toc;
% fprintf([num2str(elapsedTime), ' s\n']);
% 
% %%compute psnr
% PSNR = SLcomputePSNR(X,Xrec);
% 
% fprintf(['PSNR: ', num2str(PSNR) , ' db\n']);
% 
% figure;
% colormap gray;
% subplot(1,3,1);
% imagesc(X);
% title('original image');
% axis image;
% 
% subplot(1,3,2);
% imagesc(Xnoisy);
% title(['noisy image, sigma = ', num2str(sigma)]);
% axis image;
% 
% subplot(1,3,3);
% imagesc(Xrec);
% title(['denoised image, PSNR = ',num2str(PSNR), ' db']);
% axis image;



%
%  Copyright (c) 2014. Rafael Reisenhofer
%
%  Part of ShearLab3D v1.1
%  Built Mon, 10/11/2014
%  This is Copyrighted Material
%
%  If you use or mention this code in a publication please cite the website www.shearlab.org and the following paper:
%  G. Kutyniok, W.-Q. Lim, R. Reisenhofer
%  ShearLab 3D: Faithful Digital SHearlet Transforms Based on Compactly Supported Shearlets.
%  ACM Trans. Math. Software 42 (2016), Article No.: 5.
