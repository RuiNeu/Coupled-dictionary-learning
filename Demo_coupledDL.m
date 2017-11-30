clear all; clc; close all;
addpath(genpath('improveDL_tool'));

%% Training sample
img_path_1 = 'Data/Training_focus/'; 
img_path_2 = 'Data/Tranining_blur/';
pSize = 8;           
nSmp = 10000;       

p.ep = 0.15;
p.ds = 256;
p.itern = 2;   % itern 
p.DUC = 1;
    
% randomly capturing image patches
dir_1 = dir(fullfile(img_path_1, '*.jpg'));
dir_2= dir(fullfile(img_path_2, '*.jpg'));
Xf = [];
Xb = [];
iN = length(dir_1);
pn_img = zeros(1, iN);
for ii = 1:length(dir_1),
    im_1 = imread(fullfile(img_path_1, dir_1(ii).name));
    [im_m,im_n]=size(im_1);
    im_1=im_1(1:im_m,1:im_n);
    pn_img(ii) = prod(size(im_1));
end
pn_img = floor(pn_img * nSmp/sum(pn_img));

for ii =1:iN
    pn = pn_img(ii);
    im_1 = imread(fullfile(img_path_1, dir_1(ii).name));
    im_2 = imread(fullfile(img_path_2, dir_2(ii).name));
    [im_m,im_n]=size(im_1);
    im_1 = im_1(1:im_m,1:im_n);
    im_2 = im_2(1:im_m,1:im_n);
    [Fp, Bp] = sampPatch(im_1,im_2,pSize, pn);
    Xf = [Xf, Fp];
    Xb = [Xb, Bp];
end

pvars = var(Xf, 0, 1);
idx = pvars > 10;
Xf = Xf(:, idx);
Xb = Xb(:, idx);
%Xf1 = Xf * diag(1./sqrt(sum(Xf.*Xf)));
%Xb1 = Xb * diag(1./sqrt(sum(Xb.*Xb)));
Xf = normcols(Xf);
Xb = normcols(Xb);

rIndex= randperm(size(Xf,2));

D_focus = Xf(:,rIndex(1:p.ds));
D_blur = Xb(:,rIndex(1:p.ds));
err =zeros(1,5);
[Df, Db, err] = trainCoupledD(D_focus,D_blur, Xf, Xb, p);
dict_path = ['CoupledD/coupledD_' num2str(p.ds) '_' num2str(pSize) '_' num2str(p.ep) '.mat' ];
save(dict_path, 'Df', 'Db','err');
plot(err);

figure;
subplot(121); displayDictionaryElementsAsImage(Df, 16, 16, pSize, pSize, 0 );
title('The Focus dictionary');

subplot(122); displayDictionaryElementsAsImage(Db, 16, 16, pSize, pSize,0 );
title('The Blur dictionary');