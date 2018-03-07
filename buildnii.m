
close all;
 clear;
load_nii sMurillo-0002-00001-000001-01;

a = ans;



 %load 'a1.mat';
fTE1 = fopen('tpirec_DeNoise_ySh5_kw0.csos','r', 'b');

N=64;

rEcho = fread(fTE1, 2*N*N*N, 'float32', 'ieee-le');

image = reshape(rEcho, [2,N,N,N]);
image = squeeze(image(1,:,:,:));

a.img = image(:,64:-1:1,64:-1:1);
a.hdr.dime.bitpix = 220/64;
a.hdr.dime.datatype = 64;
a.hdr.dime.dim(2) = 64;
a.hdr.dime.dim(3) = 64;
a.hdr.dime.dim(4) = 64;
a.hdr.dime.dim(6) =220/64;
a.hdr.dime.dim(7) = 220/64;
a.hdr.dime.dim(8) = 220/64;
a.hdr.dime.pixdim(2) = 220/64;
a.hdr.dime.pixdim(3) = 220/64;
a.hdr.dime.pixdim(4) = 220/64;
a.hdr.hist.originator(1) =32;
a.hdr.hist.originator(2) =32;
a.hdr.hist.originator(3) =32;
save_nii(a, 'mynii3.nii')