close all

%load('J')
J = gpuArray(J);

%ss=svd(J);
%figure;semilogy(ss);

offsets = normrnd(0,1,size(vel));
vel_mod = vel+offsets;
seis_mod = a2d_mod_abc24(vel_mod,nbc,dx,nt,dt,s,coord,isFS);

diff = seis_mod-seis;

diff_alt = reshape(J*offsets(:), size(diff));

rms(diff(:)-diff_alt(:))/rms(diff(:)+diff_alt(:))
% err = seis(:)-seis_ref(:);
% offsets = J\err;
% 
% vel_mod = vel+reshape(offsets,size(vel));
% seis_mod=a2d_mod_abc24(vel_mod,nbc,dx,nt,dt,s,coord,isFS);
% 
% rms(seis_mod(:)-seis_ref(:))

rms(diff(:))
rms(diff_alt(:))