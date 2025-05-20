n

v = py.numpy.load('F:\seismic\data\train_samples\FlatFault_B/vel6_1_0.npy').double();
vel = squeeze(v(n,1,:,:));

class(vel)
[seis,seis_diff] = a2d_mod_abc24(vel,nbc,dx,nt,dt,s,coord,offsets);

offsets = normrnd(0,0.01,size(vel));
vel_mod = vel+offsets;
seis_mod = a2d_mod_abc24(vel_mod,nbc,dx,nt,dt,s,coord);
diff = seis_mod-seis;

vel = single(vel);
[seis,seis_diff2] = a2d_mod_abc24(vel,nbc,dx,nt,dt,s,coord,offsets);


rms(diff(:)-seis_diff(:))/rms(diff(:)+seis_diff(:))

rms(seis_diff2(:)-seis_diff(:))/rms(seis_diff2(:)+seis_diff(:))




% global check
% global check_diff
% a2d_mod_abc24(vel,nbc,dx,nt,dt,s,coord,offsets);
% c1 = check;
% d = check_diff;
% a2d_mod_abc24(vel_mod,nbc,dx,nt,dt,s,coord,offsets);
% c2 = check;
% dr = c2-c1;
% 
% rms(d(:))
% rms(dr(:))
% rms(d(:)-dr(:))/rms(d(:)+dr(:))