close all
clearvars

i_todo = 5;
'minvel cheat'
for n=20:20

    v = py.numpy.load('F:\seismic\data\train_samples\FlatFault_B/vel6_1_0.npy').single();
    vel = squeeze(v(n,1,:,:));

    s = py.numpy.load('F:\seismic\data\train_samples\FlatFault_B/seis6_1_0.npy').single();
    seis_ref = squeeze(s(n,i_todo,:,:));

    nz=70; nx=70;
    dx=10; nbc=120; nt=1000; dt=0.001;
    freq=15; s=ricker(freq,dt); isFS=false;
    coord.sx = 69*dx; coord.sz = 1*dx;
    coord.gx=(0:nx-1)*dx; coord.gz=1*ones(size(coord.gx))*dx;

    %vel_extend = zeros([size(vel,1)+1, size(vel,2)]);
    %vel_extend(2:end,:) = vel;
    %vel_extend(1,:) = vel_extend(2,:);
    vel_mod = vel(:,1:end-1);
    vel_mod = [vel(:,1) vel];
    seis=a2d_mod_abc24(vel,nbc,dx,nt,dt,s,coord,isFS);
    %seis2=a2d_mod_abc24(vel_mod,nbc,dx,nt,dt,s,coord,isFS);
    %err1=seis-seis2;

    %seis = -seis+2*seis2;

    %figure;imagesc(vel);colorbar;
    %figure;imagesc(seis);colorbar;
    %figure;imagesc(seis_ref);colorbar;
    figure;imagesc(seis-seis_ref);colorbar;
    %figure;imagesc(seis2-seis);colorbar;
    %% clim([-10,0]);
    %figure;imagesc(diff(seis,1,2));colorbar;
    %figure;imagesc(diff(seis,1,1));colorbar;

    rms(seis(:)-seis_ref(:))
    drawnow

end
return

J = zeros(length(seis(:)), length(vel(:)));
cur_col = 1;
i_row = 35;
i_col = 35;
figure;hold on;
for step_size = [1,3,10,30,100]
    vel_mod = vel;
    vel_mod(i_row,i_col) = vel_mod(i_row,i_col)+step_size;
    seis_mod=a2d_mod_abc24(vel_mod,nbc,dx,nt,dt,s,coord,isFS);
    J(:,cur_col) = (seis_mod(:)-seis(:))/step_size;
    plot(J(:,cur_col));
    rms(J(:,cur_col))
    %cur_col=cur_col+1;
end

% load('J')
% J = gpuArray(J);
% 
% vel_wrong = vel+normrnd(0,1,size(vel));
% %seis_wrong = seis+reshape(J*(vel_wrong(:)-vel(:)),size(seis)); 
% seis_wrong = a2d_mod_abc24(vel_wrong,nbc,dx,nt,dt,s,coord,isFS);
% 
% offsets = gather(J\(gpuArray(seis(:)-seis_wrong(:))));
% vel_restored = vel_wrong+reshape(offsets,size(vel_wrong));
% rms(vel(:)-vel_wrong(:))
% rms(vel(:)-vel_restored(:))
% 
% 
% 
