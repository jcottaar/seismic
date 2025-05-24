close all
clearvars

i_todo = 3;
for n=5:5

    %v = py.numpy.load('F:\seismic\data\train_samples\FlatFault_B/vel6_1_0.npy').double();
    v = py.numpy.load('F:\seismic\data\train_samples\CurveVel_B\model\model1.npy').double();
    vel = squeeze(v(n,1,:,:));

    s = py.numpy.load('F:\seismic\data\train_samples\CurveVel_B/data\data1.npy').double();
    seis_ref = squeeze(s(n,i_todo,1:999,:));

    nz=70; nx=70;
    dx=10; nbc=120; nt=999; dt=0.001;
    freq=15; s=ricker(freq,dt);
    coord.sx = 34*dx; coord.sz = 1*dx;
    coord.gx=(0:nx-1)*dx; coord.gz=1*ones(size(coord.gx))*dx;

    %vel_extend = zeros([size(vel,1)+1, size(vel,2)]);
    %vel_extend(2:end,:) = vel;
    %vel_extend(1,:) = vel_extend(2,:);
    seis=a2d_mod_abc24(vel,nbc,dx,nt,dt,s,coord);

    figure;imagesc(vel);colorbar;
    figure;imagesc(seis);colorbar;
    figure;imagesc(seis_ref);colorbar;
    figure;imagesc(seis-seis_ref);colorbar;
    % %figure;imagesc(log(abs(seis-seis_ref))/log(10));colorbar;
    % %clim([-4,0]);
    % %figure;imagesc(diff(seis,1,2));colorbar;
    % %figure;imagesc(diff(seis,1,1));colorbar;
    % 
    diff = seis-seis_ref;
    rms(seis(:)-seis_ref(:))
    drawnow

end

return


% % [c,ia,ic] = unique(vel(:));
% % c_orig = c;
% % vel_new = vel;
% % 
% % for ii=1:1
% % 
% %     step_size = 0.1;
% %     J = zeros(length(seis(:)), length(c));
% %     for ii=1:length(c)
% %         vel_mod = vel_new(:);
% %         vel_mod(ic==ii) = vel_mod(ic==ii)+step_size;
% %         vel_mod = reshape(vel_mod, size(vel_new));
% % 
% %         seis_mod=a2d_mod_abc24(vel_mod,nbc,dx,nt,dt,s,coord,isFS);
% %         J(:,ii) = (seis_mod(:)-seis(:))/step_size;
% %     end
% % 
% %     err = seis-seis_ref;
% %     err = err(:);
% % 
% %     offsets=(J'*J)\(J'*err);
% % 
% %     c=c-offsets;
% %     vel_built = 0*vel_new(:);
% %     for ii=1:length(c)
% %         vel_built(ic==ii)=c(ii);
% %     end
% %     vel_built = reshape(vel_built, size(vel));
% %     vel_new = vel_built;
% % 
% %     seis2=a2d_mod_abc24(vel_built,nbc,dx,nt,dt,s,coord,isFS);
% % 
% %     figure;imagesc(seis2-seis_ref);colorbar;
% % 
% %     rms(seis2(:)-seis_ref(:))
% % 
% % end
% % 
%figure;plot(c_orig,offsets)

offsets = normrnd(0,0.0001,size(vel));
vel_mod = vel+offsets;
% seis_mod = a2d_mod_abc24(vel_mod,nbc,dx,nt,dt,s,coord);
% diff = seis_mod-seis;
% 
% [seis,seis_diff] = a2d_mod_abc24(vel,nbc,dx,nt,dt,s,coord,offsets);
% 
% 
% rms(diff(:)-seis_diff(:))/rms(diff(:)+seis_diff(:))
% 


[c1,d]=a2d_mod_abc24(vel,nbc,dx,nt,dt,s,coord,offsets);
c2=a2d_mod_abc24(vel_mod,nbc,dx,nt,dt,s,coord);
dr = c2-c1;

rms(d(:))
rms(dr(:))
rms(d(:)-dr(:))/rms(d(:)+dr(:))

% class(vel)
% [c1,d_single] = a2d_mod_abc24(single(vel),nbc,dx,nt,dt,s,coord,offsets);
% 
% rms(d(:)-d_single(:))/rms(d(:)+d_single(:))

J = zeros(length(seis(:)), length(vel(:)));
step_size = 1;
cur_col = 1;
for i_col = 1:size(vel,2)
    i_col
    for i_row = 1:size(vel,1)
        v_diff = 0*vel;
        v_diff(i_row,i_col) = 1;
        [~,seis_diff]=a2d_mod_abc24(vel,nbc,dx,nt,dt,s,coord,v_diff);
        J(:,cur_col) = seis_diff(:);
        cur_col=cur_col+1;
    end
end
save('J', 'J', '-v7.3');

load('J')
J = gpuArray(J);
%rng(0,'twister');
%J = normrnd(0,1,size(J));
%J = gpuArray(J);

ss=svd(J);
figure;semilogy(ss);

rng(0,'twister');
offsets = normrnd(0,100,size(vel));
%offsets = 0*vel;
%offsets(1,1)=1;
%offsets(1,2)=1;
vel_mod = vel+offsets;
[c1,d]=a2d_mod_abc24(vel,nbc,dx,nt,dt,s,coord,offsets);
seis_mod = a2d_mod_abc24(vel_mod,nbc,dx,nt,dt,s,coord);

diff = seis_mod-c1;

diff_alt = reshape(J*offsets(:), size(diff));

rms(diff(:)-diff_alt(:))/rms(diff(:)+diff_alt(:))

diff = diff_alt;

lambda_vals = 10.^linspace(-18,-10,20)
res =[];
res2 = [];
for lambda = lambda_vals
    lambda
    %cost_before = lambda*norm(vel_mod(:))^2 + norm(diff(:))^2
    cost_before = norm(diff(:))^2
    proposed_offset = (lambda*eye(length(vel_mod(:))) + J'*J)\(-J'*diff(:)-0*lambda*vel_mod(:));
    cost_after = lambda*norm(proposed_offset)^2 + norm(diff(:)+J*proposed_offset)^2
    cost_use_correct = lambda*norm(offsets(:))^2 + norm(diff(:)+J*(-offsets(:)))^2
    res(end+1)=rms(proposed_offset);
    res2(end+1) = rms(proposed_offset+offsets(:));
    res
    res2
end
figure;loglog(lambda_vals,res);hold on;loglog(lambda_vals,res2);

proposed_offset = reshape(proposed_offset, size(vel));
figure;imagesc(proposed_offset);colorbar();

figure;imagesc(-offsets);colorbar();

figure;imagesc(proposed_offset+offsets);colorbar();

rms(vel_mod(:))