function seis=a2d_mod_abc28(v,nbc,dx,nt,dt,s,coord,isFS)
%  Copyright (C) 2010 Center for Subsurface Imaging and Fluid Modeling (CSIM),
%  King Abdullah University of Science and Technology, All rights reserved.
%
%  author:   Xin Wang
%  email:    xin.wang@kaust.edu.sa
%  date:     Sep 26, 2012
%  purpose:  2DTDFD solution to acoustic wave equation with accuracy of 2-8
%            use the absorbing boundary condition
%
%  seis=a2d_mod_abc28(v,nbc,dx,nt,dt,s,coord,isFS)
%
%  IN   v(:,:) -- velocity,      nbc  -- grid number of boundary 
%       dx     -- grid intervel, nt   -- number of sample
%       dt     -- time interval, s(:) -- wavelet
%       coord: coordinates, including sx,sz,gx(:),gz(:)
%       isFS   -- Free surface condition
%  OUT  seis(:,:) : Output seismogram
%
%  Example 1: three layer model
%
%    nz=81; nx=201; vel=zeros(nz,nx)+1000; vel(31:60,:)=1700; vel(61:end,:)=1500; 
%    dx=5;nbc=40; nt=2001; dt=0.0005; freq=15; s=ricker(freq,dt); isFS=false;
%    coord.sx = (nx-1)/2*dx; coord.sz = 0; 
%    coord.gx=(1:2:nx)*dx; coord.gz=zeros(size(coord.gx));
%    tic;seis=a2d_mod_abc28(vel,nbc,dx,nt,dt,s,coord,isFS);toc;
%    figure;subplot(311);imagesc(x,z,vel);colorbar;
%    xlabel('X (m)'); ylabel('Z (m)'); title('velocity');
%    subplot(312);plot((0:numel(s)-1)*dt,s);
%    xlabel('Time (s)'); ylabel('Amplitude');title('wavelet');
%    t=(0:nt-1)*dt; g=1:numel(coord.gx);
%    subplot(313);colormap(gray);imagesc(g,t,seis);
%    if isFS, figure_title='Data with Free Surface'; 
%    else figure_title='Data without Free Surface'; end;
%    title(figure_title);ylabel('Time (s)');xlabel('g #');caxis([-0.25 0.25]);

seis=zeros(nt,numel(coord.gx));
ng=numel(coord.gx);
c1 = -49/18;    % center weight
c2 = 3/2;       % ±1 offsets
c3 = -3/20;     % ±2 offsets
c4 = 1/90;      % ±3 offsets

% setup ABC and temperary variables
v=padvel(v,nbc);
abc=AbcCoef2D(v,nbc,dx);
alpha=(v*dt/dx).^2; kappa=abc*dt;
temp1=2+2*c1*alpha-kappa; temp2=1-kappa;
beta_dt = (v*dt).^2;
s=expand_source(s,nt);
[isx,isz,igx,igz]=adjust_sr(coord,dx,nbc);
p1=zeros(size(v)); p0=zeros(size(v)); [nzbc,nxbc]=size(v);nzp=nzbc-nbc;nxp=nxbc-nbc;

% Time Looping
%p1 = gpuArray(p1);p0=gpuArray(p0);
temp1 = gpuArray(temp1);temp2 = gpuArray(temp2);
alpha=gpuArray(alpha);
p1 = gpuArray(p1);p0=gpuArray(p0);

maxd = 3;
kernSz = 2*maxd + 1;
h = zeros(kernSz, kernSz, 'like', p1);
center = maxd + 1;
cs = [c2 c3 c4];

for d = 1:numel(cs)
    k = cs(d);
    off = d;   % because your c2→distance=1, c3→dist=2, etc.
    h(center,    center+off) = k;
    h(center,    center-off) = k;
    h(center+off, center   ) = k;
    h(center-off, center   ) = k;
end

% Push both data and kernel to the GPU
hg  = gpuArray(h);

for it=1:nt
    
    %class(p1)
    %class(p0)
    % p=temp1.*p1-temp2.*p0+alpha.*...
    %     (c2*(circshift(p1,[0,1,0])+circshift(p1,[0,-1,0])+circshift(p1,[1,0,0])+circshift(p1,[-1,0,0]))...
    %     +c3*(circshift(p1,[0,2,0])+circshift(p1,[0,-2,0])+circshift(p1,[2,0,0])+circshift(p1,[-2,0,0]))...
    %     +c4*(circshift(p1,[0,3,0])+circshift(p1,[0,-3,0])+circshift(p1,[3,0,0])+circshift(p1,[-3,0,0]))...
    %     +c5*(circshift(p1,[0,4,0])+circshift(p1,[0,-4,0])+circshift(p1,[4,0,0])+circshift(p1,[-4,0,0])));
    %class(p)

    %--- 1) build your 2D stencil kernel once ---

    % do the convolution in one go, with circular (wrap) boundaries
    lapg = imfilter(p1, hg, 'circular', 'conv');
    
    % finish your update
    p = temp1.*p1 - temp2.*p0 + alpha.*lapg;

    %std(p(:)-p_alt(:))/std(p(:))
   
    
    p(isz,isx) = p(isz,isx) + beta_dt(isz,isx) * s(it);
    
    % dipole source
    %p(isz-2,isx) = p(isz-2,isx) - beta_dt(isz-2,isx) *wavelet(it);
    if isFS
        p(nbc+1,:)=0.0;
        p(nbc:-1:nbc-3,:) = - p(nbc+2:nbc+5,:);
    end
    %p=gather(p);
    % snapshopt
    %if (mod(it,100)==1)
    %    imagesc(p(nbc+1:nzp,nbc+1:nxp));caxis([-2.5 2.5]);
    %    title(['Time=',num2str((it-1)*dt),'s']); pause(0.05);
    %end
    pp=gather(p);
    for ig=1:ng
        seis(it,ig)=pp(igz(ig),igx(ig));
    end
    p0=p1;
    p1=p;
end

end

function v=padvel(v0,nbc)
v=[repmat(v0(:,1),1,nbc), v0, repmat(v0(:,end),1,nbc)];
v=[repmat(v(1,:),nbc,1); v; repmat(v(end,:),nbc,1)];
end

function s=expand_source(s0,nt)
nt0=numel(s0);
if nt0<nt
    s=zeros(nt,1);s(1:nt0)=s0;
else
    s=s0;
end
end

function [isx,isz,igx,igz]=adjust_sr(coord,dx,nbc)
% set and adjust the free surface position
isx=round(coord.sx/dx)+1+nbc;isz=round(coord.sz/dx)+1+nbc;
igx=round(coord.gx/dx)+1+nbc;igz=round(coord.gz/dx)+1+nbc;
if abs(coord.sz) <0.5, isz=isz+1; end
igz=igz+(abs(coord.gz)<0.5)*1;
end