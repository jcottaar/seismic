clear all; close all; clc;

%% Define model and acquisition geometry
nz=201; nx=401;
vel=zeros(nz,nx)+1000; vel(81:140,:)=1700; vel(141:end,:)=1500; dx=5.0;
x = (0:nx-1)*dx; z = (0:nz-1)*dx; nbc=80; nt=3001; dt=0.0005;
freq=15; s=ricker(freq,dt); isFS=false; 
coord.sx = (nx-1)/2*dx; coord.sz = 0;
coord.gx=(1:2:nx)*dx; coord.gz=zeros(size(coord.gx));
t=(0:nt-1)*dt; g=1:numel(coord.gx);

%% FD modeling to see the snapshot
figure; set(gcf,'position',[200,200,400,400]);
tic;seis=a2d_mod_abc28(vel,nbc,dx,nt,dt,s,coord,isFS);toc;

%% Ploting
figure; set(gcf,'position',[200,200,800,800]);
subplot(311);imagesc(x,z,vel);colorbar;
xlabel('X (m)'); ylabel('Z (m)'); title('velocity');
subplot(312);plot((0:numel(s)-1)*dt,s);
xlabel('Time (s)'); ylabel('Amplitude');title('wavelet');
subplot(313);colormap(gray);imagesc(g,t,seis);
title('Seismogram');ylabel('Time (s)');xlabel('g #');caxis([-0.5 0.5]);