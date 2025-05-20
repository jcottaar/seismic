function [damp,damp_diff]=AbcCoef2D(vel,nbc,dx,vel_diff)

do_gradient = exist('vel_diff', 'var');

[nzbc,nxbc]=size(vel);
%velmin=min(vel(:));
% if do_gradient
%     inds = vel(:)==velmin;
%     vd = vel_diff(:);
%     velmin_diff = 0*min(vd(inds));
% end
velmin = 1535;
velmin_diff = 0;
nz=nzbc-2*nbc;nx=nxbc-2*nbc;
a=(nbc-1)*dx;
kappa = 3.0 * velmin * log(10000000.0) / (2.0 * a);
if do_gradient
    kappa_diff = 3.0 * velmin_diff * log(10000000.0) / (2.0 * a);
end
% setup 1D BC damping array
damp1d=kappa*((0:(nbc-1))*dx/a).^2;
% setup 2D BC damping array
damp=zeros(nzbc,nxbc);
% divide the whole area to 9 zones, and 5th is the target zone
%  1   |   2   |   3
%  ------------------
%  4   |   5   |   6
%  ------------------
%  7   |   8   |   9
% fulltill zone 1, 4, 7 and 3, 6, 9
for iz=1:nzbc
    damp(iz,1:nbc)=damp1d(nbc:-1:1);
    damp(iz,nx+nbc+1:nx+2*nbc)=damp1d(:);
end
% full fill zone 2 and 8
for ix=nbc+1:nbc+nx
    damp(1:nbc,ix)=damp1d(nbc:-1:1);
    damp(nbc+nz+1:nz+2*nbc,ix)=damp1d(:);
end

if do_gradient
    % setup 1D BC damping array
    damp1d_diff=kappa_diff*((0:(nbc-1))*dx/a).^2;
    % setup 2D BC damping array
    damp=zeros(nzbc,nxbc);
    % divide the whole area to 9 zones, and 5th is the target zone
    %  1   |   2   |   3
    %  ------------------
    %  4   |   5   |   6
    %  ------------------
    %  7   |   8   |   9
    % fulltill zone 1, 4, 7 and 3, 6, 9
    for iz=1:nzbc
        damp_diff(iz,1:nbc)=damp1d_diff(nbc:-1:1);
        damp_diff(iz,nx+nbc+1:nx+2*nbc)=damp1d_diff(:);
    end
    % full fill zone 2 and 8
    for ix=nbc+1:nbc+nx
        damp_diff(1:nbc,ix)=damp1d_diff(nbc:-1:1);
        damp_diff(nbc+nz+1:nz+2*nbc,ix)=damp1d_diff(:);
    end
end
end