%% INFORMATION
%  author: Tony Saad
%  date:   April 1, 2014
%
% Given a turbulent energy spectrum, this script generates a random
% velocity field that produces the input spectrum up to a certain wave
% number limit. This wave number limit is determined by the grid size
%
% The method used to generate the data is based on Lars Davidson's work for
% generating inlet turbulence. This code is loosly based on code obtained
% from his website, and can be found here: 
% http://www.tfd.chalmers.se/~lada/projects/inlet-boundary-conditions//proright.html#starten
% and here:
% http://www.tfd.chalmers.se/~lada/projects/inlet-boundary-conditions//synt_main.m
%
% This version of the script extends the method to 3D and uses more
% efficient matlab matrix-vector multiplication.
%
% Literature:
%
% L. Davidson, "Using Isotropic Synthetic Fluctuations as Inlet Boundary Conditions for Unsteady
% Simulations", "Advances and Applications in Fluid Mechanics",
% Vol 1, No  =1, pp. 1-35, 2007
%
% L. Davidson, "HYBRID LES-RANS:  Inlet Boundary Conditions for Flows With Recirculation",
% "Advances in Hybrid RANS-LES Modelling",
% Notes on Numerical Fluid Mechanics and Multidisciplinary Design,
% Springer Verlag, pp. 55-66, Vol. 97, 2008.
%
% L. Davidson, "Hybrid LES-RANS: Inlet Boundary Conditions for Flows Including Recirculation",
% 5th International Symposium on Turbulence and Shear Flow Phenomena,
% Vol. 2, pp. 689-694, Munich, Germany, 2007.

%% Main program
close all
clear all

% load an experimental specturm. Alternatively, specify it via a function call
load cbc_spectrum.txt         % cbc classic dataset
kcbc=cbc_spectrum(:,1)*100; 
ecbc=cbc_spectrum(:,2)*1e-6; 

%% USER INPUT
%set the number of modes.
nmodes =400; 

Lx = 9*2*pi/100; % domain size in the x direction
Ly = 9*2*pi/100; % domain size in the y direction
Lz = 9*2*pi/100; % domain size in the z direction

nx = 32;         % number of cells in the x direction
ny = 32;         % number of cells in the y direction
nz = 32;         % number of cells in the z direction

% generate cell centered x-grid
dx = Lx/nx;
xc = dx/2 + (0:nx-1)*dx;

% generate cell centered y-grid
dy = Ly/ny;
yc = dy/2 + (0:ny-1)*dy;

% generate cell centered z-grid
dz = Lz/nz;
zc = dz/2 + (0:nz-1)*dz; % cell centered coordinates

% smallest wavenumber - either specified via
wn1 = 15; %determined here from spectrum properties

%% START THE FUN!
tic
wn=zeros(nmodes,1); % wave number array

% compute random angles
psi  = 2*pi.*rand(nmodes,1);
fi   = 2*pi.*rand(nmodes,1);
alfa = 2*pi.*rand(nmodes,1);
teta = pi.*rand(nmodes,1);
%ang  = rand(nmodes,1);
%teta = acos(1 - ang./0.5);

% highest wave number
wnn = max(2*pi/dx, max(2*pi/dy, 2*pi/dz));

% wavenumber step
dk = (wnn-wn1)/nmodes;

% wavenumber at faces
wnf = wn1 + [0:nmodes].*dk;   

% wavenumber at cell centers
wn = wn1 + 0.5*dk + [0:nmodes-1].*dk;
wn = wn';
dkn = ones(nmodes,1).*dk;

%   wavenumber vector from random angles   
kx = sin(teta).*cos(fi).*wn;
ky = sin(teta).*sin(fi).*wn;
kz = cos(teta).*wn;

% sigma is the unit direction which gives the direction of the synthetic
% velocity field
sxm = cos(fi).*cos(teta).*cos(alfa) - sin(fi).*sin(alfa);
sym = sin(fi).*cos(teta).*cos(alfa) + cos(fi).*sin(alfa);
szm = -sin(teta).*cos(alfa);   

% another way of computing sigma. The previous method seems to give better
% results
% sxm = cos(fi).*cos(teta).*cos(alfa) - sin(fi).*sin(alfa).*cos(teta);
% sym = sin(fi).*cos(teta).*cos(alfa) + cos(fi).*sin(alfa).*cos(teta);
% szm = -sin(teta).*cos(alfa);   

% verify that the wave vector and sigma are perpendicular
kk = kx.*sxm + ky.*sym + kz.*szm;
disp('Orthogonality of k and sigma:');
disp(norm(kk));

% get the modes   
km = sqrt(kx.^2 + ky.^2 + kz.^2);

% now create an interpolant for the spectrum. this is needed for
% experimentally-specified spectra
espec = interp1(kcbc,ecbc,km,'cubic');
plot(kcbc,ecbc,'b')
hold on
plot(km,espec,'*')
   
um = 2.*sqrt(espec.*dkn);
u_ = zeros(nx,ny,nz);
v_ = zeros(nx,ny,nz);
w_ = zeros(nx,ny,nz);

% now loop through the grid
for k=1:nz
    for j=1:ny
        for i=1:nx
            % for every grid point (i,j,k) do the fourier summation 
            arg = kx.*xc(i) + ky.*yc(j) + kz.*zc(k) - psi;  
            bm = um.*cos(arg);              
            u_(i,j,k) = sum(bm.*sxm);  
            v_(i,j,k) = sum(bm.*sym);  
            w_(i,j,k) = sum(bm.*szm);
        end
    end
end
str=['Done generating fluctuations. It took me ' , num2str(toc), 's'];
display(str);

%% CALCULATE TURBULENT KE SPECTRUM TO MAKE SURE THINGS MAKE SENSE
[wn,vt]=tke_spectrum(u_,v_,w_,Lx, Ly, Lz); 
% plot the energy spectrum
figure(3)
n = round((nx*ny*nz)^(1/3));
loglog(wn(1:n-1),vt(1:n-1),'*-r');
hold on
loglog(kcbc,ecbc,'k');

%% CONVERT TO STAGGERED GRID - IF NECESSARY
u = zeros(nx+1,ny,nz);
v = zeros(nx,ny+1,nz);
w = zeros(nx,ny,nz+1);
for i=1:nx-1
    u(i+1,:,:) = 0.5*(u_(i+1,:,:) + u_(i,:,:));    
end
% set periodic condition
u(1,:,:) = 0.5*(u_(1,:,:) + u_(nx,:,:));
u(nx+1,:,:) = 0.5*(u_(1,:,:) + u_(nx,:,:));

for j=1:ny-1
    v(:,j+1,:) = 0.5*(v_(:,j+1,:) + v_(:,j,:));    
end
% set periodic condition
v(:,1,:) = 0.5*(v_(:,1,:) + v_(:,ny,:));
v(:,ny+1,:) = 0.5*(v_(:,1,:) + v_(:,ny,:));

for k=1:nz-1
    w(:,:,k+1) = 0.5*(w_(:,:,k+1) + w_(:,:,k));    
end
% set periodic condition
w(:,:,1) = 0.5*(w_(:,:,1) + w_(:,:,nz));
w(:,:,nz+1) = 0.5*(w_(:,:,1) + w_(:,:,nz));

% %set the additional point as periodic
% u(1:N,:,:) = u_;
% u(N+1,:,:) = u(1,:,:);
% 
% v(:,1:N,:) = v_;
% v(:,N+1,:) = v(:,1,:);
% 
% w(:,:,1:N) = w_;
% w(:,:,N+1) = w(:,:,1);

%% PLOT A FEW THINGS TO SATISFY THE EYES
figure(2)
contourf(u_(:,:,1),10);

%% COMPUTE DIVERGENCE
% Note: I don't know why I am getting non-zero divergence here although the
% orthogonality condition for divergence (k.sigma = 0) is satisfied.
div=zeros(nx,ny,nz);
for k=1:nz
    for j=1:ny
        for i=1:nx
            div(i,j,k) = div(i,j,k) + (u(i+1,j,k) - u(i,j,k))/dx + (v(i,j+1,k) - v(i,j,k))/dy + (w(i,j,k+1) - w(i,j,k))/dz;
        end
    end
end
divn = reshape(div,nx*ny*nz,1);
disp('Divergence:');
disp(norm(divn));

%% WRITE TO DISK
fName = 'cbc32_uvw';
uFileName = strcat(fName,'_wasatch_u');
uFileID = fopen(uFileName,'w');

for k=1:nz
    for j=1:ny
        for i=1:nx+1
            x = (i-1)*dx;
            y = (j-1)*dy + dy/2;
            z = (k-1)*dz + dz/2;
            fprintf( uFileID,'%.16f %.16f %.16f %.16f\n',x,y,z,u(i,j,k) );
        end
    end
end
fclose(uFileID);
gzip(uFileName);
delete(uFileName);

vFileName = strcat(fName,'_wasatch_v');
vFileID = fopen(vFileName,'w');
for k=1:nz
    for j=1:ny + 1
        for i=1:nx
            x = (i-1)*dx + dx/2;
            y = (j-1)*dy;
            z = (k-1)*dz + dz/2;
            fprintf( vFileID,'%.16f %.16f %.16f %.16f\n',x,y,z,v(i,j,k) );
        end
    end
end
fclose(vFileID);
gzip(vFileName);
delete(vFileName);


wFileName = strcat(fName,'_wasatch_w');
wFileID = fopen(wFileName,'w');
for k=1:nz + 1
    for j=1:ny
        for i=1:nx
            x = (i-1)*dx + dx/2;
            y = (j-1)*dy + dy/2;
            z = (k-1)*dz;
            fprintf( wFileID,'%.16f %.16f %.16f %.16f\n',x,y,z,w(i,j,k) );
        end
    end
end
fclose(wFileID);
gzip(wFileName);
delete(wFileName);