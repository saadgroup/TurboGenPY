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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ##     ##  ######  ######## ########     #### ##    ## ########  ##     ## ######## 
% ##     ## ##    ## ##       ##     ##     ##  ###   ## ##     ## ##     ##    ##    
% ##     ## ##       ##       ##     ##     ##  ####  ## ##     ## ##     ##    ##    
% ##     ##  ######  ######   ########      ##  ## ## ## ########  ##     ##    ##    
% ##     ##       ## ##       ##   ##       ##  ##  #### ##        ##     ##    ##    
% ##     ## ##    ## ##       ##    ##      ##  ##   ### ##        ##     ##    ##    
%  #######   ######  ######## ##     ##    #### ##    ## ##         #######     ##    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%set the number of modes.
nmodes =100; 

% file name base
fNameBase = 'cbc32_uvw';

% write to file
enableIO = false;

% compute the mean of the fluctuations
computeMean = true;

%Lx = 9*2*pi/100; % domain size in the x direction
%Ly = 9*2*pi/100; % domain size in the y direction
%Lz = 9*2*pi/100; % domain size in the z direction
% input domain size in the x, y, and z directions
Lx = 2*pi/15;
Ly = 2*pi/15;
Lz = 2*pi/15;

% input number of cells (cell centered control volumes)
nx = 32;         % number of cells in the x direction
ny = 32;         % number of cells in the y direction
nz = 32;         % number of cells in the z direction

% enter the smallest wavenumber represented by this spectrum
wn1 = 15; %determined here from cbc spectrum properties

%% grid generation
% generate cell centered x-grid
dx = Lx/nx;
xc = dx/2 + (0:nx-1)*dx;

% generate cell centered y-grid
dy = Ly/ny;
yc = dy/2 + (0:ny-1)*dy;

% generate cell centered z-grid
dz = Lz/nz;
zc = dz/2 + (0:nz-1)*dz; % cell centered coordinates

%% START THE FUN!
tic
wn=zeros(nmodes,1); % wave number array

% compute random angles
psi   = 2*pi.*rand(nmodes,1);
phi   = 2*pi.*rand(nmodes,1);
alfa  = 2*pi.*rand(nmodes,1);
theta = pi.*rand(nmodes,1);
%ang  = rand(nmodes,1);
%theta = acos(1 - ang./0.5);

% highest wave number that can be represented on this grid (nyquist limit)
wnn = max(pi/dx, max(pi/dy, pi/dz));
display(['I will generate data up to wave number: ', num2str(wnn)]);

% wavenumber step
dk = (wnn-wn1)/nmodes;

% wavenumber at faces
wnf = wn1 + [0:nmodes].*dk;   

% wavenumber at cell centers
wn = wn1 + 0.5*dk + [0:nmodes-1].*dk;
wn = wn';
dkn = ones(nmodes,1).*dk;

%   wavenumber vector from random angles   
kx = sin(theta).*cos(phi).*wn;
ky = sin(theta).*sin(phi).*wn;
kz = cos(theta).*wn;

% sigma is the unit direction which gives the direction of the synthetic
% velocity field
sxm = cos(phi).*cos(theta).*cos(alfa) - sin(phi).*sin(alfa);
sym = sin(phi).*cos(theta).*cos(alfa) + cos(phi).*sin(alfa);
szm = -sin(theta).*cos(alfa);   

% another way of computing sigma. The previous method seems to give better
% results
% sxm = cos(phi).*cos(theta).*cos(alfa) - sin(phi).*sin(alfa).*cos(theta);
% sym = sin(phi).*cos(theta).*cos(alfa) + cos(phi).*sin(alfa).*cos(theta);
% szm = -sin(theta).*cos(alfa);   

% verify that the wave vector and sigma are perpendicular
kk = kx.*sxm + ky.*sym + kz.*szm;
disp('Orthogonality of k and sigma (divergence in wave space):');
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
display(['Done generating fluctuations. It took me ' , num2str(toc), 's']);

%% CALCULATE TURBULENT KE SPECTRUM TO MAKE SURE THINGS MAKE SENSE
[wn,vt]=tke_spectrum(u_,v_,w_,Lx, Ly, Lz); 
vtsmooth = smooth(wn,vt,5,'moving');
vtsmooth(1:5) = vt(1:5);
% plot the energy spectrum
figure(3)
n = round((nx*ny*nz)^(1/3));
loglog(kcbc,ecbc,'-k');
hold on
%loglog(wn(1:n-1),vt(1:n-1),'.-r');
loglog(wn(1:n-1),vtsmooth(1:n-1),'*-r');

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

%% compute mean velocities
if (computeMean)
    umean = mean(mean(mean(u)));
    vmean = mean(mean(mean(v)));
    wmean = mean(mean(mean(w)));
    display(['Fluctuation Means: <u> = ' , num2str(umean), ', <v> = ', num2str(vmean), ', <w> = ', num2str(wmean)]);
end

%% PLOT A FEW THINGS TO SATISFY THE EYES
%figure(2)
%contourf(u_(:,:,1),10);

%% COMPUTE DIVERGENCE
% Note: I don't know why I am getting non-zero divergence here although the
% orthogonality condition for divergence (k.sigma = 0) is satisfied.
% div=zeros(nx,ny,nz);
% for k=1:nz
%     for j=1:ny
%         for i=1:nx
%             div(i,j,k) = div(i,j,k) + (u(i+1,j,k) - u(i,j,k))/dx + (v(i,j+1,k) - v(i,j,k))/dy + (w(i,j,k+1) - w(i,j,k))/dz;
%         end
%     end
% end
% divn = reshape(div,nx*ny*nz,1);
% disp('Divergence:');
% disp(norm(divn));

%% WRITE TO DISK
% u velocity

if(enableIO)

    uFileName = strcat(fNameBase,'_u');
    uFileID = fopen(uFileName,'w');

    nt = (nx+1)*ny*nz;
    xx = zeros(nx+1,ny,nz);
    yx = zeros(nx+1,ny,nz);
    zx = zeros(nx+1,ny,nz);
    for k=1:nz
        for j=1:ny
            for i=1:nx+1
                xx(i,j,k) = (i-1)*dx;
                yx(i,j,k) = (j-1)*dy + dy/2;
                zx(i,j,k) = (k-1)*dz + dz/2;
            end
        end
    end

    xx = reshape(xx,1,nt);
    yx = reshape(yx,1,nt);
    zx = reshape(zx,1,nt);
    u = reshape(u,1,nt);
    A=[xx;yx;zx;u];

    % tic
    % xx1=(0:nx)*dx;
    % xx1=repmat(xx1,1,ny*nz);
    % 
    % yx1 = dy/2 + (0:ny-1)*dy;
    % yx1 = yx1';
    % yx1 = repmat(yx1,1,nx+1);
    % yx1=yx1';
    % yx1 = repmat(yx1,[1,1,nz]);
    % yx1 = reshape(yx1,1,nt);
    % 
    % 
    % zx1 = ones(nx+1,ny).*dz/2;
    % 
    % zx1 = repmat(zx1,1,nx+1);
    % %zx1 = zx1';
    % zx1 = repmat(zx1,[1,1,nz]);
    % zx1 = reshape(zx1,1,nt);
    % A1=[xx1;yx1;zx1;u];
    % display(['Done generating secon xstaggered grid. It took me ' , num2str(toc), 's']);

    tic
    fprintf( uFileID,'%.16f %.16f %.16f %.16f\n',A );
    fclose(uFileID);
    clear xx yx zx
    gzip(uFileName);
    delete(uFileName);
    display(['Done writing data for u velocity. It took me ' , num2str(toc), 's']);

    %% WRITE TO DISK
    % v velocity
    vFileName = strcat(fNameBase,'_v');
    vFileID = fopen(vFileName,'w');

    nt = (ny+1)*nx*nz;
    xy = zeros(nx,ny+1,nz);
    yy = zeros(nx,ny+1,nz);
    zy = zeros(nx,ny+1,nz);

    for k=1:nz
        for j=1:ny + 1
            for i=1:nx
                xy(i,j,k) = (i-1)*dx + dx/2;
                yy(i,j,k) = (j-1)*dy;
                zy(i,j,k) = (k-1)*dz + dz/2;            
            end
        end
    end


    xy = reshape(xy,1,nt);
    yy = reshape(yy,1,nt);
    zy = reshape(zy,1,nt);
    v = reshape(v,1,nt);

    A=[xy;yy;zy;v];

    tic
    fprintf( vFileID,'%.16f %.16f %.16f %.16f\n',A );
    fclose(vFileID);
    clear xy yy zy
    gzip(vFileName);
    delete(vFileName);
    display(['Done writing data for v velocity. It took me ' , num2str(toc), 's']);

    %% WRITE TO DISK
    % w velocity
    wFileName = strcat(fNameBase,'_w');
    wFileID = fopen(wFileName,'w');

    nt = nx*ny*(nz+1);
    xz = zeros(nx,ny,nz+1);
    yz = zeros(nx,ny,nz+1);
    zz = zeros(nx,ny,nz+1);

    for k=1:nz + 1
        for j=1:ny
            for i=1:nx
                xz(i,j,k) = (i-1)*dx + dx/2;
                yz(i,j,k) = (j-1)*dy + dy/2;
                zz(i,j,k) = (k-1)*dz;
            end
        end
    end

    xz = reshape(xz,1,nt);
    yz = reshape(yz,1,nt);
    zz = reshape(zz,1,nt);
    w = reshape(w,1,nt);

    A=[xz;yz;zz;w];
    
    tic
    fprintf( wFileID,'%.16f %.16f %.16f %.16f\n',A );
    fclose(wFileID);
    clear xz yz zz A
    gzip(wFileName);
    delete(wFileName);
    display(['Done writing data for w velocity. It took me ' , num2str(toc), 's']);

end
