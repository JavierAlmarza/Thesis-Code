function xs = Invert(y, Qz, a, b, By, Bz, K, zc, Scz, yc, Scy, sz, ffgg, ...
    nF, sy, nG, lambda, z_star,nsz,Iz,type_f,ncz,type_v,ndv,gamma_z, nsy,Iy,type_g,ncy,gamma_y, nk)
% Simulates rho(x|z) for z = z_tar.
% z_star=cell(dz,1);
% for j=1:dz
%     z_star{j}=rand;
% end
% % z_star{1}=0.8;
% z_star{1}=0.35;
% z_star{2}=0.37;
% % z_star{3}=0.2;
% % z_star{4}=0.4;
% % z_star{5}=-0.1;
% % z_star{6}=0.4;
% % z_star{1}=-0.2;
% % z_star{2}=0.35;


[m, dy]=size(y);
dz = length(z_star);

[F, ~, ~, ~] = FigureOutF(nsz,Iz,type_f,ncz,z_star, type_v, ndv, ...
    true, zc, Scz,gamma_z, m);
nj=size(F,2);
F=F-sz;
for j=1:nj
    F(:,j)=F(:,j)/nF(j);
end
QQz=F*Bz;

f_star=QQz*a;

Py=zeros(m,dy);
%
% [G, Gy, ~, ~] = ...
%     FigureOut_G_and_Gy(nsy,Iy,type_g,ncy,y, true, yc, Scy,gamma_y,true);

 [G, Gy, ~, ~, ~] = ...
        FigureOut_G_Gy_and_Gyy(nsy,Iy,type_g,ncy,y, true, yc, Scy, gamma_y,true,false);

% G=G-sy;
for j=1:size(G,2)
%     G(:,j)=G(:,j)/nG(j);
    for k=1:length(Gy)
        Gy{k}(:,j)=Gy{k}(:,j)/nG(j);
    end
end
% Qy=G*By;
%
% ff = Qz*a;
% gg = Qy*b;
%
% [~,bb,ff,gg,K] = abfg_K(Qz,Qy,b,nk,K,false);
% ffgg = sum(ff(:,1:K).*gg(:,1:K),1);
% bb=b;

bbb=By*b(:,1:K);
for k=1:dy
    Py(:,k)=Py(:,k)+sum(ffgg.*(f_star(:,1:K).*(Gy{k}*bbb)),2);
end

xs=y+2*lambda*Py;

end
