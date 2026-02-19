function [G, Gy, Gyy, yc, Sc] = FigureOut_G_Gy_and_Gyy(nsy,Iy,type_g,ncy,y, things_known, yc, Sc, g_ext,needGy,needGyy)
% Determines the matrix F.

% nsy Number of subgroups of columns of G.
% For each subgoup k:
% Iy{k}: subset of the y this group depends on,
% type_g{k}: corresponding type of function,
% if type_f{k} = kernel, ncy{k}: number of centers
%                    and, for each element l in Iz{k},
%                        type_v{l}: the corresponding variable type.

% dy=length(y);
[m,dy]=size(y);
if(~things_known)
    yc=cell(nsy,1);
    Sc=cell(nsy,1);
end

% Number of colums of G.

ncol=0;

for k=1:nsy
    switch type_g{k}
        case 'kernel'
            ncol=ncol+ncy(k); % One column per center.
        case 'linear'
            Ny=length(Iy{k});
            ncol=ncol+Ny; % Ny linear terms.
        case 'linear and quadratic'
            Ny=length(Iy{k});
            ncol=ncol+Ny+(Ny*(Ny+1))/2; % Ny linear, (Ny*(Ny+1))/2 quadratic terms.
    end
end

% Columns of G and Gy{s}

G=zeros(m,ncol);
Gy=cell(dy,1);
Gyy=cell(dy,dy);
if(needGy)
    for k=1:dy
        Gy{k}=G;
    end
end
if(needGyy)
    for k=1:dy
        for l=1:dy
            Gyy{k,l}=G;
        end
    end
end

jcol=0;

for k=1:nsy
    Ny=length(Iy{k}); % Number of variables in the group
    
    switch type_g{k}
        case 'kernel'
            yy=[];
            for l=1:Ny
                jy=Iy{k}(l);
                yy=[yy y(:,jy)];
            end
            
            if(~things_known)
                [yc{k}, Sc{k}] = CentersAndCovariances(yy,ncy(k),g_ext(k));
            end
            [Ky, KyD, KyDD] = KernelMultiD_D_and_DD(yy,yc{k},Sc{k},needGy,needGyy);
            G(:,jcol+1:jcol+ncy(k)) = Ky;
            if(needGy)
                for l=1:Ny
                    Gy{Iy{k}(l)}(:,jcol+1:jcol+ncy(k))=KyD{l};
                end
            end
            if(needGyy)
                for l=1:Ny
                    for h=1:Ny
                        Gyy{Iy{k}(l),Iy{k}(h)}(:,jcol+1:jcol+ncy(k))=KyDD{l,h};
                    end
                end
            end
            jcol=jcol+ncy(k);
        case 'linear'
            yy=[];
            for l=1:Ny
                jy=Iy{k}(l);
                yy=[yy y(:,jy)];
            end
            G(:,jcol+1:jcol+Ny)=yy;
            
            if(needGy)
                for l=1:Ny
                    Gy{Iy{k}(l)}(:,jcol+l)=ones(m,1);
                end
            end
%             if(needGyy)
%                 for l=1:Ny
%                     for h=1:Ny
%                         Gyy{Iy{k}(l),Iy{k}(h)}(:,jcol+l)=zeros(m,1);
%                     end
%                 end
%             end
            jcol=jcol+Ny;
        case 'linear and quadratic'
            yy=[];
            for l=1:Ny
                jy=Iy{k}(l);
                yy=[yy y(:,jy)];
            end
            G(:,jcol+1:jcol+Ny)=yy;
            
            if(needGy)
                for l=1:Ny
                    Gy{Iy{k}(l)}(:,jcol+l)=ones(m,1);
                end
            end
            if(needGyy)
                for l=1:Ny
                    for h=1:Ny
                        Gyy{Iy{k}(l),Iy{k}(h)}(:,jcol+l)=zeros(m,1);
                    end
                end
            end
            
            jcol=jcol+Ny;
            for kk=1:Ny
                for l=kk:Ny
                    jcol=jcol+1;
                    G(:,jcol)=yy(:,kk).*yy(:,l);
                    if(needGy)
                        Gy{Iy{k}(kk)}(:,jcol)=Gy{Iy{k}(kk)}(:,jcol)+yy(:,l);
                        Gy{Iy{k}(l)}(:,jcol)=Gy{Iy{k}(l)}(:,jcol)+yy(:,kk);
                    end
                    if(needGyy)
                        Gyy{Iy{k}(kk),Iy{k}(l)}(:,jcol)=Gyy{Iy{k}(kk),Iy{k}(l)}(:,jcol)+1;
                        Gyy{Iy{k}(l),Iy{k}(kk)}(:,jcol)=Gyy{Iy{k}(l),Iy{k}(kk)}(:,jcol)+1;
                    end
                end
            end
    end
end
end