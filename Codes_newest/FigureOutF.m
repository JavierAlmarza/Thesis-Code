function [F, zc, Sc, zzz] = FigureOutF(nsz,Iz,type_f,ncz,z, type_v, ndv, things_known, zc, Sc, g_ext, m_h)
%
% Determines the matrix F.
% (Under the assumption that, if there is a kernel, ncz=1.)

% nsz Number of subgroups of columns of F.
% For each subgoup k:
% Iz{k}: subset of the z this group depends on,
% type_f{k}: corresponding type of function,
% if type_f{k} = kernel, ncz{k}: number of centers
%                    and, for each element l in Iz{k},
%                        type_v{l}: the corresponding variable type,
%                             and, if categorical,
%                                   ndv(l): number of distinct discrete
%                                           values.

NZ=length(z); 
m=size(z{1},1);

% Expand discrete and periodic variables into R^d and R^2 

for l=1:NZ
    switch type_v{l}
        case 'Real'
            zz{l}=z{l};
        case 'Categorical'
            zz{l} = project_to_equidistance(z{l},ndv(l));
        case 'Periodic'
            zz{l}=[cos(z{l}) sin(z{l})];
    end
end

if(~things_known)
    zc=cell(nsz,1);
    Sc=cell(nsz,1);
end


% Number of colums of F.

ncol=0;

for k=1:nsz
    switch type_f{k}
        case 'kernel'
            ncol=ncol+ncz(k); % One column per center.
        case 'linear and quadratic'
            Nz=0;
            for l=Iz{k}
                Nz=Nz+size(zz{l},2);
            end
            ncol=ncol+Nz+(Nz*(Nz+1))/2; %Nz linear, (Nz*(Nz+1))/2 quadratic terms.
        case 'RKHS'
            ncol=ncol+m_h; % Each sample point is a center.
    end
end

% Columns of F.

F=zeros(m,ncol);

jcol=0;

for k=1:nsz
    Nz=length(Iz{k}); % Number of variables in the group
    zzz=[];
    for l=1:Nz
        zzz=[zzz zz{Iz{k}(l)}];
    end
    Nz=size(zzz,2);
    
    switch type_f{k}
        case 'kernel'
            if(~things_known)
                [zc{k}, Sc{k}] = CentersAndCovariances(zzz, ncz(k), g_ext(k));
            % else
            %     if(update)
            %         for j=1:3     
            %             [Sc{k},f_max] = Update_S(Sc{k},zzz,zc{k},Qy,nk, vK);
            %         end
            %     end
            end
            Kz = KernelMultiD(zzz,zc{k},Sc{k},false);
            F(:,jcol+1:jcol+ncz(k)) = Kz;
            jcol=jcol+ncz(k);
        case 'linear and quadratic'
            % zz=[];
            % for j=1:Nz
            %     zz=[zz z{Iz{k}(j)}];
            % end
            F(:,jcol+1:jcol+Nz)=zzz;
            jcol=jcol+Nz;
            for kk=1:Nz
                for l=kk:Nz
                    jcol=jcol+1;
                    F(:,jcol)=zz(:,kk).*zzz(:,l);
                end
            end
        case 'RKHS'
            if(~things_known)
                Sc{k} = Covariance(zzz, g_ext(k));
                zc{k} = zzz;
            end
            Kz = KernelMultiD(zzz,zc{k},Sc{k},true); %
            F(:,jcol+1:jcol+m_h) = Kz;
            jcol=jcol+m_h;
    end
end

