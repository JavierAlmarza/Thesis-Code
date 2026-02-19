function [yc, Sv2] = CentersAndCovariances(y,m,g_ext)
%
% This function returns a set of m centers yc and corresponding
% local covariance matrix inverses S to adequately represent smooth functions on the data
% points y.

[n,d] = size(y);
if(n < 1000)   % If there are too many points, we do k-means of a smaller subsample.
    yy=y;
else
    yy=y(randi(n,1000,1),:);
end

[I,yc]=kmeans(yy,m);

% ymin=min(y); ymax=max(y); Dy=(ymax-ymin)/(3*m);
% [I,yc]=kmeans(yy,m-2);
% yc = [ymin-Dy; yc; ymax+Dy]; % k-means enriched by 

ym=mean(yc);

Sigma=zeros(d,d);
for k=1:d
    for l=1:d
        Sigma(k,l)=sum((yc(:,k)-ym(k)).*(yc(:,l)-ym(l)))/m;
    end
end

S=(4/d+2)^(-2/(d+4))*inv(Sigma)*m^(2/(d+4)); % Rule of thumb.

% Alphas to try (we can replace this by gradient descent,
% alpha is a global stretching factor for the covariance matrix.) 

alphas=0.1:0.1:2;

LLmax=-1000000000000;

Sv=zeros(d,d,m);
for alpha=alphas
    Sa=S/alpha^2;
    Kz = KernelMultiD(yc,yc,Sa,true);
    Kzt=Kz-diag(diag(Kz));
    LL=sum(log(sum(Kzt,1)),2)-m*log(alpha^d);
    if(LL > LLmax)
        LLmax=LL;
        alpha_max=alpha;
    end
end

Sa1=S/(alpha_max)^2;
Kz = KernelMultiD(yc,yc,Sa1,true);

rho=sum(Kz,1);

rd=rho.^(2/d); % j-dependent factor 
                  % so that the support of every kernel 
                  % has roughly the same nukmber of points. 
rd=m*rd/sum(rd);  % Rougly normalized version.

Sva=Sv;
for j=1:m
    Sva(:,:,j)=Sa1*rd(j); % Locally adjusted inverse covariance matrix.
end

% Gammas to try (we can replace this by gradient descent.

gammas=[0.2:0.1:5];

LLmax=-1000000;

J=1./(ones(m,1)*rho.^(1/2)); % Normalizing factor for the kernels.

for gamma=gammas
    Sv=Sva/gamma^2;
    Kz = J.*KernelMultiD(yc,yc,Sv,false);
    Kzt=Kz-diag(diag(Kz));
    LL=sum(log(sum(Kzt,1)),2)-m*log(gamma^d);
    if(LL > LLmax)
        LLmax=LL;
        gamma_max=gamma;
    end
end

Sv2=Sva/(gamma_max)^2;
Sv2=Sv2/(g_ext)^2; % An extrenally provided correction, say from cross-validation.

end

