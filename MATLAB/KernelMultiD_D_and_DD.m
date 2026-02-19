function [Ky, KyD, KyDD] = KernelMultiD_D_and_DD(y,yc,S,needGy,needGyy)
% Writes down the Kernel matrix Ky and its first derivative, using centers yc and inverse covariance
% S

[n,d] = size(y);
[m,d] = size(yc);

Karg=zeros(n,m);
Kder=cell(d,1);
Kdder=cell(d,d);
KyD=Kder;
KyDD=Kdder;

if(needGy)
    for k=1:d
        Kder{k}=zeros(n,m);
    end
end
if(needGyy)
    for k=1:d
        for l=1:d
            Kdder{k,l}=zeros(n,m);
        end
    end
end

for j=1:m
    yd=y-yc(j,:);
    Karg(:,j)=sum((yd*S(:,:,j)).*yd,2);
    if(needGy)
        for k=1:d
            Kder{k}(:,j)=S(k,:,j)*yd';
        end
    end
    if(needGyy)
        for k=1:d
            for l=1:d
                Kdder{k,l}(:,j)=Kder{k}(:,j).*Kder{l}(:,j)-S(k,l,j);
            end
        end
    end
end

Ky=exp(-Karg/2);

if(needGy)
    for k=1:d
        KyD{k}=-Kder{k}.*Ky;
    end
end
if(needGyy)
    for k=1:d
        for l=1:d
            KyDD{k,l}=Kdder{k,l}.*Ky;
        end
    end
end
end
