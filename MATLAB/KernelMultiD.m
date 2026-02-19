function K = KernelMultiD(y,yc,S,oneS)
% Writes down the Kernel matrix K, using centers yc and inverse covariance
% S

[n,~] = size(y);
[m,~] = size(yc);

Karg=zeros(n,m);

if(oneS)
    for j=1:m
        yd=y-yc(j,:);
        Karg(:,j)=sum((yd*S).*yd,2);
    end
else
    for j=1:m
        yd=y-yc(j,:);
        Karg(:,j)=sum((yd*S(:,:,j)).*yd,2);
    end
end
K=exp(-Karg/2);  % A Gaussian kernel matrix.
end