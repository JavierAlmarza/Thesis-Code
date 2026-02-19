function [Py,a,b,fg,f,K] = L_y(Qz,Qy,m,d,Gy,By,K_max,doPy)
% Objective function L (penalty component) and its gradient.

A=Qz'*Qy;
[a,s,b]=svds(A,K_max);
s=diag(s); K=length(s);

f = Qz*a;
g = Qy*b;

K=min(K, size(f,2));

fg = sum(f(:,1:K).*g(:,1:K),1);
msk=fg<0;

b(:,msk) = -b(:,msk);
g(:,msk) = -g(:,msk);
fg(msk)=-fg(msk);

Py=zeros(m,d,K);
bb=By*b(:,1:K);
if(doPy)
    for k=1:d
        Py(:,k,:)=f(:,1:K).*(Gy{k}*bb);
    end
else
    Py=[];
end

end