function [Q,B] = QB(K)
% Reduces a matrix K to an orthogonal one Q = KB
% with the same effective colum space.

[n,m]=size(K);

[U,S,V] = svd(K);
s=diag(S);

nK=sum(sum(K.^2)); % Norm of K.

s2sum=cumsum(s.^2);

s2thr=0.995*nK;

npc=find(s2sum>s2thr,1);

% 
% s2thr=nK/(m^2*n); % Threshold on accepted squared singular values.
% 
% npc=sum(s.^2>s2thr); % Number of principal components to use.

Q=U(:,1:npc); % Orthogonal matrix generating "all smooth zero-mean functions". 

% Matrix B to recover Q from K through Q = K*B. 

B=V(:,1:npc);
for j=1:npc
    B(:,j)=B(:,j)/s(j);
end

end