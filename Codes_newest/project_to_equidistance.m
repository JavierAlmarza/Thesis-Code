function [z_new] = project_to_equidistance(z,k)
%[z_new] = project_to_equidistance(z)
%   turns categorical variable into continous variable by imbedding in
%   higher dimensional space

n = size(z,1);
groups = 1:k;
% k = length(groups);
z_new = zeros(n,k-1);
for i = 1:k-1
    z_new(z==groups(i),i)=1;
end
z_new(z==groups(k),:) = (1+sqrt(k))/(k-1);

end