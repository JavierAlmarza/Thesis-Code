function [F, Fy, Fz, Fyy, Fyz, Fzz] = FF(y,z,example,parameters)
% For problems where y minimizes F(y, z) and z minimizes G(z, y).
switch example
    case 1
        alpha=parameters{1};
        lambda=parameters{2};
        F = y^2 + lambda*((y-z)^2 - alpha*z^4);
        Fy = 2*(y + lambda*(y-z));
        Fz = 2*lambda*((z-y) - 2*alpha*z^3);
        Fyy = 2*(1+lambda);
        Fyz = -2*lambda;
        Fzz = 2*lambda*(1-6*alpha*z^2);

    case 2
        lambda=parameters{1};
        F = y^2 + lambda*(y - sin(z)^2)^2;
        Fy = 2*y + 2*lambda*(y - sin(z)^2);
        Fz = -2*lambda*(y - sin(z)^2)*sin(2*z);
        Fyy = 2*(1+lambda);
        Fyz = -2*lambda*sin(2*z);
        Fzz = 2*lambda*(sin(2*z)^2-2*(y - sin(z)^2)*cos(2*z));
end
end