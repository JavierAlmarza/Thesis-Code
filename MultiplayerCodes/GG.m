function [G, Gy, Gz, Gyy, Gyz, Gzz] = GG(y,z,example,parameters)
% For problems where y minimizes F(y, z) and z minimizes G(z, y).

switch example
    case 1
        alpha=parameters{1}
        G = -((y-z)^2 - alpha*z^4);
        Gy = -2*(y-z);
        Gz = -2*(z-y) + 4*alpha*z^3;
        Gyy = -2;
        Gyz = 2;
        Gzz = -2 + 12*alpha*z^2;

    case 2
        G = -y^2;
        Gy=-2*y;
        Gz=0;
        Gyy = -2;
        Gyz=0;
        Gzz=0;

end
end