% Straight, independent gradient descent for the problem
% y = argmin F(y, z)
% z = argmin G(z, y)

lambda=1; alpha=1; % Problem's parameters.
% For these values of the paremeters, the exact solution has z=1/2, y=1/4
eta_min=0.001; grad_min=0.0001; ns=30; eps=0.0000000001;

F_pr=zeros(ns,1); G_pr=F_pr; y_pr=F_pr; z_pr=F_pr;

y0=0.3; z0=0.6; % Initial guess

eta_y=0.1; eta_z=0.1;

not_done=true;

y=y0; z = z0;

s=0;

while not_done

    s=s+1;

    y_pr(s) = y; z_pr(s) = z;

    [F, Fy, Fz, Fyy, Fyz, Fzz] = FF(y,z,lambda,alpha);
    F0=F;
    F_pr(s) = F;

    accept_1=false;
    while (~accept_1 & eta_y > eta_min)
        ys=y - eta_y*Fy;
        [F1, ~, ~, ~, ~, ~] = FF(ys,z,lambda,alpha);
        if(F1 <= F0+eps-eta_y*Fy^2/2)
            accept_1=true;
        else
            eta_y = 0.5*eta_y;
        end
    end
    if(accept_1)
        F0=F1;
        y=ys;
        eta_y=1.1*eta_y;
    else
        eta_y=2.1*eta_min;
    end

    [G, Gy, Gz, Gyy, Gyz, Gzz] = GG(y,z,alpha);
    G0=G;
    G_pr(s) = G;

    accept_2=false;
    while (~accept_2 & eta_z > eta_min)
        zs=z - eta_z*Gz;
        [G1, ~, ~, ~, ~, ~] = GG(y,zs,alpha);
        if(G1 <= G0+eps-eta_z*Gz^2/2)
            accept_2=true;
        else
            eta_z = 0.5*eta_z;
        end
    end
    if(accept_2)
        G0=G1;
        z=zs;
        eta_z=1.1*eta_z;
    else
        eta_z=2.1*eta_min;
    end

    not_done=(accept_1 | accept_2) & (abs(Fy)+abs(Gz) > grad_min) & s < ns; 

end

F_pr=F_pr(1:s); G_pr=G_pr(1:s); y_pr=y_pr(1:s); z_pr=z_pr(1:s); S=1:s;

figure(1)
plot(S, y_pr)
xlabel('step');
ylabel('y')

figure(2)
plot(S, z_pr)
xlabel('step');
ylabel('z')

figure(3)
plot(S, F_pr)
xlabel('step');
ylabel('F')

figure(4)
plot(S, G_pr)
xlabel('step');
ylabel('G')


