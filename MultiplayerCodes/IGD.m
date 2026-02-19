% Implicit, simultaneous gradient descent for the problem
% y = argmin F(y, z)
% z = argmin G(z, y)

lambda=1; alpha=1; % Problem's parameters.
% For these values of the paremeters, the exact solution has z=1/2, y=1/4
eta_min=0.001; grad_min=0.0001; ns=20; eps=0.0000000001;

F_pr=zeros(ns,1); G_pr=F_pr; y_pr=F_pr; z_pr=F_pr;

y0=0.3; z0=0.6; % Initial guess

eta_y=0.5; eta_z=0.5;

not_done=true;

y=y0; z = z0;

s=0;

while not_done

    s=s+1;

    y_pr(s) = y; z_pr(s) = z;

    [F, Fy, Fz, Fyy, Fyz, Fzz] = FF(y,z,lambda,alpha);
    [G, Gy, Gz, Gyy, Gyz, Gzz] = GG(y,z,alpha);

    F_pr(s) = F;
    G_pr(s) = G;

    accept=false;

    x=[z; y];

    while(~accept && (eta_y + eta_z) > 2*eta_min)
        eta_z=max(eta_z,eta_min);
        eta_y=max(eta_y,eta_min);

        D=[eta_z*Gz; eta_y*Fy];
        H=eye(2) + eta_z*[Gzz Gyz; 0 0] + eta_y*[0 0; Fyz Fyy];

        xs = x - H \ D;
        zs=xs(1); ys=xs(2);


        [F0, ~, ~, ~, ~, ~] = FF(y,zs,lambda,alpha);
        [F1, ~, ~, ~, ~, ~] = FF(ys,zs,lambda,alpha);

        accept_y = (F1 <= F0);

        [G0, ~, ~, ~, ~, ~] = GG(ys,z,alpha);
        [G1, ~, ~, ~, ~, ~] = GG(ys,zs,alpha);

        accept_z = (G1 <= G0);

        accept=accept_z & accept_y;

%        [F0 F1 G0 G1]

        if(~accept)
            if(~accept_z)
                eta_z = 0.5*eta_z;
            else
                eta_z=0.9*eta_z;
            end
            if(~accept_y)
                eta_y=0.5*eta_y;
            else
                eta_y=0.9*eta_y;
            end
        end
 %       [accept eta_z eta_y]
    end
    if(accept)
        eta_z = 1.1*eta_z;
        eta_y = 1.1*eta_y;
        z=zs; y=ys;
    end
    not_done= accept & (abs(Fy)+abs(Gz) > grad_min) & s < ns; 
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


