% Implicit, simultaneous gradient descent for the problem
% y = argmin F(y, z)
% z = argmin G(z, y)
% with external player.

example=2;

switch example
    case 1

        lambda=1; alpha=1; % Problem's parameters.
        parameters=cell(2,1);
        parameters{1}=alpha;
        parameters{2}=lambda;
        % For these values of the paremeters, the exact solution has z=1/2, y=1/4
        % for the non-anticipatory game and when y is the external player. When the
        % external player is z, the solution has z=1/2^(3/2), y = 1/2^(5/2).
        y0=0.23; z0=0.6; % Initial guess

    case 2
        lambda=1;
        parameters=cell(1,1);
        parameters{1}=lambda;
        % For these values of the paremeters, the exact solution has
        % z=(n+1)pi/2, y = 1/2.
        y0=0.23; z0=1.; % Initial guess

end

z_ext=true;
y_ext=false;
    
eta_min=0.001; grad_min=0.0001; ns=40; eps=0.0000000001;

F_pr=zeros(ns,1); G_pr=F_pr; y_pr=F_pr; z_pr=F_pr;



eta_y=0.1; eta_z=0.1;

not_done=true;

y=y0; z = z0;

s=0;

while not_done

    s=s+1;

    y_pr(s) = y; z_pr(s) = z;

    [F, Fy, Fz, Fyy, Fyz, Fzz] = FF(y,z,example,parameters);
    [G, Gy, Gz, Gyy, Gyz, Gzz] = GG(y,z,example,parameters);

    F_pr(s) = F;
    G_pr(s) = G;

    accept=false;

    x=[z; y];

    while(~accept && (eta_y + eta_z) > 2*eta_min)
        eta_z=max(eta_z,eta_min);
        eta_y=max(eta_y,eta_min);

        D=[eta_z*Gz; eta_y*Fy];
        H=eye(2) + eta_z*[Gzz Gyz; 0 0] + eta_y*[0 0; Fyz Fyy];
        if(z_ext)
            D(1)=D(1)-eta_z*(Fyz/Fyy)*Gy;
            H(1,:) = H(1,:) -eta_z*(Fyz/Fyy)*[Gyz Gyy];
        end
        if(y_ext)
            D(2)=D(2)-eta_y*(Gyz/Gzz)*Fz;
            H(2,:) = H(2,:) -eta_y*(Gyz/Gzz)*[Fzz Fyz];
        end
        xs = x - H \ D;
        zs=xs(1); ys=xs(2);


        [F0, ~, ~, ~, ~, ~] = FF(y,zs,example,parameters);
        [F1, ~, ~, ~, ~, ~] = FF(ys,zs,example,parameters);
        [G0, ~, ~, ~, ~, ~] = GG(ys,z,example,parameters);
        [G1, ~, ~, ~, ~, ~] = GG(ys,zs,example,parameters);

        if(y_ext)
            accept_y = (F1 <= F) | abs(Gz) > grad_min;
        else
            accept_y = (F1 <= F0) ;
        end

        if(z_ext)
            accept_z = (G1 <= G) | abs(Fy) > grad_min;
        else
            accept_z = (G1 <= G0);
        end

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
    not_done=accept & s < ns;
    if(z_ext)
        not_done=not_done & ((abs(Fy) > grad_min) | (abs(Gz - (Fyz/Fyy)*Gy) > grad_min));
    else
        not_done=not_done & abs(Gz) > grad_min;
    end
    if(y_ext)
        not_done=not_done & ((abs(Gz) > grad_min) | (abs(Fy - (Gyz/Gzz)*Fz) > grad_min));
    else
        not_done=not_done & abs(Fy) > grad_min;
    end
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


