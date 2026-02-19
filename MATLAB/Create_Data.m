% This routine provides the data, can be replaced by simply reading the
% data from some file. The output is:

% m: number of samples
% dx, dz: number of variables x and factors z.
% type_v{1:dz}: type of variable (for the factor only, x is always real).
% z{1:dz}: Values of the dz cofactors, m for each.
% x(1:m,1:dx): Values of the dx outcome variables.

% The routine may also plot the data when feasible.

% m=1590; % Number of samples
%
% data_case='first';
% data_case='second';
% data_case='third';
% data_case='fourth';
% data_case='fifth';
% data_case='sixth';
% data_case='seventh';
% data_case='eighth';
% data_case='ninth';
% data_case='tenth';
% data_case='eleventh';
% data_case='twelfth';
% data_case='thirteenth';
% data_case='fourteenth';

switch data_case
    case 'first'

        % One dimensional z and x.

        dx=1; dz=1;
        z=cell(dz,1); type_v=z; ndv=zeros(dz,1);
        type_v{1}='Real'; ndv(1)=0;

        % z{1}=rand(m,1)-1/2; % a uniform z.
        z{1}=randn(m,1)/4;
        x = cos(2*pi*z{1}) + 0.05*(1./(sin(0.1*(z{1}+0.2).^2)+0.25)).*randn(m,1); % z-dependent mean and variance.

        figure(1)
        plot(z{1},x,'b*')
        xlabel('z')
        ylabel('x')
        title('Data')

        exact_available=true;

    case 'second'

        % Two dimensional z, one dimensional x.

        dx=1; dz=2;
        z=cell(dz,1); type_v=z; ndv=zeros(dz,1);
        type_v{1}='Real'; ndv(1)=0;
        type_v{2}='Real'; ndv(1)=0;
        z=cell(dz,1);

        z{1} = rand(m,1) - (1/2)*ones(m,1);
        z{2} = rand(m,1) - (1/2)*ones(m,1);
        x = cos(2*pi*z{1}) + sin(pi*z{2}) + 0.2*(1-2*z{1}).^(1/2).*(1-2*z{2}).^(1/2).*randn(m,1);
        figure(1)
        subplot(211)
        plot3(z{1},z{2},x,'*b')
        xlabel('z1')
        ylabel('z2')
        zlabel('x')
        title('Data')

        min1=min(z{1}); max1=max(z{1}); d1=(max1-min1)/20; z11=min1:d1:max1;
        min2=min(z{2}); max2=max(z{2}); d2=(max2-min2)/20; z22=min2:d2:max2;
        [z1,z2]=meshgrid(z11,z22);
        xg=griddata(z{1},z{2},x,z1,z2);

        subplot(212)
        contour(z1,z2,xg)
        xlabel('z_1')
        ylabel('z_2')
        title('x(z)')
        colorbar

        exact_available=true;

    case 'third'

        % One dimensional z, two dimensional x.

        dx=2; dz=1;
        z=cell(dz,1); type_v=z; ndv=zeros(dz,1);
        type_v{1}='Real'; ndv(1)=0;
        z=cell(dz,1);

        x=zeros(m,2);

        z{1} = rand(m,1)-1/2; % a uniform z.
        x(:,1) = cos(2*pi*z{1}) + 0.02*(1./(z{1}.^2+0.05)).*randn(m,1); % z-dependent mean and variance.
        %         x(:,2) = cos(2*pi*z) + 0.02*(1./(z.^2+0.05)).*randn(m,1); % z-dependent mean and variance.
        x(:,2) = sin(2*pi*z{1}) + 0.02*(1./(z{1}.^2+0.05)).*randn(m,1); % z-dependent mean and variance.

        figure(1)
        subplot(311)
        plot(z{1},x(:,1),'b*')
        xlabel('z')
        ylabel('x_1')
        title('Data')
        subplot(312)
        plot(z{1},x(:,2),'b*')
        xlabel('z')
        ylabel('x_2')
        subplot(313)
        plot(x(:,1),x(:,2),'b*')
        xlabel('x_1')
        ylabel('x_2')

        exact_available=true;

    case 'fourth'

        dx=2; dz=1;
        z=cell(dz,1); type_v=z; ndv=zeros(dz,1);
        type_v{1}='Real'; ndv(1)=0;
        z=cell(dz,1);

        nc_gmm=3; % Number of components of the Gaussian mixture model.
        x=zeros(m,dx);
        Si=zeros(dx,dx,nc_gmm);
        mui=zeros(nc_gmm,dx);
        pii=zeros(nc_gmm);
        S_gmm=cell(nc_gmm,2);
        mu_gmm=cell(nc_gmm,2);
        pi_gmm=zeros(nc_gmm,2);
        %         z = 2*pi*(rand(m,1)-1/2); % a uniform z.
        rt=(rand(m,1)>0.5);
        z{1}=0.2*randn(m,1)+2*rt;

        for k=1:nc_gmm
            for l=1:2
                [S_gmm{k,l}, mu_gmm{k,l}]=GMM(dx);
                pi_gmm(k,l)=rand;
            end
        end

        for i=1:m
            for k=1:nc_gmm
                Si(:,:,k)=S_gmm{k,1}+S_gmm{k,2}*z{1}(i)^2;
                mui(k,:)=mu_gmm{k,1}+mu_gmm{k,2}*z{1}(i)^2;
                pii(k)=pi_gmm(k,2)+pi_gmm(k,2)*z{1}(i)^2;
            end
            pii=pii/sum(pii);
            GM=gmdistribution(mui,Si,pii);
            x(i,:)=random(GM);
        end

        figure(1)
        scatter(x(:,1),x(:,2),20,z{1},'filled')

        exact_available=true;

    case 'fifth'

        % One dimensional x and binary z for OT.

        dx=1; %dz=1;
        n0=450; n1=m-n0;
        x0=randn(n0,1);%source
        x1=4+randn(n1,1);%target
        x=[x0;x1];%stack data

        figure(1)
        histogram(x(1:n0)); hold on;
        histogram(x(n0+1:end));
        xlabel('x')
        legend('source 0','target 1');
        title('Data')

        exact_available=true;

    case 'sixth'

        % Binary z, two dimensional x.

        dx=2;
        %         dz=1;

        n0=450;n1=m-n0;

        % source 0
        x0=mvnrnd([-2;-2],1.*eye(2),n0);
        % target 1
        x1=mvnrnd([3;2],2*eye(2),n1);
        x=[x0; x1]; % stack data

        figure(1)
        subplot(311)
        plot(x(1:n0,1),'b*'); hold on;
        plot(x(n0+1:end,1),'g*')
        xlabel('z')
        ylabel('x_1')
        title('Data')
        subplot(312)
        plot(x(1:n0,2),'b*'); hold on;
        plot(x(n0+1:end,2),'g*')
        xlabel('z')
        ylabel('x_2')
        subplot(313)
        plot(x(1:n0,1),x(1:n0,2),'b*'); hold on;
        plot(x(n0+1:end,1),x(n0+1:end,2),'g*');
        xlabel('x_1')
        ylabel('x_2')

        exact_available=true;

    case 'seventh'

        % Two dimensional z, one dimensional x.

        dx=1; dz=2;
        z=cell(dz,1); type_v=z; ndv=zeros(dz,1);
        type_v{1}='Categorical'; ndv(1)=2;
        type_v{2}='Periodic'; ndv(2)=0;
        z=cell(dz,1);

        z{1} = randi(2,m,1);
        z{2} = 2*pi*rand(m,1);
        x = (z{1}-1).*cos(pi*cos(z{2}+pi/6).^2) + (2-z{1}).*cos(z{2}).^2 + 0.2*randn(m,1);

        figure(1)
        plot3(z{1},z{2},x,'*b')
        xlabel('z1')
        ylabel('z2')
        zlabel('x')
        title('Data')

        exact_available=true;

    case 'eighth'

        % one dimensional z, one dimensional x.

        dx=1; dz=1;
        z=cell(dz,1); type_v=z; ndv=zeros(dz,1);
        type_v{1}='Periodic'; ndv(1)=0;
        z=cell(dz,1);

        z{1} = 2*pi*rand(m,1);
        x = cos(pi*cos(z{1}+pi/6).^2) + 0.2*randn(m,1);

        figure(1)
        plot(z{1},x,'*b')
        xlabel('z')
        zlabel('x')
        title('Data')

        exact_available=true;

    case 'ninth'

        % dz-dimensional z, one dimensional x.

        dx=1; dz=6;
        z=cell(dz,1); type_v=z; ndv=zeros(dz,1);
        for h=1:dz
            type_v{h}='Real'; ndv(h)=0;
            z{h} = rand(m,1) - (1/2)*ones(m,1);
        end
        %         x = cos(2*pi*z{1}) + (0.2+z{2}.^2).*randn(m,1);

        x = cos(pi*(z{1}-sin(2*abs(z{2}).*z{2}))) + 0.2*randn(m,1);

        figure(1)
        subplot(211)
        plot3(z{1},z{2},x,'*b')
        xlabel('z1')
        ylabel('z2')
        zlabel('x')
        title('Data')

        min1=min(z{1}); max1=max(z{1}); d1=(max1-min1)/20; z11=min1:d1:max1;
        min2=min(z{2}); max2=max(z{2}); d2=(max2-min2)/20; z22=min2:d2:max2;
        [z1,z2]=meshgrid(z11,z22);
        xg=griddata(z{1},z{2},x,z1,z2);

        subplot(212)
        contour(z1,z2,xg)
        xlabel('z_1')
        ylabel('z_2')
        title('x(z)')
        colorbar

        exact_available=true;

    case 'tenth'

        % dz-dimensional z, dy-dimensional x.

        dx=3; dz=3;
        z=cell(dz,1); type_v=z; ndv=zeros(dz,1);
        for h=1:dz
            type_v{h}='Real'; ndv(h)=0;
            z{h} = rand(m,1) - (1/2)*ones(m,1);
        end
        %         x = cos(2*pi*z{1}) + (0.2+z{2}.^2).*randn(m,1);

        z_signal = cos(pi*(z{1}-sin(2*abs(z{2}).*z{2})));

        x=0.3*randn(m,dx);

        x(:,1)=x(:,1)+sin(pi*z_signal/2);
        x(:,2)=x(:,2)+sin(pi*z_signal/2).^3;

        min1=min(z{1}); max1=max(z{1}); d1=(max1-min1)/20; z11=min1:d1:max1;
        min2=min(z{2}); max2=max(z{2}); d2=(max2-min2)/20; z22=min2:d2:max2;
        [z1,z2]=meshgrid(z11,z22);
        xg=griddata(z{1},z{2},x(:,1),z1,z2);

        figure(1)

        subplot(311)
        contour(z1,z2,xg)
        xlabel('z_1')
        ylabel('z_2')
        title('x_1(z)')
        colorbar

        xg=griddata(z{1},z{2},x(:,2),z1,z2);
        subplot(312)
        contour(z1,z2,xg)
        xlabel('z_1')
        ylabel('z_2')
        title('x_2(z)')
        colorbar

        subplot(313)
        plot(x(:,1),x(:,2),'*')
        xlabel('x1')
        ylabel('x2')

    case 'eleventh'

        % dz-dimensional z, 1-dimensional x.

        dx=1; dz=4;
        z=cell(dz,1); type_v=z; ndv=zeros(dz,1);
        for h=1:dz
            type_v{h}='Real'; ndv(h)=0;
            z{h} = rand(m,1) - (1/2)*ones(m,1);
        end
        %         x = cos(2*pi*z{1}) + (0.2+z{2}.^2).*randn(m,1);

        z_signal = cos(pi*(z{1}-sin(2*abs(z{2}).*z{2})));

        ampl_signal=3.5;

        x=ampl_signal*z_signal+0.2*randn(m,1);

        min1=min(z{1}); max1=max(z{1}); d1=(max1-min1)/20; z11=min1:d1:max1;
        min2=min(z{2}); max2=max(z{2}); d2=(max2-min2)/20; z22=min2:d2:max2;
        [z1,z2]=meshgrid(z11,z22);
        xg=griddata(z{1},z{2},x,z1,z2);

        figure(1)

        contour(z1,z2,xg)
        xlabel('z_1')
        ylabel('z_2')
        title('x_1(z)')
        colorbar

        exact_available=true;

    case 'twelfth'
        % A Gaussian mixture, dx = dz = 1.

        dx=1; dz=1;

        z=cell(dz,1); type_v=z; ndv=zeros(dz,1);
        for h=1:dz
            type_v{h}='Real'; ndv(h)=0;
        end
        x=zeros(m,dx);

        nc_gmm=2; % Number of components of the Gaussian mixture model.
        Si=zeros(dx,dx,nc_gmm); % Individual covariance matrices.
        mui=zeros(nc_gmm,dx); % Individual means.
        pii=zeros(nc_gmm); % Mass of each component.

%         rt=(rand(m,1)>0.5)-0.5;
        % z{1}=0.2*randn(m,1)+ rt; % A close to discrete z.
%         z{1}=2*(rand(m,1)-1/2);

%         z{1}=max(min(randn(m,1),2.5),-2.5);
%         z{1}=randn(m,1);
%         z{1}=min(z{1},(7*z{1}+2.5)/8);
%         z{1}=max(z{1},(7*z{1}-2.5)/8);
        z{1}=5*(rand(m,1)-0.5);

        for i=1:m
            mui(1,1) = 3 + 2.*z{1}(i);
            mui(2,1) = -z{1}(i).^2+z{1}(i)/2;
            Si(1,1,1) = 0.5*exp(z{1}(i));
            Si(1,1,2) = 0.25 - 0.1*z{1}(i);
%             Si(1,1,2) = max(0.2 - 0.1*z{1}(i),0.01);
            pii(1) = 1; % (1.5+z{1}(i))^2;
            pii(2) = 1;

            pii=pii/sum(pii);
            GM=gmdistribution(mui,Si,pii);
            x(i,:)=random(GM);
        end

        if(dx==1)
            figure(1)
            plot(z{1},x,'b*')
            xlabel('z')
            ylabel('x')
            title('Data')
        end

        exact_available=true;

    case 'thirteenth'
        % A Gaussian mixture, dx = 2, dz = 1.

        dx=2;
        dz=1;

        z=cell(dz,1);
        type_v=z;
        ndv=zeros(dz,1);
        for h=1:dz
            type_v{h}='Real';
            ndv(h)=0;
        end
        x=zeros(m,dx);

        nc_gmm=2; % Number of components of the Gaussian mixture model.
        Si=zeros(dx,dx,nc_gmm); % Individual covariance matrices.
        mui=zeros(nc_gmm,dx); % Individual means.
        pii=zeros(nc_gmm); % Mass of each component.

        z{1}=5*(rand(m,1)-0.5);

        for i=1:m
            mui(1,1) = 3 + 2.*z{1}(i);
            mui(1,2) = 2 + z{1}(i);
            mui(2,1) = -z{1}(i).^2+z{1}(i)/2;
            mui(2,2) = -3;
            Si(1,1,1) = 0.5*exp(z{1}(i));
            Si(2,2,1)=0.5;
            Si(1,1,2) = 0.25 - 0.1*z{1}(i);
            Si(2,2,2)=0.25 + 0.1*z{1}(i);
            pii(1) = 1;
            pii(2) = 1;

            pii=pii/sum(pii);
            GM=gmdistribution(mui,Si,pii);
            x(i,:)=random(GM);
        end

        figure(1)
        subplot(311)
        plot(z{1},x(:,1),'b*')
        xlabel('z')
        ylabel('x_1')
        title('Data')
        subplot(312)
        plot(z{1},x(:,2),'b*')
        xlabel('z')
        ylabel('x_2')
        subplot(313)
        plot(x(:,1),x(:,2),'b*')
        xlabel('x_1')
        ylabel('x_2')

        exact_available=true;

    case 'fourteenth'

        %  GARCH(1,1) simulation
        %  x is r_t, z is r_{t-1}

        T = 600;
        m = T - 100;

        dx = 1; dz = 1;
        z      = cell(dz,1);
        type_v = z;
        ndv    = zeros(dz,1);
        type_v{1} = 'Real';

        % parameters
        sigma0 = 0.5;
        alpha0 = 0.5;
        alpha1 = 0.5;
        beta   = 0.5;

        % simulate the process
        r     = zeros(T,1);     % r_1 … r_T
        sigma = zeros(T,1);     % σ_1 … σ_T
        epsi  = randn(T,1);

        sigma(1) = sigma0;
        r(1)     = sigma(1) * epsi(1);

        for t = 2:T
            sigma(t) = sqrt(alpha0 + alpha1*r(t-1)^2 + beta*sigma(t-1)^2);
            r(t)     = sigma(t) * epsi(t);
        end


        % build output (z,x)
        start_idx = T - m + 1;  % First usable index for x

        z{1} = r(start_idx:end-1).^2;      % r_{t-1} , length m
        x    = r(start_idx:end).^2;        % r_t      , length m  (m × dx with dx = 1)


        figure(1)
        plot(z{1}, x, 'b*')
        xlabel('r_{t-1}')
        ylabel('r_t')
        title('GARCH(1,1) simulation')

        exact_available = true;

    case 'fifteenth'
        % Product of two independent Gaussians: x = z * e
        m = 500; % Number of independent samples
        dx = 1; dz = 1;

        z = cell(dz,1);
        type_v = z;
        ndv = zeros(dz,1);
        type_v{1} = 'Real';
        ndv(1) = 0;


        % Generate z and e ~ N(0,1)
        z{1} = randn(m,1);
        epsi = randn(m,1);

        x = z{1} .* epsi;

        % Optional plot to visualize the joint distribution
        figure(1); clf
        scatter(z{1}, x, 'b.')
        xlabel('z'),
        ylabel('x = z*e')
        title('x = z*e with z, e ~ N(0,1) independent')

        exact_available=true;

    case 'sixteenth'

        %  ARCH simulation
        %  x is r_t, z is r_{t-1}

        oilbool = false;

        T = 1500;
        m = T-100;
        start_idx = T - m + 1;
        q=1;

        dx = 1;
        if oilbool
            dz = q*2+1;
        else
            dz = q;
        end
        z      = cell(dz,1);
        type_v = z;
        ndv    = zeros(dz,1);
        for i = 1:dz
            type_v{i} = 'Real'; ndv(i) = 0;
        end

        % parameters
        alpha0 = 0.0001;
        alpha = 0.25 * ones(q,1);
        sigma0 = sqrt(alpha0 / (1 - sum(alpha)))

        % simulate the process
        r     = zeros(T,1);     % r_1 … r_T
        sigma = zeros(T,1);
        Fz  = zeros(T,1);
        epsi = randn(T,1);     % σ_1 … σ_T

        % Commodity simulation

        % Parameters
        n = T;          % number of steps (e.g., daily for 1 year)
        phi = 0.1;       % mean reversion strength
        mu = 25;          % long-term mean
        alpha0oil = 0.0001;    % GARCH params
        alpha1oil = 0.1;
        beta1oil = 0.45;
        sigma_0 = 0.02;     % initial volatility
        lambda = 0.005;    % jump probability
        mu_J = 0;        % mean jump size
        sigma_J = 0.15;     % jump std deviation
        X0 = 25;          % initial price

      

% Simulate
        %xoil = simulate_ou_garch_jumps(n, phi, mu, alpha0oil, alpha1oil, beta1oil, sigma0oil, lambda, muJ, sigmaJ, X0);

        xoil = simulate_price_level_OU_GARCH_Jumps(n, phi,mu, alpha0oil, alpha1oil, beta1oil, sigma_0, lambda, mu_J, sigma_J, X0);


        % Sensitivity
        FSens = 0.2;

        retoil = diff(xoil)./xoil(1:end-1);
        rxoil = [0;retoil(:)];

        sigma(1:q) = sigma0;
        r(1:q) = sigma(1:q) .* epsi(1:q);
        Fz(1:q) = alpha0 +  sum(alpha .* r(1:q).^2);


        for t = q+1:T
            Fz(t)    = alpha0+sum(alpha .* r(t-1:-1:t-q).^2);
            sigma(t) = sqrt( Fz(t) );
            r(t)     = sigma(t) * epsi(t);
        end
        
        if oilbool
            for i = q+1:dz
                z{i} = rxoil(start_idx-i+q+1:T-i+q+1);
            end
        end
        for i = 1:q
            z{i} = r(start_idx - i : T - i);  % r_{t-i}
        end
        if oilbool
            FSens = FSens;
        else
            FSens = 0;
        end
        dz
        FSens
        x1 = FSens*rxoil + r;
        x = x1(start_idx:T);

        figure(1); clf
        scatter(Fz(start_idx:T), x, 'b.')
        %scatter(z{1}, x, 'b.')
        xlabel('F(z)'),
        ylabel('x=R^2 = F(z)* e^2')
        axis([0 4 -4 4])
        axis equal
        title('R^2 = F(z)* e^2 with z 9-lags of R, e ~ N(0,1) independent')

        figure(17)
        plot(Fz)
        ylabel('F(z)'),



        % Plot Oil
        figure(40);clf
        plot(xoil, 'LineWidth', 1.5);
        xlabel('Time');
        ylabel('Price');
        title('Discrete OU + GARCH + Jumps Simulation');
        grid on;





        exact_available = true;


    case 'seventeenth'

        %  ARCH simulation
        %  x is r_t, z is r_{t-1}


        T = 1500;
        m = T-100;
        start_idx = T - m + 1;
        q=1;

        lagbool = false;

        dx = 1;
        if lagbool
            dz = q+1;
        else
            dz = 1;
        end
        z = cell(dz,1);
        type_v = z;
        ndv    = zeros(dz,1);
        for i = 1:dz
            type_v{i} = 'Real'; ndv(i) = 0;
        end

        % parameters
        alpha0 = 0.0001;
        alpha = 0.25 * ones(q,1);
        sigma0 = sqrt(alpha0 / (1 - sum(alpha)))

        % simulate the process
        r     = zeros(T,1);     % r_1 … r_T
        sigma = zeros(T,1);
        Fz  = zeros(T,1);
        epsi = randn(T,1);     % σ_1 … σ_T

        % Commodity simulation
        

        % Parameters
        n = T;          % number of steps (e.g., daily for 1 year)
        phi = 0.1;       % mean reversion strength
        mu = 50;          % long-term mean
        alpha0oil = 0.01;    % GARCH params
        alpha1oil = 0.1;
        beta1oil = 0.45;
        sigma0oil = 0.02;     % initial volatility
        lambda = 0.01;    % jump probability
        muJ = 1.5;        % mean jump size
        sigmaJ = 1.5;     % jump std deviation
        X0 = 50;          % initial price

      

% Simulate
        %xoil = simulate_ou_garch_jumps(n, phi, mu, alpha0oil, alpha1oil, beta1oil, sigma0oil, lambda, muJ, sigmaJ, X0);

        xoil = simulate_price_level_OU_GARCH_Jumps(n, phi,mu, alpha0oil, alpha1oil, beta1oil, sigma0oil, lambda, muJ, sigmaJ, X0);

        % Sensitivity
        FSens = 0.2;
               
        retoil = diff(xoil)./xoil(1:end-1);
        rxoil = [0;retoil(:)];

        sigma(1:q) = sigma0;
        r(1:q) = sigma(1:q) .* epsi(1:q);
        Fz(1:q) = alpha0 +  sum(alpha .* r(1:q).^2);


        for t = q+1:T
            Fz(t)    = alpha0+sum(alpha .* r(t-1:-1:t-q).^2);
            sigma(t) = sqrt( Fz(t) );
            r(t)     = sigma(t) * epsi(t);
        end
        
        z{dz} = rxoil(start_idx:T);

        if lagbool
            for i = 1:q
                z{i} = r(start_idx - i : T - i);  % r_{t-i}
            end
        end
        x1 = FSens*rxoil + r;
        x = x1(start_idx:T);
        length(x)


        figure(1); clf
        scatter(Fz(start_idx:T), x, 'b.')
        xlabel('F(z)'),
        ylabel('x=R^2 = F(z)* e^2')
        axis([0 4 -4 4])
        axis equal
        title('R^2 = F(z)* e^2 with z 9-lags of R, e ~ N(0,1) independent')

        figure(17)
        plot(Fz)
        ylabel('F(z)'),



        % Plot Oil
        figure(40);
        plot(xoil, 'LineWidth', 1.5);
        xlabel('Time');
        ylabel('Price');
        title('Discrete OU + GARCH + Jumps Simulation');
        grid on;


        exact_available = true;
    
    case 'eighteenth'
                
        %  CCC bivariate ARCH simulation
        %  x is r_t, z is r_{t-1}

        T = 1500;
        m = T-100;
        start_idx = T - m + 1;
        q=1;

        % Parameters
 
        mu1 = 0.15 / 252;
        mu2 = 0.06 / 252;
        omega = [0.0004, 0.00008];
        A = [0.15, 0.06;
             0.04, 0.1];
        rho = 0.5;

        % Allocate output

        sigma2 = zeros(T,2);
        eps = zeros(T,2);
        sigma2(1,:) = (eye(2)-A)^(-1)*omega';
        z = randn(T,2);

        for t = 2:T
            eps_prev = eps(t-1,:);

            % ARCH dynamics
            sigma2(t,1) = omega(1) + A(1,1)*eps_prev(1)^2 + A(1,2)*eps_prev(2)^2;
            sigma2(t,2) = omega(2) + A(2,1)*eps_prev(1)^2 + A(2,2)*eps_prev(2)^2;

            % Build conditional covariance
            a = sigma2(t,1);
            b = sigma2(t,2);
            c = rho*sqrt(sigma2(t,1))*sqrt(sigma2(t,2));

            % Principal symmetric square root of 2x2 covariance
            delta = sqrt(a*b - c^2);
            s = sqrt(a + b + 2*delta);
            Sigma_sqrt = [a + delta, c;
                          c, b + delta] / s;

            % Shocks
            eps(t,:) = (Sigma_sqrt * z(t,:)')';
        end

        dx = 2;
        dz = 2*q;
        z      = cell(dz,1);
        type_v = z;
        ndv    = zeros(dz,1);
        for i = 1:dz
            type_v{i} = 'Real'; ndv(i) = 0;
        end
        
        for i = 1:q
            z{2*i-1} = eps(start_idx - i : T - i,1);
            z{2*i} = eps(start_idx - i : T - i,2);
        end
        x1 = eps;
        x = x1(start_idx:T,:);
        x(:,1)=x(:,1)+mu1;
        x(:,2)=x(:,2)+mu2;

        exact_available = true;



    case 'nineteenth'
        q=1
        T=1400;
        start_idx = q+1;
        load('R_data.mat');
        dx = size(R,2);
        dz = dx*q;
        type_v=z;
        ndv=zeros(dz,1);
        for i = 1:dz
            type_v{i} = 'Real'; ndv(i) = 0;
        end
        x = R(start_idx:start_idx+T-1,:);
        for i=1:q
            for k=1:dx
                z{dx*(i-1)+k} = R(start_idx - i : start_idx+T - i-1,k);
            end
        end

        exact_available = true;

    case 'twentieth'

        %  ARCH simulation
        %  x is r_t, z is r_{t-1}
        T = 1260;
        createOilArch;
        exact_available = true;



end
dy=dx;
