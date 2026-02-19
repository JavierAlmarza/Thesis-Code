% Plans (and solves) successive OTBP.
%
% What follows is just one possible example.

m=1500; % Number of samples
% m=500; % Number of samples
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
%data_case='twelfth';
% data_case='thirteenth';
%data_case='fourteenth';
%data_case='fifteenth';
data_case='sixteenth';
%data_case='seventeenth';
%data_case='eighteenth';


Create_Data; % We can choose to run Create_Data externally to this macro 

N_runs=1; % Number of OTBP to run.

% We can run as many OTBP as we'd like, to further remove variability.
% Simularing rho(x|z) requires inverting the corresponding maps in reverse
% order.

% Create cell-space for the runs:
yy=cell(N_runs,1);
nsy=yy; Iy=yy; type_g=yy; ncy=yy; gamma_y=yy;
nsz=yy; Iz=yy; type_f=yy; ncz=yy; gamma_z=yy;
Qy=yy; Qz=yy; By=yy; Bz=yy; a=yy; b=yy; K=yy;
zc=yy; Scz=yy; yc=yy; Scy=yy;
sz=yy; sy=yy; nF=yy; nG=yy;
lambda=yy; fg=yy;
% xs=yy;
y=cell(N_runs+1,1);

vx=sum(var(x));
stx=sqrt(vx); % Used to normalize Ly.

% We can run Plot_Results at any point, to investigate how runs go.

% General parameters, fixed for the time being though we can make them depend on the run if we'dlike.

nn=10; % Maximal number of stages.
nmax = 1000; %maximum number of descent steps per stage.
nk = 100; % number of iterations for the update of a and b
eta_min = 0.000000001;
eta_max = 1000;
delta=0.2/sqrt(m); %Maximal correlation sought.
gamma=2; % gamma*delta is the threshold for termination.
K_max=20; % Maximum number of principal components of A considered.

y{1}=x;

for jr=1:N_runs
    %jr=2
    % Each run takes over from where the previous one left.

    switch jr
%         case 1
%
%             % A single kernel function
%             % in z while linea in y. Fine to capture
%             % conditional mean.
%
%             nsy{jr}(1)=1;
%             Iy{jr}{1}=1:dx;
%             type_g{jr}{1}='linear';
%             ncy{jr}=[]; gamma_y{jr}=[];
%
%             nsz{jr}(1)=1;
%             Iz{jr}{1}=1:dz;
%             type_f{jr}{1}='kernel';
%             ncz{jr}(1)=round(sqrt(m));
%             gamma_z{jr}(1)=1./sqrt(2); % Could be corrected through cross-validation.
%
%             PR=true; % Plot results.


        case 1

            % A single kernel function
            % in z while quadratic in y. Fine to capture
            % conditional mean and variance when z is low
            % dimensional.

            nsy{jr}(1)=1;
            Iy{jr}{1}=1:dx;
            type_g{jr}{1}='kernel';
            %ncy{jr}=[];
            ncy{jr}(1)=round(sqrt(m));
            %gamma_y{jr}=[];
            gamma_y{jr}(1)=2;

            nsz{jr}(1)=1;
            Iz{jr}{1}=1:dz;
            type_f{jr}{1}='kernel';
            ncz{jr}(1)=round(sqrt(m));
            gamma_z{jr}(1)=1./sqrt(2); % Could be corrected through cross-validation.

            PR=true; % Plot results.

        case 2

            % Kernels in both z and y.

            nsy{jr}(1)=1;
            Iy{jr}{1}=1:dx;
            type_g{jr}{1}='kernel';
            ncy{jr}(1)=round(sqrt(m));
            gamma_y{jr}(1)=2; % Could be corrected through cross-validation.

            nsz{jr}(1)=1;
            Iz{jr}{1}=1:dz;
            type_f{jr}{1}='kernel';
            ncz{jr}(1)=round(sqrt(m));
            gamma_z{jr}(1)=1/sqrt(2); % Could be corrected through cross-validation

            PR=true;

        case 3

            % Kernels in both z and y.

            nsy{jr}(1)=1;
            Iy{jr}{1}=1:dx;
            type_g{jr}{1}='kernel';
            ncy{jr}(1)=round(sqrt(m));
            gamma_y{jr}(1)=3.0; % Could be corrected through cross-validation.

            nsz{jr}(1)=1;
            Iz{jr}{1}=1:dz;
            type_f{jr}{1}='kernel';
            ncz{jr}(1)=round(sqrt(m));
            gamma_z{jr}(1)=1/sqrt(2); % Could be corrected through cross-validation

            PR=true;

        case 4

            % Kernels in both z and y.

            nsy{jr}(1)=1;
            Iy{jr}{1}=1:dx;
            type_g{jr}{1}='kernel';
            ncy{jr}(1)=round(sqrt(m));
            gamma_y{jr}(1)=2.; % Could be corrected through cross-validation.

            nsz{jr}(1)=1;
            Iz{jr}{1}=1:dz;
            type_f{jr}{1}='kernel';
            ncz{jr}(1)=round(sqrt(m));
            gamma_z{jr}(1)=1/sqrt(2); % Could be corrected through cross-validation

            PR=true;

    end

%     [y{jr+1}, Qy{jr}, Qz{jr}, a{jr}, b{jr}, By{jr}, Bz{jr}, K{jr}, zc{jr}, Scz{jr}, yc{jr}, Scy{jr}, sz{jr}, nF{jr}, sy{jr}, nG{jr}, lambda{jr},fg{jr}, nLp, Cp, Tp, TTp,etap,Ly] =...
%         OTBP(y{jr},z,nsz{jr},Iz{jr},type_f{jr},ncz{jr},type_v,ndv,gamma_z{jr},nsy{jr},Iy{jr},type_g{jr},ncy{jr},gamma_y{jr}, nn,nmax,nk,eta_min,eta_max,delta,gamma,stx);

    [y{jr+1}, Qy{jr}, Qz{jr}, a{jr}, b{jr}, By{jr}, Bz{jr}, K{jr}, zc{jr}, Scz{jr}, yc{jr}, Scy{jr}, sz{jr}, nF{jr}, sy{jr}, nG{jr}, lambda{jr},fg{jr}, nLp, Cp, Tp, TTp,etap,Ly] =...
        OTBP(y{jr},z,nsz{jr},Iz{jr},type_f{jr},ncz{jr},type_v,ndv,gamma_z{jr},nsy{jr},Iy{jr},type_g{jr},ncy{jr},gamma_y{jr}, nn,nmax,eta_min,eta_max,delta,gamma,stx, K_max);
    if(PR)
        PR = Plot_Results(y{jr}, y{jr+1}, z, nLp, Cp, Tp, TTp,etap,epsi,Fz);
%         pause
    end

end

YautocorrLB=ljungbox(y{jr+1})
[hve,pve,qstate,crite]=ljungbox(epsi(start_idx:T).^2);
e2autocorrLB=ljungbox(epsi(start_idx:T).^2)
XautocorrLB=ljungbox(x)

[hvx,pvx,qstatx,critx]=ljungbox(x.^2);
X2autocorrLB=[hvx,pvx,qstatx,critx]
[hv,pv,qstat,crit]=ljungbox(y{jr+1}.^2);
Y2autocorrLB=[hv,pv,qstat,crit]
Y2autocorrEng = archtest(y{jr+1}, 'Lags', [1 2 3 4])
mean(y{jr+1})
X2autocorrEng = archtest(x, 'Lags', [1 2 3 4])
mean(x)    

figure(12), clf
scatter(x,y{jr+1},10,"filled")
xlabel('x')
ylabel('y')
axis equal
grid on



% Simulation (It can be run separately after this macro, to continue
% simulating rho(x|z_star) under various z_star.

reply = input('Would you like to perform a simulation? Y/N ','s');

while(reply=='Y')
    z_star=cell(dz,1);

    figure(1) % To show the data to the user.

    for j=1:dz
        fprintf('z_*(%d) = ',j)
        z_star{j} = input(' ');
    end

    xs=y{N_runs+1};

    for jr=N_runs:-1:1
       jr
%         xs{jr} = Invert(y{jr+1}, Qz{jr}, a{jr}, b{jr}, By{jr}, Bz{jr}, K{jr}, zc{jr}, Scz{jr}, yc{jr}, Scy{jr}, sz{jr}, fg{jr}, ...
%             nF{jr}, sy{jr}, nG{jr}, lambda{jr}, z_star, nsz{jr}, Iz{jr},type_f{jr},ncz{jr},type_v,ndv,gamma_z{jr},nsy{jr},Iy{jr},type_g{jr},ncy{jr},gamma_y{jr}, nk);

       xs = Invert(xs, Qz{jr}, a{jr}, b{jr}, By{jr}, Bz{jr}, K{jr}, zc{jr}, Scz{jr}, yc{jr}, Scy{jr}, sz{jr}, fg{jr}, ...
            nF{jr}, sy{jr}, nG{jr}, lambda{jr}, z_star, nsz{jr}, Iz{jr},type_f{jr},ncz{jr},type_v,ndv,gamma_z{jr},nsy{jr},Iy{jr},type_g{jr},ncy{jr},gamma_y{jr}, nk);

    end

    PI=Plot_Inverse(xs, z_star,exact_available, data_case, dy);

    reply = input('Would you like to perform another simulation? Y/N ','s');
end

Msims=0;
if strcmp(data_case, 'sixteenth')
    fprintf( '%s','For how long would you like to simulate the process? ');
    Msims = input(' ')
    if Msims>0
      z_star=cell(dz,1);
      xsim=zeros(Msims,1);
      figure(1) % To show the data to the user.
      for j=1:dz
        fprintf('z_*(%d) = ',j)
        z_star{j} = input(' ');
        xsim(j) = z_star{j};
        z_star{j}
      end
      for ind=1:Msims
        xs=y{N_runs+1};
        for jr=N_runs:-1:1
          jr;
          xs = Invert(xs, Qz{jr}, a{jr}, b{jr}, By{jr}, Bz{jr}, K{jr}, zc{jr}, Scz{jr}, yc{jr}, Scy{jr}, sz{jr}, fg{jr}, ...
            nF{jr}, sy{jr}, nG{jr}, lambda{jr}, z_star, nsz{jr}, Iz{jr},type_f{jr},ncz{jr},type_v,ndv,gamma_z{jr},nsy{jr},Iy{jr},type_g{jr},ncy{jr},gamma_y{jr}, nk);
        end
        indrand=randi(length(xs));
        xsim(ind)=xs(indrand);
        for j=1:dz-1
          z_star{j} = z_star{j+1};
        end
        %z_star{dz} = xsim(ind)^2;
        z_star{dz} = xsim(ind);


        %PI=Plot_Inverse(xs, z_star,exact_available, data_case, dy);
      end
      figure(19),clf
      plot(xsim)
      ylabel('Simulated x')
    end



  %Plot likelihood functions

    omega_vals = linspace(0.005, 0.02, 100);  % avoid zero to prevent log(0)
    alpha_vals = linspace(0.10, 0.5, 100);       % alpha must be < 1 for stationarity

    [Omega, Alpha] = meshgrid(omega_vals, alpha_vals);

    % Evaluate log likelihood on the grid
    Zplot = arrayfun(@(o, a) arch1_loglike(o, a, x), Omega, Alpha);
    Zplot2 = arrayfun(@(o, a) arch1_loglike(o, a, xsim), Omega, Alpha);



    % Plot
    figure(30);
    contour(Omega, Alpha, Zplot);
    hold on;
    contour(Omega, Alpha, Zplot2, 'FaceAlpha', 0.5, 'EdgeColor', 'k');        
    hold off;
    xlabel('\omega'); ylabel('\alpha'); zlabel('Log-Likelihood');
    title('ARCH(1) Log-Likelihood ');

    figure(31);
    surf(Omega, Alpha, Zplot);
    hold on;
    surf(Omega, Alpha, Zplot2, 'FaceAlpha', 0.5, 'EdgeColor', 'r');        
    hold off;
    xlabel('\omega'); ylabel('\alpha'); zlabel('Log-Likelihood');
    title('ARCH(1) Log-Likelihood ');


  % Step 1: Load your data
    if Msims==0
        ytest = x;
    else
        ytest=xsim;
    end% column vector

    residuals=ytest;
    % Step 2: Fit mean model (e.g., constant mean)
    %residuals = ytest - mean(ytest);  % crude mean model

    % Step 3: ARCH-LM test with 1 lag
    [h, pValue, stat, cValue] = archtest(residuals, 'Lags', 1);
    fprintf('ARCH-LM test stat = %.4f, p-value = %.4f\n', stat, pValue);

    % Step 4: Fit ARCH(1)
    model = garch('Constant', NaN, 'ARCHLags', 1, 'GARCHLags', []);
    [estModel, estParamCov, logL, info] = estimate(model, residuals);

    % Step 4': Fit ARCH(1) for original data
    residualsbis = x;
    model = garch('Constant', NaN, 'ARCHLags', 1, 'GARCHLags', []);
    [estModel, estParamCov, logL, info] = estimate(model, residualsbis);

    % Step 5: Infer conditional variances
    [v, ~] = infer(estModel, residuals);

    % Plot
    figure(22);
    plot(v);
    title('Estimated Conditional Variance (ARCH(1))');
    xlabel('Time');
    ylabel('Variance');
end

fprintf('\nOK, see you next time! \n')



