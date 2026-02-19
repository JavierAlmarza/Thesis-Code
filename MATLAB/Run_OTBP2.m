% Plans (and solves) successive OTBP.
%
% What follows is just one possible example.


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
%data_case='fourteenth';
%data_case='fifteenth';
%data_case='sixteenth';
%data_case='seventeenth';
%data_case='eighteenth';
%data_case = 'nineteenth';
data_case = 'twentieth';

Create_Data; % We can choose to run Create_Data externally to this macro 

m = length(x);

N_runs=1; % Number of OTBP to run.

% We can run as many OTBP as we'd like, to further remove variability.
% Simularing rho(x|z) requires inverting the corresponding maps in reverse
% order.

[y, Inverse] = callOTBP(x,z,N_runs);




if strcmp(data_case, 'sixteenth')

    %Plots
    figure(13), clf
    scatter(epsi(101:end),y,10,"filled")
    xlabel('e')
    ylabel('y')
    axis equal
    grid on
    %Autocorrelation tests
    YautocorrLB=ljungbox(y)
    e2autocorrLB=ljungbox(epsi(start_idx:T).^2)
    XautocorrLB=ljungbox(x)

    [hvx,pvx,qstatx,critx]=ljungbox(x.^2);
    X2autocorrLB=[hvx,pvx,qstatx,critx]
    [hv,pv,qstat,crit]=ljungbox(y.^2);
    Y2autocorrLB=[hv,pv,qstat,crit]
    Y2autocorrEng = archtest(y, 'Lags', [1 2 3 4])
    mean(y)
    X2autocorrEng = archtest(x, 'Lags', [1 2 3 4])
    mean(x)    

    figure(12), clf
    scatter(x,y,10,"filled")
    xlabel('x')
    ylabel('y')
    axis equal
    grid on

    %Simulation

    Msims=0;
    fprintf( '%s','For how long would you like to simulate the process? ');
    Msims = input(' ')
    if Msims>0
      z_star=cell(dz,1);
      xsim=zeros(Msims,1);
      figure(1) % To show the data to the user.
      for j=1:dz
        fprintf('z_*(%d) = ',j)
        z_star{j} = input(' ');
      end
      for ind=1:Msims
        xs = zeros(100,1);
        xs = Inverse(y(1:100), z_star);
        indrand=randi(length(xs));
        xsim(ind)=xs(indrand);
        for j=2:dz
          z_star{j} = z_star{j-1};
        end
        z_star{1} = xsim(ind);
      end
      figure(19),clf
      plot(xsim)
      ylabel('Simulated x')
    end

    %Plot likelihood functions
    omega_vals = linspace(0.005, 0.02, 100);  % avoid zero to prevent log(0)
    alpha_vals = linspace(0.10, 0.5, 100);      
    [Omega, Alpha] = meshgrid(omega_vals, alpha_vals);
    Zplot = arrayfun(@(o, a) arch1_loglike(o, a, x), Omega, Alpha);
    Zplot2 = arrayfun(@(o, a) arch1_loglike(o, a, xsim), Omega, Alpha);

    figure(30); %contours
    contour(Omega, Alpha, Zplot);
    hold on;
    contour(Omega, Alpha, Zplot2, 'FaceAlpha', 0.5, 'EdgeColor', 'k');        
    hold off;
    xlabel('\omega'); ylabel('\alpha'); zlabel('Log-Likelihood');
    title('ARCH(1) Log-Likelihood ');

    figure(31); %surface
    surf(Omega, Alpha, Zplot);
    hold on;
    surf(Omega, Alpha, Zplot2, 'FaceAlpha', 0.5, 'EdgeColor', 'r');        
    hold off;
    xlabel('\omega'); ylabel('\alpha'); zlabel('Log-Likelihood');
    title('ARCH(1) Log-Likelihood ');

    % ARCH test and fit for simulated data


    if Msims==0
        ytest = x;
    else
        ytest=xsim;
    end

    residuals=ytest;
    
    [h, pValue, stat, cValue] = archtest(residuals, 'Lags', 1);
    fprintf('ARCH-LM test stat = %.4f, p-value = %.4f\n', stat, pValue);

    % Fit ARCH(1) for simulated data
    model = garch('Constant', NaN, 'ARCHLags', 1, 'GARCHLags', []);
    [estModel, estParamCov, logL, info] = estimate(model, residuals);

    % Fit ARCH(1) for original data
    residualsbis = x;
    model2 = garch('Constant', NaN, 'ARCHLags', 1, 'GARCHLags', []);
    [estModel, estParamCov, logL, info] = estimate(model, residualsbis);

    generateData = false;
    if oilbool
        generateData = false;
    end

    if generateData
      N2 = 300;
      T2 = 1008;
      z_star2=cell(dz,1);
      Msims2 = 300*1008;
      xsim2=zeros(Msims2,1);
      for j=1:dz
        z_star2{j} = 0;
      end
      for ind=1:Msims2
        xs = zeros(300,1);
        xs = Inverse(y(1:300), z_star2);
        indrand=randi(length(xs));
        xsim2(ind)=xs(indrand);
        for j=2:q
          z_star2{j} = z_star2{j-1};
        end
        z_star2{1} = xsim2(ind);
        if oilbool
            z_star{q+1} = 2
            for j=q+1:dz
                x;
            end
        end
      end
      xsimData = reshape(xsim2, T2, N2)';
    save('BaryDataMatrix.mat', 'xsimData');
    % Fit ARCH(1) for generated data
    residualsgen = xsimData(1,:)';
    model3 = garch('Constant', NaN, 'ARCHLags', 1, 'GARCHLags', []);
    [estModel, estParamCov, logL, info] = estimate(model3, residualsgen);

    end

    
end

if strcmp(data_case, 'seventeenth')
    x2 = y;
    ylags = 1;
    dz2 = ylags;
    z2 = cell(dz2,1);
    for j = 1:dz2
            z2{j} = x2(dz2-j+1:end-j);
    end
    x2 = x2(dz2+1:end);
    N_runs2 = 1;
    [w, Inverse2] = callOTBP(x2,z2,N_runs2);

    %Autocorrelation tests
    WautocorrLB=ljungbox(w)
    e2autocorrLB=ljungbox(epsi(start_idx:T).^2)
    YautocorrLB=ljungbox(x2)

    [hvx,pvx,qstatx,critx]=ljungbox(x2.^2);
    Y2autocorrLB=[hvx,pvx,qstatx,critx]
    [hv,pv,qstat,crit]=ljungbox(w.^2);
    W2autocorrLB=[hv,pv,qstat,crit]
    W2autocorrEng = archtest(w, 'Lags', [1 2 3 4])
    Y2autocorrEng = archtest(x2, 'Lags', [1 2 3 4])    

    figure(12), clf
    scatter(x2,w,10,"filled")
    xlabel('y')
    ylabel('w')
    axis equal
    grid on

    %Simulation

    m2 = length(x2);
    Msims  = m2;
    z2_star=cell(dz2,1);
    ysim=zeros(m2,1);
    for j=1:dz2
        z2_star{j} = z2{j}(1);
    end
    for ind=1:Msims
        x2s=w;
        x2s = Inverse2(x2s, z2_star);
        indrand=randi(length(x2s));
        ysim(ind)=x2s(indrand);
        for j=2:dz
            z2_star{j} = z2_star{j-1};
        end
        z2_star{1} = ysim(ind);
    end
    xsim = zeros(Msims,1);
    z_star = cell(dz,1);
    disp('z versus rxoil')
    for ind=1:Msims
        for j=1:dz
            z_star{j} = z{j}(ind+1);
        end
        xsim(ind) = Inverse(ysim(ind),z_star);
    end
    figure(19),clf
    plot(ysim)
    ylabel('Simulated y')

    figure(20),clf
    plot(xsim)
    ylabel('Simulated x')

    %Plot likelihood functions
    omega_vals = linspace(0.005, 0.02, 100);  % avoid zero to prevent log(0)
    alpha_vals = linspace(0.10, 0.5, 100);      
    [Omega, Alpha] = meshgrid(omega_vals, alpha_vals);
    
    ysimcent = ysim - mean(ysim);
    Zplot = arrayfun(@(o, a) arch1_loglike(o, a, r(2:end)), Omega, Alpha);
    Zplot2 = arrayfun(@(o, a) arch1_loglike(o, a, ysimcent), Omega, Alpha);


    % ARCH test and fit for simulated data


    residuals=ysimcent;
    
    [h, pValue, stat, cValue] = archtest(residuals, 'Lags', 1);
    fprintf('ARCH-LM test stat = %.4f, p-value = %.4f\n', stat, pValue);

    % Fit ARCH(1) for simulated data
    model = garch('Constant', NaN, 'ARCHLags', 1, 'GARCHLags', []);
    [estModel, estParamCov, logL, info] = estimate(model, residuals);

    % Fit ARCH(1) for original data
    residualsbis = r(2:end);
    model = garch('Constant', NaN, 'ARCHLags', 1, 'GARCHLags', []);
    [estModel, estParamCov, logL, info] = estimate(model, residualsbis);

    mdl = fitlm(rxoil(102:end),xsim)
    rsim = xsim-mdl.Coefficients(2,1).Estimate*rxoil(102:end);
    model = garch('Constant', NaN, 'ARCHLags', 1, 'GARCHLags', []);
    [estModel, estParamCov, logL, info] = estimate(model, rsim);

    Zplot3 = arrayfun(@(o, a) arch1_loglike(o, a, rsim), Omega, Alpha);

    figure(30); %contours
    contour(Omega, Alpha, Zplot);
    hold on;
    contour(Omega, Alpha, Zplot2, 'FaceAlpha', 0.5, 'EdgeColor', 'k');        
    hold on;
    contour(Omega, Alpha, Zplot3, 'FaceAlpha', 0.5, 'EdgeColor', 'c');        
    hold off;
    xlabel('\omega'); ylabel('\alpha'); zlabel('Log-Likelihood');
    title('ARCH(1) Log-Likelihood ');

    figure(31); %surface
    surf(Omega, Alpha, Zplot);
    hold on;
    surf(Omega, Alpha, Zplot2, 'FaceAlpha', 0.5, 'EdgeColor', 'r');        
    hold on;
    surf(Omega, Alpha, Zplot3, 'FaceAlpha', 0.5, 'EdgeColor', 'c');        
    hold off;    
    xlabel('\omega'); ylabel('\alpha'); zlabel('Log-Likelihood');
    title('ARCH(1) Log-Likelihood ');


end

if strcmp(data_case, 'eighteenth')

    %Autocorrelation tests
    [hvx,pvx,qstatx,critx]=ljungbox(x(:,1).^2);
    X2autocorrLB=[hvx,pvx,qstatx,critx]

    [hv,pv,qstat,crit]=ljungbox(y(:,1).^2);
    Y2autocorrLB=[hv,pv,qstat,crit]


    generateData = true;

    if generateData
      N2 = 300;
      T2 = 1008;
      z_star2=cell(dz,1);
      Msims2 = 300*1008;
      xsim2=zeros(Msims2,dx);
      for j=1:dz
        z_star2{j} = 0;
      end
      for ind=1:Msims2
        xs = zeros(200,dx);
        xs = Inverse(y(1:200,:), z_star2);
        indrand=randi(length(xs));
        xsim2(ind,:)=xs(indrand,:);
        for j=3:dz
          z_star2{j} = z_star2{j-2};
        end
        z_star2{1} = xsim2(ind,1);
        z_star2{2} = xsim2(ind,2);
      end
      xshaped = zeros(N2, T2, dx);
      for k = 1:dx
          xshaped(:,:,k) = reshape(xsim2(:,k), T2, N2)';  % reshape row-wise
      end
      xsimData2 = xshaped;
      save('BaryDataMatrix2.mat', 'xsimData2');

    end

    
end

if strcmp(data_case, 'nineteenth')
    fprintf('Case nineteenth')
end

if strcmp(data_case, 'twentieth')

    figure(12), clf
    scatter(x,y,10,"filled")
    xlabel('x')
    ylabel('y')
    axis equal
    grid on

    generateData = true;

    if generateData
      N2 = 1000;
      T2 = 2000;
      z_star2=cell(dz,1);
      Msims2 = N2*T2;
      xsim2=zeros(Msims2,1);
      Roil = generateOilArchMod(Msims2);
      
      z_star2{1} = x(end);
      z_star2{2} = Roil(1);
      for ind=1:Msims2
        xs = zeros(200,1);
        xs = Inverse(y(1:200), z_star2);
        indrand=randi(length(xs));
        xsim2(ind)=xs(indrand);
        z_star2{1} = xsim2(ind);
        if ind<Msims2
            z_star2{2} = Roil(ind+1);
        end
      end
      xsimDataC = reshape(xsim2, T2, N2)';
      Roil = reshape(Roil, T2, N2)';
      RoilC=Roil;
      save('BaryDataMatrixC.mat', 'xsimDataC', 'RoilC');
    end
   
end

fprintf('\nOK, see you next time! \n')
