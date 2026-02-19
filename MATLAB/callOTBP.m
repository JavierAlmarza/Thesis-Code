function [yret,Inverse] = callOTBP(x,z,N_runs)
    if ndims(x)~=2
        disp('x should be a matrix')
        return
    end
    if  ~ismember(N_runs, [1,2,3,4])
        disp('N_runs should be an integer between 1 and 4');
        return
    end
    if ~iscell(z)
        disp('z should be a cell')
        return
    end
    dz = length(z);
    dx = min(size(x))
    for i=1:dz
        if length(z{i})~=length(x)
            disp('Samples have different sizes for x and z');
            return
        end
    end
    m = length(x);
    type_v=z;
    ndv=zeros(dz,1);
    for i = 1:dz
            type_v{i} = 'Real';
            ndv(i) = 0;
    end
    
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
        % Each run takes over from where the previous one left.
        switch jr
            case 1
                % A single kernel function
                % in z while quadratic in y. Fine to capture
                % conditional mean and variance when z is low
                % dimensional.

                nsy{jr}(1)=1;
                Iy{jr}{1}=1:dx;
                type_g{jr}{1}='linear and quadratic';
                ncy{jr}=[];
                %ncy{jr}(1)=round(sqrt(m));
                gamma_y{jr}=[];
                %gamma_y{jr}(1)=2;

                nsz{jr}(1)=1;
                Iz{jr}{1}=1:dz;
                type_f{jr}{1}='kernel';
                ncz{jr}(1)=round(sqrt(m));
                gamma_z{jr}(1)=1./sqrt(2); % Could be corrected through cross-validation.

                PR=false; % Plot results.

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

        [y{jr+1}, Qy{jr}, Qz{jr}, a{jr}, b{jr}, By{jr}, Bz{jr}, K{jr}, zc{jr}, Scz{jr}, yc{jr}, Scy{jr}, sz{jr}, nF{jr}, sy{jr}, nG{jr}, lambda{jr},fg{jr}, nLp, Cp, Tp, TTp,etap,Ly] =...
        OTBP(y{jr},z,nsz{jr},Iz{jr},type_f{jr},ncz{jr},type_v,ndv,gamma_z{jr},nsy{jr},Iy{jr},type_g{jr},ncy{jr},gamma_y{jr}, nn,nmax,eta_min,eta_max,delta,gamma,stx, K_max);
        if(PR)
            PR = Plot_Results(y{jr}, y{jr+1}, z, nLp, Cp, Tp, TTp,etap);
            %        pause
        end
    end
    disp('Hello')
    yret = y{jr+1};
    
    Inverse = @(w,zstar) w;

    for jr=N_runs:-1:1
        prev = Inverse;
        Inverse = @(w,z_star) Invert(prev(w,z_star), Qz{jr}, a{jr}, b{jr}, By{jr}, Bz{jr}, K{jr}, zc{jr}, Scz{jr}, yc{jr}, Scy{jr}, sz{jr}, fg{jr}, ...
            nF{jr}, sy{jr}, nG{jr}, lambda{jr}, z_star, nsz{jr}, Iz{jr},type_f{jr},ncz{jr},type_v,ndv,gamma_z{jr},nsy{jr},Iy{jr},type_g{jr},ncy{jr},gamma_y{jr}, nk);
    end
    
end