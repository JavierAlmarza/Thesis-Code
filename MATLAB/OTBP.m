function [y, Qy, Qz, a, b, By, Bz, K, zc, Scz, yc, Scy, sz, nF, sy, nG, lambda, fg, nLp, Cp, Tp, TTp,etap, Ly] = OTBP(x,z,nsz,Iz,type_f,ncz,type_v,ndv,gamma_z,nsy,Iy,type_g,ncy,gamma_y, nn,nmax,eta_min,eta_max,delta,gamma, stx, K_max)
% Barycenter problem.

% nn=10; % Maximal number of stages.
% nmax = 1000; %maximum number of descent steps per stage.
% eta_min = 0.000000001;
% eta_max = 1000;
% delta=0.2/sqrt(m); %Maximal correlation sought.
% gamma=2; % gamma*delta is the threshold for termination.

% n_iterations=5;

[m,dy]=size(x);
vx=sum(var(x));

Cp=zeros(1,nn*nmax); % For saving C.
Tp=Cp; % For saving the test function times lambda.
TTp=zeros(1,nn*nmax); % Same for the individual test functions.
etap=zeros(1,nn*nmax); % Same for eta.
nLp=zeros(1,nn*nmax); % Same for the norm of the gradient.

% Initialization:

np=0; % Printing time.
y=x; % y starts at x, this could be modified to use preconditioning.

things_known=false; % Bandwidths, centers and the like.

[F, zc, Scz0, ~] = FigureOutF(nsz,Iz,type_f,ncz, z, type_v, ndv, ...
    things_known, [], [], gamma_z, m);
nj=size(F,2);
sz=sum(F,1)/m;
F=F-sz;
nF=(sum(F.^2,1)/m).^(1/2);
for j=1:nj
    F(:,j)=F(:,j)/nF(j);
end
[Qz,Bz] = QB(F); % Principal components of F.

% Main loop:

jj=0;
not_converged=true;

while(jj<nn&&not_converged) % Loop over stages
    jj=jj+1 %Stage
    dependent=true;
    call_it_a_day=false; % call_it_a_day is activated when the learning rate becomes too small.
    % This should not happen, if it does, one needs to change the parameters and restart the run.
    n=1; % First step of the stage.
    yo=y; % y at the onset of the stage, used as centers and to define the innner product.
    varyo=sum(var(yo));
    
    % Matrices G and Gy:
    
    Scz=Scz0;
    
    [G, Gy, ~, yc, Scy] = ...
        FigureOut_G_Gy_and_Gyy(nsy,Iy,type_g,ncy,y, false, [], [], gamma_y,true,false);
    
    sy=sum(G,1)/m; % This is only computed at the beginning of the stage, so it is independent of y.
    G=G-sy;
    % nG=(sum(G.^2,1)/m).^(1/2);
    nG=max(abs(G),[],1);
    %     nG=ones(m,1);
    for j=1:size(G,2)
        G(:,j)=G(:,j)/nG(j);
    end
    
    [Qy,By] = QB(G); % Principal components of G.
    
    % Initial iteration on a and b:
    
    [Py,a,b,fg,~,K] = L_y(Qz,Qy,m,dy,Gy,By,K_max,true);
    fg_capped=min(fg,gamma*delta);
    Pytot=zeros(m,dy);
    for k=1:K
        Pytot=Pytot+Py(:,:,k).*fg_capped(k);
    end
    
    %     Pytot=sum(Py,3);
    fgPy=sum(reshape(fg,1,1,K).*Py,3);
    fg2=sum(fg.^2);
    
    C = sum(sum((x-y).^2))/2;
    Cy=(y-x);
    nCy=2*C/m;
    
    nPy=sqrt(sum(sum(Pytot.^2))/m);
    
    %     lambda=(1./(2*delta))*sqrt(nCy+0.1*vx)/nPy;
    lambda=sqrt(nCy+0.1*vx)/nPy;
    
    done=false;
    
    Ly=Cy+2*lambda*fgPy;
    Ly2=sum(sum(Ly.^2));
    
    eta=(lambda*fg2/Ly2)/20; %The initial eta goes about 1/20 of the way to make P zero.
    
    while ~done % Loop within the stage.
        
        GoodEnough=false; % An indicator of whether the learning rate worked.
%         Pytot=zeros(m,dy);
%         fgPy=Pytot;
        
        fg2=0.; % sum of sigma_k for each sub-problem
        
        [G, Gy, ~, yc, Scy] = ...
            FigureOut_G_Gy_and_Gyy(nsy,Iy,type_g,ncy,y, true, yc, Scy, gamma_y,true,false);
        
        G=G-sy;
        for j=1:size(G,2)
            G(:,j)=G(:,j)/nG(j);
            for k=1:dy
                Gy{k}(:,j)=Gy{k}(:,j)/nG(j);
                %                 for l=1:dy
                %                     Gyy{k,l}(:,j)=Gyy{k,l}(:,j)/nG(j);
                %                 end
            end
        end
        Qy=G*By;
        
        [Py,a,b,fg,~,K] = L_y(Qz,Qy,m,dy,Gy,By,K_max,true);
        fg_capped=min(fg,gamma*delta);
        Pytot=zeros(m,dy);
        for k=1:K
            Pytot=Pytot+Py(:,:,k).*fg_capped(k);
        end
% 
%         Pytot=sum(Py,3);
        fgPy=sum(reshape(fg,1,1,K).*Py,3);
        fg2=fg2+sum(fg.^2);
        size(fg)
        isempty(fg)
        sig1=fg(1);
        
        C = sum(sum((x-y).^2))/2;
        Cy=(y-x);
        nCy=2*C/m;
        
        if(dependent)
            nPy=sqrt(sum(sum(Pytot.^2))/m);
%             delta1=sqrt(delta*fg(1));
%             lambda=(1./(2*delta1))*sqrt(nCy+0.1*vx)/nPy;
%             lambda=(1./(2*delta))*sqrt(nCy+0.1*vx)/nPy;
            lambda=sqrt(nCy+0.1*vx)/nPy;
            %             if(lambda > 1/(3*s_max))
            %                 lambda = 1/(3*s_max);
            %             end
        end
        %         lambdai=min(lambdai,lambda).*ones(1,dy);
        
        Lold=C+lambda*fg2;
        Ly=Cy+2*lambda.*fgPy;
        
        Ly2=sum(sum(Ly.^2)); % Rate of growth in the direction of descent.
        
        dL_exp=0.5*Ly2; % Amount of eta'th rate of descent we'd be happy with.
        
        while(~GoodEnough) % Loop over eta
            yt = y - eta*Ly; % Temporarily updated y.
            
            % Computing Lt=L(yt):

            [G, ~, ~, yc, Scy] = ...
                FigureOut_G_Gy_and_Gyy(nsy,Iy,type_g,ncy,yt, true, yc, Scy, gamma_y,false,false);
            G=G-sy;
            for j=1:size(G,2)
                G(:,j)=G(:,j)/nG(j);
                %                     for k=1:length(Iy{h})
                %                         Gy{h}{k}(:,j)=Gy{h}{k}(:,j)/nG{h}(j);
                %                     end
            end
            Qy=G*By;
            
            [~,at,bt,fgt,~,~] = L_y(Qz,Qy,m,dy,Gy,By,K_max,true);
            
%             if(~call_it_a_day)
%                 [~,at,bt,fgt,ft,~] = L_y(Qz,Qy,b,nk,m,dy,Gy,By,K,true,true);
%             else
%                 [~,fgt,ft] = L_y_final(Qz,Qy,a,b,m,dy,Gy,By,K,true);
%             end
            
%             [~,at,bt,fgt,ft,~] = L_y(Qz,Qy,b,nk,m,dy,Gy,By,K,false,false);
            fgt2=sum(fgt.^2);
            
            C = sum(sum((x-yt).^2))/2;

            Lt=C+lambda*fgt2;

            DL=Lold-Lt; % Actual descent.
            %             [DL dL_exp*eta]

            if(DL >= dL_exp*eta)
                GoodEnough=true;
            end

            if(~GoodEnough)
                if(eta < eta_min)
                    GoodEnough=true;
                    call_it_a_day=true;
%                             L_eta_new;
%                             stop
                else
                    eta=eta/2;
                end
            end
        end

        %         if(call_it_a_day)
        %             done=true;
        %         else
        a=at;
        b=bt;
        fg=fgt;

        np=np+1;

        y=yt;
        nLp(np)=sqrt(Ly2/m)/stx;
        Cp(np)=C;
        Tp(np)=lambda*fgt2;
        TTp(np)= sig1;
        etap(np)=eta;

        n=n+1;
        eta=min(1.1*eta,eta_max); % Increase slightly eta for the next step.

        ymyo=sum(sum((y-yo).^2))/m;
        
        if(dependent)
            dependent=(max(sig1)>gamma*delta)>0.01;
        end
        
        done= (~dependent && nLp(np)<0.05) | n >= nmax | ymyo > 0.1*varyo;
        %         end
        
        if(done)
            K
            if(call_it_a_day)
                'call_it_a_day'
                np
%                 not_converged=false;
%                 'converged'
            end
            if((nLp(np) < 0.05) && (~dependent))
                not_converged=false;
                'converged'
            else
                if(ymyo > 0.1*varyo)
                    'ymyo'
                end
            end
        end
    end
    f=Qz*a;
end

I=1:np;
nLp=nLp(I);
Cp=Cp(I);
Tp=Tp(I);
TTp=TTp(I);
etap=etap(I);

% To look at the answers, run Plot_Results and/or Invert_and_Plot

