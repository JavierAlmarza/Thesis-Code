function PI = Plot_Inverse(xs,z_star,exact_available, data_case, dy)

PI = true;

dz = length(z_star);
n=length(xs);
nh=round(sqrt(n));

if(dy==1)
    figure(20),clf
    hist(xs,nh)
    xlabel('x')
    thing={'\rho(x|z) for z = ( '};
    for k=1:dz-1
        thing = strcat(thing, int2str(z_star{k}), ' ,  ');
    end
    thing = strcat(thing, int2str(z_star{dz}), ' )');
    title(thing)
end
if(dy==2)
    figure
%     subplot(311)
%     histogram(xs(:,1),30)
%     xlabel('x_1')
%     thing={'\rho(x_1|z) for z = ( '};
%     for k=1:dz-1
%         thing = strcat(thing, string(z_star(k)), ' ,  ');
%     end
%     thing = strcat(thing, string(z_star(dz)), ' )');
%     title(thing)
%     subplot(312)
%     histogram(xs(:,2),30)
%     xlabel('x_2')
%     thing={'\rho(x_2|z) for z = ( '};
%     for k=1:dz-1
%         thing = strcat(thing, string(z_star(k)), ' ,  ');
%     end
%     thing = strcat(thing, string(z_star(dz)), ' )');
%     title(thing)
%     subplot(313)
    scatter(xs(:,1), xs(:,2), 20, 'filled')
    v=axis;
end

if(exact_available)

    switch data_case
        case 'first'
            x_pl=-2:0.05:2;
            mu=cos(2*pi*z_star{1});
            sigma=0.05*(1./(sin(0.1*(z_star{1}+0.2).^2)+0.25));
            rho_pl=pdf('Normal',x_pl,mu,sigma);
            figure
            plot(x_pl,rho_pl,'*')
            xlabel('x')
            ylabel('\rho')
        case 'second'
            x_pl=-2:0.01:2;
            mu=cos(2*pi*z_star{1}) + sin(pi*z_star{2});
            sigma= 0.2*(1-2*z_star{1}).^(1/2).*(1-2*z_star{2}).^(1/2);
%             mu=cos(2*pi*z_star{1}) + z_star{2};
%             sigma=0.2;
            rho_pl=pdf('Normal',x_pl,mu,sigma);
            figure
            plot(x_pl,rho_pl,'*')
            xlabel('x')
            ylabel('\rho')
        case 'third'
            x1_pl=-1.5:0.05:1.5;
            x2_pl=x1_pl;
            mu1=cos(2*pi*z_star{1});
            sigma1=0.02*(1./(z_star{1}^2+0.05));
            mu2=sin(2*pi*z_star{1});
            sigma2=0.02*(1./(z_star{1}^2+0.05));
            rho1_pl=pdf('Normal',x1_pl,mu1,sigma1);
            rho2_pl=pdf('Normal',x2_pl,mu2,sigma2);
            rho12_pl=rho2_pl'*rho1_pl;
            figure
            subplot(211)
            plot(x1_pl,rho1_pl,'*')
            xlabel('x1')
            ylabel('\rho_1')
            subplot(212)
            plot(x2_pl,rho2_pl,'*')
            xlabel('x2')
            ylabel('\rho_2')
        case 'fourth'
            for k=1:nc_gmm
                Si(:,:,k)=S_gmm{k,1}+S_gmm{k,2}*z_star^2;
                mui(k,:)=mu_gmm{k,1}+mu_gmm{k,2}*z_star^2;
                pii(k)=pi_gmm(k,2)+pi_gmm(k,2)*z_star^2;
            end
            pii=pii/sum(pii);
            GM=gmdistribution(mui,Si,pii);

            x1_pl=min(x(:,1)):0.2:max(x(:,1));
            x2_pl=min(x(:,2)):0.2:max(x(:,2));
            n1=length(x1_pl); n2=length(x2_pl);
            xpl=zeros(n1*n2,2);
            for i=1:n1
                for j=1:n2
                    xpl((j-1)*n1+i,:)=[x1_pl(i) x2_pl(j)];
                end
            end
            rho_pl=pdf(GM,xpl);
            rho_pl=reshape(rho_pl,n1,n2);
            xpl1=reshape(xpl(:,1),n1,n2);
            xpl2=reshape(xpl(:,2),n1,n2);
            figure
            contour(xpl1,xpl2,rho_pl,200)
        case 'seventh'
            x_pl=-3:0.05:3;
            mu=(z_star{1}-1).*cos(pi*cos(z_star{2}+pi/6).^2) + (2-z_star{1}).*cos(z_star{2}).^2;
            sigma=0.2;
            rho_pl=pdf('Normal',x_pl,mu,sigma);
            figure
            plot(x_pl,rho_pl,'*')
            xlabel('x')
            ylabel('\rho')
        case 'eighth'
            x_pl=-2:0.05:2;
            mu=cos(pi*cos(z_star{1}+pi/6).^2);
            sigma=0.2;
            rho_pl=pdf('Normal',x_pl,mu,sigma);
            figure
            plot(x_pl,rho_pl,'*')
            xlabel('x')
            ylabel('\rho')
        case 'ninth'
            x_pl=-2:0.05:2;
            mu=cos(pi*(z_star{1}-sin(2*abs(z_star{2}).*z_star{2})));
            sigma=0.2;
            rho_pl=pdf('Normal',x_pl,mu,sigma);
            figure
            plot(x_pl,rho_pl,'*')
            xlabel('x')
            ylabel('\rho')

        case 'eleventh'
            x_pl=-5:0.05:5;
            mu=ampl_signal*cos(pi*(z_star{1}-sin(2*abs(z_star{2}).*z_star{2})));
            sigma=0.2;
            rho_pl=pdf('Normal',x_pl,mu,sigma);
            figure
            plot(x_pl,rho_pl,'*')
            xlabel('x')
            ylabel('\rho')

        case 'twelfth'
%             x_pl=(-4:0.05:8)';

            mui(1,1) = 3 + 2*z_star{1};
            mui(2,1) = -z_star{1}.^2 + z_star{1}/2;
            Si(1,1,1) = 0.5*exp(z_star{1});
%             Si(1,1,2) = 0.2 - 0.1*z_star{1}(1);
            Si(1,1,2) = 0.25 - 0.1*z_star{1}(1);
            pii(1) = 1; % (1.5+z_star{1})^2;
            pii(2) = 1;

            x_left=mui(2,1)-4*sqrt(Si(1,1,2));
            x_right=mui(1,1)+4*sqrt(Si(1,1,1));

            x_pl=(x_left:0.05:x_right)';

            pii=pii/sum(pii);
            GM=gmdistribution(mui,Si,pii);
            rho_pl=pdf(GM,x_pl);

            figure
            plot(x_pl,rho_pl,'*')
            xlabel('x')
            ylabel('\rho')

        case 'thirteenth'

            mui(1,1) = 3 + 2.*z_star{1};
            mui(1,2) = 2 + z_star{1};
            mui(2,1) = -z_star{1}.^2+z_star{1}/2;
            mui(2,2) = -3;
            Si(1,1,1) = 0.5*exp(z_star{1});
            Si(2,2,1)=0.5;
            Si(1,1,2) = 0.25 - 0.1*z_star{1};
            Si(2,2,2)=0.25 + 0.1*z_star{1};
            pii(1) = 1;
            pii(2) = 1;

            pii=pii/sum(pii);
            GM=gmdistribution(mui,Si,pii);

            x1_pl=min(xs(:,1)):0.2:max(xs(:,1));
            x2_pl=min(xs(:,2)):0.2:max(xs(:,2));
            n1=length(x1_pl); n2=length(x2_pl);
            xpl=zeros(n1*n2,2);
            for i=1:n1
                for j=1:n2
                    xpl((j-1)*n1+i,:)=[x1_pl(i) x2_pl(j)];
                end
            end
            rho_pl=pdf(GM,xpl);
            rho_pl=reshape(rho_pl,n1,n2);
            xpl1=reshape(xpl(:,1),n1,n2);
            xpl2=reshape(xpl(:,2),n1,n2);
            figure
            contour(xpl1,xpl2,rho_pl,50)
            axis(v);

    end
end
