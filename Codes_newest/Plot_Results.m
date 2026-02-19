function PR = Plot_Results(x, y, z, nLp, Cp, Tp, TTp,etap,epsi,Fz)
[~, dy]=size(x);
dz = length(z);

PR = true;
% C = sum(sum((x-y).^2))/(2*m);
% P=sum(fg.^2);
% P=sum(fg2);
% Lt=C+lambda*P;
% for h=1:H
%     f=Qz*a;
%     g=Qy*b;
% end

both=false; % x and y superimposed.

figure(8)
if(dy==1&&dz==1)
    if both
        plot(z{1},x,'b*')
        %scatter(z{1},x)
        hold on
        xlabel('z')
        ylabel('x, y')
        plot(z{1},y,'r*')
        hold off
        title('x in blue, y in red')

        %     plot(z{1},x,'b*')
        %     hold on
    else
        %figure(1)
        %v=axis;
        figure(8), clf
        %plot(y,'r.')
        %axis equal
        %plot(x,y,'r*')

        plot(x,'b')
        hold on
        plot(y,'r')
        %hold on
        %plot(Fz,'g')
        title('x in blue, y in red, gays')
        %axis equal
        %axis(v);
        %title('y')
        %xlabel('x')
        %ylabel('y')
    end
end
if(dy==1&&dz>=2)
    subplot(211)
    plot3(z{1},z{2},x,'*b')
    xlabel('z1')
    ylabel('z2')
    zlabel('x, y')
    if both
        hold on
        title('x in blue, y in red')
    else
        v = axis;
        title('y')
    end
    plot3(z{1},z{2},y,'*r')
    axis(v);
    hold off


    min1=min(z{1}); max1=max(z{1}); d1=(max1-min1)/20; z11=min1:d1:max1;
    min2=min(z{2}); max2=max(z{2}); d2=(max2-min2)/20; z22=min2:d2:max2;
    [z1,z2]=meshgrid(z11,z22);

    yg=griddata(z{1},z{2},y,z1,z2);

    subplot(212)
    contour(z1,z2,yg)
    xlabel('z_1')
    ylabel('z_2')
    title('y(z)')
    colorbar
end
if(dy==2&&dz==1)
    if both
        subplot(311)
        plot(z{1},x(:,1),'b*')
        hold on
        xlabel('z')
        ylabel('x_1, y_1')
        plot(z{1},y(:,1),'r*')
        hold off
        title('x in blue, y in red')
        subplot(312)
        plot(z{1},x(:,2),'b*')
        hold on
        xlabel('z')
        ylabel('x_2, y_2')
        plot(z{1},y(:,2),'r*')
        hold off
        subplot(313)
        plot(x(:,1),x(:,2),'b*')
        hold on
        plot(y(:,1),y(:,2),'r*')
        hold off
        xlabel('y_1')
        ylabel('y_2')
    else
        figure(1)
        subplot(311)
        v=axis;
        figure(8)
        subplot(311)
        plot(z{1},y(:,1),'r*')
        axis(v);
        xlabel('z')
        ylabel('y_1')
        title('y in red')
        figure(1)
        subplot(312)
        v=axis;
        figure(8)
        subplot(312)
        plot(z{1},y(:,2),'r*')
        axis(v);
        xlabel('z')
        ylabel('y_2')
        figure(1)
        subplot(313)
        v=axis;
        figure(8)
        subplot(313)
        plot(y(:,1),y(:,2),'r*')
        axis(v);
        xlabel('y_1')
        ylabel('y_2')
    end
end

% np=length(Cp);
% I=1:np;

%figure(4)
%plot(Cp,'b')
%hold on
%plot(Tp,'r')
%plot(Cp+Tp,'k');
%hold off
%xlabel('step')
%title('Cost (b), Penalty (r), L=Cost + Penalty (k)')

%figure(10)
% for h=1:H
%    plot(TTp);
%     hold on
% end
% hold off
%xlabel('step')
%title('Individual correlations \sigma_k')

%figure(5)
%plot(etap)
%xlabel('step')
%title('Learniong rate \eta')

%figure(6)
%plot(nLp)
%xlabel('step')
%title('|L_y| / std(x)')

%figure(11)
%plot(y)
%ylabel('y')


xdz=x./z{1};
figure(13), clf
scatter(epsi(101:600),y,10,"filled")
%hold on
%scatter(epsi(101:600).^2,x,10,'g')
xlabel('e')
ylabel('y')
axis equal
grid on

% figure(2)
% if(dz==1)
%     plot(z,f,'*')
%     xlabel('z')
%     ylabel('f')
%     hold off
% end
% if(dz==2)
%     subplot(211)
%     plot3(z(:,1),z(:,2),f,'*')
%     xlabel('z1')
%     ylabel('z2')
%     zlabel('f')
%
%     ffg=griddata(z(:,1),z(:,2),f,z1,z2);
%
%     subplot(212)
%     contour(z1,z2,ffg)
%     xlabel('z_1')
%     ylabel('z_2')
%     title('f(z)')
%     colorbar
%     hold off
% end
%
% figure(3)
% if(dy==1)
%     plot(y,g,'*')
%     xlabel('y')
%     ylabel('g')
%     hold off
% end
% if(dy==2)
%     subplot(211)
%     plot3(y(:,1),y(:,2),g,'*')
%     xlabel('y_1')
%     ylabel('y_2')
%     zlabel('g')
%
%     min1=min(y(:,1)); max1=max(y(:,1)); d1=(max1-min1)/20; y11=min1:d1:max1;
%     min2=min(y(:,2)); max2=max(y(:,2)); d2=(max2-min2)/20; y22=min2:d2:max2;
%     [y1,y2]=meshgrid(y11,y22);
%     ffg=griddata(y(:,1),y(:,2),g,y1,y2);
%
%     subplot(212)
%     contour(y1,y2,ffg)
%     xlabel('y_1')
%     ylabel('y_2')
%     title('g(y)')
%     colorbar
%     hold off
% end

