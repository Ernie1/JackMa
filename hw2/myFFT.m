% time=zeros(4,25);
% precision=zeros(4,25);
% % % % 
% a=10;
% b=11;
% f=@(x) x^2-115;
% count=0;
% tic;
% while 1
%     count=count+1;
%     if f((a+b)/2)==0
%         res=(a+b)/2;
%         break;
%     end
%     time(1,count)=toc;
%     precision(1,count)=abs(a-b);
%     if abs(a-b)<=5e-7
%         res=(a+b)/2;
%         break;
%     end
%     if f(a)*f((a+b)/2)<0
%         b=(a+b)/2;
%     else
%         a=(a+b)/2;
%     end
% end
% % % % 
% f=@(x) x^2-115;
% fd=@(x) 2*x;
% xk=11;
% count=0;
% tic;
% while 1
%    x=xk-f(xk)/fd(xk);
%    count=count+1;
%    time(2,count)=toc;
%    precision(2,count)=abs(x-xk);
%    if abs(x-xk)<5e-7
%        break;
%    end
%    xk=x;
% end
% % % % 
% f=@(x) x^2-115;
% fd=@(x) 2*x;
% xk=11;
% fd0=fd(xk);
% count=0;
% tic;
% while 1
%    x=xk-f(xk)/fd0;
%    count=count+1;
%    time(3,count)=toc;
%    precision(3,count)=abs(x-xk);
%    if abs(x-xk)<5e-7
%        break;
%    end
%    xk=x;
% end
% % % % 
% f=@(x) x^2-115;
% xkm1=10;
% xk=11;
% count=0;
% tic;
% while 1
%    x=xk-f(xk)/(f(xk)-f(xkm1))*(xk-xkm1);
%    count=count+1;
%    time(4,count)=toc;
%    precision(4,count)=abs(x-xk);
%    if abs(x-xk)<5e-7
%        break;
%    end
%    xkm1=xk;
%    xk=x;
% end
% 
% 
% count=[1:10];
% plot(count,precision(1,1:10),'-',count,precision(2,1:10),'-',count,precision(3,1:10),'-',count,precision(4,1:10),'-');
% legend('���ַ�','ţ�ٷ�','��ţ�ٷ�','�ҽط�');
% times=0;

% A=[1.1,1.1^2,1.1^3;1.2,1.2^2,1.2^3;1.3,1.3^2,1.3^3;1.4,1.4^2,1.4^3;1.5,1.5^2,1.5^3;];
% precision=zeros(1,10000);
% % ����100�鷽��
% for j=1:1
%     A=randn(10000,10);
%     b=rand(10000,1);
%     x=rand(10,1);
%     P=1e10*eye(10);
%     for k=1:10000
%         Q=P*A(k,:)'/(1+A(k,:)*P*A(k,:)');
%         xk=x+Q*(b(k)-A(k,:)*x);
%         P=(eye(10)-Q*A(k,:))*P;
%         % ����������x�ĵ�ǰ����һ���Ĳ�Ķ���������
%         precision(k)=precision(k)+norm(xk-x,2);
%         x=xk;
%     end
% end
% precision=precision/1;
% max_y=max(precision);
% max_x=find(precision==max_y);
% figure;
% plot(1:10000,precision(1,:),'-');
% line([max_x,max_x],[0,max_y],'linestyle',':');
% set(gca,'XTick',[0,max_x,10000]);

% f=@(x) x^4-3*x^3+2*x^2-tan(x*(x-2));
% x=zeros(8,1);
% tem=0.125;
% for i=1:8
%     x(i)=f(tem);
%     tem=tem+0.25;
% end
% a=fft(x);
% S=@(x) real(a(1))+real(a(2))cosx+real(a(2))cos(x)
% fft([1;2;-1;3;]);
% s=@(x) 5/2-2/2*cos(x)-1/2*sin(x)-5/2*cos(2*x)-cos(3*x)+1/2*sin(3*x);
% s(1)


% T=1;                                        %����ʱ���յ�
% N=1024;                                      %����N������ʱ��
% t=linspace(0,T,N);
% x=12*sin(2*pi*10*t+pi/4)+5*cos(2*pi*40*t);  %�������������ֵx
% dt=t(2)-t(1);                               %��������
% f=1/dt;                                     %����Ƶ��
% X=mFFT(x,N);                                %����x�Ŀ��ٸ���Ҷ�任X
% F=X(1:N/2+1);                               %���|F(k)|����N/2�Գƣ�ֻ��ʾ0~N/2
% f=f*(0:N/2)/N;                              %ʹƵ����f��0��ʼ
% subplot(1,2,1)                              %����Ƶ��-���ͼ
% plot(f,abs(F),'-*');
% xlabel('Frequency');
% ylabel('|F(k)|');
% ix=real(ifft(X));                           %��X�Ŀ��ٸ���Ҷ��任������ԭ�������бȽ�
% subplot(1,2,2)                              %������任�����ԭ����ͼ����бȽ�
% plot(t,x,'r:',t,ix,'g*')
% 
% function X=mFFT(x,N)
%     Xl=zeros(1,N);
%     X=zeros(1,N);
%     W=@(n) exp(-2i*pi/n);
%     % �������������е�λ��任
%     wide=log2(N);
%     for i=0:N-1
%         Xl(i+1)=x(bin2dec(reverse(dec2bin(i,wide)))+1);
%     end
%     for n=1:wide
%         curN=2^n;
%         start=curN/2+1;
%         i=1;
%         while i<=N
%             % ��������
%             curNh=curN/2;
%             for loop=0:curNh-1
%                 X(i)=Xl(start+loop-curNh)+W(curN)^(loop)*Xl(start+loop);
%                 i=i+1;
%             end
%             for loop=0:curNh-1
%                 X(i)=Xl(start+loop-curNh)-W(curN)^(loop)*Xl(start+loop);
%                 i=i+1;
%             end
%             start=start+curN;
%         end
%         Xl=X;
%     end
% end

% ����ø������ι�ʽ�븴������ɭ��ʽ������ sin(x)/x ��[0, 1]��Χ�ڵĻ��֡���������ĿΪ 5��9��17��33��
% ��������
% f=@(x) if isnan(sin(x)/x)
% T(@f,5) % 0.94451352166538957155772493479162
% T(@f,9) % 0.94569086358270136594139643193557
% T(@f,17) % 0.94598502993438620539690120381238
% T(@f,33) % 0.94605856096276816011303445463909
% function y=f(x)
%     temf=@(x) sin(x)/x;
%     y=temf(x);
%     if isnan(y)
%         syms temx;
%         y=limit(sin(temx)/temx,temx,x);
%     end
% end
% function I=T(f,n)
%     x=linspace(0,1,n);
%     h=(1-0)/(n-1);
%     I=0;
%     for i=2:(n-1)
%         I=I+f(x(i));
%     end
%     I=vpa(h/2*(f(x(1))+2*I+f(x(n))));
% end
% ��������ɭ��ʽ
% S(@f,5) % 0.9460869339517937182115758787404
% S(@f,9) % 0.94608331088847189005441862263979
% S(@f,17) % 0.94608308538494759650413120274
% S(@f,33) % 0.94608307130556207100354389657089
% function y=f(x)
%     temf=@(x) sin(x)/x;
%     y=temf(x);
%     if isnan(y)
%         syms temx;
%         y=limit(sin(temx)/temx,temx,x);
%     end
% end
% function I=S(f,n)
%     x=linspace(0,1,n);
%     h=2*(1-0)/(n-1);
%     I=0;
%     for i=2:2:(n-1)
%         I=I+f(x(i));
%     end
%     temI=0;
%     for i=3:2:(n-2)
%         temI=temI+f(x(i));
%     end
%     I=vpa(h/6*(f(x(1))+4*I+2*temI+f(x(n))));
% end

% x=(0:0.1:1);
% dy=@(x,y) y-2*x/y;
% % ǰ��ŷ����
% y1=zeros(1,11);
% y1(1)=1;
% for i=2:11
%     y1(i)=y1(i-1)+0.1*dy(x(i-1),y1(i-1));
% end
% % ����ŷ����
% y2=zeros(1,11);
% y2(1)=1;
% for i=2:11
%     y2(i)=y2(i-1)+0.1*dy(x(i-1),y2(i-1));
%     while 1
%         ynp1=y2(i-1)+0.1*dy(x(i),y2(i));
%         if abs(ynp1-y2(i))<5e-6
%             y2(i)=ynp1;
%             break;
%         end
%         y2(i)=ynp1;
%     end
% end
% % ���η���
% y3=zeros(1,11);
% y3(1)=1;
% for i=2:11
%     y3(i)=y3(i-1)+0.1*dy(x(i-1),y3(i-1));
%     while 1
%         ynp1=y3(i-1)+0.1/2*(dy(x(i-1),y3(i-1))+dy(x(i),y3(i)));
%         if abs(ynp1-y3(i))<5e-6
%             y3(i)=ynp1;
%             break;
%         end
%         y3(i)=ynp1;
%     end
% end
% % �Ľ�ŷ������
% y4=zeros(1,11);
% y4(1)=1;
% for i=2:11
%     y4(i)=y4(i-1)+0.1*dy(x(i-1),y4(i-1));
%     yc=y4(i-1)+0.1*dy(x(i),y4(i));
%     y4(i)=(y4(i)+yc)/2;
% end
% % ����
% uitable(gcf,'Data',[x',y1',y2',y3',y4'],'Position',[20 20 410 210],'Columnname',["x";"y1";"y2";"y3";"y4"]);


