% function main()
%     computeTime=zeros(2,4);
%     size=[10,50,100,200];
%     %ÿ�ּ���1000��
%     times=1;
%     for j=1:4
%         for i=1:times
%             [time1,time2]=produceSolveComputeTime(size(j));
%             computeTime(1,j)=computeTime(1,j)+time1;
%             computeTime(2,j)=computeTime(2,j)+time2;
%         end
%     end
%     computeTime=computeTime/times;
%     plot(size(:),computeTime(1,:),'-',size(:),computeTime(2,:),'-');
%     xlim([0,210]);
%     legend('��˹��ȥ��','����Ԫ��ȥ��');
% end
[times1,time1,times2,time2,times3,time3,times4,time4]=produceSolveComputeTime(200)
function [times1,time1,times2,time2,times3,time3,times4,time4] = produceSolveComputeTime(size)
    %��������ԽǾ�����������������������Գ���������A��ͬʱ����2D-A����
    while 1
        v=diag(rand(size,1));
        u=orth(randn(size));
        A=u'*v*u;
        %p1Ϊ0ʱA����
        [R,p1]=chol(A);
        %2D-A
        [R,p2]=chol(2*diag(diag(A))-A);
        if p1==0&&p2==0
            break;
        end
    end
    x=rand(size,1);
    b=A*x;
    tic;
    [x,times1]=Jacobi(A,b);
    time1=toc;
    tic;
    [x,times2]=GaussSeidel(A,b);
    time2=toc;
    tic;
    [x,times3]=SOR(A,b,1.23);
    time3=toc;
    tic;
    [x1,times4]=CG(A,b);
    time4=toc;
end

function [x,times]=Jacobi(A,b)
    dim=size(A,1);
    x=zeros(dim,1);
    D=diag(diag(A));
    %J=D^(-1)*(L+U)
    J=D\(-tril(A,-1)-triu(A,1));
    %f=D^(-1)*b
    f=D\b;
    x_k=zeros(dim,1);
    times=0;
    while 1
        x=J*x_k+f;
        %���ȣ��������������������Ԫ�صľ���ֵ�����ģ�
         if norm(x-x_k,inf)<1e-5
            break;
        end
        x_k=x;
        %������������
        times=times+1;
        if times==29000
            error('����������������');
        end
    end
end

function [x,times]=GaussSeidel(A,b)
    dim=size(A,1);
    x=zeros(dim,1);
    D=diag(diag(A));
    %G=(D-L)^(-1)*U
    G=(D+tril(A,-1))\(-triu(A,1));
    %f=D^(-1)*b
    f=(D+tril(A,-1))\b;
    x_k=zeros(dim,1);
    times=0;
    while 1
        x=G*x_k+f;
        %���ȣ��������������������Ԫ�صľ���ֵ�����ģ�
         if norm(x-x_k,inf)<1e-5
            break;
        end
        x_k=x;
        %������������
        times=times+1;
        if times==29000
            error('����������������');
        end
    end
end

function [x,times]=SOR(A,b,w)
    dim=size(A,1);
    x=zeros(dim,1);
    D=diag(diag(A));
    L=-tril(A,-1);
    U=-triu(A,1);
    %Lw=(D-wL)^(-1)((1-w)D+wU)
    Lw=(D-w*L)\((1-w)*D+w*U);
    %f=w(D-wL)^(-1)b
    f=w*(D-w*L)^(-1)*b;
    x_k=zeros(dim,1);
    times=0;
    while 1
        x=Lw*x_k+f;
        %���ȣ��������������������Ԫ�صľ���ֵ�����ģ�
        if norm(x-x_k,inf)<1e-5
            break;
        end
        x_k=x;
        %������������
        times=times+1;
        if times==29000
            error('����������������');
        end
    end
end

function [x,times]=CG(A,b)
    dim=size(A,1);
    x=zeros(dim,1);
    %r0=b-A*x0
    r=b-A*x;
    %p0=r0
    p=r;
    times=1;
    while 1
        %r=0��(p,Ap)=0������ֹͣ
        if norm(r,2)<1e-10||dot(p,A*p)<1e-10
            break;
        end
        a=dot(r,r)/dot(p,A*p);
        x=x+a*p;
        r_kp1=r-a*A*p;
        beta=dot(r_kp1,r_kp1)/dot(r,r);
        p=r_kp1+beta*p;
        r=r_kp1;
        %������������
        times=times+1;   
        if times==29000
            error('����������������');
        end
    end
end