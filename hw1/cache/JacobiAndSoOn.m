% function main()
%     computeTime=zeros(2,4);
%     size=[10,50,100,200];
%     %每种计算1000个
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
%     legend('高斯消去法','列主元消去法');
% end
[times1,time1,times2,time2,times3,time3,times4,time4]=produceSolveComputeTime(200)
function [times1,time1,times2,time2,times3,time3,times4,time4] = produceSolveComputeTime(size)
    %利用随机对角矩阵和随机正交矩阵生成随机对称正定矩阵A，同时满足2D-A正定
    while 1
        v=diag(rand(size,1));
        u=orth(randn(size));
        A=u'*v*u;
        %p1为0时A正定
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
        %精度，用无穷范数（向量的所有元素的绝对值中最大的）
         if norm(x-x_k,inf)<1e-5
            break;
        end
        x_k=x;
        %迭代次数限制
        times=times+1;
        if times==29000
            error('超出迭代次数限制');
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
        %精度，用无穷范数（向量的所有元素的绝对值中最大的）
         if norm(x-x_k,inf)<1e-5
            break;
        end
        x_k=x;
        %迭代次数限制
        times=times+1;
        if times==29000
            error('超出迭代次数限制');
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
        %精度，用无穷范数（向量的所有元素的绝对值中最大的）
        if norm(x-x_k,inf)<1e-5
            break;
        end
        x_k=x;
        %迭代次数限制
        times=times+1;
        if times==29000
            error('超出迭代次数限制');
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
        %r=0或(p,Ap)=0，计算停止
        if norm(r,2)<1e-10||dot(p,A*p)<1e-10
            break;
        end
        a=dot(r,r)/dot(p,A*p);
        x=x+a*p;
        r_kp1=r-a*A*p;
        beta=dot(r_kp1,r_kp1)/dot(r,r);
        p=r_kp1+beta*p;
        r=r_kp1;
        %迭代次数限制
        times=times+1;   
        if times==29000
            error('超出迭代次数限制');
        end
    end
end