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

produceSolveComputeTime(500);

function time1 = produceSolveComputeTime(size)
    while 1
        %A=sprandn(size,size,0.4);
        v=diag(rand(1,size));
        u=orth(randn(size));
        A=u*v*u^-1;
%         vrho(A)%特征值
        [D p1]=chol(A);
        p1
        %tem=2D-A
        tem=2*diag(diag(A))-A;
        [D p2]=chol(tem);
        p2
        if p1==0&&p2==0
            break;
        end
    end
        x=rand(size,1);
%         x=randn(size,1);
        b=A*x;
%  A=[4,    -1,    -1,     0;
% 
%     -1,     4,     0,    -1;
% 
%     -1,     0,     4,    -1;
% 
%      0,    -1,    -1,     4];
%  b=[1;2;0;1];
        [x,hasSol,time]=Jacobi(A,b)
%         if hasSol==true
%             break;
%         end
%      tic;
%     Jacobi(A,b)
%      time1=toc;
end

function [x,hasSol,time]=Jacobi(A,b)
    dim=size(A,1);
    x=zeros(dim,1);
    D=diag(diag(A));
    %J=D^(-1)*(L+U);
    J=D\(-tril(A,-1)-triu(A,1));
    hasSol=true;
%     try
%         if max(abs(eig(J)))>=1
%             hasSol=false;
%             return;
%         end
%     catch
%         hasSol=false;
%         return;
%     end
    f=D\b;
    x_k=zeros(dim,1);
    time=0;
    while 1
        x=J*x_k+f;
        %精度，用无穷范数
         if norm(x-x_k,inf)<1e-5
            break;
        end
        x_k=x;
        %迭代次数限制
        time=time+1;
        if time==2000000000
            error('超出迭代次数限制');
        end
    end
end