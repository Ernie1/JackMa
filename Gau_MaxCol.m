main();

function main()
    computeTime=zeros(2,4);
    size=[10,50,100,200];
    %每种计算1000个
    times=1000;
    for j=1:4
        for i=1:times
            [time1,time2]=produceSolveComputeTime(size(j));
            computeTime(1,j)=computeTime(1,j)+time1;
            computeTime(2,j)=computeTime(2,j)+time2;
        end
    end
    %平均
    computeTime=computeTime/times;
    plot(size(:),computeTime(1,:),'-',size(:),computeTime(2,:),'-');
    xlim([0,210]);
    legend('高斯消去法','列主元消去法');
end

%随机生成独立同分布的正态分布的随机A和b，解，返回时间
function [time1,time2] = produceSolveComputeTime(size)
    A=randn(size);
    x=randn(size,1);
    b=A*x;
    tic;
    GaussianElimination(A,b);
    time1=toc;
    tic;
    EliminationWithMaximalColumnPivoting(A,b);
    time2=toc;
end

%高斯消去法
function x = GaussianElimination(A,b);
    dim=size(A,1);
    %消去
    for i=1:dim
        if A(i,i)==0
            error('主元素=0，消去法无法进行');
            return;
        end
        for j=i+1:dim            
            m=A(j,i)/A(i,i);
            %A(j,:)=A(j,:)-m*A(i,:);
            A(j,i+1:dim)=A(j,i+1:dim)-m*A(i,i+1:dim);
            b(j)=b(j)-m*b(i);
        end
    end
    
    %回代
    x=zeros(dim,1);
    x(dim)=b(dim)/A(dim,dim);
    for i=dim-1:-1:1
        x(i)=(b(i)-A(i,i+1:dim)*x(i+1:dim))/A(i,i);
    end
end

%列主元消去法
function x = EliminationWithMaximalColumnPivoting(A,b);
    dim=size(A,1);
    
    %消去
    for i=1:dim
        %选最大
        mcp=find(abs(A(i:dim,i))==max(abs(A(i:dim,i))))+i-1;
        if A(mcp,i)==0
            error('主元素=0，消去法无法进行');
            return;
        end
        %交换
        tem=A(mcp,:);
        A(mcp,:)=A(i,:);
        A(i,:)=tem;
        tem=b(mcp);
        b(mcp)=b(i);
        b(i)=tem;
        for j=i+1:dim
            m=A(j,i)/A(i,i);
            %A(j,:)=A(j,:)-m*A(i,:);
            A(j,i+1:dim)=A(j,i+1:dim)-m*A(i,i+1:dim);
            b(j)=b(j)-m*b(i);
        end
    end
    
    %回代
    x=zeros(dim,1);
    x(dim)=b(dim)/A(dim,dim);
    for i=dim-1:-1:1
        x(i)=(b(i)-A(i,i+1:dim)*x(i+1:dim))/A(i,i);
    end
end