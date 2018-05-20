main();

function main()
    computeTime=zeros(2,4);
    size=[10,50,100,200];
    %ÿ�ּ���1000��
    times=1000;
    for j=1:4
        for i=1:times
            [time1,time2]=produceSolveComputeTime(size(j));
            computeTime(1,j)=computeTime(1,j)+time1;
            computeTime(2,j)=computeTime(2,j)+time2;
        end
    end
    %ƽ��
    computeTime=computeTime/times;
    plot(size(:),computeTime(1,:),'-',size(:),computeTime(2,:),'-');
    xlim([0,210]);
    legend('��˹��ȥ��','����Ԫ��ȥ��');
end

%������ɶ���ͬ�ֲ�����̬�ֲ������A��b���⣬����ʱ��
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

%��˹��ȥ��
function x = GaussianElimination(A,b);
    dim=size(A,1);
    %��ȥ
    for i=1:dim
        if A(i,i)==0
            error('��Ԫ��=0����ȥ���޷�����');
            return;
        end
        for j=i+1:dim            
            m=A(j,i)/A(i,i);
            %A(j,:)=A(j,:)-m*A(i,:);
            A(j,i+1:dim)=A(j,i+1:dim)-m*A(i,i+1:dim);
            b(j)=b(j)-m*b(i);
        end
    end
    
    %�ش�
    x=zeros(dim,1);
    x(dim)=b(dim)/A(dim,dim);
    for i=dim-1:-1:1
        x(i)=(b(i)-A(i,i+1:dim)*x(i+1:dim))/A(i,i);
    end
end

%����Ԫ��ȥ��
function x = EliminationWithMaximalColumnPivoting(A,b);
    dim=size(A,1);
    
    %��ȥ
    for i=1:dim
        %ѡ���
        mcp=find(abs(A(i:dim,i))==max(abs(A(i:dim,i))))+i-1;
        if A(mcp,i)==0
            error('��Ԫ��=0����ȥ���޷�����');
            return;
        end
        %����
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
    
    %�ش�
    x=zeros(dim,1);
    x(dim)=b(dim)/A(dim,dim);
    for i=dim-1:-1:1
        x(i)=(b(i)-A(i,i+1:dim)*x(i+1:dim))/A(i,i);
    end
end