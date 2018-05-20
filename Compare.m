%�Ƚ� Jacobi ��������Gauss-Seidel ����������γ��ɳڵ������������ݶȷ����˹��ȥ��������Ԫ��ȥ���ļ���ʱ�䡣
compareSolveTime();
function compareSolveTime()
    size=[10,50,100,200];
    %ÿ�ּ���100��
    times=100;
    timeArr1=zeros(1,4);
    timeArr2=zeros(1,4);
    timeArr3=zeros(1,4);
    timeArr4=zeros(1,4);
    timeArr5=zeros(1,4);
    timeArr6=zeros(1,4);
    for j=1:4
        for i=1:times
            %��������ԽǾ�����������������������Գ���������A��ͬʱ����2D-A����
            while 1
                v=diag(rand(size(j),1));
                u=orth(randn(size(j)));
                A=u'*v*u;
                %p1Ϊ0ʱA����
                [R,p1]=chol(A);
                %2D-A
                [R,p2]=chol(2*diag(diag(A))-A);
                if p1==0&&p2==0
                    break;
                end
            end
            x=rand(size(j),1);
            b=A*x;
            tic;
            Jacobi(A,b);
            time1=toc;
            tic;
            GaussSeidel(A,b);
            time2=toc;
            tic;
            SOR(A,b,1.23);
            time3=toc;
            tic;
            CG(A,b);
            time4=toc;
            tic;
            GaussianElimination(A,b);
            time5=toc;
            tic;
            EliminationWithMaximalColumnPivoting(A,b);
            time6=toc;
            timeArr1(j)=timeArr1(j)+time1;
            timeArr2(j)=timeArr2(j)+time2;
            timeArr3(j)=timeArr3(j)+time3;
            timeArr4(j)=timeArr4(j)+time4;
            timeArr5(j)=timeArr5(j)+time5;
            timeArr6(j)=timeArr6(j)+time6;
        end
    end
    %ƽ��
    timeArr1=timeArr1/times;
    timeArr2=timeArr2/times;
    timeArr3=timeArr3/times;
    timeArr4=timeArr4/times;
    timeArr5=timeArr5/times;
    timeArr6=timeArr6/times;
    plot(size(:),timeArr1(:),'-',size(:),timeArr2(:),'-',size(:),timeArr3(:),'-',size(:),timeArr4(:),'-',size(:),timeArr5(:),'-',size(:),timeArr6(:),'-');
    xlim([0,210]);
    set(gca,'XTick',[10,50,100,200]);
    legend('Jacobi','Gauss-Seidel','SOR','CG','GaussianElimination','EliminationWithMaximalColumnPivoting');
end

function Jacobi(A,b)
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
        if times==290000
            error('����������������');
        end
    end
end

function GaussSeidel(A,b)
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
        if times==290000
            error('����������������');
        end
    end
end

function SOR(A,b,w)
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
        if times==290000
            error('����������������');
        end
    end
end

function CG(A,b)
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
        if times==290000
            error('����������������');
        end
    end
end

function GaussianElimination(A,b)
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

function EliminationWithMaximalColumnPivoting(A,b)
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