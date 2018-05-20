produceSolveTimes(10,50);
produceSolveTimes(50,100);
produceSolveTimes(100,150);
produceSolveTimes(200,200);
%limit: ������������
function produceSolveTimes(size,limit)
    %ÿ�ּ���1000��
    times=1000;
    timesArr1=zeros(1,limit);
    timesArr2=zeros(1,limit);
    timesArr3=zeros(1,limit);
    timesArr4=zeros(1,limit);
    for i=1:times
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
        timesArr1=timesArr1+Jacobi(A,b,x,limit);
        timesArr2=timesArr2+GaussSeidel(A,b,x,limit);
        timesArr3=timesArr3+SOR(A,b,1.23,x,limit);
        timesArr4=timesArr4+CG(A,b,x,limit);
    end
    %ƽ��
    timesArr1=timesArr1/times;
    timesArr2=timesArr2/times;
    timesArr3=timesArr3/times;
    timesArr4=timesArr4/times;
    figure;
    plot(timesArr1(1,:),'-');
    hold on;
    plot(timesArr2(1,:),'-');
    hold on;
    plot(timesArr3(1,:),'-');
    hold on;
    plot(timesArr4(1,:),'-');
    legend('Jacobi','Gauss-Seidel','SOR','CG');
    xlim([0,limit+1]);
end

function timesArr=Jacobi(A,b,xSharp,limit)
    timesArr=zeros(1,limit);
    dim=size(A,1);
    x=zeros(dim,1);
    D=diag(diag(A));
    %J=D^(-1)*(L+U)
    J=D\(-tril(A,-1)-triu(A,1));
    %f=D^(-1)*b
    f=D\b;
    times=0;
    while 1
        x=J*x+f;
        %������������
        times=times+1;
        timesArr(times)=abs(x-xSharp)'/abs(xSharp)';
        if times==limit
            break;
        end
    end
end

function timesArr=GaussSeidel(A,b,xSharp,limit)
    timesArr=zeros(1,limit);
    dim=size(A,1);
    x=zeros(dim,1);
    D=diag(diag(A));
    %G=(D-L)^(-1)*U
    G=(D+tril(A,-1))\(-triu(A,1));
    %f=D^(-1)*b
    f=(D+tril(A,-1))\b;
    times=0;
    while 1
        x=G*x+f;
        %������������
        times=times+1;
        timesArr(times)=abs(x-xSharp)'/abs(xSharp)';
        if times==limit
            break;
        end
    end
end

function timesArr=SOR(A,b,w,xSharp,limit)
    timesArr=zeros(1,limit);
    dim=size(A,1);
    x=zeros(dim,1);
    D=diag(diag(A));
    L=-tril(A,-1);
    U=-triu(A,1);
    %Lw=(D-wL)^(-1)((1-w)D+wU)
    Lw=(D-w*L)\((1-w)*D+w*U);
    %f=w(D-wL)^(-1)b
    f=w*(D-w*L)^(-1)*b;
    times=0;
    while 1
        x=Lw*x+f;
        %������������
        times=times+1;
        timesArr(times)=abs(x-xSharp)'/abs(xSharp)';
        if times==limit
            break;
        end
    end
end

function timesArr=CG(A,b,xSharp,limit)
    timesArr=zeros(1,limit);
    dim=size(A,1);
    x=zeros(dim,1);
    %r0=b-A*x0
    r=b-A*x;
    %p0=r0
    p=r;
    times=0;
    while 1
        a=dot(r,r)/dot(p,A*p);
        x=x+a*p;
        r_kp1=r-a*A*p;
        beta=dot(r_kp1,r_kp1)/dot(r,r);
        p=r_kp1+beta*p;
        r=r_kp1;
        %������������
        times=times+1;
        timesArr(times)=abs(x-xSharp)'/abs(xSharp)';
        if times==limit
            break;
        end
    end
end