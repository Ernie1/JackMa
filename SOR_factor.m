% ʵ��Jacobi ��������Gauss-Seidel ����������γ��ɳڵ������������ݶȷ���
% AΪ�Գ���������������ֵ���Ӷ���ͬ�ֲ���[0,1]��ľ��ȷֲ�;b�е�Ԫ�ط��Ӷ���ͬ�ֲ�����̬�ֲ����� n=10��50��100��200���ֱ���Ƴ��㷨���������ߣ�������Ϊ����������������Ϊ�����
produceSolveTimes(10);
produceSolveTimes(50);
produceSolveTimes(100);
produceSolveTimes(200);
% limit: ������������
function produceSolveTimes(size)
    % ÿ�ּ���100��
    times=100;
    timesArr=zeros(1,100);
    w=zeros(1,100);
    for j=1:100
       w(j)=1+0.01*(j-1); 
    end
    for i=1:times
        % ��������ԽǾ�����������������������Գ���������A��ͬʱ����2D-A����
        while 1
            v=diag(rand(size,1));
            u=orth(randn(size));
            A=u'*v*u;
            % p1Ϊ0ʱA����
            [R,p1]=chol(A);
            % 2D-A
            [R,p2]=chol(2*diag(diag(A))-A);
            if p1==0&&p2==0
                break;
            end
        end
        x=rand(size,1);
        b=A*x;
        for j=1:100
            timesArr(j)=timesArr(j)+SOR(A,b,w(j));
        end
    end
    % ƽ��
    timesArr=timesArr/times;
    min_y=min(timesArr);
    min_x=1+0.01*(find(timesArr==min_y)-1);
    figure;
    plot(w(:),timesArr(:),'-');
    xlim([0.99,2]);
    line([0,min_x],[min_y,min_y],'linestyle',':');
    line([min_x,min_x],[0,min_y],'linestyle',':');
    set(gca,'XTick',[1,min_x,2]);
end

function times=SOR(A,b,w)
    dim=size(A,1);
    x=zeros(dim,1);
    D=diag(diag(A));
    L=-tril(A,-1);
    U=-triu(A,1);
    % Lw=(D-wL)^(-1)((1-w)D+wU)
    Lw=(D-w*L)\((1-w)*D+w*U);
    % f=w(D-wL)^(-1)b
    f=w*(D-w*L)^(-1)*b;
    x_k=zeros(dim,1);
    times=0;
    while 1
        x=Lw*x_k+f;
        % ���ȣ��������������������Ԫ�صľ���ֵ�����ģ�
        if norm(x-x_k,inf)<1e-5
            break;
        end
        x_k=x;
        % ������������
        times=times+1;
        if times==290000
            error('����������������');
        end
    end
end