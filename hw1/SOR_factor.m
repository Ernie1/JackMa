% 实现Jacobi 迭代法、Gauss-Seidel 迭代法、逐次超松弛迭代法、共轭梯度法。
% A为对称正定矩阵，其特征值服从独立同分布的[0,1]间的均匀分布;b中的元素服从独立同分布的正态分布。令 n=10、50、100、200，分别绘制出算法的收敛曲线，横坐标为迭代步数，纵坐标为相对误差。
produceSolveTimes(10);
produceSolveTimes(50);
produceSolveTimes(100);
produceSolveTimes(200);
% limit: 迭代次数限制
function produceSolveTimes(size)
    % 每种计算100个
    times=100;
    timesArr=zeros(1,100);
    w=zeros(1,100);
    for j=1:100
        % 松弛因子测试范围为[1,2)，精度为1e-2
        w(j)=1+0.01*(j-1); 
    end
    for i=1:times
        % 利用随机对角矩阵和随机正交矩阵生成随机对称正定矩阵A，同时满足2D-A正定
        while 1
            v=diag(rand(size,1));
            u=orth(randn(size));
            A=u'*v*u;
            % p1为0时A正定
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
    % 平均
    timesArr=timesArr/times;
    % 标出最少迭代次数对应的松弛因子
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
        % 精度，用无穷范数（向量的所有元素的绝对值中最大的）
        if norm(x-x_k,inf)<1e-5
            break;
        end
        x_k=x;
        % 迭代次数限制
        times=times+1;
        if times==290000
            error('超出迭代次数限制');
        end
    end
end