## 一
### 问题描述
求解线性方程组 Ax=b，其中 A 为 nxn 维的已知矩阵，b 为 n 维的已知向量，x 为 n 维的未知向量。A 与 b 中的元素服从独立同分布的正态分布。
### 算法设计
1. 高斯消去法
首先，要将 L_1 以下的等式中的 x 消除，然后再将 L_k (k=1,2,...,n-1) 以下的等式中的 y 消除。这样可使整个方程组变成一个三角形似的格式。之后再将已得出的答案一个个地代入已被简化的等式中的未知数中，就可求出其余的答案了。
```matlab
function x = GaussianElimination(A,b)
    dim=size(A,1);
    % 消去
    for i=1:dim
        if A(i,i)==0
            error('主元素=0，消去法无法进行');
            return;
        end
        for j=i+1:dim            
            m=A(j,i)/A(i,i);
            % A(j,:)=A(j,:)-m*A(i,:);
            A(j,i+1:dim)=A(j,i+1:dim)-m*A(i,i+1:dim);
            b(j)=b(j)-m*b(i);
        end
    end
    
    % 回代
    x=zeros(dim,1);
    x(dim)=b(dim)/A(dim,dim);
    for i=dim-1:-1:1
        x(i)=(b(i)-A(i,i+1:dim)*x(i+1:dim))/A(i,i);
    end
end
```
2. 列主元消去法
在进行第 k (k=1,2,...,n-1) 步消元时，从第 k 列的 a_{kk} 及其以下的各元素中选取绝对值最大的元素，然后通过行变换将它交换到主元素 a_{kk} 的位置上，再进行消元。
```matlab
function x = EliminationWithMaximalColumnPivoting(A,b)
    dim=size(A,1);
    
    % 消去
    for i=1:dim
        % 选最大
        mcp=find(abs(A(i:dim,i))==max(abs(A(i:dim,i))))+i-1;
        if A(mcp,i)==0
            error('主元素=0，消去法无法进行');
            return;
        end
        % 交换
        tem=A(mcp,:);
        A(mcp,:)=A(i,:);
        A(i,:)=tem;
        tem=b(mcp);
        b(mcp)=b(i);
        b(i)=tem;
        for j=i+1:dim
            m=A(j,i)/A(i,i);
            % A(j,:)=A(j,:)-m*A(i,:);
            A(j,i+1:dim)=A(j,i+1:dim)-m*A(i,i+1:dim);
            b(j)=b(j)-m*b(i);
        end
    end
    
    % 回代
    x=zeros(dim,1);
    x(dim)=b(dim)/A(dim,dim);
    for i=dim-1:-1:1
        x(i)=(b(i)-A(i,i+1:dim)*x(i+1:dim))/A(i,i);
    end
end
```
### 数值实验
令 n=10、50、100、200，测试计算时间并绘制曲线。
```matlab
main();

function main()
    computeTime=zeros(2,4);
    size=[10,50,100,200];
    % 每种计算100个
    times=100;
    for j=1:4
        for i=1:times
            [time1,time2]=produceSolveComputeTime(size(j));
            computeTime(1,j)=computeTime(1,j)+time1;
            computeTime(2,j)=computeTime(2,j)+time2;
        end
    end
    % 平均
    computeTime=computeTime/times;
    plot(size(:),computeTime(1,:),'-',size(:),computeTime(2,:),'-');
    xlim([0,210]);
    set(gca,'XTick',[10,50,100,200]);
    legend('高斯消去法','列主元消去法');
end

% 随机生成独立同分布的正态分布的随机A和b，解，返回时间
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
```
### 结果分析
![](fig/Gau_MaxCol.svg)
观察计算时间，列主元消去法时间略长，因为涉及主元的寻找和行变换。

```matlab
% 在 Epinions 社交数据集(https://snap.stanford.edu/data/soc-Epinions1.html)中， 每个网络节点可以选择信任其它节点。
% 借鉴 Pagerank 的思想编写程序，对网络节点的受信任程度进行评分。
% start
% 手动将soc-Epinions1.txt导入数据到socEpinions1，socEpinions1是table类型，先转为数组
socEpinions1=table2array(socEpinions1);
% 由于数据的节点从0开始，而matlab索引从1开始，为了方便，先给数据的节点编号都加一
socEpinions1=socEpinions1+1;
% 统计每个节点发出数，即L(p_j)，存入socEpinions数组
socEpinions=zeros(75888,1);
for i=1:length(socEpinions1)
    tem=socEpinions1(i,1);
    socEpinions(tem)=socEpinions(tem)+1;
end
% 每个元素变成倒数，socEpinions(p_i)即为特殊矩阵中l(:,p_i)不为0时的值，保证链出概率和为1
socEpinions=1./socEpinions;
% 创建稀疏矩阵A即特殊矩阵
A=sparse(75888,75888);
for i=1:length(socEpinions1)
    A(sub2ind(size(A),socEpinions1(i,2),socEpinions1(i,1)))=socEpinions(socEpinions1(i,1));
end
% 观察一下A
spy(A);
% 用幂法求出R，得出PageRank
R=ones(75888,1);
d=0.85;
other=(1-d)/7;
times=0;
while 1
    times=times+1;
    R_kp1=other+d*A*R;
    if norm(R_kp1-R,inf)<1e-5
        R=R_kp1;
        break;
    end
    R=R_kp1;
end
% 给R加上原来从0开始的节点编号
R=[[0:75887]',R];
% 进行降序排序
R=sortrows(R,2,'descend');
% 与直接通过各节点链入数的排序进行比较
socEpinionsIn=zeros(75888,1);
for i=1:length(socEpinions1)
    tem=socEpinions1(i,2);
    socEpinionsIn(tem)=socEpinionsIn(tem)+1;
end
socEpinionsIn=[[0:75887]',socEpinionsIn];
socEpinionsIn=sortrows(socEpinionsIn,2,'descend');
```