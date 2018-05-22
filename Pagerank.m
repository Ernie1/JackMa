% 在 Epinions 社交数据集(https://snap.stanford.edu/data/soc-Epinions1.html)中， 每个网络节点可以选择信任其它节点。
% 借鉴 Pagerank 的思想编写程序，对网络节点的受信任程度进行评分。
% start
% 手动将soc-Epinions1.txt导入数据到socEpinions1，socEpinions1是table类型，先转为数组
% socEpinions1=table2array(socEpinions1);
% 由于数据的节点从0开始，而matlab索引从1开始，为了方便，先给数据的节点编号都加一
socEpinions1=socEpinions1-1;
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
other=(1-d)/75888;
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