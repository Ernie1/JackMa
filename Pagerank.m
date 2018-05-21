% % former=[   0  ,  1 , 1 , 1 , 1 , 0 ,1;
% % 
% %    1    ,0 , 0 , 0  ,0 , 0,  0;
% % 
% %     1   , 1 , 0,  0 , 0 , 0 , 0;
% % 
% %     0,    1 , 1 , 0  ,1 , 0,  0;
% % 
% %     1  ,  0  ,1  ,1,  0  ,1 , 0;
% % 
% %     1   , 0 , 0  ,0 , 1  ,0 , 0;
% % 
% %     0  ,  0 , 0 , 0  ,1,  0 , 0;];

% % a=[
% % 0  ,   1   , 0.5 ,   0   ,    0.25   , 0.5 ,   0;
% % 0.2 , 0  ,  0.5  ,  1/3   ,  0  ,     0   ,   0;
% % 0.2 , 0  ,  0   ,   1/3    , 0.25 ,   0    ,  0;
% % 0.2 , 0  ,  0   ,   0     ,  0.25 ,   0   ,   0;
% % 0.2,  0   , 0    ,  1/3   ,  0   ,    0.5 ,   1;
% % 0  ,   0  ,  0    ,  0    ,   0.25 ,   0 ,     0;
% % 0.2 , 0   , 0     , 0    ,  0     ,  0   ,   0;];
% % size(socEpinions1)
% % socEpinions1.Properties.VariableNames{1} = 'FromNodeId';
% % socEpinions1.Properties.VariableNames{2} = 'ToNodeId';

% start
% 手动导入数据到socEpinions1
% socEpinions1=table2array(socEpinions1);
% 节点编号都加一
% socEpinions1=socEpinions1+1;
% 统计发出数
% socEpinions=zeros(75888,1);
% for i=1:length(socEpinions1)
%     tem=socEpinions1(i,1);
%     socEpinions(tem)=socEpinions(tem)+1;
% end
% 每个元素变成倒数
% socEpinions=1./socEpinions;
% 创建稀疏矩阵A
% A=sparse(75888,75888);
% for i=1:length(socEpinions1)
%     A(sub2ind(size(A),socEpinions1(i,2),socEpinions1(i,1)))=socEpinions(socEpinions1(i,1));
% end
% 观察一下A
% spy(A);
% R=ones(75888,1);
% d=0.85;
% other=(1-d)/7;
% times=0;
% while 1
%     times=times+1;
%     R_kp1=other+d*A*R;
%     if norm(R_kp1-R,inf)<1e-5
%         R=R_kp1;
%         break;
%     end
%     R=R_kp1;
% end

% R=[[0:75887]',R];
% R=sortrows(R,2,'descend');

% 比较
% socEpinionsIn=zeros(75888,1);
% for i=1:length(socEpinions1)
%     tem=socEpinions1(i,2);
%     socEpinionsIn(tem)=socEpinionsIn(tem)+1;
% end
% socEpinionsIn=[[0:75887]',socEpinionsIn];
% socEpinionsIn=sortrows(socEpinionsIn,2,'descend');