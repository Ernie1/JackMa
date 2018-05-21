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

% 手动导入数据到socEpinions1
% A=sparse(75888,75888);
% socEpinions1=table2array(socEpinions1);
% socEpinions1=socEpinions1+1;
% A(sub2ind(size(A),socEpinions1(:,2),socEpinions1(:,1)))=1;

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
% sortrows(R,2,'descend');