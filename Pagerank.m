% % size(socEpinions1)
% % socEpinions1.Properties.VariableNames{1} = 'FromNodeId';
% % socEpinions1.Properties.VariableNames{2} = 'ToNodeId';

% �ֶ��������ݵ�socEpinions1
% A=sparse(75888,75888);
% socEpinions1=table2array(socEpinions1);
% socEpinions1=socEpinions1+1;
A(sub2ind(size(A),socEpinions1(:,2),socEpinions1(:,1)))=1;