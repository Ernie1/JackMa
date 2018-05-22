% �� Epinions �罻���ݼ�(https://snap.stanford.edu/data/soc-Epinions1.html)�У� ÿ������ڵ����ѡ�����������ڵ㡣
% ��� Pagerank ��˼���д���򣬶�����ڵ�������γ̶Ƚ������֡�
% start
% �ֶ���soc-Epinions1.txt�������ݵ�socEpinions1��socEpinions1��table���ͣ���תΪ����
% socEpinions1=table2array(socEpinions1);
% �������ݵĽڵ��0��ʼ����matlab������1��ʼ��Ϊ�˷��㣬�ȸ����ݵĽڵ��Ŷ���һ
socEpinions1=socEpinions1-1;
% ͳ��ÿ���ڵ㷢��������L(p_j)������socEpinions����
socEpinions=zeros(75888,1);
for i=1:length(socEpinions1)
    tem=socEpinions1(i,1);
    socEpinions(tem)=socEpinions(tem)+1;
end
% ÿ��Ԫ�ر�ɵ�����socEpinions(p_i)��Ϊ���������l(:,p_i)��Ϊ0ʱ��ֵ����֤�������ʺ�Ϊ1
socEpinions=1./socEpinions;
% ����ϡ�����A���������
A=sparse(75888,75888);
for i=1:length(socEpinions1)
    A(sub2ind(size(A),socEpinions1(i,2),socEpinions1(i,1)))=socEpinions(socEpinions1(i,1));
end
% �۲�һ��A
spy(A);
% ���ݷ����R���ó�PageRank
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
% ��R����ԭ����0��ʼ�Ľڵ���
R=[[0:75887]',R];
% ���н�������
R=sortrows(R,2,'descend');
% ��ֱ��ͨ�����ڵ���������������бȽ�
socEpinionsIn=zeros(75888,1);
for i=1:length(socEpinions1)
    tem=socEpinions1(i,2);
    socEpinionsIn(tem)=socEpinionsIn(tem)+1;
end
socEpinionsIn=[[0:75887]',socEpinionsIn];
socEpinionsIn=sortrows(socEpinionsIn,2,'descend');