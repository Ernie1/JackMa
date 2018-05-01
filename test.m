%生成随机矩阵
while 1
  A=rand(10);
  if abs(det(A))>1e-8
    break;
  end
end

b=A*[1;2;3;4;5;2;1;2;9;10];

tic;
x=EliminationWithMaximalColumnPivoting(A,b)
t1=toc;

disp("计算时间: "+t1);

% function [outputArg1,outputArg2] = untitled3(inputArg1,inputArg2)
function x = EliminationWithMaximalColumnPivoting(A,b);
    if abs(det(A))<=1e-8
        error('该矩阵是奇异矩阵');
        return;
    end
    dim=size(A,1);
    %消去
    for i=1:dim
        %选最大
        mcp=find(abs(A(i:dim,i))==max(abs(A(i:dim,i))))+i-1;
        %交换
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
    %回代
    x=zeros(dim,1);
    x(dim)=b(dim)/A(dim,dim);
    for i=dim-1:-1:1
        x(i)=(b(i)-A(i,i+1:dim)*x(i+1:dim))/A(i,i);
    end
end