%生成随机矩阵
while 1
  A = rand(10)
  if abs(det(A))>1e-8
    break;
  end
end

% 
function EliminationWithMaximalColumnPivoting(A, b)
	if abs(det(A))<=1e-8
        error('该矩阵是奇异矩阵');
        return;
    end
	dim=size(A,1);
    for i=1:dim
        %选最大
        mcp=find(abs(A(i:dim,i))==max(abs(A(i:dim,i))));
        %交换
        tem=A(mcp,:);
        A(
        for j=i+1:m
            
        end
    end
end
tic
t1=toc