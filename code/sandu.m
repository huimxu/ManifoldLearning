function [SW SB] = sandu( Input,Target )
% Ipuut:    n*d matrix,each row is a sample;
% Target:   n*1 matrix,each is the class label

% 初始化
[n dim]=size(Input);
ClassLabel=unique(Target);
k=length(ClassLabel);

nGroup=NaN(k,1);            % group count
GroupMean=NaN(k,dim);       % the mean of each value
SB=zeros(dim,dim);          % 类间离散度矩阵
SW=zeros(dim,dim);          % 类内离散度矩阵

% 计算类内离散度矩阵和类间离散度矩阵
for i=1:k
    group=(Target==ClassLabel(i));
    nGroup(i)=sum(double(group));
    GroupMean(i,:)=mean(Input(group,:));
    tmp=zeros(dim,dim);
    for j=1:n
        if group(j)==i
            t=Input(j,:)-GroupMean(i,:);
            tmp=tmp+t'*t;
        end
    end
    SW=SW+tmp;
end
m=mean(GroupMean);
for i=1:k
    tmp=GroupMean(i,:)-m;
    p=1/nGroup(i);
    SB=SB+p*tmp'*tmp;
end
end

