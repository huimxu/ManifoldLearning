clear all;
clc;

%导入数据集
database = load('ORL_32.mat');
X = database.fea;
Y = database.gnd;

%划分数据集
k = rand(1,10);
[m n] = sort(k);
trainX = [];
trainY = [];
testX = [];
testY = [];
classNum = 40;
for i = 1:classNum
    num = n+10*(i-1);
    trainX = [trainX;X(num(1:5),:)];
    trainY = [trainY;Y(num(1:5),:)];
    testX = [testX;X(num(6:10),:)];
    testY = [testY;Y(num(6:10),:)];
end
clear num n m k i

%pca降维
[pc,score,latent] = pca(trainX);
sumE = sum(latent); 
sumNow = 0;
for idx = 1:length(latent)
    sumNow = sumNow + latent(idx);
    if (sumNow/sumE>0.95)
        break;
    end
end
eigvector = pc(:,1:idx);
Xpca = trainX*eigvector;
[m n]=size(Xpca);

%稀疏表示
%一个样本可以用同类样本的线性组合表示
W = [];
for k=1:classNum
    s = zeros(5,5);     
    for i =5*(k-1)+1:5*k
        x = Xpca(5*(k-1)+1:5*k,:);
        xn = mod(i,5);
        if(xn == 0) 
            xn = 5;
        end
        x(xn,:) = zeros(1,n);
        s(:,xn) = SolveFISTA(x',Xpca(xn,:)');
    end
    W = blkdiag(W,s);
end
%另一方面，一个样本可以用其余不同类的样本的线性组合表示
B = zeros(m,m); 
for k=1:40
    for i =5*(k-1)+1:5*k
        x = Xpca;
        x(5*(k-1)+1:5*k,:) = zeros(5,n);
        B(:,i) = SolveFISTA(x',Xpca(i,:)');
    end
end

W = (W+W')/2;
B = (B+B')/2;

Dw=zeros(m,m);
for i=1:m
    for j=1:m
        Dw(i,i)=Dw(i,i)+W(i,j);
    end
end
for i=1:m
    for j=1:m
        Lw(i,j)=Dw(i,j)-W(i,j);
    end
end

Db=zeros(m,m);
for i=1:m
    for j=1:m
        Db(i,i)=Db(i,i)+B(i,j);
    end
end
for i=1:m
    for j=1:m
        Lb(i,j)=Db(i,j)-B(i,j);
    end
end

Sw = Xpca'*Lw*Xpca;
Sb = Xpca'*Lb*Xpca; 

ac = [0];
for dim=1:5:80
    %计算特征值、特征向量
    
    [vec val] = eigs(Sb,Sw,dim,'lm');
    
    %特征提取
    train = trainX*eigvector*vec;
    test = testX*eigvector*vec;
    
    %1-NN分类
%    testy = knnclassify(test,train,trainY);
    dis = zeros(200,200);
    testy = [];
    for i=1:200
        for j=1:200
            dis(i,j) = norm(test(i,:)-train(j,:));
        end
        [min id] = sort(dis(i,:));
        testy = [testy;trainY(id(1))];
    end
    ac = [ac,sum(testy==trainY)/200];
end

dim = 0:5:80;
plot(dim,ac,'-*');

