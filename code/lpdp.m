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
for i = 1:40
    num = n+10*(i-1);
    trainX = [trainX;X(num(1:5),:)];
    trainY = [trainY;Y(num(1:5),:)];
    testX = [testX;X(num(6:10),:)];
    testY = [testY;Y(num(6:10),:)];
end

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

%构建邻接图
Xpca = zscore(Xpca);
W = constructW(Xpca);


D=zeros(200,200);
for i=1:200
    for j=1:200
        D(i,i)=D(i,i)+W(i,j);
    end
end
L = D-W;
%MMC判别信息
r = 0.1;
[sw sb] = sandu(Xpca,trainY);
lt = Xpca'*L'*Xpca-r*(sb-sw);
rt = Xpca'*D'*Xpca;

ac = [0];
for dim=1:5:80
    %计算特征值、特征向量
    
    [vec val] = eigs(lt,rt,dim,'sm');
  
    %特征提取
    train = trainX*eigvector*vec;
    test = testX*eigvector*vec;
    
    %1-NN分类
   testy = knnclassify(test,train,trainY);
    
    count = 0;
    for i=1:200
        if(testy(i)==testY(i))
            count = count+1;
        end
    end
    ac = [ac,count/200];
end

dim = 0:5:80;
plot(dim,ac,'-o');
hold on

