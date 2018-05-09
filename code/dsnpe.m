clear all;
clc;

%�������ݼ�
database = load('ORL_32.mat');
X = database.fea;
Y = database.gnd;

%�������ݼ�
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

%pca��ά
[pc,score,latent] = pca(trainX);
sumE = sum(latent); 
sumNow = 0;
for idx = 1:length(latent)
    sumNow = sumNow + latent(idx);
    if (sumNow/sumE>0.98)
        break;
    end
end
eigvector = pc(:,1:idx);
Xpca = trainX*eigvector;
Xpca = zscore(Xpca);
[m n] = size(Xpca);

%SRϡ���ʾ
[m n]=size(Xpca);
S = [];
for k=1:40
    s = zeros(5,5);     
    for i =5*(k-1)+1:5*k
        xpca = Xpca(5*(k-1)+1:5*k,:);
        xn = mod(i,5);
        if(xn == 0) 
            xn = 5;
        end
        xpca(xn,:) = zeros(1,n);
        s(:,xn) = SolveFISTA(xpca',Xpca(xn,:)');
    end
    S = blkdiag(S,s);
end

I = zeros(200,200);
for i=1:200
    I(i,i)=1;
end
Salph = I-S-S'+S'*S;

[sb sw] = sandu(Xpca,trainY);
r = 0.1;
lt = Xpca'*Salph'*Xpca-r*(sb-sw);
rt = Xpca'*Xpca;

ac = [0];
for dim=1:5:80
    %��������ֵ����������
    
    [vec val] = eigs(lt,rt,dim,'lm');
  
    %������ȡ
    train = trainX*eigvector*vec;
    test = testX*eigvector*vec;
    
    %1-NN����
   dis = zeros(200,200);
    testy = [];
    for i=1:200
        for j=1:200
            dis(i,j) = norm(test(i,:)-train(j,:));
        end
        [min id] = sort(dis(i,:));
        testy = [testy;trainY(id(1))];
    end
    
    count = 0;
    for i=1:200
        if(testy(i)==testY(i))
            count = count+1;
        end
    end
    ac = [ac,count/200];
end

dim = 0:5:80;
plot(dim,ac,'-*');






