clc
clear all
close all
%%
data=xlsread('nn.xlsx');
figure;
plot(data(:,5),data(:,6),'*');
a=randperm(80);
for j=1:80
    data(j,:)=data(a(j),:);
end
input=data(:,1:5);
p=input';
inputw=data(:,1:5);
pw=inputw';
output=data(:,6);
t=output';
for i=1:5
p(i,:)=p(i,:)/max(t(1,:));
end
for i=1:4
pw(i,:)=pw(i,:)/max(t(1,:));
end
t(1,:)=t(1,:)/max(t(1,:));
 net = feedforwardnet(20,'trainlm');
 net = train(net,p,t);
 view(net)
 y = net(p);
 %%%
 netw = train(net,pw,t);
 view(netw)
 yw = net(pw);
 %%%
 [trainInd,valInd,testInd]=divideblock(80,0.7,0.15,0.15);
net.trainParam.epochs=60;               %Maximum number of epochs to train
net.trainParam.goal=1e-5;                    %Performance goal
net.trainParam.lr=.05;                    %Learning rate
net.trainParam.max_fail=10;                %Maximum validation failures
net.trainParam.min_grad=1e-10;             %Minimum performance gradient
net.trainParam.show=10;                    %Epochs between showing progress
net.trainparam.mu_max=1e10;
tv.P=p(:,69:80);
tv.Pw=pw(:,69:80);
tv.T=net(tv.P);
tv.Tw=net(tv.Pw);
target=t(:,69:80);
% %%                   Calculate MSE
e_test = target-tv.T;
e_testw = target-tv.Tw;
Caluclat_MSE = mse(e_test)
Caluclat_MSEw = mse(e_testw)
%%                  Calculate  Absolute Error (AE) and Mean Absolute Error (MAE)
Caluclat_AE=sum(abs(target-tv.T));
Caluclat_AEw=sum(abs(target-tv.Tw));
Caluclat_MAE=mean(abs(target-tv.T));
Caluclat_MAEw=mean(abs(target-tv.Tw));
MAE = mae(target-tv.T)
MAEw = mae(target-tv.Tw)
x=mae(abs((target-tv.T)./target))
xw=mae(abs((target-tv.Tw)./target))


plot(tv.T,target,'*','Color','R');
title('result with 5 features');
xlabel('nn output')
ylabel('target')

figure;
plot(tv.Tw,target,'*','Color','R');
title('result without 5 features');
xlabel('nn output')
ylabel('target')
