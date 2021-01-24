clc
clear all
close all
%%
data=xlsread('nn.xlsx');
input=data(:,1:5);
p=input';
%%%%
inputw=data(:,1:4);
pw=inputw';
%%%%
output=data(:,6);
t=output';
for i=1:5
p(i,:)=p(i,:)/max(p(i,:));
end
%%
for i=1:4
pw(i,:)=pw(i,:)/max(pw(i,:));
end
%%
t(1,:)=t(1,:)/max(t(1,:));
train_p=p(:,1:80);
train_pw=pw(:,1:80);
train_t=t(:,1:80);
%%
test_p=p(:,69:80);
test_pw=pw(:,69:80);
test_t=t(:,69:80);
%%%%
ptr=[train_p train_p train_p train_p train_p train_p train_p train_p train_p train_p];
ptrw=[train_pw train_pw train_pw train_pw train_pw train_pw train_pw train_pw train_pw train_pw];
ttr=[train_t train_t train_t train_t train_t train_t train_t train_t train_t train_t];
tv.P=[test_p test_p test_p test_p test_p test_p test_p test_p test_p test_p ];
tv.Pw=[test_pw test_pw test_pw test_pw test_pw test_pw test_pw test_pw test_pw test_pw ];
tv.T=[test_t test_t test_t test_t test_t test_t test_t test_t test_t test_t];
net = newrb(ptr,ttr,10^-10);
netw = newrb(ptrw,ttr,10^-5);
output_RBF = sim(net,tv.P);
output_RBFw = sim(netw,tv.Pw);
%%                      calculate Mse
e_test = tv.T-output_RBF;
e_testw = tv.T-output_RBFw;
Caluclat_MSE = mse(e_test)
Caluclat_MSEw = mse(e_testw)
%%%%%%%%%%%%%%%%%

%                  Calculate  Absolute Error (AE) and Mean Absolute Error (MAE)
Caluclat_AE=sum(abs(output_RBF-tv.T));
Caluclat_AEw=sum(abs(output_RBFw-tv.T));
Caluclat_MAE=mean(abs(output_RBF-tv.T))
Caluclat_MAEw=mean(abs(output_RBFw-tv.T))
MAE = mae(output_RBF-tv.T)
MAEw = mae(output_RBFw-tv.T)
x=mae(abs((output_RBF-tv.T)./output_RBF))
xw=mae(abs((output_RBFw-tv.T)./output_RBFw))


plot(tv.T,output_RBF,'*','Color','R')
title('result with 5 features');
xlabel('nn output')
ylabel('target')

figure;
plot(tv.T,output_RBFw,'*','Color','R');
title('result without 5 features');
xlabel('nn output')
ylabel('target')



