
clc
clear all
close all
%%%%normalization
t0 = cputime;
load X_testlc.mat
load X_trainlc.mat
X_train = X_trainlc(:, :)';
X_test = X_testlc(:, :)';

Max_X_train = max(X_train,[],1);
Min_X_train = min(X_train,[],1);

X_train_norm = 2*((X_train - Min_X_train(ones(1, size(X_train, 1)), :)) ./ (Max_X_train(ones(1, size(X_train, 1)), :) - Min_X_train(ones(1, size(X_train, 1)), :))) - ones(size(X_train));
X_test_norm = 2*((X_test - Min_X_train(ones(1, size(X_test, 1)), :)) ./ (Max_X_train(ones(1, size(X_test, 1)), :) - Min_X_train(ones(1, size(X_test, 1)), :))) - ones(size(X_test));

%%%%  class label(train)
X_train_new_norm = zeros(size(X_train_norm, 1), size(X_train_norm, 2)+1);
for i = 1 : size(X_trainlc, 3)
    for j = 1 : size(X_trainlc, 2)
        X_train_new_norm(((i-1)*size(X_trainlc, 2) + j), :) = [i, X_train_norm(((i-1)*size(X_trainlc, 2) + j), :)];
    end
end
%%%% class label(test)
X_test_new_norm = zeros(size(X_test_norm, 1), size(X_test_norm, 2)+1);
for i = 1 : size(X_testlc, 3)
    for j = 1 : size(X_testlc, 2)
        X_test_new_norm(((i-1)*size(X_testlc, 2) + j), :) = [i, X_test_norm(((i-1)*size(X_testlc, 2) + j), :)];
    end
end

%%%%%%%% Initializing variables
p = 0.001;          %%%% Step size
epoch = 100;         %%%% number of epoch
tst_ems = zeros(epoch, 1);
trn_ems = zeros(epoch, 1);

num_neron = 50;                     %%%% number of hidden layer neurons
class_num = size(X_trainlc, 3);            %%%% number of classes
N = size(X_trainlc,1);                     %%%% Number of features

Confusion_mat_epoch_test = zeros(class_num, class_num);
Confusion_mat_epoch_percent_test = zeros(class_num, class_num);
True_decision_test = zeros(epoch, 1);
Confusion_mat_epoch_train = zeros(class_num, class_num);
Confusion_mat_epoch_percent_train = zeros(class_num, class_num);
True_decision_train = zeros(epoch, 1);

X_1 = zeros(1, N+1);
Wnm = (2*rand(N+1, num_neron) - ones(N+1, num_neron))/2;
dWnm = zeros(N+1, num_neron);

Y_1 = zeros(1, num_neron+1);
Umj = (2*rand(num_neron+1, class_num) - ones(num_neron+1, class_num))/2;
dUmj = zeros(num_neron+1, class_num);

R = zeros(1, num_neron);
S = zeros(1, class_num);
Z = zeros(1, class_num);
t0 = cputime - t0; 

%%%%%%%% Training the Neural Network
for i = 1:epoch
    t(i) = cputime;
    for j = 1 : size(X_train_new_norm, 1)
        T = -0.9*ones(1, class_num);  T(X_train_new_norm(j, 1)) = 0.9;
        X_1 = [1, X_train_new_norm(j, [2 : N+1])];
        R = X_1*Wnm;
        Y_1 = [1, tanh(R)];
        S = Y_1*Umj;
        Z = tanh(S);
        dUmj = 2*Y_1'*((Z-T).*(1-Z.^2));
        dWnm = 2*X_1'*((Umj([2:num_neron+1], :)*((Z-T).*(1-Z.^2))').*(1-Y_1(:, [2:num_neron+1]).^2)')';
        Wnm = Wnm - p*dWnm;
        Umj = Umj - p*dUmj;
    end
    
    %%%% Train data classificiation
    Confusion_mat_train = zeros(class_num, class_num);
    for j = 1 : size(X_train_new_norm, 1)
        T = -0.9*ones(1, class_num);  T(X_train_new_norm(j, 1)) = 0.9;
        X_1 = [1, X_train_new_norm(j, [2 : N+1])];
        R = X_1*Wnm;
        Y_1 = [1, tanh(R)];
        S = Y_1*Umj;
        Z = tanh(S);
        [maximum index] = max(Z);
        f = X_train_new_norm(j, 1);
        Confusion_mat_train(index,f) = Confusion_mat_train(index,f) + 1;
        trn_ems(i) = trn_ems(i) + norm(Z-T);
    end
    Confusion_mat_epoch_train = Confusion_mat_train;
    sum_rows = ones(class_num, 1)*sum(Confusion_mat_train);
    Confusion_mat_epoch_percent_train = 100 * Confusion_mat_train ./ sum_rows;
    True_decision_train(i) = trace(Confusion_mat_epoch_percent_train)/class_num;
    
    %%%% Test data classificiation
    t_tst(i) = cputime;
    Confusion_mat_test = zeros(class_num, class_num);
    for j = 1 : size(X_test_new_norm, 1)
        T = -0.9*ones(1, class_num);
        T(X_test_new_norm(j, 1)) = 0.9;
        X_1 = [1, X_test_new_norm(j, [2 : N+1])];
        R = X_1*Wnm;
        Y_1 = [1, tanh(R)];
        S = Y_1*Umj;
        Z = tanh(S);
        [maximum index] = max(Z);
        f = X_test_new_norm(j, 1);
        Confusion_mat_test(index, f) = Confusion_mat_test(index, f) + 1;
        tst_ems(i) = tst_ems(i) + norm(Z-T);
    end
    Confusion_mat_epoch_test = Confusion_mat_test;
    sum_rows = ones(class_num, 1)*sum(Confusion_mat_test);
    Confusion_mat_epoch_percent_test = 100 * Confusion_mat_test ./ sum_rows;
    True_decision_test(i) = trace(Confusion_mat_epoch_percent_test)/class_num;
    t_tst(i) = cputime - t_tst(i);

    %%%% Shuffling train data
    shuffle = randperm(size(X_train_new_norm, 1));
    X_train_new_norm = X_train_new_norm(shuffle,:);

    %%%% Shuffling test data
    shuffle = randperm(size(X_test_new_norm, 1));
    X_test_new_norm = X_test_new_norm(shuffle,:);

    t(i) = cputime - t(i);
end
t0 = cputime - t0; 

%%%%%%%% Display
trn_confusion = Confusion_mat_train/size(X_trainlc,2);
trn_CCR = 100*trace(trn_confusion)/sum(sum(trn_confusion));
trn_ems = trn_ems/size(X_train_new_norm, 1);
trn_confusion = round(trn_confusion*100)/100;

disp(['Train CCR: ' num2str(trn_CCR)]);

tst_confusion = Confusion_mat_test/size(X_testlc,2);
tst_CCR = 100*trace(tst_confusion)/sum(sum(tst_confusion));
tst_ems = tst_ems/size(X_test_new_norm, 1);
tst_confusion = round(tst_confusion*100)/100;

disp(['Test CCR: ' num2str(tst_CCR)]);

disp(['Trainig Time: ' num2str(t0) ' s']);

figure(1);
%%%%subplot(2,1,1);  
plot(trn_ems,'r')
xlabel('Epoch');  ylabel('ems');  title('Train ems per Epoch');  grid on
%%%%subplot(2,1,2);  
figure(2)
plot(tst_ems,'r')
xlabel('Epoch');  ylabel('ems');  title('Test ems per Epoch');  grid on

figure(3)
%%%%subplot(2,1,1);  
plot(True_decision_train,'b')
xlabel('Epoch');  ylabel('CCR')
title('Train CCR per Epoch');  grid on
%%%%subplot(2,1,2);  
figure(4)
plot(True_decision_test,'b')
xlabel('Epoch');  ylabel('CCR');  
title('Test CCR per Epoch');  grid on

t(1) = t(1)+t0;  t = cumsum(t);  t_tst = t_tst+t0;