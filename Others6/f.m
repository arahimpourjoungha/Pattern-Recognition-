clear all
close all
clc

%%%normalization
t = cputime;
load X_testlc.mat
load X_trainlc.mat
X_train = X_trainlc;
X_test = X_testlc;
pca = 7;        % number of features PCA
for cnt = 1:8
    trn_hlp(:,:,cnt) = X_trainlc(:,:,cnt)';
    tst_hlp(:,:,cnt) = X_testlc(:,:,cnt)';
end
X_trainlc = trn_hlp;
X_testlc = tst_hlp; 
%%%%%%%% Whitening
all_class = [];
for cnt = 1:8
    all_class = [all_class; X_trainlc(:,:,cnt)];
end
b = mean(all_class)';               %%%% Bias term
train_cov = cov(all_class);
[U,S,V] = svd(train_cov);
S = diag(S);
num_whitened = find((S/max(S)) >= 0.0001,1,'last');          %%%% Number of whitened features
a = diag(S(1:num_whitened).^-0.5)*V(:,1:num_whitened)';     %%%% Transformation
%%%% Whitening train & test data
for cnt = 1:8
    hlp_train(:,:,cnt) = (X_trainlc(:,:,cnt)-repmat(b',size(X_trainlc(:,:,cnt),1),1))*a';
    hlp_test(:,:,cnt) = (X_testlc(:,:,cnt)-repmat(b',size(X_testlc(:,:,cnt),1),1))*a';
end
X_trainlc = hlp_train;
X_testlc = hlp_test;

%%%%%%%% PCA
Sw = 0;
Sb = 0;
mu = 0;
N = 0;
for cnt = 1:8
    Sw = Sw + cov(X_trainlc(:,:,cnt));
    Sb = Sb + mean(X_trainlc(:,:,cnt))'*mean(X_trainlc(:,:,cnt))*size(X_trainlc(:,:,cnt),1);
    mu = mu + mean(X_trainlc(:,:,cnt))'*size(X_trainlc(:,:,cnt),1);
    N = N + size(X_trainlc(:,:,cnt),1);
end
mu = mu/N;
Sb = (Sb - N*mu*mu')/N;
[V,S] = eig(inv(Sw)*Sb);
S = diag(S);
%%%% Performing PCA transformation on train & test data
for cnt = 1:8
    hlp_X_trainlc(:,:,cnt) = (X_trainlc(:,:,cnt)*V(:,1:pca))';
    hlp_X_testlc(:,:,cnt) = (X_testlc(:,:,cnt)*V(:,1:pca))';
end
X_trainlc = hlp_X_trainlc;
X_testlc = hlp_X_testlc;
X_train = X_trainlc(:, :)';
X_test = X_testlc(:, :)';

Max_X_train = max(X_train,[],1);
Min_X_train = min(X_train,[],1);

X_train_norm = 2*((X_train - Min_X_train(ones(1, size(X_train, 1)), :)) ./ (Max_X_train(ones(1, size(X_train, 1)), :) - Min_X_train(ones(1, size(X_train, 1)), :))) - ones(size(X_train));
X_test_norm = 2*((X_test - Min_X_train(ones(1, size(X_test, 1)), :)) ./ (Max_X_train(ones(1, size(X_test, 1)), :) - Min_X_train(ones(1, size(X_test, 1)), :))) - ones(size(X_test));

%%%% Add class label to train data
X_train_new_norm = zeros(size(X_train_norm, 1), size(X_train_norm, 2)+1);
for i = 1 : size(X_trainlc, 3)
    for j = 1 : size(X_trainlc, 2)
        X_train_new_norm(((i-1)*size(X_trainlc, 2) + j), :) = [i, X_train_norm(((i-1)*size(X_trainlc, 2) + j), :)];
    end
end
%%%% Add class label to test data
X_test_new_norm = zeros(size(X_test_norm, 1), size(X_test_norm, 2)+1);
for i = 1 : size(X_testlc, 3)
    for j = 1 : size(X_testlc, 2)
        X_test_new_norm(((i-1)*size(X_testlc, 2) + j), :) = [i, X_test_norm(((i-1)*size(X_testlc, 2) + j), :)];
    end
end

%%%%%%%% Initializing variables
p = 0.00001;    %%%% Step size
Epoch = 200;
trn_ems = zeros(Epoch, 1);
tst_ems = zeros(Epoch, 1);

M1 = 10;         %%%% number of neurons in first hidden layer
M2 = 9;         %%%% number of neurons in second hidden layer
J = size(X_trainlc, 3); %%%% number of classes
N = size(X_train_new_norm, 2) - 1; %%%% Number of features

Confusion_mat_epoch_test = zeros(J, J);
Confusion_mat_epoch_percent_test = zeros(J, J);
True_decision_test = zeros(Epoch, 1);
Confusion_mat_epoch_train = zeros(J, J);
Confusion_mat_epoch_percent_train = zeros(J, J);
True_decision_train = zeros(Epoch, 1);

X_1 = [ones(size(X_train_new_norm, 1), 1), X_train_new_norm(:, [2 : N+1])];
T = -0.9*ones(size(X_train_new_norm, 1), J);
for i = 1 : size(X_train_new_norm, 1)
    T(i, X_train_new_norm(i, 1)) = 0.9;
end

Wnl = rand(N+1, M1) - 0.5*ones(N+1, M1);
Wnl_old = zeros(N+1, M1);
dWnl = zeros(N+1, M1);
dWnl_old = zeros(N+1, M1);
PWnl = zeros(N+1, M1);
PWnl_old = zeros(N+1, M1);

Qlm = rand(M1+1, M2) - 0.5*ones(M1+1, M2);
Qlm_old = zeros(M1+1, M2);
dQlm = zeros(M1+1, M2);
dQlm_old = zeros(M1+1, M2);
PQlm = zeros(M1+1, M2);
PQlm_old = zeros(M1+1, M2);

Umj = rand(M2+1, J) - 0.5*ones(M2+1, J);
Umj_old = zeros(M2+1, J);
dUmj = zeros(M2+1, J);
dUmj_old = zeros(M2+1, J);
PUmj = zeros(M2+1, J);
PUmj_old = zeros(M2+1, J);

R1 = zeros(1, M1);
R2 = zeros(1, M2);
S = zeros(1, J);
Z = zeros(1, J);

SSE = zeros(Epoch, 1);
K = zeros(Epoch, 1);


for i = 1 : Epoch
    ddWnl = zeros(N+1, M1);
    ddQlm = zeros(M1+1, M2);
    ddUmj = zeros(M2+1, J);
    E = 0;
    
    R1 = X_1* Wnl;
    Y_1 = [ones(size(X_train_new_norm, 1), 1), tanh(R1)];
    R2 = Y_1*Qlm;
    Y_1_1 = [ones(size(X_train_new_norm, 1), 1), tanh(R2)];
    S = Y_1_1*Umj;
    Z = tanh(S);
    
    
    ddUmj = 2*Y_1_1'*((Z-T).*(1-Z.^2));
    ddQlm = 2*Y_1'*((Umj([2:M2+1], :)*((Z-T).*(1-Z.^2))').*(1-Y_1_1(:, [2:M2+1]).^2)')';
    ddWnl = 2*X_1'*((Qlm([2:M1+1],:)*((Umj([2:M2+1], :)*((Z-T).*(1-Z.^2))').*(1-Y_1_1(:, [2:M2+1]).^2)'))'.*(1-Y_1(:, [2:M1+1]).^2));
    
    if i == 1
            PWnl_old = ddWnl;
            PQlm_old = ddQlm;
            PUmj_old = ddUmj;
            PWnl = ddWnl;
            PQlm = ddQlm;
            PUmj = ddUmj;
            dWnl_old = ddWnl;
            dQlm_old = ddQlm;
            dUmj_old = ddUmj;
            Wnl = Wnl - p*ddWnl;
           Qlm = Qlm - p*ddQlm;
            Umj = Umj - p*ddUmj;
    end
        
    if i ~= 1
        A = sum(sum(dWnl_old.^2));
        if A == 0
            A = eps;
        end
        B = sum(sum(dQlm_old.^2));
        if B == 0
            B = eps;
        end
        C = sum(sum(dUmj_old.^2));
        if C == 0
            C = eps;
        end
        PWnl = ddWnl - ((sum(sum(ddWnl.^2)))/(A))*PWnl_old;
        PQlm = ddQlm - ((sum(sum(ddQlm.^2)))/(B))*PQlm_old;
        PUmj = ddUmj - ((sum(sum(ddUmj.^2)))/(C))*PUmj_old;
        PWnl_old = PWnl;
        PQlm_old = PQlm;
        PUmj_old = PUmj;
        ddWnl_old = ddWnl;
        ddQlm_old = ddQlm;
        ddUmj_old = ddUmj;
        
        Umj_old = Umj;
        Qlm_old = Qlm;
        Wnl_old = Wnl;

        Umj = Umj - p*PUmj;
        Qlm = Qlm - p*PQlm;
        Wnl = Wnl - p*PWnl;
    end
    
    SSE(i) = 0;
    R1 = X_1* Wnl;
    Y_1 = [ones(size(X_train_new_norm, 1), 1), tanh(R1)];
    R2 = Y_1*Qlm;
    Y_1_1 = [ones(size(X_train_new_norm, 1), 1), tanh(R2)];
    S = Y_1_1*Umj;
    Z = tanh(S);
    SSE(i) = sum(sum((Z-T).^2));
    E = SSE(i);
    
    if i>1 && SSE(i-1) < E
        Umj = Umj_old;
        Qlm = Qlm_old;
        Wnl = Wnl_old;
    else
        while SSE(i)<= E
            Umj_old = Umj;
            Qlm_old = Qlm;
            Wnl_old = Wnl;
            Umj = Umj - p*PUmj;
            Qlm = Qlm - p*PQlm;
            Wnl = Wnl - p*PWnl;
            K(i) = K(i) + 1;
            
            E = SSE(i);
            R1 = X_1* Wnl;
            Y_1 = [ones(size(X_train_new_norm, 1), 1), tanh(R1)];
            R2 = Y_1*Qlm;
            Y_1_1 = [ones(size(X_train_new_norm, 1), 1), tanh(R2)];
            S = Y_1_1*Umj;
            Z = tanh(S);
            SSE(i) = sum(sum((Z-T).^2));
        end
        
        Umj = Umj_old;
        Qlm = Qlm_old;
        Wnl = Wnl_old;
        Umj = Umj + 0.2*p*PUmj;
        Qlm = Qlm + 0.2*p*PQlm;
        Wnl = Wnl + 0.2*p*PWnl;
        R1 = X_1* Wnl;
        Y_1 = [ones(size(X_train_new_norm, 1), 1), tanh(R1)];
        R2 = Y_1*Qlm;
        Y_1_1 = [ones(size(X_train_new_norm, 1), 1), tanh(R2)];
        S = Y_1_1*Umj;
        Z = tanh(S);
        EE = sum(sum((Z-T).^2));
        
        if EE < E
            k = 0.2;
        else
            o = 1;
            k = -0.2;
            Umj = Umj_old;
            Qlm = Qlm_old;
            Wnl = Wnl_old;
        end

        while EE <= E || o == 1
            o = 0;
            Umj_old = Umj;
            Qlm_old = Qlm;
            Wnl_old = Wnl;
            Umj = Umj - k*p*PUmj;
            Qlm = Qlm - k*p*PQlm;
            Wnl = Wnl - k*p*PWnl;

            E = EE;
            R1 = X_1* Wnl;
            Y_1 = [ones(size(X_train_new_norm, 1), 1), tanh(R1)];
            R2 = Y_1*Qlm;
            Y_1_1 = [ones(size(X_train_new_norm, 1), 1), tanh(R2)];
            S = Y_1_1*Umj;
            Z = tanh(S);
            EE = sum(sum((Z-T).^2));  
        end
        
        Umj = Umj_old;
        Qlm = Qlm_old;
        Wnl = Wnl_old;
        Umj = Umj + 0.02*p*PUmj;
        Qlm = Qlm + 0.02*p*PQlm;
        Wnl = Wnl + 0.02*p*PWnl;
        R1 = X_1* Wnl;
        Y_1 = [ones(size(X_train_new_norm, 1), 1), tanh(R1)];
        R2 = Y_1*Qlm;
        Y_1_1 = [ones(size(X_train_new_norm, 1), 1), tanh(R2)];
        S = Y_1_1*Umj;
        Z = tanh(S);
        EE = sum(sum((Z-T).^2));
        
        if EE < E
            k = 0.02;
        else
            o = 1;
            k = -0.02;
            Umj = Umj_old;
            Qlm = Qlm_old;
            Wnl = Wnl_old;
        end
        
        while EE <= E || o == 1
            o = 0;
            Umj_old = Umj;
            Qlm_old = Qlm;
            Wnl_old = Wnl;
            Umj = Umj - k*p*PUmj;
            Qlm = Qlm - k*p*PQlm;
            Wnl = Wnl - k*p*PWnl;

            E = EE;
            R1 = X_1* Wnl;
            Y_1 = [ones(size(X_train_new_norm, 1), 1), tanh(R1)];
            R2 = Y_1*Qlm;
            Y_1_1 = [ones(size(X_train_new_norm, 1), 1), tanh(R2)];
            S = Y_1_1*Umj;
            Z = tanh(S);
            EE = sum(sum((Z-T).^2));
        end
        
        Umj = Umj_old;
        Qlm = Qlm_old;
        Wnl = Wnl_old;
        Umj = Umj + 0.001*p*PUmj;
        Qlm = Qlm + 0.001*p*PQlm;
        Wnl = Wnl + 0.001*p*PWnl;
        R1 = X_1* Wnl;
        Y_1 = [ones(size(X_train_new_norm, 1), 1), tanh(R1)];
        R2 = Y_1*Qlm;
        Y_1_1 = [ones(size(X_train_new_norm, 1), 1), tanh(R2)];
        S = Y_1_1*Umj;
        Z = tanh(S);
        EE = sum(sum((Z-T).^2));
        
        if EE < E
            k = 0.001;
        else
            o = 1;
            k = -0.001;
            Umj = Umj_old;
            Qlm = Qlm_old;
            Wnl = Wnl_old;
        end
        
        while EE <= E || o == 1
            o = 0;
            Umj_old = Umj;
            Qlm_old = Qlm;
            Wnl_old = Wnl;
            Umj = Umj - k*p*PUmj;
            Qlm = Qlm - k*p*PQlm;
            Wnl = Wnl - k*p*PWnl;

            E = EE;
            R1 = X_1* Wnl;
            Y_1 = [ones(size(X_train_new_norm, 1), 1), tanh(R1)];
            R2 = Y_1*Qlm;
            Y_1_1 = [ones(size(X_train_new_norm, 1), 1), tanh(R2)];
            S = Y_1_1*Umj;
            Z = tanh(S);
            EE = sum(sum((Z-T).^2));
        end
        
        Umj = Umj_old;
        Qlm = Qlm_old;
        Wnl = Wnl_old;
        Umj = Umj + 0.0001*p*PUmj;
        Qlm = Qlm + 0.0001*p*PQlm;
        Wnl = Wnl + 0.0001*p*PWnl;
        R1 = X_1* Wnl;
        Y_1 = [ones(size(X_train_new_norm, 1), 1), tanh(R1)];
        R2 = Y_1*Qlm;
        Y_1_1 = [ones(size(X_train_new_norm, 1), 1), tanh(R2)];
        S = Y_1_1*Umj;
        Z = tanh(S);
        EE = sum(sum((Z-T).^2));
        
        if EE < E
            k = 0.0001;
        else
            o = 1;
            k = -0.0001;
            Umj = Umj_old;
            Qlm = Qlm_old;
            Wnl = Wnl_old;
        end
        
        while EE <= E || o == 1
            o = 0;
            Umj_old = Umj;
            Qlm_old = Qlm;
            Wnl_old = Wnl;
            Umj = Umj - k*p*PUmj;
            Qlm = Qlm - k*p*PQlm;
            Wnl = Wnl - k*p*PWnl;

            E = EE;
            R1 = X_1* Wnl;
            Y_1 = [ones(size(X_train_new_norm, 1), 1), tanh(R1)];
            R2 = Y_1*Qlm;
            Y_1_1 = [ones(size(X_train_new_norm, 1), 1), tanh(R2)];
            S = Y_1_1*Umj;
            Z = tanh(S);
            EE = sum(sum((Z-T).^2));
        end
        
        Umj = Umj_old;
        Qlm = Qlm_old;
        Wnl = Wnl_old;
        
        if K(i) < 30
            p = p*0.99;
        end
                
    end
   
    %%%% Train data classificiation
    Confusion_mat_train = zeros(J, J);
    for j = 1 : size(X_train_new_norm, 1)
        T2 = -0.9*ones(1, J);
        T2(X_train_new_norm(j, 1)) = 0.9;
        X_12 = [1, X_train_new_norm(j, [2 : N+1])];
        R12 = X_12* Wnl;
        Y_12 = [1, tanh(R12)];
        R22 = Y_12*Qlm;
        Y_1_12 = [1, tanh(R22)];
        S12 = Y_1_12*Umj;
        Z12 = tanh(S12);
        [maximum index] = max(Z12);
        f = X_train_new_norm(j, 1);
        Confusion_mat_train(index, f) = Confusion_mat_train(index, f) + 1;
        trn_ems(i) = trn_ems(i) + norm(Z12-T2);
    end
    Confusion_mat_epoch_train = Confusion_mat_train;
    sum_rows = ones(J, 1)*sum(Confusion_mat_train);
    Confusion_mat_epoch_percent_train = 100 * Confusion_mat_train ./ sum_rows;
    True_decision_train(i) = trace(Confusion_mat_epoch_percent_train)/J;
    
    %%%% Test data classificiation
    Confusion_mat_test = zeros(J, J);
    for j = 1 : size(X_test_new_norm, 1)
        T1 = -0.9*ones(1, J);
        T1(X_test_new_norm(j, 1)) = 0.9;
        X_11 = [1, X_test_new_norm(j, [2 : N+1])];
        R11 = X_11* Wnl;
        Y_11 = [1, tanh(R11)];
        R21 = Y_11*Qlm;
        Y_1_11 = [1, tanh(R21)];
        S11 = Y_1_11*Umj;
        Z11 = tanh(S11);
        [maximum index] = max(Z11);
        f = X_test_new_norm(j, 1);
        Confusion_mat_test(index, f) = Confusion_mat_test(index, f) + 1;
        tst_ems(i) = tst_ems(i) + norm(Z11-T1);
    end
    Confusion_mat_epoch_test = Confusion_mat_test;
    sum_rows = ones(J, 1)*sum(Confusion_mat_test);
    Confusion_mat_epoch_percent_test = 100 * Confusion_mat_test ./ sum_rows;
    True_decision_test(i) = trace(Confusion_mat_epoch_percent_test)/J;
    %%%% Shuffling train data
    shuffle = randperm(size(X_train_new_norm, 1));
    X_train_new_norm = X_train_new_norm(shuffle,:);

    %%%% Shuffling test data
    shuffle = randperm(size(X_test_new_norm, 1));
    X_test_new_norm = X_test_new_norm(shuffle,:);  
   
end
t = cputime - t;

%%%%%%%% Display
trn_confusion = Confusion_mat_train/size(X_trainlc,2);
trn_CCR = 100*trace(trn_confusion)/sum(sum(trn_confusion));
trn_ems = trn_ems/size(X_train_new_norm, 1);
trn_confusion = round(trn_confusion*100)/100;
open trn_confusion
disp(['Train CCR: ' num2str(trn_CCR)]);

tst_confusion = Confusion_mat_test/size(X_testlc,2);
tst_CCR = 100*trace(tst_confusion)/sum(sum(tst_confusion));
tst_ems = tst_ems/size(X_test_new_norm, 1);
tst_confusion = round(tst_confusion*100)/100;
open tst_confusion
disp(['Test CCR: ' num2str(tst_CCR)]);

disp(['Trainig Time: ' num2str(t) ' s']);

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