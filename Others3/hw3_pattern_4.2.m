clc
close all
clear all

%%% Computing the pdf s
load X_train.mat
k_NN = 1;
N_tr = 1400;
N_te = 400;
h=1;
no_feature=50;
dfnc_tr = zeros(N_tr,8,8);      
dfnc_te = zeros(N_te,8,8);      

for i = 1:8
    for j=1:N_tr
        for k=1:8
            dis = sum((X_train(:,:,k)-repmat(X_train(:,j,i),1,N_tr)).^2);
            [dtr_so,index_tr] = sort(dis,'ascend');
            h = max(abs((X_train(:,index_tr(k_NN),k)-X_train(:,j,i))))*2;
           dfnc_tr(j,k,i) =k_NN/(N_tr*h^no_feature);
        end
    end
end
           
for i = 1:8
    for j=1:N_te
        for k=1:8
            dis = sum((X_train(:,:,k)-repmat(X_test(:,j,i),1,N_tr)).^2);
            [dte_so,index_te] = sort(dis,'ascend');
            h = max(abs((X_train(:,index_te(k_NN),k)-X_test(:,j,i))))*2;
           dfnc_te(j,k,i) =k_NN/(N_te*h^no_feature);
        end
    end
end
%%% Classifying the train data
confidence = zeros(8,8);
confusion = zeros(8,8);
for i = 1:8
    for j = 1:N_tr
        [so_dfnc_tr,index] = sort(dfnc_tr(j,:,i),'descend');
        confusion(index(1),i) = confusion(index(1),i)+1;
        confidence(index(1),i) = confidence(index(1),i)+(so_dfnc_tr(1)-so_dfnc_tr(2))/so_dfnc_tr(1);            
    end
end         % for data in class cnt
tr_confidence = confidence./confusion(1:8,:);
tr_confusion = confusion/N_tr;
tr_confusion=round(tr_confusion*10000)/10000;
tr_CCR = sum(diag(tr_confusion))/sum(sum(tr_confusion));
tr_confidence(isnan(tr_confidence))=0;
tr_confidence=round(tr_confidence*10000)/10000;

%%% Classifying the test data
confidence = zeros(8,8);
confusion = zeros(8,8);
for i = 1:8
    for j = 1:N_te
        [so_dfnc_te,index] = sort(dfnc_te(j,:,i),'descend');
        confusion(index(1),i) = confusion(index(1),i)+1;
        confidence(index(1),i) = confidence(index(1),i)+(so_dfnc_te(1)-so_dfnc_te(2))/so_dfnc_te(1);            
    end
end         % for data in class cnt
te_confidence = confidence./confusion(1:8,:);
te_confusion = confusion/N_te;
te_confusion=round(te_confusion*10000)/10000;
te_CCR = sum(diag(te_confusion))/sum(sum(confusion));
te_confidence(isnan(te_confidence))=0;
te_confidence=round(te_confidence*10000)/10000;