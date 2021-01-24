


%%% Computing the pdf s
load X_train.mat
N_tr = 1400;
N_te = 400;
h=0.8;
no_feature=50;
dfnc_tr = zeros(N_tr,8,8);     
dfnc_te = zeros(N_te,8,8);      

for i = 1:8
    for j=1:N_tr
        for k=1:8
            dis = max(abs(X_train(:,:,k)-repmat(X_train(:,j,i),1,N_tr)),[],1);
           dfnc_tr(j,k,i) = sum(dis < h/2)/(N_tr*h^no_feature);
        end
    end
end
           
for i = 1:8
    for j=1:N_te
        for k=1:8
            dis = max(abs(X_train(:,:,k)-repmat(X_test(:,j,i),1,N_tr)),[],1);
           dfnc_te(j,k,i) = sum(dis < h/2)/(N_tr*h^no_feature);
        end
    end
end
% %% Classifying the train data
confidence = zeros(8,8);
confusion = zeros(8,8);
for i = 1:8
    for j = 1:N_tr
        [so_dfnc_tr,index] = sort(dfnc_tr(j,:,i),'descend');
        confusion(index(1),i) = confusion(index(1),i)+1;
        confidence(index(1),i) = confidence(index(1),i)+(so_dfnc_tr(1)-so_dfnc_tr(2))/so_dfnc_tr(1);            
    end
end         % for data in class cnt
tr_confidence = confidence./confusion;
tr_confusion = confusion/N_tr;
tr_CCR = sum(diag(tr_confusion))/sum(sum(tr_confusion));
tr_confidence(isnan(tr_confidence))=0;

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
te_confidence = confidence./confusion;
te_confusion = confusion/N_te;
te_CCR = sum(diag(te_confusion))/sum(sum(te_confusion));
te_confidence(isnan(te_confidence))=0;