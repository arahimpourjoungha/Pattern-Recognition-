
load X_train.mat
pdf_mean = zeros(1,50,8);
pdf_var = zeros(50,50,8);
N_tr = 1400;
N_te = 400;
for i = 1:8
    k=X_train(:,:,i);
    pdf_mean(:,:,i) = (sum(k')./N_tr);
    pdf_var(:,:,i) = cov(k',1);
end

%%% Classifying the train data
dfnc = zeros(N_tr,8,8);      % Discriminant Function values
confidence = zeros(8,8);
confusion = zeros(8,8);
for i = 1:8
    for j = 1:8
        dfnc(:,j,i) = mvnpdf(X_train(:,:,i)',pdf_mean(:,:,j),pdf_var(:,:,j));
    end     
    for j = 1:N_tr
        [so_dfnc_tr,index] = sort(dfnc(j,:,i),'descend');
        confusion(index(1),i) = confusion(index(1),i)+1;
        confidence(index(1),i) = confidence(index(1),i)+(so_dfnc_tr(1)-so_dfnc_tr(2))/so_dfnc_tr(1);            
    end
end        
tr_confidence = confidence./confusion;
tr_confusion = confusion/N_tr;
tr_confusion=round(tr_confusion*10000)/10000;
tr_CCR = sum(diag(tr_confusion))/sum(sum(tr_confusion));
tr_confidence(isnan(tr_confidence))=0;
tr_confidence=round(tr_confidence*10000)/10000;

%%% Classifying the test data
dfnc = zeros(N_te,8,8); 
confidence = zeros(8,8);
confusion = zeros(8,8);
for i = 1:8
    for j = 1:8
        dfnc(:,j,i) = mvnpdf(X_test(:,:,i)',pdf_mean(:,:,j),pdf_var(:,:,j));
    end     
    for j = 1:N_te
        [so_dfnc_te,index] = sort(dfnc(j,:,i),'descend');
        confusion(index(1),i) = confusion(index(1),i)+1;
        confidence(index(1),i) = confidence(index(1),i)+(so_dfnc_te(1)-so_dfnc_te(2))/so_dfnc_te(1);            
    end
end         % for data in class cnt
te_confidence = confidence./confusion;
te_confusion = confusion/N_te;
te_confusion=round(te_confusion*10000)/10000;
te_CCR = sum(diag(te_confusion))/sum(sum(te_confusion));
te_confidence(isnan(te_confidence))=0;
te_confidence=round(te_confidence*10000)/10000;
