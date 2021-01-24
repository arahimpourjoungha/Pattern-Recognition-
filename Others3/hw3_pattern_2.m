


load AllData.mat
pdf_mean = zeros(1,50,10);
pdf_var = zeros(50,50,10);
N_tr = 1500;
N_te = 500;
for i = 1:10
    k=Train_Data(:,:,i);
    pdf_mean(:,:,i) = (sum(k')./N_tr);
    pdf_var(:,:,i) = cov(k',1);
end
%%%
landa=zeros(10,10);
for i=1:10
    for j=1:10
        if i==j
            landa(i,j)=0;
        else
            if i>=7
                landa(i,j)=10;
            else
                landa(i,j)=1;
              end
        end
    end
end
landa(7,8)=100;
%%% Classifying the train data
dfnc = zeros(N_tr,10,10);      % Discrimint Function values
confidence = zeros(10,10);
confusion = zeros(10,10);
f = zeros(N_tr,10,10);      % pdf values
for i = 1:10
    for j = 1:10
        f(:,j,i) = mvnpdf(Train_Data(:,:,i)',pdf_mean(:,:,j),pdf_var(:,:,j));
    end     % for computing propability in pdf cnt1
    for j = 1:N_tr
        for k = 1:10
            dfnc(j,k,i) = f(j,:,i)*landa(:,k);
        end     % for computing discriminent function value
        [so_dfnc_tr,index] = sort(dfnc(j,:,i),'ascend');
        confusion(index(1),i) = confusion(index(1),i)+1;
        confidence(index(1),i) = confidence(index(1),i)+(so_dfnc_tr(2)-so_dfnc_tr(1))/so_dfnc_tr(2);            
    end
end         % for data in class cnt
tr_confidence = confidence./confusion;
tr_confusion = confusion/N_tr;
tr_confusion=round(tr_confusion*10000)/10000;
tr_CCR = sum(diag(tr_confusion))/sum(sum(tr_confusion));
tr_confidence(isnan(tr_confidence))=0;
tr_confidence=round(tr_confidence*10000)/10000;
%%% Classifying the test data
dfnc = zeros(N_te,10,10); 
f = zeros(N_te,10,10); 
confidence = zeros(10,10);
confusion = zeros(10,10);
for i = 1:10
    for j = 1:10
        f(:,j,i) = mvnpdf(Test_Data(:,:,i)',pdf_mean(:,:,j),pdf_var(:,:,j));
    end     % for computing propability in pdf cnt1
    for j = 1:N_te
        for k = 1:10
            dfnc(j,k,i) = f(j,:,i)*landa(:,k);
        end     % for computing discriminent function value
        [so_dfnc_te,index] = sort(dfnc(j,:,i),'ascend');
        confusion(index(1),i) = confusion(index(1),i)+1;
        confidence(index(1),i) = confidence(index(1),i)+(so_dfnc_te(2)-so_dfnc_te(1))/so_dfnc_te(2);            
    end
end         % for data in class cnt
te_confidence = confidence./confusion;
te_confusion = confusion/N_te;
te_confusion=round(te_confusion*10000)/10000;
te_CCR = sum(diag(te_confusion))/sum(sum(te_confusion));
te_confidence(isnan(te_confidence))=0;
te_confidence=round(te_confidence*10000)/10000;


