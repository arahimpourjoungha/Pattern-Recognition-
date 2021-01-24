

%%% Computing the pdf s
load X_train.mat
k_NN = 8;
N_tr = 1400;
N_te = 400;
h=1;
confidence = zeros(8,8);
confusion = zeros(8,8);
for i = 1:8
    for j = 1:N_tr        
        % finding K nearest neighbor from data in class k
        dis = zeros(N_tr,8);
        for k = 1:8
            dis(:,k) = sum((X_train(:,:,k)-repmat(X_train(:,j,i),1,N_tr)).^2);
        end
        dist=dis(:);
        [sorted_d,index] = sort(dist,'ascend');
        class = zeros(1,8);
         for k = 1:k_NN
            class(fix((index(k)-1)/N_tr)+1) = class(fix((index(k)-1)/N_tr)+1)+1;
         end
        [s_class,index_class] = sort(class,'descend');
        confusion(index_class(1),i) = confusion(index_class(1),i)+1; 
        confidence(index_class(1),i) = confidence(index_class(1),i)+(s_class(1)-s_class(2))/s_class(1);    
    end
end
tr_confidence = confidence./confusion(1:8,:);
tr_confusion = confusion/N_tr;
tr_CCR = sum(diag(tr_confusion))/sum(sum(tr_confusion));
tr_confusion=round(tr_confusion*10000)/10000;
tr_confidence(isnan(tr_confidence))=0;
tr_confidence=round(tr_confidence*10000)/10000;

%%%
confidence = zeros(8,8);
confusion = zeros(8,8);
for i = 1:8
    for j = 1:N_te       
        % finding K nearest neighbor from data in class k
        dis = zeros(N_tr,8);
        for k = 1:8
            dis(:,k) = sum((X_train(:,:,k)-repmat(X_test(:,j,i),1,N_tr)).^2);
        end
        dist=dis(:);
        [sorted_d,index] = sort(dist,'ascend');
        class = zeros(1,8);
         for k = 1:k_NN
            class(fix(index(k)/N_tr)+1) = class(fix(index(k)/N_tr)+1)+1;
         end
        [s_class,index_class] = sort(class,'descend');
        confusion(index_class(1),i) = confusion(index_class(1),i)+1; 
        confidence(index_class(1),i) = confidence(index_class(1),i)+(s_class(1)-s_class(2))/s_class(1);    
    end
end
te_confidence = confidence./confusion;
te_confusion = confusion/N_te;
te_CCR = sum(diag(te_confusion))/sum(sum(te_confusion));
te_confidence(isnan(te_confidence))=0;
