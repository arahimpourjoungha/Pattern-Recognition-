

load X_train.mat
class = 8;
kmax = 100;
tol = 0.001;
eta = 1.5;
W = zeros(51,class-1);
for i= 1:class-1
    o=other_label(i,X_train)';
    points=X_train(:,:,i)';
    W(:,i) = Ho_Kashyap(points,o,kmax,tol,eta);  
end

%%% Train classification
N_tr = 1400;
confusion = zeros(8,8);
for i = 1:8
    for j = 1:N_tr
        dfnc = [1 X_train(:,j,i)']*W;
        l = class;
        for k = 1:class-1
            if dfnc(k) >= 0
                l = k;
                break;
            end
        end
        confusion(l,i) = confusion(l,i)+1;
    end
end
tr_confidence = confidence./confusion(1:8,:);
tr_confusion = confusion/N_tr;
tr_CCR = sum(diag(tr_confusion))/sum(sum(tr_confusion));
tr_confusion=round(tr_confusion*10000)/10000;
tr_confidence(isnan(tr_confidence))=0;
tr_confidence=round(tr_confidence*10000)/10000;

%%% Test classification
N_te = 400;
confusion = zeros(8,8);
for i = 1:8
    for j = 1:N_te
        dfnc = [1 X_test(:,j,i)']*W;
        l = class;
        for k = 1:class-1
            if dfnc(k) >= 0
                l = k;
                break;
            end
        end
        confusion(l,i) = confusion(l,i)+1;
    end
end
te_confidence = confidence./confusion;
te_confusion = confusion/N_te;
te_CCR = sum(diag(te_confusion))/sum(sum(te_confusion));
te_confidence(isnan(te_confidence))=0;
te_confusion=round(te_confusion*10000)/10000;
te_confidence=round(te_confidence*10000)/10000;
