
load X_train.mat
class = 8;
kmax = 100;
tol = 0.001;
eta = 1.5;
W = [];
label=[];
for i = 1:class-1
    for j = i+1:class
        o=X_train(:,:,j)';
        points=X_train(:,:,i)';
        W = [W Ho_Kashyap(points,o,kmax,tol,eta)];
        label =[label [i;j]];
    end
end

%%% Train classification
N_tr=1400;
confusion = zeros(8,8);
for i = 1:8
    for j = 1:N_tr
        dfnc = [1 X_train(:,j,i)']*W;
        codition = [label(1,find(dfnc>=0)) label(2,find(dfnc<0))];
        clabel = zeros(8,1);
        for k = 1:8
            clabel(k) = sum(codition == k);
        end
        [sort_clabel,index] = sort(clabel,'descend');
        confusion(index(1),i) = confusion(index(1),i)+1;
    end
end
tr_confusion = confusion/N_tr;
tr_CCR = sum(diag(tr_confusion))/sum(sum(tr_confusion));
tr_confusion=round(tr_confusion*10000)/10000;

%%% Test classification
N_te = 400;
confusion = zeros(8,8);
for i = 1:8
    for j = 1:N_te
        dfnc = [1 X_test(:,j,i)']*W;
        codition = [label(1,find(dfnc>=0)) label(2,find(dfnc<0))];
        clabel = zeros(8,1);
        for k = 1:8
            clabel(k) = sum(codition == k);
        end
        [sort_clabel,index] = sort(clabel,'descend');
        confusion(index(1),i) = confusion(index(1),i)+1;
    end
end
te_confusion = confusion/N_te;
te_CCR = sum(diag(te_confusion))/sum(sum(te_confusion));
te_confusion=round(te_confusion*10000)/10000;
