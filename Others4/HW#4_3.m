load X_trainlc.mat
load X_testlc.mat
Numclass = 10;
coefmax = 50;
tolerance = 0.01;
e = 0.01;
W = [];
label=[];
time1=cputime;
for i = 1:Numclass-1
    for j = i+1:Numclass
        o=X_trainlc(:,:,j)';
        point=X_trainlc(:,:,i)';
        W = [W Ho_Kashyap(point,o,coefmax,tolerance,e)];
        label =[label [i;j]];
    end
end
time2=cputime;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N_training=1500;
confusion = zeros(10,10);
for i = 1:10
    for j = 1:N_training
        dfunc = [1 X_trainlc(:,j,i)']*W;
        condition = [label(1,find(dfunc>=0)) label(2,find(dfunc<0))];
        classlabel = zeros(10,1);
        for k = 1:10
            classlabel(k) = sum(condition == k);
        end
        [sort_clabel,index] = sort(classlabel,'descend');
        confusion(index(1),i) = confusion(index(1),i)+1;
    end
end
time3=cputime;
training_confusion = confusion/N_training;
training_CCR = sum(diag(training_confusion))/sum(sum(training_confusion));
training_confusion=round(training_confusion*10000)/10000;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N_tec = 500;
confusion = zeros(10,10);
for i = 1:10
    for j = 1:N_tec
        dfunc = [1 X_testlc(:,j,i)']*W;
        condition = [label(1,find(dfunc>=0)) label(2,find(dfunc<0))];
        classlabel = zeros(10,1);
        for k = 1:10
            classlabel(k) = sum(condition == k);
        end
        [sort_clabel,index] = sort(classlabel,'descend');
        confusion(index(1),i) = confusion(index(1),i)+1;
    end
end
time4=cputime;
tec_confusion = confusion/N_tec;
tec_CCR = sum(diag(tec_confusion))/sum(sum(tec_confusion));
tec_confusion=round(tec_confusion*10000)/10000;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('training time=%.2f(sec)\n',time2-time1);
fprintf('evaluted time(training data)=%.2f(sec)\n',time3-time2);
fprintf('evaluted time(testing data)=%.2f(sec)\n',time4-time3);