close all
clear all
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load X_trainlc.mat
load X_testlc.mat
data_train=[];
data_test=[];
class_train=[];
class_test=[];
for i=1:10
    data_train=[data_train ;X_trainlc(:,:,i)'];
    class_train=[class_train ;i*ones(1500,1)];  
    data_test=[data_test ;X_testlc(:,:,i)'];
    class_test=[class_test ;i*ones(500,1)];  
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n=500;
numbclass=10;
[n1,n2]=size(data_train);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
c=900;
lambda = 1e-7;
kernelop=10;
cernel='poly';
ver=1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
time1=cputime;
[xsup,w,b,nbsv]=svmmulticlassoneagainstone(data_train,class_train,numbclass,c,lambda,cernel,kernelop,ver);
time2=cputime;
[label_train,max_dfnc_train] = svmmultivaloneagainstone(data_train,xsup,w,b,nbsv,cernel,kernelop);
time3=cputime;
[labeltest,maxdfnctest] = svmmultivaloneagainstone(data_test,xsup,w,b,nbsv,cernel,kernelop);
time4=cputime;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N_train=1500;
confusion = zeros(10,10);   
    for j = 1:N_train*10
        confusion(label_train(j),class_train(j)) = confusion(label_train(j),class_train(j))+1;    
    end
tr_confusion = confusion/N_train;
tr_confusion=round(tr_confusion*10000)/10000;
tr_CCR = sum(diag(tr_confusion))/sum(sum(tr_confusion));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N_tec=500;
confusion = zeros(10,10);   
    for j = 1:N_tec*10
        confusion(labeltest(j),class_test(j)) = confusion(labeltest(j),class_test(j))+1;        
    end
te_confusion = confusion/N_tec;
te_confusion=round(te_confusion*10000)/10000;
te_CCR = sum(diag(te_confusion))/sum(sum(te_confusion));
fprintf('training time=%.2f(sec)\n',time2-time1);
fprintf('evaluted time(training data)=%.2f(sec)\n',time3-time2);
fprintf('evaluted time(testing data)=%.2f(sec)\n',time4-time3);



