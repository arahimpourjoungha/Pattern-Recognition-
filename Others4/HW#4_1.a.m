close all
clear all
%%%%%%%%%%%%%%%%%%%%%%%%
load X_trainlc.mat
load X_testlc.mat
data_train=[];
data_test=[];
class_train=[];
class_test=[];
%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:10
    data_train=[data_train ;X_trainlc(:,:,i)'];
    class_train=[class_train ;i*ones(1500,1)]; 
    data_test=[data_test ;X_testlc(:,:,i)'];
    class_test=[class_test ;i*ones(500,1)];  
end

n=1500;

numbclass=10;
[n1,n2]=size(data_train);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
c = 900;
lambda = 1e-6;
kerneloption=10;
kernel='poly';
ver =0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
t1=cputime;
[xsup,w,b,nbsv]=svmmulticlassoneagainstall(data_train,class_train,numbclass,c,lambda,kernel,kerneloption,ver);
t2=cputime
[label_train,max_dfnc_train,dfnc_train] = svmmultival(data_train,xsup,w,b,nbsv,kernel,kerneloption);
t3=cputime;
[label_test,max_dfnc_test,dfnc_test] = svmmultival(data_test,xsup,w,b,nbsv,kernel,kerneloption);
 t4=cputime;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Numb_tr=1500;
dfunc=zeros(Numb_tr,numbclass);
dfunc = dfnc_train;     
confusion = zeros(10,10);   
    for j = 1:Numb_tr*10
        i=floor((j-1)/Numb_tr)+1;
        [so_dfnc_tr,index] = sort(dfunc(j,:),'descend')
        confusion(index(1),class_train(j)) = confusion(index(1),class_train(j))+1;           
    end
tr_confusion = confusion/Numb_tr;
tr_CCR = sum(diag(tr_confusion))/sum(sum(tr_confusion));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N_t=400;
dfunc=zeros(N_t,numbclass);
dfunc = dfnc_test;      % Discrimint Function values
confusion = zeros(10,10);   
    for j = 1:N_t*10
        [so_dfnc_tr,index] = sort(dfunc(j,:),'descend')
        confusion(index(1),class_test(j)) = confusion(index(1),class_test(j))+1;           
    end
te_confusion = confusion/N_t;
te_CCR = sum(diag(te_confusion))/sum(sum(te_confusion));
fprintf('training time=%.2f(sec)\n',t2-t1);
fprintf('evaluted time(training data)=%.2f(sec)\n',t3-t2)
fprintf('evaluted time(testing data)=%.2f(sec)\n',t4-t3);
