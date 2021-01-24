load X_trainlc.mat
load X_testlc.mat
datatrain=[];
datatest=[];
classtrain=[];
classtest=[];
for i=1:10
    datatrain=[datatrain ;X_trainlc(:,:,i)'];
    classtrain=[classtrain ;i*ones(1500,1)];  
    datatest=[datatest ;X_testlc(:,:,i)'];
    classtest=[classtest ;i*ones(500,1)];  
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
numb=1500;
numbclass=10;
[n1,n2]=size(datatrain);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
c = 1000;
landa = 1e-7;
kernelop= 1.1;
kernel='gaussian';
ver=1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
time1=cputime;
[xsup,w,b,nbsv]=svmmulticlassoneagainstone(datatrain,classtrain,numbclass,c,landa,kernel,kernelop,ver);
time2=cputime;
[label_train,max_dfnc_train] = svmmultivaloneagainstone(datatrain,xsup,w,b,nbsv,kernel,kernelop);
time3=cputime;
[label_test,max_dfnc_test] = svmmultivaloneagainstone(datatest,xsup,w,b,nbsv,kernel,kernelop);
time4=cputime;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N_training=1500;
confusion = zeros(10,10);   
    for j = 1:N_training*10
        confusion(label_train(j),classtrain(j)) = confusion(label_train(j),classtrain(j))+1;    
    end
training_confusion = confusion/N_training;
training_confusion=round(training_confusion*10000)/10000;
train_CCR = sum(diag(training_confusion))/sum(sum(training_confusion));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N_tec=400;
confusion = zeros(10,10);   
    for j = 1:N_tec*10
        confusion(label_test(j),classtest(j)) = confusion(label_test(j),classtest(j))+1;        
    end
teC_confusion = confusion/N_tec;
teC_confusion=round(teC_confusion*10000)/10000;
teC_CCR = sum(diag(teC_confusion))/sum(sum(teC_confusion));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('training time=%.2f(sec)\n',time2-time1);
fprintf('evaluted time(training data)=%.2f(sec)\n',time3-time2);
fprintf('evaluted time(testing data)=%.2f(sec)\n',time4-time3);
