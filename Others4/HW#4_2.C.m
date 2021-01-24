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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
numb=1500;
numbclass=10;
[n1,n2]=size(datatrain);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
c = 1000;
lambda = 1e-7;
kernelop= [0.001,0];
kernel='mlp';
ver = 0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
time2=cputime;
[label_train,max_dfnc_train] = svmmultivaloneagainstone(datatrain,xsup,w,b,nbsv,kernel,kernelop);
time3=cputime;
[label_test,max_dfnc_test] = svmmultivaloneagainstone(datatest,xsup,w,b,nbsv,kernel,kernelop);
time4=cputime;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N_train=1500;
confusion = zeros(10,10);   
    for j = 1:N_train*10
        confusion(label_train(j),classtrain(j)) = confusion(label_train(j),classtrain(j))+1;    
    end
training_confusion = confusion/N_train;
training_confusion=round(training_confusion*10000)/10000;
training_CCR = sum(diag(training_confusion))/sum(sum(training_confusion));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N_te=500;
confusion = zeros(10,10);   
    for j = 1:N_te*10
        confusion(label_test(j),classtest(j)) = confusion(label_test(j),classtest(j))+1;        
    end
tec_confusion = confusion/N_te;
tec_confusion=round(tec_confusion*10000)/10000;
tec_CCR = sum(diag(tec_confusion))/sum(sum(tec_confusion));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('training time=%.2f(sec)\n',time2-t1);
fprintf('evaluted time(training data)=%.2f(sec)\n',time3-time2);
fprintf('evaluted time(testing data)=%.2f(sec)\n',time4-time3);
)
