clc;
tic
load X_test_input;
load X_test_target;
X_test_input=X_test_input';
X_test_target=X_test_target';
X_train_input=X_train_input';
X_train_target=X_train_target';
TrainingDataSet=prtDataSetClass(X_train_input,X_train_target);
TestDataSet=prtDataSetClass(X_test_input,X_test_target);
classifier= prtClassRvm;
classifier = classifier.train(TrainingDataSet);
clc;

classified = run(classifier, TestDataSet);
confusionmatrix=zeros(2,2);
for i=1:210
    if classified.data(i)>.5
        confusionmatrix(X_test_target(i),2)=confusionmatrix(X_test_target(i),2)+1;
    else
        confusionmatrix(X_test_target(i),1)=confusionmatrix(X_test_target(i),1)+1;
    end
end
display(confusionmatrix);
toc