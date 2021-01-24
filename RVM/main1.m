tic
clear all;
clc;
load X_test_input;
load X_test_target;
load X_train_input;
load X_train_target;
X_test_input=X_test_input';
X_test_target=X_test_target';
X_train_input=X_train_input';
X_train_target=X_train_target';

TrainingDataSet=prtDataSetClass(X_train_input,X_train_target);
TestDataSet=prtDataSetClass(X_test_input,X_test_target);
classifier= prtClassRvm;
classifier = classifier.train(TrainingDataSet);
classified_test = run(classifier, TestDataSet);
clc;

classified_train = run(classifier, TrainingDataSet);

confusionmatrix_test=zeros(2,2);
for i=1:210
    if classified_test.data(i)>.5
        confusionmatrix_test(X_test_target(i),2)=confusionmatrix_test(X_test_target(i),2)+1;
    else
        confusionmatrix_test(X_test_target(i),1)=confusionmatrix_test(X_test_target(i),1)+1;
    end
end
display(confusionmatrix_test);

confusionmatrix_train=zeros(2,2);
for i=1:489
    if classified_train.data(i)>.5
        confusionmatrix_train(X_train_target(i),2)=confusionmatrix_train(X_train_target(i),2)+1;
    else
        confusionmatrix_train(X_train_target(i),1)=confusionmatrix_train(X_train_target(i),1)+1;
    end
end
display(confusionmatrix_train);
toc
