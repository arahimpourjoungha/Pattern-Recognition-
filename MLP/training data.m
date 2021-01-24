clc
clear all
load Data
X_trainlc = zeros(51,1500,10);
X_testlc = zeros(51,500,10);
for i = 1:10
A = 3*X_train(14,:,i)+2*X_train(22,:,i)-2*X_train(36,:,i);
X_trainlc(:,:,i) = cat(1,X_train(:,:,i),A);
end
for i = 1:10
A = 3*X_test(14,:,i)+2*X_test(22,:,i)-2*X_test(36,:,i);
X_testlc(:,:,i) = cat(1,X_test(:,:,i),A);
end
save X_testlc
save X_trainlc