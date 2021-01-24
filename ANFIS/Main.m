clc
clear all
tic
Data=xlsread('nn.xlsx');
numOfData=size(Data,1);
figure;
plot(Data(:,5),Data(:,6),'*');
Data=Data(randperm(numOfData),:);
Data=removeDC(Data);

[V,E,D] =pca(Data(:,1:5)');
%feature=1;
input=Data(:,1:5)';%V(1:feature,:)*Data(:,1:5)';
Data=[ input' Data(:,6)];
plot(Data(:,1),Data(:,2),'*');
ddd='';
input(ddd);

numOfTrainData=80;
 numMFs = 2;
 mfType = 'gbellmf';
 epoch_n = 20;
 in_fis = genfis1(Data(1:numOfTrainData,:),numMFs,mfType);
out_fis = anfis(Data(1:numOfTrainData,:));
a=evalfis(Data(69:numOfData,1:5),out_fis);
b=Data(69:numOfData,6);
MSE=mean((b-a).^2)
MAE_MAT_with=abs(b-a);
MAE=mean(MAE_MAT_with)
MRE=abs((b-a)./b)
x=mae(MRE)
figure;
plot (Data(69:80,6),a,'*','Color','R');
title('result with 5 features');
xlabel('nn output')
ylabel('target')
Data=xlsread('nn.xlsx');
numOfData=size(Data,1);
Data=Data(randperm(numOfData),:);
Data=removeDC(Data);

[V,E,D] =pca(Data(:,1:5)');
%feature=3;
input=Data(:,1:4)';%V(1:feature,:)*Data(:,1:5)';
Data=[ input' Data(:,6)];

numOfTrainData=80;
 numMFs = 2;
 mfType = 'gbellmf';
 epoch_n = 20;
 in_fis = genfis1(Data(1:numOfTrainData,:),numMFs,mfType);
out_fis = anfis(Data(1:numOfTrainData,:));
a=evalfis(Data(69:numOfData,1:4),out_fis);
b=Data(69:numOfData,5);
MSEw=mean((b-a).^2)
MAE_MAT_without=abs(b-a)
MAEw=mean(MAE_MAT_with)
xw=mae(abs((b-a)./b))
figure;
plot (Data(69:80,5),a,'*','Color','R');
title('result without 5th feature');
xlabel('nn output')
ylabel('target')
return
gamma=0.75;
eps1=0.005;
m2=8; 
 
a=-1;b=1; 
w0=a+(b-a)*rand(1,m1*m2); 
R1=1;
R2=1;
K0=1;
d=1;
S11_file=csvread('S11.csv');
S21_file=csvread('S21.csv');
s11=zeros(numOfFrequence,1);
s21=zeros(numOfFrequence,1);
for i=1:numOfFrequence
   s11(i)=S11_file(i,2)+S11_file(i,3)*i;
   s21(i)=S21_file(i,2)+S21_file(i,3)*i;
end
Zw=sqrt((1+S11/R1^2).^2 - s21.^2*(R1^2*R2^2))./...
    (sqrt((1-S11/R1^2).^2 - s21.^2*(R1^2*R2^2)));
T=(s21/(R1*R2))./(1-(S11/R1^2).*((Zw-1)./(Zw+1)));
n=(imag(ln(T))+2*pi-real(ln(T).*i))/k0*d;
epsilon=n./Zw;
mu=n.*Zw; 
for i=1:2 
    switch i 
        case 1,beta=0.75;
        otherwise,beta=0.25; 
    end 
    c=[2/7*(0:m1-1)-1;2/7*(0:m2-1)-1];
    sigma=0.1213*ones(2,m1);
    w=w0; 
    mu=zeros(2,m1);
    alpha=zeros(1,m1*m2);
    alpha_=zeros(1,m1*m2);
 
    delta2=zeros(2,m1);
    dw=zeros(1,m1*m2);
    dc=zeros(2,m1);
    dsigma=zeros(2,m1);
 

    err=1;
    er=[]; 
    counter=0;
    while(err>=eps1) 
        Par_E_w=zeros(1,m1*m2);
        Par_E_c=zeros(2,m1);
        Par_E_sigma=zeros(2,m1);
        E=0; 
        for x1=-1:2/19:1 
            for x2=-1:2/19:1 
                yd=sin(pi*x1)*cos(pi*x2);
               
                mu(1,:)=exp(-(x1-c(1,:)).^2./sigma(1,:).^2);
                mu(2,:)=exp(-(x2-c(2,:)).^2./sigma(2,:).^2); 
                s=zeros(2,m1,m1*m2);
                for m=1:m1 
                    for n=1:m2 
                        alpha((m-1)*m2+n)=min(mu(1,m),mu(2,n));
                        if mu(1,m)<=mu(2,n) 
                            s(1,m,(m-1)*m2+n)=1; 
                        end 
                        if mu(1,m)>=mu(2,n) 
                            s(2,n,(m-1)*m2+n)=1; 
                        end 
                    end 
                end 
                alpha_=alpha/sum(alpha);
             
                y=alpha_*w.';
                E=E+1/2*(yd-y)^2;
                
                delta5=yd-y; 
                delta4=delta5*w; 
 
                for k=1:m1*m2 
                    delta3(k)=delta4(k)*(sum(alpha)-alpha(k))./sum(alpha)^2; 
                end 
                for m=1:2 
                    for n=1:m1 
                        delta2(m,n)=0; 
                        for l=1:m1*m2 
                            delta2(m,n)=delta2(m,n)+delta3(l)*s(m,n,l)*mu(m,n); 
                        end 
                    end 
                end 
             
                Par_E_w=Par_E_w-delta5*alpha_;
                Par_E_c(1,:)=Par_E_c(1,:)-2*delta2(1,:).*(x1-c(1,:))./sigma(1,:).^2; 
                Par_E_c(2,:)=Par_E_c(2,:)-2*delta2(2,:).*(x2-c(2,:))./sigma(2,:).^2; 
                Par_E_sigma(1,:)=Par_E_sigma(1,:)-2*delta2(1,:).*(x1-c(1,:)).^2./sigma(1,:).^3; 
                Par_E_sigma(2,:)=Par_E_sigma(2,:)-2*delta2(2,:).*(x2-c(2,:)).^2./sigma(2,:).^3; 
            end 
        end 
   
        num=20*20; 
        Par_E_w=Par_E_w/num; 
        Par_E_c=Par_E_c/num; 
        Par_E_sigma=Par_E_sigma/num; 
     
        dw=-beta*Par_E_w+gamma*dw; 
        dc=-beta*Par_E_c+gamma*dc; 
        dsigma=-beta*Par_E_sigma+gamma*dsigma; 
     
        w=w+dw; 
        c=c+dc; 
        sigma=sigma+dsigma; 
     
        counter=counter+1; 
        er(counter)=E/num; 
        err=E/num; 
     if counter>1000 
         break; 
     end 
    end 
 
  
    xx1=-1:2/19:1; 
    xx2=-1:2/19:1; 
    yd1=zeros(20,20); 
    for m=1:20 
        for n=1:20 
            yd1(m,n)=sin(pi*xx1(m))*cos(pi*xx2(n));
         
            mu(1,:)=exp(-(xx1(m)-c(1,:)).^2./sigma(1,:).^2);
            mu(2,:)=exp(-(xx2(n)-c(2,:)).^2./sigma(2,:).^2); 
         
            for k=1:m1         
                for l=1:m2 
                    alpha((k-1)*m2+l)=min(mu(1,k),mu(2,l)); 
                end 
            end 
         
            alpha_=alpha/sum(alpha);
            yr(m,n)=alpha_*w.';     
        end 
    end 
    errorf1=1/2*(yd1-yr).^2;
     
     
    xxx1=-1:2/11:1; 
    xxx2=-1:2/11:1; 
    yd2=zeros(12,12); 
    for m=1:12 
        for n=1:12 
            yd2(m,n)=sin(pi*xxx1(m))*cos(pi*xxx2(n));
         
            mu(1,:)=exp(-(xxx1(m)-c(1,:)).^2./sigma(1,:).^2);
            mu(2,:)=exp(-(xxx2(n)-c(2,:)).^2./sigma(2,:).^2); 
         
            for k=1:m1              
                for l=1:m2 
                    alpha((k-1)*m2+l)=min(mu(1,k),mu(2,l)); 
                end 
            end 
         
            alpha_=alpha/sum(alpha);
            yr2(m,n)=alpha_*w.';     
        end 
    end 
    errorf2=1/2*(yd1-yr).^2;
     
     

    figure(i); 
    sn=sprintf(beta,gamma); 
     
    X=ones(size(xx2.'))*xx1; 
    Y=xx2.'*ones(size(xx1)); 
     
    subplot(2,2,1); 
    surf(X,Y,yd1); 
    xlabel('x1'); 
    ylabel('x2'); 
     
    title(sn); 
     
    subplot(2,2,3); 
    surf(X,Y,yr); 
    xlabel('x1'); 
    ylabel('x2'); 
   
    title(sn); 
     
    subplot(2,2,2); 
    plot(er); 
    xlabel('ÑµÁ·²½Êý'); 
    ylabel('Îó²î'); 
    title(sn); 
     
    subplot(2,2,4); 
    surf(X,Y,errorf1); 
    xlabel('x1'); 
    ylabel('x2'); 
    zlabel('Îó²î'); 
    title(sn); 
     
    figure(i+2); 
    sn=sprintf(beta,gamma); 
 
    X=ones(size(xxx2.'))*xxx1; 
    Y=xxx2.'*ones(size(xxx1)); 
    subplot(2,1,1); 
    surf(X,Y,yd2); 
    xlabel('x1'); 
    ylabel('x2'); 
     
    title(sn); 
     
    subplot(2,1,2); 
    surf(X,Y,yr2); 
    xlabel('x1'); 
    ylabel('x2'); 
    
    title(sn); 
     
    beta 
    counter     
end

toc