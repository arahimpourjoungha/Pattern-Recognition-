clc
close all
clear all
J=3;
% Sample 
Samples = [];
for k = 1:250/4
    pre_Samples=[normrnd(1.5,sqrt(0.01)); normrnd(1.5,sqrt(0.01));normrnd(1.0,sqrt(0.01));normrnd(2.0,sqrt(0.04))];
    Samples = [Samples; pre_Samples];
end
N=length(Samples);
%Initializ
cov=zeros(J,J);
for j=1:J
    mu(j)=j;
    cov(j,j)=j;
    P(j)=1/J;
end
theta(1,:)=mu;
theta(2,:)=[cov(1,1),cov(2,2),cov(3,3)];
theta(3,:)=P;
while(1)
    for j=1:J
        for k=1:N
            f(k,j)=(1/sqrt(2*pi*cov(j,j)))*exp((-(Samples(k)-mu(j))^2)/2*cov(j,j));
%             f(k,2)=(1/sqrt(2*pi*cov(2,2)))*exp((-(Samples-mu(2))^2)/2*cov(2,2));
%             f(k,3)=(1/sqrt(2*pi*cov(3,3)))*exp((-(Samples-mu(3))^2)/2*cov(3,3));
        end
    end
    h=0;
    for j=1:J
        h=h+sum(f(:,j))*P(j);
    end
    for j=1:J
        for k=1:N
            Pj(k,j)=(f(k,j)*P(j))/h;
        end
    end
    for j=1:J
        mu(j)=0;
        cov(j,j)=0;
        for k=1:N
            mu(j)=mu(j)+Pj(k,j)*Samples(k);
            cov(j,j)=cov(j,j)+Pj(k,j)*(Samples(k)-theta(2,j))^2;
        end
        mu(j)=mu(j)/sum(Pj(:,j));
        cov(j,j)=cov(j,j)/sum(Pj(:,j));
        P(j)=(1/N)*sum(Pj(:,j));
    end
    if(norm(theta-[mu;cov(1,1) cov(2,2) cov(3,3);P])<0.1)
        break
    end
    theta=[mu;cov(1,1) cov(2,2) cov(3,3);P];
    end
        
    
    