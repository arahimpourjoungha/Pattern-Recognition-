function [V,E,D] = pca(X)

covarianceMatrix = X*X'/size(X,2);
[E, D] = eig(covarianceMatrix);
[~,order] = sort(diag(-D));
E = E(:,order);
d = diag(D); 
dT = real(d.^(-0.5));
DT = diag(dT(order));
D = diag(d(order));
V = DT*E';