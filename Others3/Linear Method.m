function W = Ho_Kashyap(x,y,kmax,tol,eta)
% Design a linear classifier according to the Ho-Kashyap methods
Y = [ones(size(x,1),1) x; -ones(size(y,1),1) -y];
b = ones(size(Y,1),1);
W = inv(Y'*Y)*Y'*b;
for k = 1:kmax
    e = Y*W-b;
    ep = (e+abs(e))/2;
    b = b+2*eta*ep;
    W = inv(Y'*Y)*Y'*b;
    if max(abs(e)) <= tol
        break;
    end
end
