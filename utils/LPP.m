function P = LPP(data,W,options)
data = double(data);
W = double(W);
D = diag(sum(W,2));
L = D - W;
Sl = data'*L*data;
Sd = data'*D*data;
Sl = (Sl+Sl')/2;
Sd = (Sd+Sd')/2;

Sl = Sl + options.alpha*eye(size(Sl,2));
opts.disp = 0;
[P,Diag] = eigs(double(Sd),double(Sl),options.ReducedDim,'la',options);
for i = 1:size(P,2)
    if (P(1,i)<0)
        P(:,i) = P(:,i)*-1;
    end
end
end
function K = kernel_meda(ker,X,sigma)
    switch ker
        case 'linear'
            K = X' * X;
        case 'rbf'
            n1sq = sum(X.^2,1);
            n1 = size(X,2);
            D = (ones(n1,1)*n1sq)' + ones(n1,1)*n1sq -2*X'*X;
            K = exp(-D/(2*sigma^2));        
        case 'sam'
            D = X'*X;
            K = exp(-acos(D).^2/(2*sigma^2));
        otherwise
            error(['Unsupported kernel ' ker])
    end
end