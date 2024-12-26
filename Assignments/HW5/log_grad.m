function G=log_grad(y, X, B) 
    [n,d] = size(X);%n: number of samples, d: number of features
    K = size(B,2) + 1; %Total number of classes

%compute gradient 
    XB = X * B;
    expXB = exp(XB);
    prob = expXB ./ (1 + sum(expXB, 2));

    prob = [prob, 1 - sum(prob, 2)];
    
    G = zeros(d,K-1);
    for k = 1:K-1
        indicator = (y == k);  % Indicator vector for class k
        G(:, k) = X' * (indicator - prob(:, k));  % Gradient for class k
    end
  

end