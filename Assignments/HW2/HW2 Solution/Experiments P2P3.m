%Mohammad Alshurbaji
%Problem 2: Assignment 2
n = 100; %Number of samples
p = 1000; %Number of features, p>>n
x = randn(n,p); %Assuming that each column is centered
%Normal Method:
tic; %Timer start
cov1 = (x'*x) / (n-1);%covariance of the Normal method
[eigVecs1, eigVals1] = eig(cov1); %eig decomposition
[eigVals1, idx1] = sort(diag(eigVals1), 'descend');%Sorting
eigVecs1 = eigVecs1(:, idx1); %Sort eigenvectors
end_time1 = toc;

%Fast approach:
tic; %Timer start
cov2 = (x*x') / (n-1);%covariance of the Fast method
[eigVecs2, eigVals2] = eig(cov2); %eig decomposition
[eigVals2, idx2] = sort(diag(eigVals2), 'descend');%Sorting
eigVecs2 = eigVecs2(:, idx2); %Sort eigenvectors

eigVecs_final = x'*eigVecs2;% to find for x'x
%Normalization
for i = 1:size(eigVecs_final,2)
    eigVecs_final(:,i) = eigVecs_final(:,i) / norm(eigVecs_final(:,i));
end
end_time2 = toc;

fprintf('Time for Normal Approach: %.4f seconds \n', end_time1);
fprintf('Time for Fast Approach: %.4f seconds \n', end_time2);