close all;clear all;

load usps_digital.mat

%% lambda=0

lambda=0;
c=1e-4;
[B_0, te_err_0, tr_err_0, obj_0]=log_reg(tr_y, tr_X, te_y, te_X, lambda, c);

%% plot result

figure;
plot(1:length(obj_0)-1, obj_0(1:end-1), 'b-o', 'LineWidth',1, 'MarkerSize', 1);
xlabel('Number of Iterations','FontSize',15);
ylabel('Objective Value','FontSize',15);
title(['Objective Value for \lambda = ', num2str(lambda)], 'FontSize', 15);
set(gca, 'FontSize',10);
drawnow
saveas(gcf, ['obj_lambda_' num2str(lambda) '.fig']);

 
figure;
plot(1:length(tr_err_0)-1, tr_err_0(1:end-1), 'b-o', 'LineWidth', 1, 'MarkerSize', 1);
xlabel('Number of Iterations','FontSize',15);
ylabel('Training Accuracy ','FontSize',15);
 title(['Training Accuracy for \lambda = ', num2str(lambda)], 'FontSize', 15);
set(gca, 'FontSize',15);
drawnow
saveas(gcf, ['train_err_lambda_' num2str(lambda) '.fig']);
 

figure;
plot(1:length(te_err_0)-1, te_err_0(1:end-1), 'b-o', 'LineWidth', 1, 'MarkerSize', 1);
xlabel('Number of Iterations','FontSize',15);
ylabel('Testing Accuracy ','FontSize',15);
title(['Testing Accuracy for \lambda = ', num2str(lambda)], 'FontSize', 15);
set(gca, 'FontSize',15);
drawnow
saveas(gcf, ['test_err_lambda_' num2str(lambda) '.fig']);

%% lambda=1
lambda=1;
[B_1, te_err_1, tr_err_1, obj_1]=log_reg(tr_y, tr_X, te_y, te_X, lambda, c);

%% plot result

figure;
plot(1:length(obj_1)-1, obj_1(1:end-1), 'b-o', 'LineWidth',1, 'MarkerSize', 1);
xlabel('Number of Iterations','FontSize',15);
ylabel('Objective Value','FontSize',15);
title(['Objective Value for \lambda = ', num2str(lambda)], 'FontSize', 15);
set(gca, 'FontSize',10);
drawnow
saveas(gcf, ['obj_lambda_' num2str(lambda) '.fig']);

 
figure;
plot(1:length(tr_err_1)-1, tr_err_1(1:end-1), 'b-o', 'LineWidth', 1, 'MarkerSize', 1);
xlabel('Number of Iterations','FontSize',15);
ylabel('Training Accuracy ','FontSize',15);
 title(['Training Accuracy for \lambda = ', num2str(lambda)], 'FontSize', 15);
set(gca, 'FontSize',15);
drawnow
saveas(gcf, ['train_err_lambda_' num2str(lambda) '.fig']);
 

figure;
plot(1:length(te_err_1)-1, te_err_1(1:end-1), 'b-o', 'LineWidth', 1, 'MarkerSize', 1);
xlabel('Number of Iterations','FontSize',15);
ylabel('Testing Accuracy ','FontSize',15);
title(['Testing Accuracy for \lambda = ', num2str(lambda)], 'FontSize', 15);
set(gca, 'FontSize',15);
drawnow
saveas(gcf, ['test_err_lambda_' num2str(lambda) '.fig']);

%% lambda=10
lambda=10;
[B_10, te_err_10, tr_err_10, obj_10]=log_reg(tr_y, tr_X, te_y, te_X, lambda, c);

%% plot result

figure;
plot(1:length(obj_10)-1, obj_10(1:end-1), 'b-o', 'LineWidth',1, 'MarkerSize', 1);
xlabel('Number of Iterations','FontSize',15);
ylabel('Objective Value','FontSize',15);
title(['Objective Value for \lambda = ', num2str(lambda)], 'FontSize', 15);
set(gca, 'FontSize',10);
drawnow
saveas(gcf, ['obj_lambda_' num2str(lambda) '.fig']);

 
figure;
plot(1:length(tr_err_10)-1, tr_err_10(1:end-1), 'b-o', 'LineWidth', 1, 'MarkerSize', 1);
xlabel('Number of Iterations','FontSize',15);
ylabel('Training Accuracy ','FontSize',15);
 title(['Training Accuracy for \lambda = ', num2str(lambda)], 'FontSize', 15);
set(gca, 'FontSize',15);
drawnow
saveas(gcf, ['train_err_lambda_' num2str(lambda) '.fig']);
 

figure;
plot(1:length(te_err_10)-1, te_err_10(1:end-1), 'b-o', 'LineWidth', 1, 'MarkerSize', 1);
xlabel('Number of Iterations','FontSize',15);
ylabel('Testing Accuracy ','FontSize',15);
title(['Testing Accuracy for \lambda = ', num2str(lambda)], 'FontSize', 15);
set(gca, 'FontSize',15);
drawnow
saveas(gcf, ['test_err_lambda_' num2str(lambda) '.fig']);
%% lambda=100
lambda=100;
[B_100, te_err_100, tr_err_100, obj_100]=log_reg(tr_y, tr_X, te_y, te_X, lambda, c);

%% plot result

figure;
plot(1:length(obj_100)-1, obj_100(1:end-1), 'b-o', 'LineWidth',1, 'MarkerSize', 1);
xlabel('Number of Iterations','FontSize',15);
ylabel('Objective Value','FontSize',15);
title(['Objective Value for \lambda = ', num2str(lambda)], 'FontSize', 15);
set(gca, 'FontSize',10);
drawnow
saveas(gcf, ['obj_lambda_' num2str(lambda) '.fig']);

 
figure;
plot(1:length(tr_err_100)-1, tr_err_100(1:end-1), 'b-o', 'LineWidth', 1, 'MarkerSize', 1);
xlabel('Number of Iterations','FontSize',15);
ylabel('Training Accuracy ','FontSize',15);
 title(['Training Accuracy for \lambda = ', num2str(lambda)], 'FontSize', 15);
set(gca, 'FontSize',15);
drawnow
saveas(gcf, ['train_err_lambda_' num2str(lambda) '.fig']);
 

figure;
plot(1:length(te_err_100)-1, te_err_100(1:end-1), 'b-o', 'LineWidth', 1, 'MarkerSize', 1);
xlabel('Number of Iterations','FontSize',15);
ylabel('Testing Accuracy ','FontSize',15);
title(['Testing Accuracy for \lambda = ', num2str(lambda)], 'FontSize', 15);
set(gca, 'FontSize',15);
drawnow
saveas(gcf, ['test_err_lambda_' num2str(lambda) '.fig']);
%% lambda=200
lambda=200;
[B_200, te_err_200, tr_err_200, obj_200]=log_reg(tr_y, tr_X, te_y, te_X, lambda, c);

%% plot result

figure;
plot(1:length(obj_200)-1, obj_200(1:end-1), 'b-o', 'LineWidth',1, 'MarkerSize', 1);
xlabel('Number of Iterations','FontSize',15);
ylabel('Objective Value','FontSize',15);
title(['Objective Value for \lambda = ', num2str(lambda)], 'FontSize', 15);
set(gca, 'FontSize',10);
drawnow
saveas(gcf, ['obj_lambda_' num2str(lambda) '.fig']);

 
figure;
plot(1:length(tr_err_200)-1, tr_err_200(1:end-1), 'b-o', 'LineWidth', 1, 'MarkerSize', 1);
xlabel('Number of Iterations','FontSize',15);
ylabel('Training Accuracy ','FontSize',15);
 title(['Training Accuracy for \lambda = ', num2str(lambda)], 'FontSize', 15);
set(gca, 'FontSize',15);
drawnow
saveas(gcf, ['train_err_lambda_' num2str(lambda) '.fig']);
 

figure;
plot(1:length(te_err_200)-1, te_err_200(1:end-1), 'b-o', 'LineWidth', 1, 'MarkerSize', 1);
xlabel('Number of Iterations','FontSize',15);
ylabel('Testing Accuracy ','FontSize',15);
title(['Testing Accuracy for \lambda = ', num2str(lambda)], 'FontSize', 15);
set(gca, 'FontSize',15);
drawnow
saveas(gcf, ['test_err_lambda_' num2str(lambda) '.fig']);











