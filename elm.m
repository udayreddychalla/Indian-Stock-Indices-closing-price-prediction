clc;
close all;
clear all;

%%%%%% choosing hyper  parameters %%%%%%%%%%%% 

hidden_neurons = 10;

days_pred = 1;
wd_size = 10;
    

%%%% Loading training and testing data %%%%%
train_data = xlsread('technical_indicators_nse_5');
data = xlsread('NSE_data_5');

%%% normalizing function %%%
norm = @(v,X)((v - min(X))/(max(X) - min(X)));

%%%%% de - normalization
de_norm = @(v,X)((v*(max(X)- min(X))+ min(X)));




%%% splitting 80% training and 20%testing
total_samp = size(train_data,1);
no_train = round(0.8*total_samp);
no_test = size(train_data,1) - no_train  - days_pred;

%%% storing the 10 technical indicators %%%
tech_ind = zeros(total_samp,10);
for i = 1:4
    tech_ind(:,i) = train_data(:,i);
end;
tech_ind(:,3) = train_data(:,9);
for i = 5:7
    tech_ind(:,i) = train_data(:,i+1);
end;
tech_ind(:,8) = train_data(:,10);
tech_ind(:,9) = train_data(:,12);
tech_ind(:,10) = train_data(:,13);

%%% Training data %%%
train_inp = tech_ind(1:no_train,:);
train_out = norm(data(wd_size + days_pred : no_train + wd_size + days_pred - 1, 4), data(:,4));

%%% Testing data %%%
test_inp = tech_ind(no_train+1:end-days_pred,:);
test_out = norm(data(no_train + wd_size + days_pred  : end , 4), data(:,4));
c = test_out;
% check same number of patterns in training input and target data
if size(train_inp,1) ~= size(train_out,1)
    disp('ERROR: data mismatch')
   return 
end    
% check same number of patterns in testing and target data
if size(test_inp,1) ~= size(test_out,1)
    disp('ERROR: data mismatch')
   return 
end 

%%%%% Training Phase %%%%%%

%read how many patterns
patterns = size(train_inp,1);

%read how many inputs
inputs = size(train_inp,2);


% ---------- set random initital weights -----------------


weight_input_hidden = (rand(inputs, hidden_neurons) - 0.5);
bias = rand(hidden_neurons,1) - 0.5;

H = tanh((train_inp*weight_input_hidden)+ repmat(bias',no_train,1));


beta = (inv(H'*H)*H')*train_out;


%%%%%%%%%%%% Testing Phase %%%%%%%%%%%%%%%%%%%%%%



H = tanh(test_inp*weight_input_hidden + repmat(bias',no_test,1));

%disp('kk')
test_pred = H*beta;

MSE = mean((test_out - test_pred).^2);

disp('RMSE')
disp(sqrt(MSE)*100)

%%%%% de - normalization



test_pred = de_norm(test_pred, data(:,4))';
test_out = de_norm(test_out, data(:,4))';


MAPE = 0;
for i = 1: size(test_pred,1)
    
    MAPE = MAPE + abs((test_out(i) - test_pred(i))/test_out(i));
end;
MAPE = (MAPE/size(test_pred,1))*100;
format longG;
disp('MAPE')
disp(MAPE)

figure(1)
plot(test_out,'-b','LineWIdth',2);hold on;
plot(test_pred, '--r','LineWIdth',2)
xlabel('No. of days','FontSize' , 18)
ylabel('NSE closing price','FontSize' , 18)
legend({'actual','estimated'},'FontSize' , 16);
