clc;
close all;
clear all;

%%%%%% choosing hyper  parameters %%%%%%%%%%%% 


hidden_neurons = 10;
epochs = 500;

wd_size = 10;

Mu = 0.1;

new_err = [];

%%%% Loading training and testing data %%%%%

train_data = xlsread('technical_indicators_bse_5');
data = xlsread('BSE_data_5');

%%% normalizing function %%%
norm = @(v,X)((v - min(X))/(max(X) - min(X)));

%%%%% de - normalization
de_norm = @(v,X)((v*(max(X)- min(X))+ min(X)));


days_pred = 0;
for d = 1:1

    %days_pred = days_pred + 2^(d-1);
    days_pred = 5;
    if d == 1
        days_pred = 15;
    end;
    if d == 3
        days_pred = 15;
    end;
%%% splitting 80% training and 20%testing

total_samp = size(train_data,1);
no_train = round(0.8*total_samp);
no_test = size(train_data,1) - no_train - days_pred;

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
train_out = norm(data(wd_size + days_pred : no_train + wd_size + days_pred -1, 4),data(:,4));

%%% Testing data %%%
test_inp = tech_ind(no_train+1:end-days_pred,:);
test_out = norm(data(no_train + wd_size + days_pred : end, 4), data(:,4));


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


%do a number of epochs



    % ---------- set random initital weights -----------------
    
     weight_input_hidden = (rand(inputs,hidden_neurons) - 0.5);
     weight_hidden_output = (rand(hidden_neurons,1) - 0.5);
    

%do a number of epochs
MSE = [];
for iter = 1:epochs
       err = [];
       delta_HO = 0;
       delta_IH = zeros(inputs,hidden_neurons);
       
    %loop through the patterns, selecting randomly
    for j = 1:patterns
        
        %select a sequential pattern
         patnum = round((rand * patterns) + 0.5);
        if patnum > patterns
            patnum = patterns;
        elseif patnum < 1
            patnum = 1;    
        end
        
        %set the current pattern
        this_pat = train_inp(patnum,:);
        act = train_out(patnum,1);
        
        %calculate the current error for this pattern
        hval = (tanh(this_pat*weight_input_hidden))';
        pred = hval'*weight_hidden_output;
        pred = tanh(pred);
        error = act-pred;
        
        
        err = [err error^2];
     
        % adjust weight hidden - output
        delta_HO = delta_HO + ((1-pred^2)/2)*error.*Mu.*hval;
        
        
        % adjust the weights input - hidden
        delta_IH= delta_IH +( Mu.*error.*weight_hidden_output.*((1-(hval.^2))/2)*this_pat)';
       
        
    end;
    MSE = [MSE mean(err)];
    weight_hidden_output = weight_hidden_output + delta_HO/patterns;
    weight_input_hidden = weight_input_hidden +  delta_IH/patterns;
    
end;



figure(1)

if d == 1
plot(MSE,'-b','LineWidth',2);
hold on;
elseif d ==2
plot(MSE,'--g','LineWIdth',2);
hold on;
elseif d ==3
plot(MSE,':r','LineWidth',2);
hold on;
end;
if d == 3
    
    legend({'1 day ahead','5 days ahead','15 days ahead'},'FontSize' , 16); %,'7 days ahead','15 days ahead'
   
    title('Mean squared error plot of training data','FontSize' , 18)
    xlabel('No. of epochs','FontSize' , 18)
    ylabel('Mean squared error','FontSize' , 18)
end;



%%%%%%%%%%%% Testing Phase %%%%%%%%%%%%%%%%%%%%%%

patterns = size(test_inp,1);
test_err = [];
test_pred = [];
for j = 1:patterns
      
        patnum =j;
        
        %set the current pattern
        this_pat = test_inp(patnum,:);
        act = test_out(patnum,1);
        
        %calculate the current error for this pattern
        hval = (tanh(this_pat*weight_input_hidden))';
        pred = tanh(hval'*weight_hidden_output);
        
        test_pred = [test_pred pred];
        error = act-pred;
        test_err = [test_err error^2];
end;


disp('RMSE')
disp((sqrt(mean(test_err)))*100)


test_pred = de_norm(test_pred, data(:,4));
test_out = de_norm(test_out, data(:,4))';


MAPE = 0;
for i = 1: size(test_pred,1)
    
    MAPE = MAPE + abs((test_out(i) - test_pred(i))/test_out(i));
end;

MAPE = (MAPE/size(test_pred,1))*100;

format longG;
disp('MAPE')
disp(MAPE)
end;




