clc;
close all;
clear all;

%%%%%% choosing hyper  parameters %%%%%%%%%%%% 

hidden_neurons = 10;
epochs = 1000;
days_pred = 1;

wd_size = 10;

Mu = 0.01;

%%%% Loading training and testing data %%%%%
train_data = xlsread('technical_indicators_nse_5');
data = xlsread('NSE_data_5');


norm = @(v,X)((v*2 - (max(X)+min(X)))/(max(X) + min(X)));

de_norm = @(v,X)((v*(max(X)+min(X))+(max(X) + min(X)))/2);


for d = 1:1

    %days_pred = days_pred + 2^(d-1);
    days_pred = 5;
    if d == 1
        days_pred = 1;
    end;
    if d == 3
        days_pred = 15;
    end;
%%% splitting 80% training and 20%testing
total_samp = size(train_data,1);
no_train = round(0.8*total_samp);
no_test = size(train_data,1) - no_train-days_pred;


%%% storing the 10 technical indicators %%%
tech_ind = zeros(total_samp,10);
for i = 1:4
    tech_ind(:,i) = train_data(:,i);
end;
%tech_ind(:,3) = train_data(:,9);
for i = 5:7
    tech_ind(:,i) = train_data(:,i+1);
end;
tech_ind(:,8) = train_data(:,10);
tech_ind(:,9) = train_data(:,12);
tech_ind(:,10) = train_data(:,13);


%%% Training data %%%
train_inp = tech_ind(1:no_train,:);
train_out = norm(data(wd_size + days_pred : no_train + wd_size + days_pred -1, 4), data(:,4));

%%% Testing data %%%
test_inp = tech_ind(no_train+1:end-days_pred,:);
test_out = norm(data(no_train + wd_size + days_pred : end, 4), data(:, 4));


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
weights = rand(50,1) - 0.5;%zeros(50,1);
MSE = [];

%do a number of epochs
for iter = 1:epochs
    
    delta = zeros(50,1);
    errs = [];
    
    %loop through the patterns, selecting randomly
    for j = 1:patterns
        
        %select pattern
        patnum = j;
       
        %set the current pattern
        this_pat = train_inp(patnum,:);
        act = train_out(patnum,1);
        
        inp = [];

        inp = [inp this_pat];
        inp = [inp sin(pi*this_pat)];
        inp = [inp sin(3*pi*this_pat)];
        %inp = [inp sin(5*pi*this_pat)];
        %inp = [inp sin(7*pi*this_pat)];
        
        inp = [inp cos(pi*this_pat)];
        inp = [inp cos(3*pi*this_pat)];
        %inp = [inp cos(5*pi*this_pat)];
        %inp = [inp cos(7*pi*this_pat)];
        %calculate the current error for this pattern
        pred = (tanh(dot(weights,inp)));
         
        error = act-pred;
       
        errs = [errs error^2];
        
        delta = delta + 2*Mu*error*(inp');
    end
    
    MSE = [MSE mean(errs)];
    weights = weights +delta/patterns;
    
end

figure(1)
if d==1
plot(MSE,'-b','LineWidth',2);
hold on;
elseif d==2
plot(MSE,'--g','LineWidth',2);
hold on;
elseif d==3
plot(MSE,':r','LineWidth',2);
hold on;
end;

if d == 3
    %linVec = {'--', '-.','+', '-+','..', '-,'};
    legend('1 day ahead','5 days ahead','15 days ahead','FontSize',18); %,'7 days ahead','15 days ahead'
   
    title('Mean squared error plot of training data','FontSize',20)
    xlabel('No. of epochs','FontSize',20)
    ylabel('Mean squared error','FontSize',20)
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
        inp = [];

        inp = [inp this_pat];
        inp = [inp sin(pi*this_pat)];
        inp = [inp sin(3*pi*this_pat)];
        %inp = [inp sin(5*pi*this_pat)];
        %inp = [inp sin(7*pi*this_pat)];
        
        inp = [inp cos(pi*this_pat)];
        inp = [inp cos(3*pi*this_pat)];
        %inp = [inp cos(5*pi*this_pat)];
        %inp = [inp cos(7*pi*this_pat)];
        
        %calculate the current error for this pattern
        pred = (tanh(dot(weights,inp)));
        test_pred = [test_pred pred];
         
        error = act-pred;
        test_err = [test_err error^2];
end;


disp('RMSE')
disp(sqrt(mean(test_err))*100)



test_pred = de_norm(test_pred, data(:,4));
test_out = de_norm(test_out, data(:,4))';


MAPE = 0;
for i = 1: size(test_pred,1)
    
    MAPE = MAPE + abs((test_out(i) - test_pred(i))/test_out(i));
end;


MAPE =( MAPE/size(test_pred,1))*100;

format longG;
disp('MAPE')
disp(MAPE)

end;
%figure(2)
%plot(test_out,'-');hold on;
%plot(test_pred,'--')
%xlabel('No. of days','FontSize',20)
%ylabel('closding Price','FontSize',20)
%legend('Actual','Estimated','FontSize',18)