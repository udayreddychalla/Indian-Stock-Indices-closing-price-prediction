clc;
close all;
clear all;

stock_data = xlsread('BSE_data');

days = size(stock_data,1);
attributes = 6;
data = stock_data(1:days,1:attributes);

wd_size = 10; %%% window size
%%%%%%% Feature extraction %%%%%%%%%%%

train  = round((days - wd_size + 1));
%test = days - train;


OBV = 0;
EMA = 0;
SMA_data = [];
EMA_data = [];
ADO_data = [];
STO_K_data = [];
STO_D_data = [];
OBV_data = [];
WLJMS_R_data = [];
RSI9_data = [];
RSI14_data = [];
PROC12_data = [];
PROC27_data = [];
CPACC_data = [];
HPACC_data = [];


for day = 1:train
    
    
    SMA = mean(data(day : wd_size+day-1, 4));
    SMA_data = [SMA_data SMA];  %%% Simple Moving Average
    
    P = data(wd_size+day-1,4); %%%current price
    A = 2/(wd_size+1);            %%%Smoothing factor
    EMA = (P*A) + (EMA*(1-A));
    EMA_data = [EMA_data EMA];  %%% Exponential moving Average
    
    
    CP = data(wd_size+day-1, 4); %%% Closing prise
    HP = data(wd_size+day-1, 2); %%% Highest price
    LP = data(wd_size+day-1, 3); %%% Lowest price
    P_vol = sum(data(day:wd_size+day-1, 6));  %%% period volume
    ADO = ((CP-LP)-(HP-CP))/((HP-LP)*P_vol);
    ADO_data = [ADO_data ADO];  %%% Accumilation Distribution Oscillator
    
    
    LLP = min(data(day : wd_size+day-1,3)); %%% Lowest of Lowest prices
    HHP = max(data(day : wd_size+day-1,2)); %%% Highest of Highest prices
    STO_K = ((CP - LLP)/(HHP-LLP))*100;
    STO_K_data = [STO_K_data STO_K];   %%% Stocastic Oscillator %K
    
    
    temp = 0;
    for i=day:wd_size+day-1
        CP = data(day, 4);     %%% Closing price
        temp = temp + ((CP-LLP)/(HHP-LLP))*100;
    end;   
    STO_D = temp/wd_size;     %%%%%%%%%%%doubt K-period
    STO_D_data = [STO_D_data STO_D];   %%% Stocastic oscillator %D     
    
    
    T_vol = data(wd_size+day-1, 6);  %%% Volume of the present day
    if(data(wd_size+day-1,4) > data(wd_size+day-2,4))
        OBV = OBV + T_vol;     
    else
        OBV = OBV - T_vol;
    end;    
    OBV_data = [OBV_data OBV]; %%% On balance Volume
    
    CP = data(wd_size+day-1,4);
    WLJMS_R = ((HHP - CP)/(HHP - LLP))*100; 
    WLJMS_R_data = [WLJMS_R_data WLJMS_R];  %%% WILLIJAMS's %R
    
    
    %%%RSI for 9 days
    x = 9;
    gain = 0;
    loss = 0;
    for i = day-(x-wd_size+1):x+day-1 %% day:wd_size+day-2
        change = data(i+1,4) - data(i,4);
        if change >0
            gain  = gain + change;
        else
            loss = loss -change;
        end;
    end;
    U = gain/9;
    D = loss/9;
    RSI9 = 100 - (100/(1+(U/D)));    
    RSI9_data = [RSI9_data RSI9];  %%% Relative Strength index
    
    %%% RSI for 14 days %%
    
    
    x = 14;
    if(day > x-wd_size+1 )  %% satrt from day 5
        for i = day-(x-wd_size+1): x+day-(x-wd_size+1)-1
            change = data(i+1,4) - data(i,4);
            if change >0
                gain  = gain + change;
            else
                loss = loss -change;
            end;
        end;
        U = gain/14;
        D = loss/14;
        RSI14 = 100 - (100/(1+(U/D)));    
        RSI14_data = [RSI14_data RSI14];  %%% Relative Strength index
    else
        RSI14_data = [RSI14_data 0];
    end;
    
    x = 12;
    if(day > x-wd_size+1) %%satrts from day 4
        CP_12 = data(wd_size+day-1 - x, 4);  %%% closing price 12 days ago
        PROC12 = ((CP - CP_12)/(CP_12))*100; 
        PROC12_data = [PROC12_data PROC12];   %%% Price Rate of Change
    else
        PROC12_data = [PROC12_data 0];
    end;
    
    x = 27;
    if(day > x-wd_size+1)
        CP_27 = data(wd_size+day-1 - x, 4);    %%% closing price 27 days ago
        PROC27 = ((CP - CP_12)/(CP_12))*100; 
        PROC27_data = [PROC27_data PROC27];   %%% Price Rate of Change
    else
        PROC27_data = [PROC27_data 0];
    end;    
    
    if(day>1) %% after one day
        CP_wd = data(wd_size+day-1 - wd_size, 4);
        CPACC = ((CP - CP_wd)/(CP_wd))*100;
        CPACC_data = [CPACC_data CPACC];    %%% Closing Price Acceleration


        HP_wd = data(wd_size+day-1 - wd_size, 2);
        HPACC = ((HP - HP_wd)/HP_wd)*100;
        HPACC_data = [HPACC_data HPACC];    %%% High Price Acceleration
    else
        CPACC_data = [CPACC_data 0];  
        HPACC_data = [HPACC_data 0];
    end;
    
  
end;


%%%%%%%%%Normalization%%%%%%%%%%%%

norm = @(X)((X - min(X))/(max(X) - min(X)));

SMA_data1 = norm(SMA_data)';

EMA_data1 = norm(EMA_data)';

ADO_data1 = norm(ADO_data)';

STO_K_data1 = norm(STO_K_data)';

STO_D_data1 = norm(STO_D_data)';

OBV_data1 = norm(OBV_data)';

WLJMS_R_data1 = norm(WLJMS_R_data)';

RSI9_data1 = norm(RSI9_data)';

RSI14_data1 = norm(RSI14_data)';

PROC12_data1 = norm(PROC12_data)';

PROC27_data1 = norm(PROC27_data)';

CPACC_data1 = norm(CPACC_data)';

HPACC_data1 = norm(HPACC_data)';

final_data = zeros(days-wd_size + 1,13);


    final_data(:,1) = SMA_data1(:,1);
    final_data(:,2) = EMA_data1(:,1);
    final_data(:,3) = ADO_data1(:,1);
    final_data(:,4) = STO_K_data1(:,1);
    final_data(:,5) = STO_D_data1(:,1);
    final_data(:,6) = OBV_data1(:,1);
    final_data(:,7) = WLJMS_R_data1(:,1);
    final_data(:,8) = RSI9_data1(:,1);
    final_data(:,9) = RSI14_data1(:,1);
    final_data(:,10) = PROC12_data1(:,1);
    final_data(:,11) = PROC27_data1(:,1);
    final_data(:,12) = CPACC_data1(:,1);
    final_data(:,13) = HPACC_data1(:,1);
    
xlswrite('technical_indicators_bse1', final_data);