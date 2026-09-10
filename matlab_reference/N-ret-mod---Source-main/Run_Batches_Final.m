%% Run Fyn-MST Final
clear all; close all; clc



%% Batch-run: ILM_oct2023

cert_fun_choice_ILM = 'ILM_oct2023';


NRBatch = 50;% Number of reals per batch
for i = 1:2
    close all;

    redo_preprocessing = 1;
    redo_themes = 1;
    redo_local_themes = 1;
 
    seed = i;
    Nreals_pre = (i-1)*NRBatch;
    kriging_name = 'BeforeKrigingWorkSpace30-Oct-2023_Fyn-MST_cert_fun_ILM_oct2023.mat';

    if i == 1;
        Main(22,0,0,redo_preprocessing,redo_themes,redo_local_themes,0,1,'rbm@geus.dk',seed,NRBatch,cert_fun_choice_ILM)
    else
        Main(22,0,0,redo_preprocessing,redo_themes,redo_local_themes,0,1,'rbm@geus.dk',seed,NRBatch,cert_fun_choice_ILM,Nreals_pre,kriging_name)
    end
end


%% Batch-run: RBM_oct2023

cert_fun_choice_RBM = 'RBM_oct2023';

NRBatch = 50;% Number of reals per batch
for i = 1:2
    close all;

    redo_preprocessing = 0;
    redo_themes = 1;
    redo_local_themes = 1;
 
    seed = i;
    Nreals_pre = (i-1)*NRBatch+1000;
    kriging_name = 'BeforeKrigingWorkSpace20-Oct-2023_Fyn-MST_cert_fun_RBM_oct2023.mat';
    
    if i == 1;
        Main(22,0,0,redo_preprocessing,redo_themes,redo_local_themes,0,0,'rbm@geus.dk',seed,NRBatch,cert_fun_choice_RBM)
    else
        Main(22,0,0,redo_preprocessing,redo_themes,redo_local_themes,0,0,'rbm@geus.dk',seed,NRBatch,cert_fun_choice_RBM,Nreals_pre,kriging_name)
     end
end


%% Run AnholtLæsø Final
clear all; close all; clc

cert_fun_choice = 'ILM_sep2023';

% Batch-run
NRBatch = 50;% Number of reals per batch
for i = 1:2
    close all
    seed = i;
    Nreals_pre = (i-1)*NRBatch;
    Main(4,0,0,0,0,0,1,1,'rbm@geus.dk',seed,NRBatch,cert_fun_choice,Nreals_pre,'BeforeKrigingWorkSpace09-Oct-2023_AnholtLæsø.mat')
end



%% Run Fyn Final
clear all; close all; clc

cert_fun_choice = 'ILM_sep2023';

% Batch-run
NRBatch = 25;% Number of reals per batch
for i = 1:4
    close all
    seed = i;
    Nreals_pre = (i-1)*25;
    Main(2,0,0,0,0,0,1,1,'rbm@geus.dk',seed,NRBatch,cert_fun_choice,Nreals_pre,'BeforeKrigingWorkSpace22-Sep-2023_Fyn.mat')
end


%% Run Sjælland Final
clear all; close all; clc

cert_fun_choice = 'ILM_sep2023';

% Batch-run
NRBatch = 20;% Number of reals per batch
for i = 1:5
    close all
    redo_preprocessing = 0;
    redo_themes = 0;
    redo_local_themes = 0;

    seed = i;
    Nreals_pre = (i-1)*NRBatch;
%    if i == 1;
%        Main(3,0,0,redo_preprocessing,redo_themes,redo_local_themes,1,1,'rbm@geus.dk',seed,NRBatch,cert_fun_choice,Nreals_pre)
%    else
        Main(3,0,0,redo_preprocessing,redo_themes,redo_local_themes,1,1,'rbm@geus.dk',seed,NRBatch,cert_fun_choice,Nreals_pre,'BeforeKrigingWorkSpace06-Nov-2023_Sjælland_cert_fun_ILM_sep2023.mat')
%    end
end

%% Run Jylland Final
clear all; close all; clc

cert_fun_choice = 'ILM_sep2023';

% Batch-run
NRBatch = 15;% Number of reals per batch
for i = 2:7
    close all
    seed = i;
    Nreals_pre = (i-1)*NRBatch;
    if i == 1;
        Main(1,0,0,1,0,0,1,1,'rbm@geus.dk',seed,NRBatch,cert_fun_choice,Nreals_pre)
    else
        Main(1,0,0,0,0,0,1,1,'rbm@geus.dk',seed,NRBatch,cert_fun_choice,Nreals_pre,'BeforeKrigingWorkSpace05-oct-2023_Jylland.mat')
    end
end


