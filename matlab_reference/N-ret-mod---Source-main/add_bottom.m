function [model_out, reals_out] = add_bottom(mean_model,real_models)
%Adds 50m of chalk/limestone below the mean model and all realizations


ny = size(mean_model,1);
nx = size(mean_model,2);
Nlay = size(mean_model,3);

%First add bottom layer to mean model
model_out = zeros(ny,nx,Nlay+1);
model_out(:,:,1:Nlay) = mean_model;
model_out(:,:,Nlay+1) = mean_model(:,:,Nlay)-50;

%Then add bottom layer to realizations
Nreal = size(real_models,4);


%keyboard
%reals_out = zeros(ny,nx,Nlay,Nreal);
%reals_out(:,:,1:Nlay,:) = real_models(:,:,1:Nlay,:);
reals_out = real_models;
clear real_models

%keyboard

for i = 1:Nreal
    reals_out(:,:,Nlay+1,i) = reals_out(:,:,Nlay,i)-50;
    disp(['Added bottom for real ' num2str(i)])
end

end