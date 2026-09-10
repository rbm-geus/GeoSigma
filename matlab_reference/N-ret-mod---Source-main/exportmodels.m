function exportmodels(xs,ys,mean_model,realization_models,layer_names,reg,wbr,destination_path,Nreal_pre)

% Nreal_pre: preexisting number of reals (Default = 0)
if nargin < 9
    Nreal_pre = 0;
end


destination_reals = [destination_path, '\realizations'];
destination_mean = [destination_path,'\mean_model'];

if isfolder(destination_path)==0
    mkdir(destination_path)
end

if isfolder(destination_reals)==0
    mkdir(destination_reals)
end

if isfolder(destination_mean)==0
    mkdir(destination_mean)
end

Nlay = size(mean_model,3);
Nreal = size(realization_models,4);
Nnames = numel(layer_names);

if Nnames == Nlay
    disp('Saving Mean Model')
%     %Start by saving the mean model
%     for i = 1:Nlay
%         name = [destination_mean,'\',layer_names{i}];
%         surface = mean_model(:,:,i);
% 
%         write_grid_ascii(name,surface,xs,ys);
%     end
    
    try
        disp('Saving Realizations')
        for j = 1:Nreal
            cur_real_path = [destination_reals,'\realization',num2str(j+Nreal_pre)];
    
            if isfolder(cur_real_path)==0
                mkdir(cur_real_path)
            end
    
            waitbar(j/Nreal,wbr,['Saving Realization ',num2str(j),' of ',num2str(Nreal)])
    
            for i = 1:Nlay
                progress = ((j-1)*Nlay+i)/(Nreal*Nlay);
                waitbar(progress,wbr,['Saving Layer ',num2str(i), ' of ',num2str(Nlay)])
                name = [cur_real_path,'\',layer_names{i}];
                surface = realization_models(:,:,i,j);
    
                write_grid_ascii(name,surface,xs,ys)
            end
    
            disp(['Real ',num2str(j),' exported! as realization ',num2str(j+Nreal_pre)])
        end
    catch
        keyboard
    end

else
    disp('number of names does not match number of layers!')
end

end