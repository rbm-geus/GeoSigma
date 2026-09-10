function [mean_model,realcube,krig_mean]=get_layered_models(kriged_structs,terrain,region,usepreQ,Onshoremask)


landsdel = region_to_landsdel(region);

if nargin < 4 % Default not using preQ
    usepreQ = 0;
end

%Determine number of Layers
if nargin < 3 % Compatability with old version
    Nlay = numel(kriged_structs);
else
    [~,~,Nlay,~,~,Npreq,~,~,~,~] = landsdel_switch(landsdel,0,0);
end

if usepreQ == 1
    %Determine number of Quarternary Layers (region dependent)
    NQlay = Npreq+1;
else
    NQlay = Nlay+1;
    if landsdel == 4
        NQlay = Nlay;
    end
end

%First make the kriging mean model. Gather Structs in a single 3D matrix


% NEW SOLUTION
if landsdel == 4
    krig_mean = zeros(kriged_structs{1}.Ny,kriged_structs{1}.Nx,Nlay);
else
    krig_mean = zeros(kriged_structs{1}.Ny,kriged_structs{1}.Nx,Nlay+1);
end


krig_mean(:,:,1) = terrain;

for ilay = 1:Nlay   
    s = kriged_structs{ilay};
    krig_mean(:,:,ilay+1) = s.img_m_est+s.img_means;
end

disp('Kriged mean done!')

%Now produce realizations

disp('Initializing arrays')
%Determine the size of one model
if landsdel == 4
    ms = [kriged_structs{1}.Ny, kriged_structs{1}.Nx, Nlay];
else
    ms = [kriged_structs{1}.Ny, kriged_structs{1}.Nx, Nlay+1];
end

%Determine the number of realizations
NumReal = size(kriged_structs{1}.img_reals,2);

%Prepare matrix to hold realizations
realcube = NaN(ms(1),ms(2),ms(3),NumReal);


disp('Producing reals')
%Produce models 1 at a time
%Start with realization 1

%keyboard
for r = 1:NumReal
    realcube(:,:,1,r) = terrain;

    s = kriged_structs{NQlay-1};
    preqQGrid = s.img_m_est+s.img_means+s.img_reals{r};

    for ilay = 1:Nlay
        s = kriged_structs{ilay};
        %if ilay ~= NQlay-1
        Grid = s.img_m_est+s.img_means+s.img_reals{r}; % Current layer
        %Grid = s.img_means; % Current layer
        
        Grid(s.img_m_est+s.img_reals{r}==0 & Onshoremask) = nan; % Insert nan where layer is not simulated

        %else
        %    Grid = ones(size(s.img_m_est))*mean(mean(s.img_m_est+s.img_means+s.img_reals{r}));
        %end
        %testreal(:,:,ilay+1) = Grid; % Insert layer in cube
        %if ilay == 1
        prevGrid = realcube(:,:,ilay,r); % Prev. layer
        %end

        indexchange = Grid > prevGrid | isnan(Grid); % Old layer is lower than current or current is nan (not-simulated)
        tempGrid = Grid;
        tempGrid(indexchange) = prevGrid(indexchange);
        
        % Bind to preQuarternary (if in Quaternary)
        if ilay < NQlay && ilay >= 2
            indexchange = Grid < preqQGrid;
            tempGrid(indexchange) = preqQGrid(indexchange);
        end
        realcube(:,:,ilay+1,r) = tempGrid; % Use old layer in cube
        clear tempGrid
        %ilay
    end


    
% %    OBSOLETE
%     for i = 1:Nlay
%         s = kriged_structs{i};
%         Grid = s.img_m_est+s.img_means+s.img_reals{r};
%     
%         realcube(:,:,i+1,r) = Grid;
%     
%             for k = 1:s.Ny
%                 for j = 1:s.Nx
%     
%                     val = realcube(k,j,i+1,r);
%                     val2 = realcube(k,j,i,r);
%     
%                     if val > val2
%                         val = val2;
%                     end
%     
%                     if isnan(val)
%                        val = val2;
%                     end
%     
%                     realcube(k,j,i+1,r) = val;
%     
%                 end
%             end
%         i
%     end

%    keyboard
    disp(['Produced: ' num2str(r) ' of ' num2str(NumReal)])
end

mean_model = mean(realcube,4);



% %
% figure(7); clf(7)
% plot(squeeze(realcube(500,200:1200,1,1)),'k','linewidth',3); hold on
% plot(squeeze(realcube(500,200:1200,2:NQlay-1,1)),'r')
% plot(squeeze(realcube(500,200:1200,NQlay,1)),'b','linewidth',3); hold on
% plot(squeeze(realcube(500,200:1200,NQlay+1:NQlay+5,1)),'g','linewidth',1); hold on
% plot(squeeze(preqQGrid(500,200:1200)),'y','linewidth',1); hold on
% 
% 
% ylim([-300,100])
% 
% 
