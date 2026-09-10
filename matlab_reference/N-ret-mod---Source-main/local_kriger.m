function [GLOBAL,Out] = local_kriger(workdir,S,unique_name,wellthemes,clustmods,nreals,modeltheme)

% Function that takes a struct containing surface, output from ML and
% clustering plus mask layers and does local kriging

doplot = 0;

% Check for figure subfolder, otherwise make one
if isfolder([workdir,'Figures'])==0
    mkdir([workdir,'Figures'])
end
figdir = [workdir,'Figures/'];


% Statmodels and masks
disp('Unpacking clustering result')
[GLOBAL] = setup_GLOBAL_structure(S); clear S;
map = zeros(GLOBAL.Ny,GLOBAL.Nx);

% Uncertainty map
GLOBAL.img_unc = wellthemes; clear wellthemes;

use_stds = 1;
ranges_use = GLOBAL.img_ranges(~isnan(GLOBAL.img_variances));
variances_use = GLOBAL.img_variances(~isnan(GLOBAL.img_variances));
stds_use = GLOBAL.img_stds(~isnan(GLOBAL.img_variances));
statmodspoints2 = GLOBAL.img_clust(~isnan(GLOBAL.img_variances));

Nclust = numel(unique(GLOBAL.clust));


% GLOBAL IMAGES
if doplot == 1
    disp('Plotting images from clustering and maps')
    figure(1); clf(1);
    set(gcf,'color','w');
    set(gcf,'Units','normalized','Position',[0,0,1,1])
    %cmapwhite_parula = [1,1,1;parula(2000)];
    cmapwhite_jet = [1,1,1;jet(2000)];
    plot_GLOBAL_images(GLOBAL,cmapwhite_jet)
    exportgraphics(gcf,[figdir,'GLOBAL_images_',unique_name,'.png'],'Resolution',250)
end

% STATMODELS HISTOGRAMS
disp('Histograms of clusters')
gcf2 = figure(2); clf(2)
set(gcf2,'color','w');
set(gcf2,'Units','normalized','Position',[0,0,1,1])
%use_stds = 0;
[mode_statmods] = plot_and_compute_statmods(statmodspoints2,use_stds,Nclust,variances_use,stds_use,ranges_use);
exportgraphics(gcf,[figdir,'statmodels_Nclust' num2str(Nclust),'_',unique_name,'.png'],'Resolution',250)



% KRIGING INPUT PLOT
if doplot == 1
    disp('Kriging input plot')
    gcf3 = figure(3); clf(3)
    set(gcf3,'color','w');
    set(gcf3,'Units','normalized','Position',[0,0,1,1])
    plot_kriging_setup(GLOBAL,cmapwhite_jet)
    exportgraphics(gcf,[figdir,'kriging_overview_Nclust' num2str(Nclust) '.png'],'Resolution',250)
end



disp('Labeling regions of clusters')
% LABEL REGIONS OF STATMODSGRID
if doplot == 1
    gcf5 = figure(5); clf(5)
    set(gcf5,'color','w');
    set(gcf5,'Units','normalized','Position',[0,0,1,1])
end

clear statelems
clear statelems_bin

k = 1;
for ic = 1:Nclust

    statmodcur = GLOBAL.img_clust==ic;
    statelems.labels{ic} = bwlabel(statmodcur,4);
    statelems.Nlab{ic} = max(statelems.labels{ic}(:));
    for i = 1:statelems.Nlab{ic}
        statelems_bin.img{k} =  find(statelems.labels{ic}==i);
        statelems_bin.area(k) =  size(statelems_bin.img{k}(:),1);
        statelems_bin.statmod(k) =  ic;
        k = k + 1;
    end
    if doplot == 1
        subplot(2,Nclust/2,ic)
        imagesc(statelems.labels{ic}); hold on
        set(gca,'Ydir','normal')
        xlabel('UTMX')
        ylabel('UTMY')
        colorbar()
        axis equal
        title(['Statmodel ' num2str(ic) ' elements'])
    end
end

Nclel = k-1;

disp(['Found ' num2str(Nclel) ' cluster elements'])
disp(['Largest element is ' num2str(max(statelems_bin.area)) ' gridcells'])

if doplot == 1
    exportgraphics(gcf,[figdir,'individual_clusters' '_Nclust' num2str(Nclust) '.png'],'Resolution',250)
end

% Calculating buffer zones
disp('Calculating buffer zones...')

doplot = 0;
if doplot == 1
    gcf6 = figure(6); clf(6)
    set(gcf6,'color','w');
    set(gcf6,'Units','normalized','Position',[0,0,1,1])
end
if doplot == 1
    h = waitbar(0,'Please wait...','Name',['Finding ' num2str(Nclel) ' buffer zones ']);

    map = zeros(GLOBAL.Ny,GLOBAL.Nx);

    tic()
    for ibin = 1:Nclel

        map = map*0;
        tempvar = statelems_bin.img{ibin};
        map(tempvar) = 1;

        bufz_img = calculate_buffer_zones(map,GLOBAL.Nx,GLOBAL.Ny);
        ip_buf = bufz_img>0 & GLOBAL.img_point==1; % point index within buffer
        i_buf = bufz_img>0; % index within buffer
        if doplot == 1
            subplot(3,4,ibin)
            hold on
            %imagesc(bufz_img'); hold on
            scatter(GLOBAL.xx_norm(i_buf),GLOBAL.yy_norm(i_buf),5,bufz_img(i_buf),'filled')
            scatter(GLOBAL.xx_norm(ip_buf),GLOBAL.yy_norm(ip_buf),5,'k')
            set(gca,'Ydir','normal')
            colorbar()
            axis equal
            xlim([1,GLOBAL.Nx])
            ylim([1,GLOBAL.Ny])
            title(['Statmodel ' num2str(statelems_bin.statmod(ibin)) ' buffer elements'])
        end
        if mod(ibin,20) == 0
            curelap = toc();
            waitbar(ibin / Nclel,h,[num2str(ibin / Nclel*100) '% complete - approx. ' num2str(round(curelap/60/ibin*(Nclel-ibin))) ' mins'])
        end
    end
    close(h)
end

% Making buffer overlap map
disp('making buffer overlap map...')

GLOBAL.img_buf_sum = zeros(size(GLOBAL.img_FOHM));

h = waitbar(0,'Please wait...','Name',['Making buffer overlap map']);

map = zeros(GLOBAL.Ny,GLOBAL.Nx);

tic()
% Make buffer overlap map
for ibin = 1:Nclel

    map = map*0;
    tempvar = statelems_bin.img{ibin};
    map(tempvar) = 1;

    bufz_img = calculate_buffer_zones(map,GLOBAL.Nx,GLOBAL.Ny);
    GLOBAL.img_buf_sum = GLOBAL.img_buf_sum+bufz_img;

    if mod(ibin,20) == 0
        curelap = toc();
        waitbar(ibin / Nclel,h,[num2str(ibin / Nclel*100) '% complete - approx. ' num2str(round(curelap/60/ibin*(Nclel-ibin))) ' mins'])
    end
end

close(h)


save([workdir,'GLOBAL_FOR_KRIGING_Nclust' num2str(Nclust) '.mat'],'GLOBAL')
disp('Buffer overlap map saved')



% Inversion
disp('Starting local kriging...')

GLOBAL.img_m_est = zeros(size(GLOBAL.img_FOHM));
GLOBAL.img_post_std = zeros(size(GLOBAL.img_FOHM));

for i = 1:nreals
    GLOBAL.img_reals{i} = zeros(size(GLOBAL.img_FOHM));
end

tic()
h = waitbar(0,'Please wait...','Name',['Kriging']);

Out.SubClusterNumbers = 1:Nclel;

CovarianceMatrices = cell(1,Nclel);
SubClusterBufferedValues = cell(1,Nclel);
SubClusterBufferedIndices = cell(1,Nclel);


for ibin = 1:Nclel

    map = map*0;
    tempvar = statelems_bin.img{ibin};
    map(tempvar) = 1;

    bufz_img = calculate_buffer_zones(map,GLOBAL.Nx,GLOBAL.Ny);
    i_buf = bufz_img>0; % index within buffer
    ip_buf = i_buf==1 & GLOBAL.img_point==1 & ~isnan(GLOBAL.img_FOHM_mask); % point index within buffer


    % Current sill and range
    curstatmod = statelems_bin.statmod(ibin);
    %% CHANGE THIS FREDERIK
    % currange and curvariance are set up in a 9-by-3 array
    % array. One row for each cluster. Column 1 contains the cluster index
    % while column 2 contains the range and column 3 contains the sill.

    % !!!!! COLUMN 1 MUST BE IN NUMERICAL ORDER !!!!!
    try
        clustind = clustmods(curstatmod,1);
    catch
        curstatmod = curstatmod-1;
        clustind = clustmods(curstatmod,1);
        disp(['Hotfix for clustermods applied for cluster: ',num2str(curstatmod)])
    end
    if clustind ~= curstatmod
        keyboard
        disp('error - check clustmods array')
    end

    curv = modeltheme(i_buf);

    currange = clustmods(curstatmod,2);
    curvariance = mean(curv(~isnan(curv)));

    %     curvariance = nanmean(GLOBAL.img_variances(i_buf));

    %%

    % Ratio between correlated and uncorrelated noise
    cor_noise_rat = 0.5;

    if sum(ip_buf(:)) > 0
        % Setup
        [G,Cm,Cd,d_obs,m0] = local_kriging_setup_img(GLOBAL,i_buf,ip_buf,curvariance,currange,cor_noise_rat);

        [m_est,Cm_est]=least_squares_inversion(G,Cm,Cd,0,d_obs);
    else
        
        % CM
        statmod_var = sprintf('%g %s(%g)',curvariance,'Gau',currange);
        Cm = precal_cov([GLOBAL.xx_norm(i_buf)*100,GLOBAL.yy_norm(i_buf)*100],[GLOBAL.xx_norm(i_buf)*100,GLOBAL.yy_norm(i_buf)*100],statmod_var);

        m_est = zeros(sum(i_buf(:)),1);
        Cm_est = Cm;
    end

    if any([isnan(sum(m_est)) isinf(sum(m_est))])
        keyboard
    end
    
    CovarianceMatrices{ibin} = Cm_est;
    SubClusterBufferedValues = bufz_img(i_buf);
    SubClusterBufferedIndices = i_buf;

    reals = get_reals_cholesky(Cm_est,nreals);

    % Place results in inversion img and scale with buffer
    new_surface = m_est.*bufz_img(i_buf)./GLOBAL.img_buf_sum(i_buf);
    new_reals = reals.*bufz_img(i_buf)./GLOBAL.img_buf_sum(i_buf);

    old_surface = GLOBAL.img_m_est(i_buf);
    GLOBAL.img_m_est(i_buf) = old_surface+new_surface;

    for i = 1:nreals
        old_reals = GLOBAL.img_reals{i}(i_buf);
        GLOBAL.img_reals{i}(i_buf) = old_reals+new_reals(:,i);
    end

    if sum(isnan(new_surface)) > 0
        %             keyboard
    end

    new_surface = sqrt(diag(Cm_est)).*bufz_img(i_buf)./GLOBAL.img_buf_sum(i_buf);
    old_surface = GLOBAL.img_post_std(i_buf);
    GLOBAL.img_post_std(i_buf) = old_surface+new_surface;

    if mod(ibin,20) == 0
        curelap = toc();
        waitbar(ibin / Nclel,h,[num2str(ibin / Nclel*100) '% complete - approx. ' num2str(round(curelap/60/ibin*(Nclel-ibin))) ' mins'])
    end


end

Out.BufferSum = GLOBAL.img_buf_sum;
Out.img_m_est = GLOBAL.img_m_est;
Out.CovarianceMatrices = CovarianceMatrices;
Out.SubClusterBufferedValues = SubClusterBufferedValues;
Out.SubClusterBufferedIndices = SubClusterBufferedIndices;


timetot = toc();
GLOBAL.KrigingTime = timetot;
close(h)

end



