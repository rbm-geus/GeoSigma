%function [] = plot_validate_uncertainty_theme(figfolder,themes,themename,terrain,LayerStructP,reg)
function [] = plot_validate_uncertainty_theme(figfolder,themes,themename,region,XS,YS)


%keyboard

mkdir([figfolder 'uncertainty_maps_' region date])
curtime = clock();
%keyboard

Nfigs = ceil(size(themes,3)/6); % Number of figures needed

for iF = 1:Nfigs
    gcf1 = figure(1); clf(1)
    set(gcf1,'color','w');
    set(gcf1,'Units','normalized','Position',[0,0,1,1])
    
    for i = 1:6
        subplot(2,3,i);
        try
        imagesc(XS,YS,sqrt(themes(:,:,i+(iF-1)*6))); 
        title(['LAYNUM ' num2str(i+(iF-1)*6) ',stdmin: ' num2str(round(min(min(sqrt(themes(:,:,i+(iF-1)*6)))),2))]); colorbar;     
        clim([0,20])
        colormap([jet(1000);[1,1,1]])
        set(gca,'Ydir','normal')
        hold on
        plot_dk();
        catch
        end
    end
    suptitle(themename)
    export_fig(gcf1,[figfolder 'uncertainty_maps_' region date '\' themename '_' num2str(iF) '_' num2str(curtime(4)) num2str(curtime(5)) '.png'],'-m4','-r350')
end


% Figure to validate SkyTEM depth to variance conversion

%Load File based on region0
% switch reg
%     case 'Jylland'
%         filename = 'jylland_skytem_til_usikkerhed.mat';
%     case 'Fyn'
%         filename = 'fyn_skytem_til_usikkerhed.mat';
%     case 'Sjælland'
%         filename = 'Sjaelland_skytem_til_usikkerhed.mat';
% end

% numskip = 500;
% terrain_vec = terrain(:);
% theme_vec = reshape(sqrt(themes), [numel(terrain(:)) 11]);
% 
% 
% in = load(['N:\PROJEKTER\N-retentionskortlægning 2022-24\Geostatistisk modellering\N-ret-mod_data_usikkerheder\geofys_usikkerheder\',filename]);
% SkyTEM.Model_Depths = in.dkm_d_pos;
% SkyTEM.ThickGeophysModel = in.dkm_odvthk_pos;
% 
% 


% figure(2); clf(2); 
% for i = 1:11 
%     subplot(4,3,i); 
%     layers_vec = LayerStructP{i}.LayerBottom(:);
%     depth_vec = terrain_vec-layers_vec;    
%     depth_vec = depth_vec(1:numskip:end);
% 
%     scatter(depth_vec,theme_vec(1:numskip:end,i),15,'displayname','SkyTEM','MarkerEdgeColor','k','MarkerFaceColor','r','LineWidth',1); hold on
%     
% %    keyboard
%     if i > 3
%         % Theoretical for SkyTEM
%         plot(SkyTEM.Model_Depths(:,i-3),0.5*1.2*SkyTEM.ThickGeophysModel(:,i-3),'x','color','k')
%     end
%     
% 
% 
%     ylim([0,50])
% %    imagesc(sqrt(themes(:,:,i))); 
% %    title(['stdmin: ' num2str(min(min(sqrt(themes(:,:,i))))) ' stdmax: ' num2str(max(max(sqrt(themes(:,:,i)))))]); colorbar;     
% 
% end
% 
% 
% 
% 

% 
% 
% scatter(depth_vec,SkyTEM_vec,15,'displayname','SkyTEM','MarkerEdgeColor','k','MarkerFaceColor','r','LineWidth',1)
% grid on
% hold on
% scatter(depth_vec,TEM_vec,15,'displayname','TEM','MarkerEdgeColor','k','MarkerFaceColor','b','LineWidth',1)
% scatter(depth_vec,Model_vec,15,'displayname','Models','MarkerEdgeColor','k','MarkerFaceColor','g','LineWidth',1)
% scatter(depth_vec,well_vec,15,'displayname','Wells','MarkerEdgeColor','k','MarkerFaceColor','c','LineWidth',1)
% scatter(depth_vec,PACES_vec,15,'displayname','PACES','MarkerEdgeColor','k','MarkerFaceColor','m','LineWidth',1)
% scatter(depth_vec,PACEP_vec,15,'displayname','PACEP','MarkerEdgeColor','k','MarkerFaceColor','k','LineWidth',1)
% scatter(depth_vec,MEP_vec,15,'displayname','MEP','MarkerEdgeColor','k','MarkerFaceColor','y','LineWidth',1)
% axis([0 200 0 50])
% legend
% 
% xlabel('Mean-Model Depth [m]','FontSize',16)
% ylabel('2\sigma from Uncertainty Themes [m]','FontSize',16)
% 
% figure()
% grid on
% hold on
% scatter(depth_vec,final_vec,15,'displayname','Final Uncertainty Themes','MarkerEdgeColor','k','MarkerFaceColor','b','LineWidth',1)
% axis([0 200 0 50])
% legend
% 
% xlabel('Mean-Model Depth [m]','FontSize',16)
% ylabel('2\sigma from Uncertainty Theme [m]','FontSize',16)