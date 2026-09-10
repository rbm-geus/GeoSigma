numskip = 500;
terrain_vec = terrain(:);
layers_vec = reshape(mean_mod(:,:,2:end), [numel(mean_mod(:,:,1)) 11]);

depth_vec = layers_vec*0;

for i = 1:11
    depth_vec(:,i) = terrain_vec-layers_vec(:,i);
end


SkyTEM_vec = reshape(SkyTEM_themes, [numel(mean_mod(:,:,1)) 11]);
TEM_vec = reshape(TEM_themes, [numel(mean_mod(:,:,1)) 11]);
Model_vec = reshape(modeltheme, [numel(mean_mod(:,:,1)) 11]);
well_vec = reshape(wellthemes, [numel(mean_mod(:,:,1)) 11]);
PACES_vec = reshape(PACES_themes, [numel(mean_mod(:,:,1)) 11]);
PACEP_vec = reshape(PACEP_themes, [numel(mean_mod(:,:,1)) 11]);
MEP_vec = reshape(MEP_themes, [numel(mean_mod(:,:,1)) 11]);
final_vec = reshape(final_variance_themes, [numel(mean_mod(:,:,1)) 11]);

depth_vec = depth_vec(1:numskip:end);
SkyTEM_vec = 2*sqrt(SkyTEM_vec(1:numskip:end));
TEM_vec = 2*sqrt(TEM_vec(1:numskip:end));
Model_vec = 2*sqrt(Model_vec(1:numskip:end));
well_vec = 2*sqrt(well_vec(1:numskip:end));
PACES_vec = 2*sqrt(PACES_vec(1:numskip:end));
PACEP_vec = 2*sqrt(PACEP_vec(1:numskip:end));
MEP_vec = 2*sqrt(MEP_vec(1:numskip:end));
final_vec = 2*sqrt(final_vec(1:numskip:end));

scatter(depth_vec,SkyTEM_vec,15,'displayname','SkyTEM','MarkerEdgeColor','k','MarkerFaceColor','r','LineWidth',1)
grid on
hold on
scatter(depth_vec,TEM_vec,15,'displayname','TEM','MarkerEdgeColor','k','MarkerFaceColor','b','LineWidth',1)
scatter(depth_vec,Model_vec,15,'displayname','Models','MarkerEdgeColor','k','MarkerFaceColor','g','LineWidth',1)
scatter(depth_vec,well_vec,15,'displayname','Wells','MarkerEdgeColor','k','MarkerFaceColor','c','LineWidth',1)
scatter(depth_vec,PACES_vec,15,'displayname','PACES','MarkerEdgeColor','k','MarkerFaceColor','m','LineWidth',1)
scatter(depth_vec,PACEP_vec,15,'displayname','PACEP','MarkerEdgeColor','k','MarkerFaceColor','k','LineWidth',1)
scatter(depth_vec,MEP_vec,15,'displayname','MEP','MarkerEdgeColor','k','MarkerFaceColor','y','LineWidth',1)
axis([0 200 0 50])
legend

xlabel('Mean-Model Depth [m]','FontSize',16)
ylabel('2\sigma from Uncertainty Themes [m]','FontSize',16)

figure()
grid on
hold on
scatter(depth_vec,final_vec,15,'displayname','Final Uncertainty Themes','MarkerEdgeColor','k','MarkerFaceColor','b','LineWidth',1)
axis([0 200 0 50])
legend

xlabel('Mean-Model Depth [m]','FontSize',16)
ylabel('2\sigma from Uncertainty Theme [m]','FontSize',16)