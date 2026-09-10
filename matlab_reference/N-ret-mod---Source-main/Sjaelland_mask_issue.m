
%%
%[xutm,yutm,maske] = dtm_read_plot('C:\N-ret-mod---Source\Flader\Jylland\Corrected_for_calibration\masks\1200_Kvartaer_sand_Bund_mask.asc');
%
%[xutm,yutm,maske] = dtm_read_plot('C:\N-ret-mod---Source\Flader\Sjælland\Corrected_for_calibration\masks\hub_kl2__mask.asc');

close all;
lnames = get_layer_names('Sjælland',0);


%%

[xutm,yutm,maske1] = dtm_read_plot('C:\N-ret-mod---Source\Flader\Sjælland\Corrected_for_calibration\masks\ub_ks1__mask.asc');
[xutm,yutm,maske2] = dtm_read_plot('C:\N-ret-mod---Source\Flader\Sjælland\Corrected_for_calibration\masks\ub_ks2__mask.asc');
[xutm,yutm,maske3] = dtm_read_plot('C:\N-ret-mod---Source\Flader\Sjælland\Corrected_for_calibration\masks\ub_ks3__mask.asc');


figure(1); clf(1)
subplot(1,3,1)
imagesc(xutm,yutm,maske1); hold on
plot_dk();
set(gca,'ydir','normal')
title('ub_ks1__mask.asc')


subplot(1,3,2)
imagesc(xutm,yutm,maske2); hold on
plot_dk();
set(gca,'ydir','normal')
title('ub_ks2__mask.asc')

subplot(1,3,3)
imagesc(xutm,yutm,maske3); hold on
plot_dk();
set(gca,'ydir','normal')
title('ub_ks3__mask.asc')


%%    

load('BeforeKrigingWorkSpace06-Nov-2023_Sjælland_cert_fun_ILM_sep2023.mat')
[xutm,yutm,maske1] = dtm_read_plot('C:\N-ret-mod---Source\Flader\Sjælland\Corrected_for_calibration\masks\ub_ks1__mask.asc');
lnames = get_layer_names('Sjælland',0);

%%

figure(2); clf(2)

for i = 1:11
    subplot(3,4,i)
    imagesc(xutm,yutm,reshape(LayerStructP{i}.indexMask_Combined,2050,1350)); hold on
 %   imagesc(xutm,yutm,maske); hold on
    plot_dk();
    set(gca,'ydir','normal')
    title(lnames{i+1})
end
    
 