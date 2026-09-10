%lname = '1200_Kvartaer_sand_Bund';
%lname = '0100_Postglacial_toerv_Bund';
lname = '5900_Klintinghoved_KRL5_Bund';

[xx_old,yy_old,G_grid_old]=dtm_read_plot(['C:\N-ret-mod---Source\Flader\Jylland\',lname,'.asc']);
[xx,yy,G_grid]=dtm_read_plot(['C:\N-ret-mod---Source\Flader\Jylland\Corrected_for_calibration\',lname,'.asc']);


%%
figure(1); clf(); 
ax1 = subplot(1,3,1)
indnewx = [35:2000];
indnewy = [1:3450];
mat_new = G_grid(indnewy,indnewx);
imagesc(xx(indnewx),yy(indnewy),mat_new)
caxis([-200,200])
hold on; plot_dk()
set(gca,'YDir','normal')
ax2 = subplot(1,3,2);
indoldx = [1:2000-34];
indoldy = [1:3450]+15;
mat_old = G_grid_old(indoldy,indoldx);
imagesc(xx_old(indoldx),yy_old(indoldy),mat_old);
%imagesc(xx_old,yy_old,G_grid_old);

hold on;plot_dk();
caxis([-200,200])
set(gca,'YDir','normal')


ax3 = subplot(1,3,3);
imagesc(xx(indnewx),yy(indnewy),mat_old-mat_new)
hold on;plot_dk();
colorbar
caxis([-50,50])
set(gca,'YDir','normal')

linkaxes([ax1,ax2,ax3],'xy')
