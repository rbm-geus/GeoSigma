%% Graphically Illustrate the Models
%laynums = [Npreq-1];
laynums = [7];
type = 'East-West';
ind = 685;
YS(ind)

plot_profile_view(laynums,type,ind,kriged_structs,LayerStructP,terrain,Sdata,region,allwells,mean_mod,realization_mods,correction_maps,Well_S,NPL)

%%
figure(123);clf;imagesc(XS,YS,Sdata.Wells(:,:,laynums)); hold on; plot_dk(); set(gca,'Ydir','normal'); colorbar
caxis([0,20])
plot([XS(1),XS(end)],[YS(ind),YS(ind)])
plot(562250,YS(ind),'ko','markersize',10)

%%
figure(122);clf;imagesc(XS,YS,Sdata.SkyTEM(:,:,laynums)); hold on; plot_dk(); set(gca,'Ydir','normal'); colorbar
caxis([0,20])
plot([XS(1),XS(end)],[YS(ind),YS(ind)])
plot(562250,YS(ind),'ko','markersize',10)

%%
figure(121);clf;imagesc(XS,YS,Sdata.PACES(:,:,laynums)); hold on; plot_dk(); set(gca,'Ydir','normal'); colorbar
caxis([0,20])
plot([XS(1),XS(end)],[YS(ind),YS(ind)])
plot(562250,YS(ind),'ko','markersize',10)

%%
figure(120);clf;imagesc(XS,YS,Sdata.PACEP(:,:,laynums)); hold on; plot_dk(); set(gca,'Ydir','normal'); colorbar
caxis([0,20])
plot([XS(1),XS(end)],[YS(ind),YS(ind)])
plot(562250,YS(ind),'ko','markersize',10)

%%
% % figure(130);clf
% % for i=4:11
% % subplot(2,4,i-3)
% % p_data=Sdata.SkyTEM(:,:,i);
% % p_data(p_data>50)=NaN;
% % pcolor(XS,YS,p_data);shading flat, hold on; plot_dk(); set(gca,'Ydir','normal'); colorbar
% % clim([0,50])
% % title(layernames(i+1))
% % end
% % % plot([XS(1),XS(end)],[YS(ind),YS(ind)])
% % % plot(592650,YS(ind),'ko','markersize',10)

%%
% % figure(135);clf
% % for i=4:11
% % subplot(2,4,i-3)
% % p_data=final_variance_themes(:,:,i);
% % p_data(p_data>50)=NaN;
% % pcolor(XS,YS,p_data);shading flat, hold on; plot_dk(); set(gca,'Ydir','normal'); colorbar
% % clim([0,50])
% % title(layernames(i+1))
% % end
% % % plot([XS(1),XS(end)],[YS(ind),YS(ind)])
% % % plot(592650,YS(ind),'ko','markersize',10)

%%
% % figure(131);clf
% % for i=4:11
% % subplot(2,4,i-3)
% % p_data=Sdata.Wells(:,:,i);
% % p_data(p_data>100)=NaN;
% % pcolor(XS,YS,p_data);shading flat, hold on; plot_dk(); set(gca,'Ydir','normal'); colorbar
% % clim([0,100])
% % title(layernames(i+1))
% % end
% % % plot([XS(1),XS(end)],[YS(ind),YS(ind)])
% % % plot(592650,YS(ind),'ko','markersize',10)

%%
% % figure(132);clf
% % for i=4:11
% % subplot(2,4,i-3)
% % imagesc(XS,YS,Sdata.MEP(:,:,i)); hold on; plot_dk(); set(gca,'Ydir','normal'); colorbar
% % clim([0,20])
% % title(layernames(i+1))
% % end
% % % plot([XS(1),XS(end)],[YS(ind),YS(ind)])
% % % plot(592650,YS(ind),'ko','markersize',10)
%%
% % figure(133);clf
% % for i=4:11
% % subplot(2,4,i-3)
% % imagesc(XS,YS,Sdata.PACES(:,:,i)); hold on; plot_dk(); set(gca,'Ydir','normal'); colorbar
% % clim([0,20])
% % title(layernames(i+1))
% % end
% % % plot([XS(1),XS(end)],[YS(ind),YS(ind)])
% % % plot(592650,YS(ind),'ko','markersize',10)

%%
% % figure(134);clf
% % for i=4:11
% % subplot(2,4,i-3)
% % imagesc(XS,YS,Sdata.PACEP(:,:,i)); hold on; plot_dk(); set(gca,'Ydir','normal'); colorbar
% % clim([0,20])
% % title(layernames(i+1))
% % end
% % % plot([XS(1),XS(end)],[YS(ind),YS(ind)])
% % % plot(592650,YS(ind),'ko','markersize',10)
%%
% % figure(135);clf
% % for i=4:11
% % subplot(2,4,i-3)
% % imagesc(XS,YS,Sdata.TEM(:,:,i)); hold on; plot_dk(); set(gca,'Ydir','normal'); colorbar
% % clim([0,20])
% % title(layernames(i+1))
% % end
% % % plot([XS(1),XS(end)],[YS(ind),YS(ind)])
% % % plot(592650,YS(ind),'ko','markersize',10)

%%
figure(124); clf;
% semilogy(1,sqrt(Sdata.PACEP(find(XS == 592650),ind,4)),'.','markersize',10); hold on;
% plot(1,sqrt(Sdata.PACES(find(XS == 592650),ind,4)),'.','markersize',10);
% plot(1,sqrt(Sdata.SkyTEM(find(XS == 592650),ind,4)),'.','markersize',10);
% plot(1,sqrt(Sdata.Wells(find(XS == 592650),ind,4)),'.','markersize',10);
% plot(1,sqrt(Sdata.MEP(find(XS == 592650),ind,4)),'.','markersize',10);
% plot(1,sqrt(Sdata.TEM(find(XS == 592650),ind,4)),'.','markersize',10);
% %plot(1,Sdata.PL(find(XS == 592650),ind,4),'.','markersize',10);
% plot(1,sqrt(final_variance_themes(find(XS == 592650),ind,4)),'.','markersize',10);

semilogy(XS,sqrt(Sdata.PACEP(ind,:,laynums)),'.-','markersize',10,'Color','m'); hold on;
plot(XS,sqrt(Sdata.PACES(ind,:,laynums)),'.-','markersize',10,'Color','c');
plot(XS,sqrt(Sdata.SkyTEM(ind,:,laynums)),'.-','markersize',10,'Color','g');
plot(XS,sqrt(Sdata.Wells(ind,:,laynums)),'.-','markersize',10,'Color','r');
plot(XS,sqrt(Sdata.MEP(ind,:,laynums)),'.-','markersize',10,'Color','k');
plot(XS,sqrt(Sdata.TEM(ind,:,laynums)),'.-','markersize',10,'Color','b');
plot(XS,sqrt(final_variance_themes(ind,:,laynums)),'.-','markersize',10,'LineWidth',1);


%plot(1,Sdata.Wells(find(XS == 592650),ind,4),'.','markersize',10);

legend([{'PACEP'},{'PACES'},{'SkyTEM'},{'Wells'},{'MEP'},{'TEM'},{'FINAL'}])


