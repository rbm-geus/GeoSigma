% Script for preparing fyn for MST fall 2023

%%%%%%%%%%%%%%% BIANCA MAIL %%%%%%%%%%%%%%%%%
% topo 	...\02_Maps\Topo\DTM_FYN_HAV_100m_resamp_0vedhav.dfs2
% Top Sand1	...\02_Maps\Geologi\Fyn\02_Top_KS1_Adjusted.dfs2
% Bund Sand1	...\02_Maps\Geologi\Fyn\03_Bund_KS1_Adjusted.dfs2
% Top Sand2	...\02_Maps\Geologi\Fyn\04_Top_KS2_Adjusted.dfs2
% Bund Sand2	...\02_Maps\Geologi\Fyn\05_Bund_KS2_Adjusted.dfs2
% Top Sand3	...\02_Maps\Geologi\Fyn\06_Top_KS3_Adjusted.dfs2
% Bund Sand3	...\02_Maps\Geologi\Fyn\07_Bund_KS3_Adjusted.dfs2
% PreQ	...\02_Maps\Geologi\Fyn\08_Top_PreQ_Adjusted.dfs2
% Topkalk	...\02_Maps\Geologi\Fyn\09_Top_Kalk_Adjusted.dfs2
%


%%%%%%%%%%% ILM MAIL %%%%%%%%%%%%%%%%%%%%%%%
% Hej Rasmus
% 
% Så fik jeg lavet filer til usikkerhedstemaer og tjekker ind til efterårsferie 😊
% 
% Der er ikke lavet en ny fil for snappede tolkningspunkter - Fyn_tolkningspunkter_med_boringssnap.mat ( heller ikke lavet en kopi).
% Når filen med snappede tolkningspunkter læses ind, kan det være, at du skal filtrere de punkter fra, som ligger uden for MST-griddet.
% 
% De er lagt samme sted som for Nret-filerne på N (n:\PROJEKTER\N-retentionskortlægning 2022-24\Geostatistisk modellering\N-ret-mod_data_usikkerheder\)
% De hedder det samme som filerne for Fyn med tilføjelsen  ”_MST”
% 
% fyn_modelleringsomraede_grid_klasser_MST.mat
% fyn_jup_boringer_til_usikkerhed_MST.mat
% 
% fyn_paces_til_usikkerhed_MST.mat
% fyn_pacep_til_usikkerhed_MST.mat
% fyn_mep_til_usikkerhed_MST.mat
% fyn_tem_til_usikkerhed_MST.mat
% fyn_skytem_til_usikkerhed_MST.mat
% 
% Mens jeg kørte mine -skripts igennem, fandt jeg ud af at MST-griddene skulle poleres lidt for at få mine skripts til at virke. Det kan også være at du kommer til det. Der er ikke havbundskoter i terrængriddet, men havoverfladen og hvor det kommer på land i Jylland er det sat til -9999. 
% I lag-griddene er der nogle steder langs kystlinjen, hvor gridnoderne ligger over terrængriddet.
% 
% Hvis der er noget, som ikke virker, så send en sms (24409280), så lukker jeg pc’en op og kigger på det.
% 
% Vh Ingelise



%% Surfaces
%[xutm,yutm,topo]= dtm_read_plot([pwd,'\Flader\AnholtLæsø\dk8_topo100m.asc']);

% Mask
%[xutm,yutm,mask]= dtm_read_plot([pwd,'\Flader\AnholtLæsø\dk8_mask_grid.asc']);

lnames = get_layer_names('Fyn-MST',5);
lnames_old = get_layer_names('Fyn',0);

rng(1)




mkdir([pwd,'\Flader\Fyn-MST\Corrected_for_calibration\masks\']);

for i = 1:length(lnames)
    try
        % Load MST surface
        [xutm,yutm,surfaces{i}]= dtm_read_plot([pwd,'\Flader\Fyn-MST\',lnames{i},'.asc']);
        % Make mask files from surfaces (mask = surface at -9999 elevation)
        masks{i} = surfaces{i}>-9999 & surfaces{i}~=0;

        % Load DK_model surface
        [xutm_old,yutm_old,surfaces_old{i}]= dtm_read_plot([pwd,'\Flader\Fyn\',lnames_old{i},'.asc']);
        
        % Make grids
        [xx_old,yy_old] = meshgrid(xutm_old,yutm_old);
        [xx,yy] = meshgrid(xutm,yutm);

        
        % Pick random points from DK-model surface
        cur_surf_old = surfaces_old{i};
        idp = randperm(numel(xx_old),floor(numel(xx_old)/1000));
        img_idp = zeros(size(cur_surf_old));
        img_idp(idp) = 1;
        img_idp(surfaces_old{1}>=0) = 0;
        img_idp = logical(img_idp);

        % Current surface MST
        cur_surf = surfaces{i};

        % Make interpolation function of DK-model (above elevation 0 in terrain AKA "Fyn-mask" + inside mask of MST-fyn)
        F_old = scatteredInterpolant([xx_old(img_idp);xx(masks{i})],[yy_old(img_idp);yy(masks{i})],[cur_surf_old(img_idp);cur_surf(masks{i})],'linear');

        % Use function to update where there is masks in MST surface
        vq = F_old(xx(~masks{i}),yy(~masks{i}));
        cur_surf(~masks{i}) = vq;
        surfaces{i} = cur_surf;
        clear F_old

        write_grid_ascii([pwd,'\Flader\Fyn-MST\Corrected_for_calibration\masks\',lnames{i},'_mask'],double(masks{i}),xutm,yutm)        
        write_grid_ascii([pwd,'\Flader\Fyn-MST\Corrected_for_calibration\',lnames{i}],double(surfaces{i}),xutm,yutm)
    catch
        disp(['Problem with ',pwd,'\Flader\Fyn-MST\Corrected_for_calibration\',lnames{i},'.asc'])
        if i == length(lnames) % Make own bottom layer for Fyn
            surfaces{i} = surfaces{i-1}-50; 
            write_grid_ascii([pwd,'\Flader\Fyn-MST\Corrected_for_calibration\',lnames{i}],double(surfaces{i}),xutm,yutm)
            masks{i} = surfaces{i}>-9999;
            write_grid_ascii([pwd,'\Flader\Fyn-MST\Corrected_for_calibration\masks\',lnames{i},'_mask'],double(masks{i}),xutm,yutm)
        end
    end
end


%%

i = 8

[xutm,yutm,surfaces_old{i}]= dtm_read_plot([pwd,'\Flader\Fyn-MST\',lnames{i},'.asc']);
[xutm_old,yutm_old,surfaces_old{i}]= dtm_read_plot([pwd,'\Flader\Fyn\',lnames_old{i},'.asc']);



% Pick random points from surface
cur_surf_old = surfaces_old{i};
idp = randperm(numel(xx_old),floor(numel(xx_old)/1000));
img_idp = zeros(size(cur_surf_old));
img_idp(idp) = 1;
img_idp(surfaces_old{1}>=0) = 0;
img_idp = logical(img_idp);

% Current surface MST
cur_surf = surfaces{i};

% Make interpolation function (above elevation 0 in DK-model + within mask Fyn-MST)
F_old = scatteredInterpolant([xx_old(img_idp);xx(masks{i})],[yy_old(img_idp);yy(masks{i})],[cur_surf_old(img_idp);cur_surf(masks{i})],'linear');
%F_old = scatteredInterpolant([xx_old(img_idp)],[yy_old(img_idp)],[cur_surf_old(img_idp)]);

% Use function to update where there is masks in surface
vq = F_old(xx(~masks{i}),yy(~masks{i}));
cur_surf(~masks{i}) = vq;



%
figure(10); clf(10);

subplot(2,3,1)
imagesc(xutm,yutm, surfaces{i}); 
hold on;
plot_dk();
set(gca,'Ydir','normal')
title([lnames{i}])
colorbar
    caxis([-160,130])

subplot(2,3,2)
imagesc(xutm,yutm,masks{i}); 
hold on;
plot_dk();
set(gca,'Ydir','normal')
title(['Mask: ',lnames{i}])
colorbar

subplot(2,3,3)
imagesc(xutm_old,yutm_old, surfaces_old{i}); 
hold on;
plot_dk();
set(gca,'Ydir','normal')
title([lnames_old{i}])
colorbar
        caxis([-160,130])



subplot(2,3,4)
scatter(xx_old(idp),yy_old(idp),20,cur_surf_old(idp),'filled'); 
hold on;
plot_dk();
set(gca,'Ydir','normal')
title([lnames_old{i}])
colorbar
caxis([-60,130])

subplot(2,3,5)
imagesc(xutm_old,yutm_old,surfaces_old{i}<0); 
hold on;
plot_dk();
set(gca,'Ydir','normal')
title([lnames_old{i}])
colorbar

subplot(2,3,6)
imagesc(xutm,yutm,cur_surf); 
hold on;
plot_dk();
set(gca,'Ydir','normal')
title([lnames{i},' updated'])
colorbar
    caxis([-160,130])



suptitle('Surfaces')


%%
figure(1); clf(1);
for i = 1:length(lnames)
    subplot(3,4,i)
    imagesc(xutm,yutm,surfaces{i}); 
    hold on;
    plot_dk();
    set(gca,'Ydir','normal')
    title([lnames{i}])
    colorbar
    caxis([-400,150])
end
suptitle('Surfaces')


figure(2); clf(2);
for i = 1:length(lnames)
    subplot(3,4,i)
    imagesc(xutm,yutm,masks{i}); 
    hold on;
    plot_dk();
    set(gca,'Ydir','normal')
    title(['Mask: ' lnames{i}])
end
suptitle('Masks')


figure(3); clf(3);
for i = 1:length(lnames)-1
    subplot(3,4,i)
    imagesc(xutm,yutm,surfaces{i+1}-surfaces{i}); 
    hold on;
    plot_dk();
    set(gca,'Ydir','normal')
    title([lnames{i+1},'-',lnames{i}])
end
suptitle('Thicknesses')


    cmapwhite = [1,1,1;parula(200)];


% Surfaces within mask
figure(4); clf(4);
for i = 1:length(lnames)
    subplot(3,4,i)
    surf_mask = nan(size(surfaces{i}));
    surf_mask(masks{i}) = surfaces{i}(masks{i});

    imagesc(xutm,yutm,surf_mask); 
    hold on;
    plot_dk();
    set(gca,'Ydir','normal')
    title([lnames{i}])
    colorbar
    colormap(gca,cmapwhite)

end
suptitle('Surfaces within masks')


% Surfaces within mask
figure(5); clf(5);
for i = 1:length(lnames)-1
    subplot(3,4,i)
    surf_mask = nan(size(surfaces{i}));
    surf_mask(masks{i}) = surfaces{i+1}(masks{i})-surfaces{i}(masks{i});

    imagesc(xutm,yutm,surf_mask); 
    hold on;
    plot_dk();
    set(gca,'Ydir','normal')
    title([lnames{i}])
    colorbar
    colormap(gca,cmapwhite)
    
end

suptitle('Thicknesses within masks')



