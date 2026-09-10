% Script for preparing Anholt and Læse (DK8)

% Hej Rasmus
% 
% Nu er jeg kommet gennem alle forberedelserne af DK8
% 
% 
% 1)	Topo-fil fra Lars T for DK8 er dk8_topo100m.asc. Findes i mappen \\netapp\grundvand\PROJEKTER\N-retentionskortlægning 2022-24\Geostatistisk modellering\DK8_grids_topo_mask\
% 2)	HS -modellen fra Læsø består af 3 lag, som lagt ind i FOHM/Nret-24 stratigrafi:
% a.	Lag1_Bund_Post_Glacial_Ler_Tørv_Gytje.asc
% b.	Lag2_Bund_Post_Glacial_Sand.asc
% c.	Lag5_Bund_Sen_Glacial_ler.asc
% 3)	HS -modellen fra Anholt består af 4 lag, som lagt ind i FOHM/Nret-24 stratigrafi:
% a.	Lag2_Bund_Post_Glacial_Sand.asc
% b.	1200_Kvartaer_sand_Bund.asc
% c.	1300_Kvartaer_ler_Bund.asc
% d.	1400_Kvartaer_sand_Bund.asc
% 4)	Samlet kommer DK8 til at bestå af 13 lag, hvoraf de 7 er tomme. Grids findes i mappen: \\netapp\grundvand\PROJEKTER\N-retentionskortlægning 2022-24\Geostatistisk modellering\DK8_grids_topo_mask\
% 5)	Maske for DK8 er dk8_mask_grid.asc. Filen findes i  mappen: \\netapp\grundvand\PROJEKTER\N-retentionskortlægning 2022-24\Geostatistisk modellering\DK8_grids_topo_mask\. Den åbner for både Læsø og Anholt samtidig, men det er nok ikke hensigtsmæssigt at tillade lag som ikke er modelleret i modellen, så 
% a.	Masken skal manipuleres så den åben for 
% i.	Læsø (nordlige halvdel) for PL_lag 1,2, og 5
% ii.	Anholt (sydlige halvdel) for PL_lag 2, FOHMlag 1200, 1300 og 1400
% 6)	Tolkningspunkter til punktudtagning og usikkerhedstema
% a.	For FOHM lag: DK8_tolkningspunkter_med_boringssnap.mat (kun punkter fra Anholt)
% b.	For PL-lag DK8_PL_tolkningspunkter.mat, som er formateret som input fra Vendsyssel
% 7)	Modelleringsusikkerhed:
% a.	Den er som for Jylland lavet med kvalitetsvurdering. Filnavn = DK8_modelleringsomraede_grid_klasser.mat
% 8)	Jupiterboringer til usikkerheder
% a.	For FOHM lag filen DK8_jup_boringer_til_usikkerhed.mat
% b.	For PL-lag filen DK8_PL_jup_boringer_til_usikkerhed.mat. Den skal indlæses på samme vis, som for Vendsyssel.
% 9)	Der er ingen geofysik i GERDA og dermed ingen filer med data til geofysikstemaer.
% 10)	Alle filer med data til punktudtagning/usikkerhedstemaer ligger i mappen \\netapp\grundvand\PROJEKTER\N-retentionskortlægning 2022-24\Geostatistisk modellering\N-ret-mod_data_usikkerheder\  
% 



%% Surfaces
[xutm,yutm,topo]= dtm_read_plot([pwd,'\Flader\AnholtLæsø\dk8_topo100m.asc']);

% Mask
[xutm,yutm,mask]= dtm_read_plot([pwd,'\Flader\AnholtLæsø\dk8_mask_grid.asc']);

lnames = get_layer_names('AnholtLæsø',5);

%
for i = 1:14
    try
        [~,~,surfaces{i}]= dtm_read_plot([pwd,'\Flader\AnholtLæsø\',lnames{i},'.asc']);
    catch

        disp([pwd,'\Flader\AnholtLæsø\',lnames{i},'.asc'])
    end
end

%%
figure(1); clf(1);

    subplot(3,5,1)
    imagesc(xutm,yutm,mask); 
    hold on;
    plot_dk();
    set(gca,'Ydir','normal')
    title('Mask')

for i = 1:14
    subplot(3,5,i+1)
    if i < 14
        imagesc(xutm,yutm,surfaces{i+1}-surfaces{i}); 
        hold on;
        plot_dk();
        set(gca,'Ydir','normal')
        title([lnames{i+1},'-',lnames{i}])
    end
end



%% Make masks for individual layers


% First divide combined AnholtLæsø mask into separate masks
figure(2); clf(2);

mask_Laeso = mask;
mask_Laeso(1:130,:) = 0;
mask_Anholt = mask;
mask_Anholt(end-240:end,:) = 0;
mask_zero = mask*0;

subplot(2,2,1)
imagesc(xutm,yutm,mask); 
hold on;
plot_dk();
set(gca,'Ydir','normal')
title('Mask')

subplot(2,2,2)
imagesc(xutm,yutm,mask_Laeso); 
hold on;
plot_dk();
set(gca,'Ydir','normal')
title('Mask Læsø')


subplot(2,2,3)
imagesc(xutm,yutm,mask_Anholt); 
hold on;
plot_dk();
set(gca,'Ydir','normal')
title('Mask Anholt')

subplot(2,2,4)
imagesc(xutm,yutm,mask_zero); 
hold on;
plot_dk();
set(gca,'Ydir','normal')
title('Zero layer')



% Loop through all layers and make correct mask
% 2)	HS -modellen fra Læsø består af 3 lag, som lagt ind i FOHM/Nret-24 stratigrafi:
% a.	Lag1_Bund_Post_Glacial_Ler_Tørv_Gytje.asc
% b.	Lag2_Bund_Post_Glacial_Sand.asc
% c.	Lag5_Bund_Sen_Glacial_ler.asc
% 3)	HS -modellen fra Anholt består af 4 lag, som lagt ind i FOHM/Nret-24 stratigrafi:
% a.	Lag2_Bund_Post_Glacial_Sand.asc
% b.	1200_Kvartaer_sand_Bund.asc
% c.	1300_Kvartaer_ler_Bund.asc
% d.	1400_Kvartaer_sand_Bund.asc

AnholtLayers = [{'Lag2_Bund_Post_Glacial_Sand'},{'1200_Kvartaer_sand_Bund'},{'1300_Kvartaer_ler_Bund'},{'1400_Kvartaer_sand_Bund'}];
LaesoLayers = [{'Lag1_Bund_Post_Glacial_Ler_Tørv_Gytje'},{'Lag2_Bund_Post_Glacial_Sand'},{'Lag5_Bund_Sen_Glacial_Ler'}];



figure(3); clf(3);
for i = 1:14
    tempmask = mask_zero; % initalize
    % Loop over læsø and Anholt to see if layer is to be simulated
    Afound = 0;
    for  iA = 1:numel(AnholtLayers)
        if strcmp(lnames{i},AnholtLayers{iA})
             Afound = 1;
        end
    end

    Lfound = 0;

    for  iL = 1:numel(LaesoLayers)

        if strcmp(lnames{i},LaesoLayers{iL})
             Lfound = 1;
        end
    end
    
    if Afound == 1
        tempmask = tempmask + mask_Anholt; 
    end
    if Lfound == 1
        tempmask = tempmask + mask_Laeso; 
    end
    
    masks{i} = tempmask>0; 
    subplot(3,5,i)
    imagesc(xutm,yutm,masks{i}); 
    hold on;
    plot_dk();
    set(gca,'Ydir','normal')
    title(['Mask for ',lnames{i}])
    
    % Write mask
    if i > 1
        write_grid_ascii([pwd,'\Flader\AnholtLæsø\Corrected_for_calibration\masks\',lnames{i},'_mask'],double(masks{i}),xutm,yutm)
%        write_grid_ascii([pwd,'\Flader\AnholtLæsø\Corrected_for_calibration\masks\','test','_mask'],masks{i},xutm,yutm)
    end
end
