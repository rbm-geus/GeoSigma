function product = Main(landsdel,redo_peatlands,redo_topography,redo_preprocessing,redo_themes,redo_local_themes,topo_APR2023,use_calibration_model,emailaddress,seed,Nreals,cert_fun_choice,Nreals_pre,kriging_name)
% DESCRIPTION
%
% This script is the main script for the geostatistical modelling codes. See
% documentation for more details. Other scripts contain a short description
% such as this one, followed by one section containing the function,
% whereas this script is split into several sections.
%
%INPUT:
%   - landsdel: 1, 2, 22, 3 or 4 for Jylland, Fyn, Fyn-MST ,Sjælland or AnholtLæsø respectively.
%   - redo_peatlands: FLAG
%   - redo_topography: FLAG
%   - redo_preprocessing: FLAG
%   - redo_themes: FLAG15
%   - redo_local_themes: FLAG
%   - topo_APR2023: FLAG
%   - use_calibration_model: FLAG
%   - emailaddress: Adress to send status updates to (default rbm@geus.dk)
%   - seed: Seed number (default 1)
%   - Nreals: Number of realizations to be made (default 10)
%   - Nreals_pre; % Number of preexisting reals (default 0)
%   - kriging_name: name of preexisting file containing kriging variables, 
%   e.g. "BeforeKrigingWorkSpace07-Sep-2023_Jylland.mat". If set, all
%   processing steps are skipped.
%
% product = Main(landsdel,redo_peatlands,redo_topography,redo_preprocessing,redo_themes,redo_local_themes,topo_APR2023,use_calibration_model,emailaddress,seed,Nreals,Nreals_pre,kriging_name)

% Set default options
if nargin < 9
    emailaddress = 'rbm@geus.dk';
end    

% Set default options
if nargin < 10
    seed = 1;
end    

% Set default options
if nargin < 11
    Nreals = 10;
end    

% Set default options
if nargin < 12
    cert_fun_choice = 'FRAFA_apr2023';
end

% Set default options
if nargin < 13
    Nreals_pre = 0; % Number of preexisting reals
end

% Set default options
if nargin < 14
    kriging_name = [];
end



% % FRAFA APRIL 2023
% if str(cert_fun_choice,'FRAFRA_apr2023')
%     cert_fun = @(dist,range,width,sill) (dist<=width).*sill+(dist>width).*sill.*(exp(-(3*power((dist-width),2))./(power(range,2))));
% end
% 
% % ILM SEP 2023
% if str(cert_fun_choice,'ILM_sep2023')
%     cert_fun = @(dist,range,width,sill) (dist<=width).*sill.*(exp(-(3*power((dist-width),2))./(power(range,2))));
% end
% 
% % ILM OCT 2023
% if str(cert_fun_choice,'ILM_oct2023')
%     cert_fun = @(dist,range,width,sill) (dist<=width).*sill.*(exp(-(3*power((dist),2))./(power(range,2))));
% end

% Check if kriging name is provided
if ~isempty(kriging_name)
    % Check the validity of kriging_name
    if exist(kriging_name) == 2
        disp('Kriging file correctly provided')
        disp(['Filename: "' kriging_name '"'])
        disp('Starting kriging directly from file without any preprocessing')
        region = kriging_name(35:end-4);
        sendolmail(emailaddress,['Main-script initiated for ' region],...
    ['The following input where chosen: <br>'...
    'landsdel:' region '<br>'...
    'starting kriging directly from file:' kriging_name '<br>'...
    'seed: ' num2str(seed) '<br>'...
    'Nreals already calculated: ' num2str(Nreals_pre) '<br>'...
    'Nreals: ' num2str(Nreals)])
    kriging_ready = 1; % Flag to mark of kriging can be started
    else
        disp('Mispelled "kriging_name" variable')
        return
    end
else
    disp('No kriging_name provided, doing pre-processing')
    kriging_ready = 0; % Flag to mark of kriging can be started
end

%Set the working directory to be the current local path
native_dir = pwd;


%% SECTION: Initialize Landsdel (Part of Denmark)
if kriging_ready == 0

disp('%% SECTION: Initialize Landsdel (Part of Denmark)')
% 1 for jylland, 2 for fyn, 3 for sjælland

%The following switch prepares the subsequent sections for the specifics of the
%currently chosen part of Denmark by initializing some variables.
[region,NPL,Nlay,layernames,Ninterfaces,Npreq,filetype,wellname_interpreted,wellname_not_interpreted,wellname]...
     = landsdel_switch(landsdel,redo_topography,topo_APR2023);

%Load Wells for this part of Denmark

wells = get_wells(wellname{1},region,Nlay,NPL,native_dir);

sendolmail(emailaddress,['Main-script initiated for ' region],...
['The following input where chosen: <br>'...
    'landsdel:' region '<br>'...
    'redo_peatlands: ' num2str(redo_peatlands) '<br>'...
    'redo_topography: ' num2str(redo_topography) '<br>'...
    'redo_preprocessing: ' num2str(redo_topography) '<br>'...
    'redo_themes: ' num2str(redo_themes) '<br>'...
    'redo_local_themes: ' num2str(redo_local_themes) '<br>'...
    'topo_APR2023: ' num2str(topo_APR2023) '<br>'...
    'use_calibration_model: ' num2str(use_calibration_model) '<br>'...
    'seed: ' num2str(seed) '<br>'...
    'Nreals: ' num2str(Nreals) '<br>'...
    'Used certainty function: ' cert_fun_choice '<br>'...
    'kriging_name: ' kriging_name])

end

%% SECTION: Read Peat-Land Layers and Write To Files
if kriging_ready == 0

    sectiontit = '%% SECTION: Read Peat-Land Layers and Write To Files';
    disp(sectiontit)
    %waitbar(0,wbr,'Preparing Peatlands')
    disp('Preparing Peatlands')
    %Put layers same folder as other layers
    tiledir = [native_dir,'\Flader\',region,'\'];
    
    %Path for the first layer (terrain).
    terrain_path = [native_dir,'\Flader\',region,'\',layernames{1},filetype];
    
    %Read terrain layer
    switch filetype
        case '.asc'
            [xt,yt,terrain] = dtm_read_plot(terrain_path);
    
        case '.grd'
            [terrain, tInfo] = ReadSurfer7(terrain_path);
            xt = tInfo.UTM_X;
            yt = tInfo.UTM_Y;
    
    end
    
    %Write Terrain Layer XY-coordinate vectors to tinfo struct
    tinfo.xs = xt;
    tinfo.ys = yt;
    
    %The following function reads tiles, combine them and produces a new model
    %layer for each tile layer. These are saved to the folder with the existing
    %model layers (tiledir).
    if NPL > 0
        if redo_peatlands
            [SnapList] = ReadWriteTiles(layernames,tiledir,terrain,tinfo,region,0);
            Snapped_PL = SnapList;
            save([region,'_peatland_snaps'],"SnapList");
        else
            load([region,'_peatland_snaps'],"SnapList");
            Snapped_PL = SnapList;
        end
    else
    %    load([region,'_peatland_snaps'],"SnapList");
        Snapped_PL = [];
    end
    
    % Please note that "tiledir" is not the directory for the GS3D-modelled tiles.
    % It is the directory for where to save the tiles after "ReadWriteTiles" is
    % done processing them. The original directory is listed inside the
    % ReadWriteTiles code.
    
    sendolmail(emailaddress,['Main-script ' region ' update: ' sectiontit ' completed'],[sectiontit])

end
    

%% SECTION: Preprocessing of Layers



if kriging_ready == 0 
    
    sectiontit = '%% SECTION: Preprocessing of Layers';
    disp(sectiontit)
    %"prepare_for_kriging" does a lot. This includes drawing points from the surface
    % as data, splitting the surface into sliding window grids, predicting the
    % effective range and sill in each sliding window grid and clustering into
    % smaller regions with similar range and sill.
    if use_calibration_model == 1
        modeldir = [native_dir,'\Flader\',region,'\Corrected_for_calibration\'];
    else
        modeldir = [native_dir,'\Flader\',region,'\'];
    end
    
    disp(['Using this directory: ' modeldir])
    
    
    if redo_preprocessing == 1
        
        loadLatest = 0;
    
        if loadLatest == 1
            load(['LayerStructs_',region,'.mat'],'cur_struct')
            startlay = 0;
            ilay = 0;
            while startlay == 0
                ilay = ilay + 1;
                if  isempty(cur_struct{ilay})
                    startlay = ilay;
                end
            end
            clear ilay
        else
            cur_struct = cell(1,Nlay);
            startlay = 1;
        end
        
    
        %keyboard
        %a = tic;
        %startlay = 4;
        for ilay = startlay:Nlay
%        for ilay = startlay:Nlay    
            peat_log = ilay<(NPL+1);
            %pr = ilay/Nlay;
            %waitbar(pr,wbr,['Preprocessing Layer ',num2str(ilay)])
            
            layername_top = layernames{ilay};
            layername_bottom = layernames{ilay+1}
            
            dir_t = [modeldir,layername_top,filetype];
            dir_b = [modeldir,layername_bottom,filetype];
    
            if ilay == 1
                top = NaN;
            else
                load(['LayerStructs_',region,'.mat'],'cur_struct')
                top = cur_struct{ilay-1}.NextTop;
            end
            
            cur_struct{ilay} = prepare_for_kriging(dir_t,dir_b,0,1,0,[ilay Nlay],wells,NPL,region,top,peat_log,Snapped_PL,layername_bottom,use_calibration_model);
            clear layername_top
            clear layername_bottom
            
           %keyboard
            
            terrain = cur_struct{1}.LayerTop;
    
            save(['LayerStructs_',region,'.mat'],'-v7.3')
            sendolmail(emailaddress,['Preprocessed layer: ' region ' ' num2str(ilay)],'')
            loadLatest = 0;
        end
    
        %preprocess_time = toc(a);
        %disp(['Time Elapsed for Preprocessing is ',num2str(preprocess_time),' seconds.'])
        %save(['LayerStructs_',region,'.mat'],'-v7.3')
    else
        load(['LayerStructs_',region,'.mat'],'cur_struct')
    end
    
    
    sendolmail(emailaddress,['Main-script ' region ' update: ' sectiontit ' completed'],[sectiontit  ' <br> used this directory for input model: ' modeldir])
    clear modeldir

end



%% SECTION: Prepare Wells

if kriging_ready == 0 
    
    sectiontit = '%% SECTION: Prepare Wells';
    disp(sectiontit)
    % This section prepares wells such that the output is two cell-arrays of
    % wells with Nlay number of cells. each cell contains a struct with
    % the wells that penetrate the corresponding layer. It includes information
    % such as UTM_X, UTM_Y, Depth and well quality. The snapped wells also have
    % information about whether or not a 0-thickness is interpreted, which is
    % used in the followin sections to force realiztions to comply at those
    % wells. The two cell-arrays represent the snapped wells and the wells
    % that are not snapped, respectively.
    
    
    if exist('cur_struct','var')
        LayerStructP = cur_struct;
    else
        [~] = load(['LayerStructs_',region,'.mat']);
        LayerStructP = cur_struct;
    end
    
    [SnapWells,JupiterWells,nothick,Well_S,allwells] = prepare_wells(wells,wellname{2},cur_struct,terrain);
    save(['nothick_wells.mat'],'nothick','-v7.3')
    
    sendolmail(emailaddress,['Main-script ' region ' update: ' sectiontit ' completed'],sectiontit)

end



%% SECTION: Uncertainty Themes, Tie-Correction Theme and Cluster Statistics
if kriging_ready == 0 
    
    sectiontit = '%% SECTION: Uncertainty Themes, Tie-Correction Theme and Cluster Statistics'; 
    disp(sectiontit)
    %Initialize variance/uncertainty theme for wells
    wellthemes = zeros([size(terrain),Nlay]);
    %wbr = waitbar(0,'Preparing Uncertainty Themes');
    disp('% Preparing Uncertainty Themes')
    
    %Initialize tie-correction theme. Ties force the simulation results to conform to a
    %zero-thickness locally. The theme is a map of where this should occur.
    correction_maps = zeros([size(terrain),Nlay]);
    
    %This loop produces the values for the themes for all layers
    S = LayerStructP{1};
    
    XS = S.Grid_UTM_X;
    YS = S.Grid_UTM_Y;
    
    complexity = import_complexitymap(XS,YS,region);
    
    figfolder_unc = 'n:\PROJEKTER\N-retentionskortlægning 2022-24\Geostatistisk modellering\Validation_plots\';
    
    % Well uncertainties quantified
    if redo_themes == 1 
        for i = 1:Nlay
            %pr = i/Nlay;
            %waitbar(pr,wbr,['Preprocessing Wells for Layer ',num2str(i)])
            disp(['% Preprocessing Wells for Layer ',num2str(i)])
            S = LayerStructP{i};
        
            XS = S.Grid_UTM_X;
            YS = S.Grid_UTM_Y;
        
            % Check if fields exist but are empty
            if isfield(SnapWells{i},'xutm')
                if  isempty(SnapWells{i}.xutm)
                    SnapWells(i) = []; % Delete old
                    SnapWells{i} = []; % Insert empty
                end
            end
            
            % If no-thick points exist make a correction map
            if ~isempty(SnapWells{i})
                wellst = SnapWells{i};
                wellsj = JupiterWells{i};
                try
                    correction_maps(:,:,i) = get_corr_maps(XS,YS,wellst);
                catch
                    try
                        indrmv = (min(YS) > wellst.yutm | max(YS) < wellst.yutm | min(XS) > wellst.xutm | max(XS) < wellst.xutm);
                        Nrmv = sum(indrmv);
                        disp(['Hotfix, removing wells (total ',num2str(Nrmv),') that was wrongly included, but are outside simulation grid'])
                        wellst.bor_qual(indrmv) = [];
                        wellst.borsnap_index(indrmv) = [];
                        wellst.dguno(indrmv) = [];
                        wellst.origin_modelno(indrmv) = [];
                        wellst.origin_modelTxt(indrmv) = [];                        
                        wellst.origin_modelyear(indrmv) = [];   
                        wellst.xutm(indrmv) = [];
                        wellst.yutm(indrmv) = [];
                        wellst.z(indrmv) = [];
                        wellst.zero_layerthk(indrmv) = [];
                        correction_maps(:,:,i) = get_corr_maps(XS,YS,wellst);
                    catch
                        keyboard
                    end
                end
            else
        
            end
            quat = i < Npreq; 
        
            %Prepare complexity map for well themes (Same complexity for preQ layers == 1!)
            if quat
                complexitymap = complexity;
            else
                complexitymap = correction_maps(:,:,1)*0+1;
            end
            
            if ~isempty(SnapWells{i})
                wellthemes(:,:,i) = get_well_theme(XS,YS,wellst,wellsj,terrain,complexitymap);
            else
                wellthemes(:,:,i) = Inf;
            end
        end
        save(['WELLS_themes_',region,'.mat'],'correction_maps','wellthemes','-v7.3')
    else
        load(['WELLS_themes_',region,'.mat'])
    end
    clear S
    
    %plot_validate_uncertainty_theme(wellthemes,terrain,LayerStructP);
    %keyboard
    
    
    
    AllThemes = zeros(size(wellthemes,1),size(wellthemes,2),Nlay,12)+Inf;
    AllThemes(:,:,:,1) = wellthemes;
    

    disp([cert_fun_choice,' chosen as certainty function'])
    
    %Get the remaining Geophysics Themes (No Geophsyics in Anholt and Læsø)
    if Nlay > NPL && ~strcmp(region,'AnholtLæsø')
        %waitbar(1/10,wbr,'Preparing Geophysics SkyTEM Themes')
        disp('% Preparing Geophysics SkyTEM Themes')
        if redo_themes == 1
            [SkyTEM_themes, SkyTEM_c] = get_SkyTEM_theme(XS,YS,terrain,complexity,region,1,Npreq,NPL,Nlay,cert_fun_choice);
            save(['SkyTEM_themes_',region,'.mat'],'SkyTEM_themes','SkyTEM_c','-v7.3')
        else
            load(['SkyTEM_themes_',region,'.mat'],'SkyTEM_themes','SkyTEM_c')
        end
        AllThemes(:,:,:,2) = SkyTEM_themes(:,:,1:Nlay);
    
       % plot_validate_uncertainty_theme(SkyTEM_themes,terrain,LayerStructP,region);
       % keyboard
    
        %waitbar(2/10,wbr,'Preparing Geophysics PACEP Uncertainty Themes')
        disp('% Preparing Geophysics PACEP Uncertainty Themes')
        if redo_themes == 1
            [PACEP_themes, PACEP_c] = get_PACEP_theme(XS,YS,terrain,complexity,region,1,Npreq,NPL,Nlay,cert_fun_choice);
            save(['PACEP_themes_',region,'.mat'],'PACEP_themes','PACEP_c','-v7.3')
        else
            load(['PACEP_themes_',region,'.mat'],'PACEP_themes','PACEP_c')
        end
        AllThemes(:,:,:,4) = PACEP_themes(:,:,1:Nlay);
    
        %waitbar(4/10,wbr,'Preparing Geophysics MEP Uncertainty Themes')
        disp('% Preparing Geophysics MEP Uncertainty Themes')
        if redo_themes == 1
            [MEP_themes, MEP_c] = get_MEP_theme(XS,YS,terrain,complexity,region,1,Npreq,NPL,Nlay,cert_fun_choice);
            save(['MEP_themes_',region,'.mat'],'MEP_themes','MEP_c','-v7.3')
        else
            load(['MEP_themes_',region,'.mat'],'MEP_themes','MEP_c')
        end
        AllThemes(:,:,:,5) = MEP_themes(:,:,1:Nlay);
    
        % Sdata struct is used for plotting
        Sdata.SkyTEM = SkyTEM_themes;
        Sdata.Wells = wellthemes;
        Sdata.PACEP = PACEP_themes;
        Sdata.MEP = MEP_themes;
    
        Sdata.cSkyTEM = SkyTEM_c;
        Sdata.cPACEP = PACEP_c;
        Sdata.cMEP = MEP_c;
    end
    
    
    % plot_validate_uncertainty_theme(figfolder_unc,wellthemes,'Wells',region,XS,YS)
    % clear wellthemes
    % plot_validate_uncertainty_theme(figfolder_unc,SkyTEM_themes,'SkyTEM',region,XS,YS)
    % clear SkyTEM_themes
    % clear SkyTEM_c
    % plot_validate_uncertainty_theme(figfolder_unc,PACEP_themes,'PACEP',region,XS,YS)
    % clear PACEP_themes
    % clear PACEP_c
    % plot_validate_uncertainty_theme(figfolder_unc,MEP_themes,'MEP',region,XS,YS)
    % clear MEP_themes
    % clear MEP_c
    
    
    %Load region-specific geophysics and logs
    %waitbar(7/10,wbr,'Preparing Geophysics Region-Specific Uncertainty Themes')
    disp('Preparing Geophysics Region-Specific Uncertainty Themes')
    switch region
        case 'Jylland'
            
            %%%% get_PACES_theme...
            disp('get_PACES_theme...')
            if redo_local_themes == 1 
                [PACES_themes, PACES_c] = get_PACES_theme(XS,YS,terrain,complexity,region,1,Npreq,NPL,Nlay);
                save(['PACES_themes_',region,'.mat'],'PACES_themes','PACES_c','-v7.3')
            else
                load(['PACES_themes_',region,'.mat'],'PACES_themes','PACES_c')
            end
            AllThemes(:,:,:,3) = PACES_themes(:,:,1:Nlay);
            disp('get_PACES_theme...Done')
    
            %%%%% get_GAMMALOG_theme...
            disp('get_GAMMALOG_theme...')
            if redo_local_themes == 1 
                [GAMMALOG_themes,GAMMALOG_c] = get_GAMMALOG_theme(XS,YS,terrain,complexity,region,1,Npreq,NPL,Nlay);
                save(['GAMMALOG_themes_',region,'.mat'],'GAMMALOG_themes','GAMMALOG_c','-v7.3')
            else
                load(['GAMMALOG_themes_',region,'.mat'],'GAMMALOG_themes','GAMMALOG_c')
            end
            AllThemes(:,:,:,6) = GAMMALOG_themes(:,:,1:Nlay);
            disp('get_GAMMALOG_theme...Done')
    
            %%%%% get_RESLOG_theme...
            disp('get_RESLOG_theme...')
            if redo_local_themes == 1 
                [RESLOG_themes,RESLOG_c] = get_RESLOG_theme(XS,YS,terrain,complexity,region,1,Npreq,NPL,Nlay);
                save(['RESLOG_themes_',region,'.mat'],'RESLOG_themes','RESLOG_c','-v7.3')
            else
                load(['RESLOG_themes_',region,'.mat'],'RESLOG_themes','RESLOG_c')
            end
            AllThemes(:,:,:,7) = RESLOG_themes(:,:,1:Nlay);
            disp('get_RESLOG_theme...Done')
    
            %%%%% get_REFSEIS_theme...
            disp('get_REFSEIS_theme...')
            if redo_local_themes == 1 
                [REFSEIS_themes,REFSEIS_c] = get_REFSEIS_theme(XS,YS,terrain,complexity,region,1,Npreq,NPL,Nlay);
                save(['REFSEIS_themes',region,'.mat'],'REFSEIS_themes','REFSEIS_c','-v7.3')
            else
                load(['REFSEIS_themes',region,'.mat'],'REFSEIS_themes','REFSEIS_c')
            end         
            AllThemes(:,:,:,8) = REFSEIS_themes(:,:,1:Nlay);
            disp('get_REFSEIS_theme...Done')
    
            %%%%% get_fewTEM_theme...
            disp('get_fewTEM_theme...')
            if redo_local_themes == 1 
                [fewTEM_themes, fewTEM_c] = get_fewTEM_theme(XS,YS,terrain,complexity,region,1,Npreq,NPL,Nlay);
                save(['fewTEM_themes',region,'.mat'],'fewTEM_themes','fewTEM_c','-v7.3')
            else
                load(['fewTEM_themes',region,'.mat'],'fewTEM_themes','fewTEM_c')
            end    
            AllThemes(:,:,:,9) = fewTEM_themes(:,:,1:Nlay);
            disp('get_fewTEM_theme...Done')
    
    
            %%%%% get_manyTEM_theme...
            disp('get_manyTEM_theme...')
            if redo_local_themes == 1 
                [manyTEM_themes, manyTEM_c] = get_manyTEM_theme(XS,YS,terrain,complexity,region,1,Npreq,NPL,Nlay);
                save(['manyTEM_themes',region,'.mat'],'manyTEM_themes','manyTEM_c','-v7.3')
            else
                load(['manyTEM_themes',region,'.mat'],'manyTEM_themes','manyTEM_c')
            end    
            AllThemes(:,:,:,10) = manyTEM_themes(:,:,1:Nlay);
            disp('get_manyTEM_theme...Done')
    
            %%%%% get_tTEM_theme...
            disp('get_tTEM_theme...')
            if redo_local_themes == 1 
                [tTEM_themes, tTEM_c] = get_tTEM_theme(XS,YS,terrain,complexity,region,1,Npreq,NPL,Nlay);
                save(['tTEM_themes_',region,'.mat'],'tTEM_themes','tTEM_c','-v7.3')
            else
                load(['tTEM_themes_',region,'.mat'],'tTEM_themes','tTEM_c')
            end    
            AllThemes(:,:,:,12) = tTEM_themes(:,:,1:Nlay);
            disp('get_tTEM_theme...Done')
    
    
            Sdata.GAMMALOG = GAMMALOG_themes;
            Sdata.RESLOG = RESLOG_themes;
            Sdata.REFSEIS = REFSEIS_themes;
            Sdata.fewTEM = fewTEM_themes;
            Sdata.manyTEM = manyTEM_themes;
    
            Sdata.cGAMMALOG = GAMMALOG_c;
            Sdata.cRESLOG = RESLOG_c;
            Sdata.cREFSEIS = REFSEIS_c;
            Sdata.cmanyTEM = manyTEM_c;
            Sdata.cfewTEM = fewTEM_c;
            Sdata.cPACES = PACES_c;
            
            %%%%% get_PL_theme..._ILM version
            disp('get_PL_theme...')
            if redo_local_themes == 1 
                [PL_themes,Data_snap,Data_other] = get_PL_themes_ILM(XS,YS,terrain,complexity,NPL,region,LayerStructP);
                save(['PL_themes_',region,'.mat'],'PL_themes','Data_snap','Data_other','-v7.3')
            else
                load(['PL_themes_',region,'.mat'],'PL_themes','Data_snap','Data_other')
            end
            disp('get_PL_theme...Done')
    
    
            AllThemes(:,:,1:NPL,1) = PL_themes(:,:,1:NPL);
    
            Sdata.cPeatSnaps = Data_snap;
            Sdata.cPeatJups = Data_other;
            Sdata.PL = PL_themes;
    
        case {'Fyn','Fyn-MST'}

            if NPL < Nlay
                %%%%% get_TEM_theme...
                disp('get_TEM_theme...')
                if redo_local_themes == 1 
                    [TEM_themes, TEM_c] = get_TEM_theme(XS,YS,terrain,complexity,region,1,Npreq,NPL,Nlay,cert_fun_choice);                    
                    save(['TEM_themes_',region,'.mat'],'TEM_themes','TEM_c','-v7.3')
                else
                    load(['TEM_themes_',region,'.mat'],'TEM_themes','TEM_c')
                end    
                AllThemes(:,:,:,6) = TEM_themes(:,:,1:Nlay);
                disp('get_TEM_theme...Done')


                %%%%% get_PACES_theme...
                disp('get_PACES_theme...')
                if redo_local_themes == 1 
                    [PACES_themes, PACES_c] = get_PACES_theme(XS,YS,terrain,complexity,region,1,Npreq,NPL,Nlay,cert_fun_choice);
                    save(['PACES_themes_',region,'.mat'],'PACES_themes','PACES_c','-v7.3')
                else
                    load(['PACES_themes_',region,'.mat'],'PACES_themes','PACES_c')
                
                end
                AllThemes(:,:,:,3) = PACES_themes(:,:,1:Nlay);
                disp('get_PACES_theme...Done')

                Sdata.cTEM = TEM_c;
                Sdata.TEM = TEM_themes;
                Sdata.PACES = PACES_themes;
                Sdata.cPACES = PACES_c;
                
                %plot_validate_uncertainty_theme(figfolder_unc,TEM_themes,'TEM',region,XS,YS)
                clear TEM_themes
                clear TEM_c
                %plot_validate_uncertainty_theme(figfolder_unc,PACES_themes,'PACES',region,XS,YS)
                clear PACES_themes
                clear PACES_c
            end
                
%            [PL_themes,Data_snap,Data_other] = get_PL_themes_ILM(XS,YS,terrain,complexity,NPL,region,LayerStructP);
%            AllThemes(:,:,1:NPL,1) = PL_themes;
            
            if strcmp(region,'Fyn')
                %%%%% get_PL_theme..._ILM version
                disp('get_PL_theme...')
                if redo_local_themes == 1 
                    [PL_themes,Data_snap,Data_other] = get_PL_themes_ILM(XS,YS,terrain,complexity,NPL,region,LayerStructP);
                    save(['PL_themes_',region,'.mat'],'PL_themes','Data_snap','Data_other','-v7.3')
                else
                    load(['PL_themes_',region,'.mat'],'PL_themes','Data_snap','Data_other')
                end
                disp('get_PL_theme...Done')
       
                AllThemes(:,:,1:NPL,1) = PL_themes(:,:,1:NPL);
    
                Sdata.cPeatSnaps = Data_snap;
                Sdata.cPeatJups = Data_other;
                Sdata.PL = PL_themes;
            end

        case 'AnholtLæsø'
            if redo_local_themes == 1 
                [PL_themes,Data_snap,Data_other] = get_PL_themes_ILM(XS,YS,terrain,complexity,NPL,region,LayerStructP);
                save(['PL_themes_',region,'.mat'],'PL_themes','Data_snap','Data_other','-v7.3')
            else
                load(['PL_themes_',region,'.mat'],'PL_themes','Data_snap','Data_other')
            end
            disp('get_PL_theme...Done')
   
            AllThemes(:,:,1:NPL,1) = PL_themes(:,:,1:NPL);

            Sdata.cPeatSnaps = Data_snap;
            Sdata.cPeatJups = Data_other;
            Sdata.PL = PL_themes;
    
        case 'Sjælland'
            [GAMMALOG_themes,GAMMALOG_c] = get_GAMMALOG_theme(XS,YS,terrain,complexity,region,1,Npreq,NPL,Nlay);
            AllThemes(:,:,:,6) = GAMMALOG_themes(:,:,1:Nlay);
    
            [RESLOG_themes,RESLOG_c] = get_RESLOG_theme(XS,YS,terrain,complexity,region,1,Npreq,NPL,Nlay);
            AllThemes(:,:,:,7) = RESLOG_themes(:,:,1:Nlay);
    
            [REFSEIS_themes,REFSEIS_c] = get_REFSEIS_theme(XS,YS,terrain,complexity,region,1,Npreq,NPL,Nlay);
            AllThemes(:,:,:,8) = REFSEIS_themes(:,:,1:Nlay);
    
            [fewTEM_themes, fewTEM_c] = get_fewTEM_theme(XS,YS,terrain,complexity,region,1,Npreq,NPL,Nlay);
            AllThemes(:,:,:,9) = fewTEM_themes(:,:,1:Nlay);
    
            [manyTEM_themes, manyTEM_c] = get_manyTEM_theme(XS,YS,terrain,complexity,region,1,Npreq,NPL,Nlay);
            AllThemes(:,:,:,10) = manyTEM_themes(:,:,1:Nlay);
    
    
            Sdata.GAMMALOG = GAMMALOG_themes;
            Sdata.RESLOG = RESLOG_themes;
            Sdata.REFSEIS = REFSEIS_themes;
            Sdata.fewTEM = fewTEM_themes;
            Sdata.manyTEM = manyTEM_themes;
    
            Sdata.cGAMMALOG = GAMMALOG_c;
            Sdata.cRESLOG = RESLOG_c;
            Sdata.cREFSEIS = REFSEIS_c;
            Sdata.cmanyTEM = manyTEM_c;
            Sdata.cfewTEM = fewTEM_c;
    end
    
    
    %Get the model Theme
    modeltheme = get_modeltheme(native_dir,region,LayerStructP,terrain,NPL);
    %modeltheme(:,:,1:NPL) = 4;
    
    AllThemes(:,:,:,11) = modeltheme;
    
    
    themes = AllThemes;
    
    
    %plot_validate_uncertainty_theme(figfolder_unc,modeltheme,'Model',region,XS,YS)
    clear AllThemes
    
    
    %MAKE SURE THEY COMBINE HERE
    final_variance_themes = 1./(sum(1./themes,4));
    
    
    
    %Enforce a minimum variance which is 0.5 the uncertainty of the best well
    minimum_map = get_minimum_map(terrain,LayerStructP);
    if NPL > 0
        final_variance_themes(:,:,1:NPL) = final_variance_themes(:,:,1:NPL)+0.0625;
    end
    floor_filter = final_variance_themes < minimum_map;
    final_variance_themes(floor_filter) = minimum_map(floor_filter);
    
    % Clear all unwanted large variables before saving
    save(['plotting_' region,'_cert_fun_',cert_fun_choice, '.mat'],'NPL','Sdata','allwells','LayerStructP','region','Well_S','correction_maps','-v7.3')
    
    clear themes
    clear PL_themes
    clear cur_struct
    clear Sdata
    clear minimum_map
    clear floor_filter
    close all
    
    
    
    sendolmail(emailaddress,['Main-script ' region ' update: ' sectiontit ' completed'],sectiontit)

end



%% -- Cluster Statistics -- and save variables for kriging
if kriging_ready == 0 
    
    sectiontit = '%% SECTION: Cluster statistics and save variables for kriging';
    disp(sectiontit)
    
    %The following line produces a list of median range values and 95th percentile
    %sill values for each cluster and saves it to a file. The algorithm will load
    %this for kriging. The values are also outputs of the function, but for each'
    %individual grid cell.
    [~,rm,sm]=prepare_clusters(LayerStructP,region,0.95,cert_fun_choice);
    
    for i = 1:Nlay
        LayerStructP{i}.ranges_bottom = rm{i};
        LayerStructP{i}.variances_bottom = sm{i};
    end
    
    % Saving variables for kriging
    disp('% Saving variables')
    kriging_name = ['BeforeKrigingWorkSpace',date,'_',region,'_cert_fun_',cert_fun_choice,'.mat'];
    save([native_dir,'\',kriging_name],'Nlay','LayerStructP','modeltheme','final_variance_themes','native_dir','region','terrain','correction_maps','cert_fun_choice','-v7.3')
    %save(['N:\PROJEKTER\N-retentionskortlægning 2022-24\Geostatistisk modellering\N-ret-mod\MappeJylland\BeforeKrigingWorkSpace',date,'_',region,'.mat'],'-v7.3')
    sendolmail(emailaddress,['Main-script ' region ' update: ' sectiontit ' completed'],[native_dir '\BeforeKrigingWorkSpace',date,'_',region,'.mat'])
end

%% SECTION: Kriging and Simulation
% Load essentiel kriging variables and otherwise clear workspace for
% performance

clearvars -except 'region' 'native_dir' 'kriging_name' 'Nreals' 'seed' 'emailaddress' 'Nreals_pre' 'cert_fun_choice'
close all;


sectiontit = '%% SECTION: Kriging and Simulation';
disp(sectiontit)

load([native_dir,'\',kriging_name])
rng(seed)


tb = tic;


per_drev_mappe = [native_dir '\Kriging\' region '\'];
mkdir(per_drev_mappe)


for i = 1:Nlay
    disp(['Currently kriging layer ' num2str(i)])
    wbkrig = waitbar(0,'Kriging');

    clear layermask
    clear Out
    clear kriged_structs
    clear real_s
    clear layermasks

    %n_drev_mappe = 'N:\PROJEKTER\N-retentionskortlægning 2022-24\Geostatistisk modellering\N-ret-mod\MappeJylland\';

    cursavfold = [per_drev_mappe 'Kriged_mats\lay_' num2str(i) '\'];

    pr = i/Nlay;
    waitbar(pr,wbkrig,['Kriging Layer ',num2str(i)])
    LayerStruct = LayerStructP{i};

    if LayerStruct.zeromask == 0
        LayerStruct.ModelPrior = sqrt(modeltheme(:,:,i));
        workdir = [native_dir,'\Kriging Mappe\'];
        %   workdir = ['N:\PROJEKTER\N-retentionskortlægning 2022-24\Geostatistisk modellering\N-ret-mod\Scripts Frederik\BACKUP\Mappe til Samlet Kode\Kriging Mappe\']
        layermask = LayerStructP{i}.indexMask_Combined;
    
        unique_name = [region,num2str(i)];
        OS = load(['ModeValues\ClusterindexModeSillRange_',region,cert_fun_choice,'_Layer_',num2str(i)]','out');
        out = OS.out;
        try
            [kriged_struct,~] = local_kriger(workdir,LayerStruct,unique_name,sqrt(final_variance_themes(:,:,i)),out,Nreals,modeltheme(:,:,i));
            %    kriging_time = toc(tb)
            if ~exist(cursavfold, 'dir')
                mkdir(cursavfold)
            end
            
            kriger_fn = ['kriging_results_',region,'_cert_fun_',cert_fun_choice,'_seed',num2str(seed),'_Nreals',num2str(Nreals)];    
            save([cursavfold,kriger_fn,'.mat'],'kriged_struct','layermask','terrain','-v7.3')
            sendolmail(emailaddress,['Main-script ' region ' update: ' sectiontit '- Done:' num2str(i)],['Done:' num2str(i)])
        catch
            keyboard
        end
    else
        sendolmail(emailaddress,['Main-script ' region ' update: ' sectiontit '- Done:' num2str(i)],['Done:' num2str(i) ' layer does not exist (zeromask = 1)'])
    end        
end

%save([cursavfold 'kriging_results_',region,'.mat'],'Nlay','terrain','-v7.3')

sendolmail(emailaddress,['Main-script ' region ' update: ' sectiontit ' completed'],[cursavfold])
close all
% %% Load and combine into single file
% native_dir = pwd;
% region = 'Fyn';
% per_drev_mappe = [native_dir '\Kriging\' region '\'];
% load(['BeforeKrigingWorkSpace30-Jun-2023_Fyn.mat'],'Nlay')
% 
% 
% kriged_structs = cell(1,Nlay);
% real_s = cell(1,Nlay);
% layermasks = cell(1,Nlay);
% 
% for i = 1:Nlay
%     cursavfold = [per_drev_mappe 'Kriged_mats\lay_' num2str(i) '\'];
%     load([cursavfold 'kriging_results_',region,'.mat'],'Out','kriged_struct','layermask','terrain')
%     real_s{i} = Out;
%     kriged_structs{i} = kriged_struct;
%     layermasks{i} = layermask;
% end
% 
% savestruct.real_s = real_s;
% savestruct.terrain = terrain;
% savestruct.kriged_s = kriged_structs;
% savestruct.layermasks = layermasks;
% 
% 
% save([pwd '\RealStructs',date,'_',region,'.mat'],'savestruct','-v7.3');
% save([pwd '\BeforeLayerCorrectionsWorkSpace',date,'_',region,'.mat'],'kriged_structs','-v7.3');
% 
% %save(['N:\PROJEKTER\N-retentionskortlægning 2022-24\Geostatistisk modellering\N-ret-mod\MappeJylland\RealStructs',date,'_',region,'.mat'],'savestruct','-v7.3')
% %save(['N:\PROJEKTER\N-retentionskortlægning 2022-24\Geostatistisk modellering\N-ret-mod\MappeJylland\BeforeLayerCorrectionsWorkSpace',date,'_',region,'.mat'],'kriged_structs','-v7.3')
% 
% 

%% SECTION: Integrating the Geostatistical Model and Realizations with background model
%  kriging_name = 'BeforeKrigingWorkSpace06-Nov-2023_Sjælland_cert_fun_ILM_sep2023.mat';
%  native_dir = pwd;
%  region = 'Sjælland';
%  Nreals = 20;
%  seed = 5;
%  emailaddress = 'rbm@geus.dk';
%  Nreals_pre = 80;
%  cert_fun_choice = 'ILM_sep2023';

clearvars -except 'region' 'native_dir' 'kriging_name' 'Nreals' 'seed' 'emailaddress' 'Nreals_pre'  'cert_fun_choice'
kriger_fn = ['kriging_results_',region,'_cert_fun_',cert_fun_choice,'_seed',num2str(seed),'_Nreals',num2str(Nreals)];    


sectiontit = '%% SECTION: Integrating the Geostatistical Model and Realizations with background model';
disp(sectiontit)




%load('BeforeKrigingWorkSpace10-May-2023_Jylland.mat','cur_struct','Nlay','region')
%load(['BeforeKrigingWorkSpace30-Jun-2023_Fyn.mat'],'LayerStructP')
load(kriging_name,'LayerStructP','Nlay')


per_drev_mappe = [native_dir '\Kriging\' region '\'];



for i = 1:Nlay

    %c_s1 = kriged_structs{i}; % FRAFA ORIGINAL!

    cursavfold = [per_drev_mappe 'Kriged_mats\lay_' num2str(i) '\'];

    if exist([cursavfold,kriger_fn,'.mat']) == 2
        load([cursavfold,kriger_fn,'.mat'],'kriged_struct','layermask','terrain')
        c_s1 = kriged_struct; % MAPPE VERSION!
        clear kriged_struct
    
        c_s2 = LayerStructP{i};
    
        cur_mean = c_s1.img_means;
        cur_m_est = c_s1.img_m_est;
    
        cur_mas = c_s2.indexMask_Combined;
        cur_mas = reshape(cur_mas,size(cur_mean));
        
    
        background_model = c_s1.img_FOHM;
    
        %Make the mean go to the background value
        merged_mean = merge_layers(cur_mean,background_model,cur_mas);
    
        %Make the kriging mean go to 0
        merged_m_est = merge_layers(cur_m_est,background_model*0,cur_mas);
    
        %Make each realization go to 0
        for j = 1:Nreals
            cur_real = c_s1.img_reals{j};
            merged_real = merge_layers(cur_real,background_model*0,cur_mas);
    
            %Put the realization back into the struct
            c_s1.img_reals{j} = merged_real;
        end
    
        %Put Model Back into Struct
        c_s1.img_means = merged_mean;
        c_s1.img_m_est = merged_m_est;
    
        %Put Struct back into "Kriged_Structs"
        %kriged_structs{i} = c_s1; %FRAFA ORIGINAL
        kriged_struct = c_s1;
        save([cursavfold,kriger_fn,'_background.mat'],'kriged_struct','layermask','terrain','-v7.3')
    else
        cursavfold_prev = [per_drev_mappe 'Kriged_mats\lay_' num2str(i-1) '\'];
        load([cursavfold_prev,kriger_fn,'_background.mat'],'kriged_struct','layermask','terrain');
        if ~exist(cursavfold, 'dir')
            mkdir(cursavfold)
        end
        save([cursavfold,kriger_fn,'_background.mat'],'kriged_struct','layermask','terrain','-v7.3')
    end
    disp(['Done:' num2str(i)])
end

sendolmail(emailaddress,['Main-script ' region ' update: ' sectiontit ' completed'],[cursavfold])


%% SECTION: Applying Ties at Absolute 0-Thickness points
% This section is here because some places with 0 thickness actually
% are not supposed to attain a thickness during simulation, although they might.
% The ties were pre-defined in a previous section and in this section
% they are applied.

% kriging_name = 'BeforeKrigingWorkSpace07-Sep-2023_Jylland.mat';
% native_dir = pwd;
% region = 'Jylland';
% Nreals = 10;
% seed = 1;
% emailaddress = 'rbm@geus.dk';

clearvars -except 'region' 'native_dir' 'kriging_name' 'Nreals' 'seed' 'emailaddress' 'Nreals_pre' 'cert_fun_choice'
kriger_fn = ['kriging_results_',region,'_cert_fun_',cert_fun_choice,'_seed',num2str(seed),'_Nreals',num2str(Nreals)];    


sectiontit = '%% SECTION: Applying Ties at Absolute 0-Thickness points';
disp(sectiontit)


per_drev_mappe = [native_dir '\Kriging\' region '\'];

load(kriging_name,'LayerStructP','Nlay','correction_maps')


cmap_lin = lines(50);

figure(101); clf(101); 

%Loop through all nreals sets of realizations and for each realization all
%Nlay layers.
for j = 1:Nlay
    
    cur_tie = ~isnan(correction_maps(:,:,j));
    
    doTie = sum(cur_tie(:))<numel(cur_tie);
    %keyboard
    
    % Layer above
    if j == 1
        cursavfold_prev = [per_drev_mappe 'Kriged_mats\lay_' num2str(1) '\'];
        load([cursavfold_prev,kriger_fn,'_background.mat'],'terrain')  
    else
        cursavfold_prev = [per_drev_mappe 'Kriged_mats\lay_' num2str(j-1) '\'];
        load([cursavfold_prev,kriger_fn,'_background.mat'],'kriged_struct')
        cur_mean1 = kriged_struct.img_means; % Folder version
        cur_m_est1 = kriged_struct.img_m_est; % Folder version
        layerabove = kriged_struct.img_reals;
    end
    clear kriged_struct
    
    % Current layer
    cursavfold = [per_drev_mappe 'Kriged_mats\lay_' num2str(j) '\'];
    load([cursavfold,kriger_fn,'_background.mat'],'kriged_struct','layermask','terrain')
    

    for i = 1:Nreals

    
        %Define the specific realization of the current layer and the
        %previous layer (the one above). real1 will be the upper layer and
        %real2 will be the lower layer.

        if j == 1
            real1 = terrain;
        else
            
            %cur_mean1 = kriged_structs{j-1}.img_means; % FRAFRA ORIGINAL
            %cur_m_est1 = kriged_structs{j-1}.img_m_est; % FRAFRA ORIGINAL
            %cur_real1 = kriged_structs{j-1}.img_reals{i}; % FRAFRA ORIGINAL

            cur_real1 = layerabove{i}; % Folder version
            real1 = cur_mean1+cur_m_est1+cur_real1;
        end
        
        %cur_mean2 = kriged_structs{j}.img_means; % FRAFRA ORIGINAL
        %cur_m_est2 = kriged_structs{j}.img_m_est; % FRAFRA ORIGINAL
        %cur_real2 = kriged_structs{j}.img_reals{i}; % FRAFRA ORIGINAL
        
        cur_mean2 = kriged_struct.img_means; % Folder version
        cur_m_est2 = kriged_struct.img_m_est; % Folder version
        cur_real2 = kriged_struct.img_reals{i}; % Folder version

        real2 = cur_mean2+cur_m_est2+cur_real2;

        %cur_tie = ~isnan(correction_maps(:,:,j)); % FRAFRA ORIGINAL

        %Apply the ties on real2 and get a new real2.
 %       if j == 6
 %           keyboard
 %       end
        if doTie % should layer be tied or not
            new_real2 = tie_realizations(real1,real2,cur_tie);
        else
            new_real2 = real2;
        end
%
        
%         ax1 = subplot(3,1,3); 
%         plot(cur_tie(741,:),'color',cmap_lin(j,:)); hold on 
%         ax2 = subplot(3,1,[1:2]); 
%         plot(terrain(741,:),'k','LineWidth',1.5); hold on;
%         plot(real2(741,:),'color',cmap_lin(j,:),'LineWidth',1.5);
%         plot(new_real2(741,:),':','color',cmap_lin(j,:),'LineWidth',1);

%        new_real2 = tie_realizations(real1,real2,cur_tie);
%        new_real3 = tie_realizations(real2,real1,cur_tie);

%        plot(new_real2(741,:),'k--','linewidth',1)
%        plot(new_real3(741,:),'r--','linewidth',1)

   %     linkaxes([ax1,ax2],'x') 

        
        %Update the model before the next layer/iteration
        %kriged_structs{j}.img_reals{i} = new_real2-cur_mean2-cur_m_est2; % FRAFRA ORIGINAL
        kriged_struct.img_reals{i} = new_real2-cur_mean2-cur_m_est2; % Folder version

    end

    save([cursavfold,kriger_fn,'_tied.mat'],'kriged_struct','layermask','terrain','-v7.3')
    disp(['Done tying model layer:' num2str(j)])
end

sendolmail(emailaddress,['Main-script ' region ' update: ' sectiontit ' completed'],[cursavfold])

%% PLot test
% figure(101); clf(101); 
% ax1 = subplot(3,1,3); plot(cur_tie(741,:)); 
% ax2 = subplot(3,1,[1:2]); 
% plot(real1(741,:),'r'); hold on; plot(real2(741,:),'k')
% new_real2 = tie_realizations(real1,real2,cur_tie);
% new_real3 = tie_realizations(real2,real1,cur_tie);
% 
% plot(new_real2(741,:),'k--','linewidth',1)
% plot(new_real3(741,:),'r--','linewidth',1)
% 
% linkaxes([ax1,ax2],'x')
% 

%% Combine to all kriged layers into combined struct
sectiontit = '%% Combine to all kriged layers into combined struct';
disp(sectiontit)

for ilay = 1:Nlay
    cursavfold = [per_drev_mappe 'Kriged_mats\lay_' num2str(ilay) '\'];
    load([cursavfold,kriger_fn,'_tied.mat'],'kriged_struct');
    kriged_structs{ilay} = kriged_struct;
    ilay
end

krig_str_name = ['kriged_structs',date,'_',region,'_cert_fun_',cert_fun_choice,'_seed',num2str(seed),'_Nreals',num2str(Nreals),'.mat'];
combined_struct_name = [per_drev_mappe,'\',krig_str_name];
save(combined_struct_name,'kriged_structs','terrain','-v7.3')

sendolmail(emailaddress,['Main-script ' region ' update: ' sectiontit ' completed'],['Name of file: ' combined_struct_name])


%% SECTION: More Layer Corrections and Assembly of Final Model


% kriging_name = 'BeforeKrigingWorkSpace19-Oct-2023_Fyn-MST_cert_fun_RBM_oct2023.mat';
% native_dir = pwd;
% region = 'Fyn-MST';
% Nreals = 50;
% seed = 1;
% emailaddress = 'rbm@geus.dk';
% cert_fun_choice= 'RBM_oct2023';
% Nreals_pre = 1000;
% dategiven = '19-Oct-2023';

clearvars -except 'region' 'native_dir' 'kriging_name' 'Nreals' 'emailaddress' 'seed' 'Nreals_pre'  'cert_fun_choice' 'dategiven'

sectiontit = '%% SECTION: More Layer Corrections and Assembly of Final Model';
disp(sectiontit)

per_drev_mappe = [native_dir '\Kriging\' region '\'];

if exist('dategiven') == 1
    load([per_drev_mappe '\kriged_structs',dategiven,'_',region,'_cert_fun_',cert_fun_choice,'_seed',num2str(seed),'_Nreals',num2str(Nreals),'.mat'],'kriged_structs','terrain')    
else
    load([per_drev_mappe '\kriged_structs',date,'_',region,'_cert_fun_',cert_fun_choice,'_seed',num2str(seed),'_Nreals',num2str(Nreals),'.mat'],'kriged_structs','terrain')
end

% Compute mean model and realizations (Includes layer correction!)
destination_path = [per_drev_mappe];
%Gather Structs in a single 3D matrix and correct layers such that they
%terminate on the upper layer where-ever they are simulated to exist above it.
%[mean_mod,realization_mods,ref_mod] = get_layered_models(kriged_structs,terrain);
if ~strcmp(region,'AnholtLæsø')
    usepreQ = 1;
else
    usepreQ = 0;
end


plot_name = ['plotting_' region,'_cert_fun_',cert_fun_choice, '.mat']; % ADD DATE???
load(plot_name,'LayerStructP')
Onshore_mask = reshape(LayerStructP{1}.indexMask_Onshore,LayerStructP{1}.Grid_Size_Rows,LayerStructP{1}.Grid_Size_Columns);
clear LayerStructP

[mean_mod,realization_mods,krig_mean] = get_layered_models(kriged_structs,terrain,region,usepreQ,Onshore_mask);

mods_name = ['mods',date,'_',region,'_seed',num2str(seed),'_Nreals',num2str(Nreals),'.mat'];
save([destination_path,'\',mods_name],'mean_mod','realization_mods','krig_mean','-v7.3')


% Add bottom
[final_model,final_reals] = add_bottom(mean_mod,realization_mods);

sendolmail(emailaddress,['Main-script ' region ' update: ' sectiontit ' completed'],['Models created'])

% %% Quick Thickness plot
% % Get landsdel
% landsdel = region_to_landsdel(region);
% % Get NPL
% [~,NPL,Nlay,layernames,~,~,~,~,~,~] = landsdel_switch(landsdel,0,0);
% 
% plot_name = ['plotting_' region '.mat']; % ADD DATE???
% load(plot_name,'NPL','Sdata','allwells','LayerStructP','region','Well_S','correction_maps')
% XS = LayerStructP{1}.Grid_UTM_X;
% YS = LayerStructP{1}.Grid_UTM_Y;
% figure(44); clf(44)
% 
% ilay = 9;
% ireal = 2;
% subplot(1,3,1); imagesc( XS,YS,(LayerStructP{ilay}.LayerTop-LayerStructP{ilay}.LayerBottom)); colorbar; hold on; plot_dk;
% caxis([0,20])
% title('Input thickness')
% subplot(1,3,2); imagesc( XS,YS,(final_reals(:,:,ilay,ireal)-final_reals(:,:,ilay+1,ireal))); colorbar; hold on; plot_dk;
% caxis([0,20])
% title('Realization thickness')
% subplot(1,3,3); imagesc( XS,YS,reshape(LayerStructP{ilay}.indexMask_Combined,1030,860)); colorbar; hold on; plot_dk;
% title('Mask')
% 
% suptitle(layernames{ilay+1})

%%


% %% Plot all surfaces of mean model
% 
% %load('BeforeKrigingWorkSpace10-May-2023_Jylland.mat','layernames')
% %load(['BeforeKrigingWorkSpace30-Jun-2023_Fyn.mat'])
% load(kriging_name)
% 
% 
% for ilay = 9
%     gcf1 = figure(1); clf(1)
%     set(gcf1,'color','w');
%     set(gcf1,'Units','normalized','Position',[0,0,1,1])
% 
%     curname = layernames{ilay+1};
%     
%     % NEW MODEL!
%     mean_surf = mean_mod(:,:,ilay+1);
%     mean_above = mean_mod(:,:,ilay);
%     
%     % OLD MODEL!    
%     [~,~,mean_surf_old]= dtm_read_plot([native_dir,'\mean_model\' layernames{ilay+1},'.asc']);
%     [~,~,mean_above_old]= dtm_read_plot([native_dir,'\mean_model\' layernames{ilay},'.asc']);
% 
%     % OLD MODEL!    
%     FOHM_surf = reshape(LayerStructP{ilay}.LayerBottom,LayerStructP{ilay}.Grid_Size_Rows,LayerStructP{ilay}.Grid_Size_Columns);
%     if ilay==1
%         FOHM_above = terrain;
%     else
%         FOHM_above = reshape(LayerStructP{ilay-1}.LayerBottom,LayerStructP{ilay-1}.Grid_Size_Rows,LayerStructP{ilay-1}.Grid_Size_Columns);
%     end
%     
%     
% 
%     %FOHM_surf(FOHM_surf<3) = nan;
%     %mean_surf(mean_surf<3) = nan;
%     %mean_above(mean_above<3) = nan;
% 
%     subplot(2,3,4)
%     
%     imagesc(LayerStructP{ilay}.Grid_UTM_X,LayerStructP{ilay}.Grid_UTM_Y,mean_surf)
%     hold on;plot_dk(0,1,1);
%     colorbar
%     set(gca,'ydir','normal')
%     
%     caxis([-400,150])
%     colormap(gca,'jet')
%     
%     title('NEW: Mean model')
%     xlabel('XUTM')
%     ylabel('YUTM')
%     
%     
%     subplot(2,3,2)
%     imagesc(LayerStructP{ilay}.Grid_UTM_X,LayerStructP{ilay}.Grid_UTM_Y,mean_surf)
%     hold on;plot_dk(0,1,1);
%     set(gca,'ydir','normal')
%     colorbar
%     
%     caxis([-400,150])
%     colormap(gca,'jet')
%     
%     title('OLD: Mean model')
%     xlabel('XUTM')
%     ylabel('YUTM')
%     
%     
%     subplot(2,3,3)
%     imagesc(LayerStructP{ilay}.Grid_UTM_X,LayerStructP{ilay}.Grid_UTM_Y,FOHM_surf)
%     hold on;plot_dk(0,1,1);
%     set(gca,'ydir','normal')
%     colorbar
%     
%     caxis([-400,150])
%     colormap(gca,'jet')
%     
%     title('DK-model/FOHM')
%     xlabel('XUTM')
%     ylabel('YUTM')
%     
%     subplot(2,3,5)
%     imagesc(LayerStructP{ilay}.Grid_UTM_X,LayerStructP{ilay}.Grid_UTM_Y,abs(mean_surf-mean_surf_old))
%     hold on;plot_dk(0,1,1);
%     set(gca,'ydir','normal')
%     colorbar
%     
%     caxis([0,25])
%     colormap(gca,flipud(bone))
%     
%     title('abs(New-Old)')
%     xlabel('XUTM')
%     ylabel('YUTM')
% 
%     subplot(2,3,6)
%     imagesc(LayerStructP{ilay}.Grid_UTM_X,LayerStructP{ilay}.Grid_UTM_Y,abs(mean_surf-FOHM_surf))
%     hold on;plot_dk(0,1,1);
%     set(gca,'ydir','normal')
%     colorbar
%     
%     caxis([0,25])
%     colormap(gca,flipud(bone))
%     
%     title('abs(New-DK/FOHM)')
%     xlabel('XUTM')
%     ylabel('YUTM')
% 
% 
%     
% 
%     %diff1 = mean_above-mean_surf;
%     %diff1(diff1<=5) = nan;
%     
%    
% 
%     gcf2 = figure(2); clf(2)
%     set(gcf2,'color','w');
%     set(gcf2,'Units','normalized','Position',[0,0,1,1])
% 
%     subplot(2,3,4)
%     imagesc(LayerStructP{ilay}.Grid_UTM_X,LayerStructP{ilay}.Grid_UTM_Y,diff1)
%     hold on;plot_dk(0,1,1);
%     set(gca,'ydir','normal')
%     colorbar
%     
%     caxis([0,25])
%     colormap(gca,flipud(hot))
%     
%     title('Mean model - Layer Thickness')
%     xlabel('XUTM')
%     ylabel('YUTM')
%     
%     
% %    diff2 = mean_above-mean_surf;
% %    diff2(diff1<=3) = nan;
%     
%  
% 
%     subplot(2,3,5)
%     imagesc(LayerStructP{ilay}.Grid_UTM_X,LayerStructP{ilay}.Grid_UTM_Y,FOHM_above-FOHM_surf)
%     hold on;plot_dk(0,1,1);
%     set(gca,'ydir','normal')
%     colorbar
%     
%     caxis([0,25])
%     colormap(gca,flipud(hot))
%     
%     title('Mean model - Layer Thickness [m]')
%     xlabel('XUTM')
%     ylabel('YUTM')
%     
%     
%     subplot(2,3,6)
%     imagesc(LayerStructP{ilay}.Grid_UTM_X,LayerStructP{ilay}.Grid_UTM_Y,abs((mean_above-mean_surf)-(FOHM_above-FOHM_surf)))
%     hold on;plot_dk(0,1,1);
%     set(gca,'ydir','normal')
%     colorbar
%     
%     caxis([0,25])
%     colormap(gca,flipud(bone))
%     
%     title('abs(Thick. Mean model - Thick. Original) [m]')
%     xlabel('XUTM')
%     ylabel('YUTM')
%     
%     
%     suptitle(curname)
%     
%     export_fig(gcf,[destination_path 'meanmodel_figs\' curname '_oldkrig_2.png'],'-m3','-r350')
% 
% end
% 



%% SECTION: Exporting models
%save(['N:\PROJEKTER\N-retentionskortlægning 2022-24\Geostatistisk modellering\N-ret-mod\MappeJylland\FullWorkSpace',date,'_',region,'.mat'],'-v7.3')

sectiontit = '%% SECTION: Exporting models';
disp(sectiontit)

wbr = waitbar(0,'Saving Final Model');
%destination_path = ['N:\PROJEKTER\N-retentionskortlægning 2022-24\Geostatistisk modellering\N-ret-mod\MappeJylland'];
%load('BeforeKrigingWorkSpace10-May-2023_Jylland.mat','NPL','XS','YS')

plot_name = ['plotting_' region,'_cert_fun_',cert_fun_choice, '.mat']; % ADD DATE???

% Get landsdel
landsdel = region_to_landsdel(region);
% Get NPL
[~,NPL,Nlay,layernames,~,~,~,~,~,~] = landsdel_switch(landsdel,0,0);
% Get XS and YS
load(plot_name,'LayerStructP')
XS = LayerStructP{1}.Grid_UTM_X;
YS = LayerStructP{1}.Grid_UTM_Y;

clear LayerStructP
clear kriged_structs
clear realization_mods

if strcmp(region,'Fyn')
    destination_path_local = ['D:\dk3\'];
    destination_path_server = ['Y:\Stochastic_hydrostratigraphy\dk3\'];
end

if strcmp(region,'Fyn-MST')
    destination_path_local = ['D:\MST Efterår 2023\'];
    destination_path_server = ['N:\PROJEKTER\MST Efterår 2023\Realization - October 2023'];
end

if strcmp(region,'Jylland')
    destination_path_local = ['D:\dk456\'];
    destination_path_server = ['Y:\Stochastic_hydrostratigraphy\dk456\'];
end

if strcmp(region,'Sjælland')
    destination_path_local = ['D:\dk12\'];
    destination_path_server = ['Y:\Stochastic_hydrostratigraphy\dk12\'];
end

if strcmp(region,'AnholtLæsø')
    destination_path_local = ['D:\dk8\'];
    destination_path_server = ['Y:\Stochastic_hydrostratigraphy\dk8\'];
end


%per_drev_mappe = [native_dir '\Kriging\' region '\'];


% Jylland
if landsdel == 1
    exportmodels(XS,YS,final_model,final_reals,layernames(1:50),region,wbr,destination_path_local,Nreals_pre)
end

% Fyn
if landsdel == 2
    exportmodels(XS,YS,final_model,final_reals,layernames,region,wbr,destination_path_local,Nreals_pre)
end

% Fyn-MST
if landsdel == 22
    exportmodels(XS,YS,final_model,final_reals,layernames,region,wbr,destination_path_local,Nreals_pre)
end

% Sjælland
if landsdel == 3
    exportmodels(XS,YS,final_model,final_reals,layernames(1:14),region,wbr,destination_path_local,Nreals_pre)
end

% AnholtLæsø
if landsdel == 4
    exportmodels(XS,YS,final_model(:,:,1:14),final_reals(:,:,1:14,:),layernames(1:14),region,wbr,destination_path_local,Nreals_pre)
end


sendolmail(emailaddress,['Main-script ' region ' update: ' sectiontit ' completed'],['Local copy on: ',destination_path_local])

% Copying realizations to server
for i = 1:Nreals
    copyfile([destination_path_local,'\realizations\realization',num2str(Nreals_pre+i)],[destination_path_server,'\realizations\realization',num2str(Nreals_pre+i)])
    disp(['Copying realization ' num2str(Nreals_pre+i),' to ',destination_path_server])
end

sendolmail(emailaddress,['Main-script ' region ' update: ' sectiontit ' completed'],['Server copy on: ',destination_path_server])




%save([destination_path '\ProcessingComplete',date,'_',region,'.mat'],'NPL','-v7.3')




% %% Load for plot
% 
% % mods_name = 'mods12-Sep-2023_Jylland.mat';
% % load([destination_path,'\',mods_name],'mean_mod','realization_mods')
% % 
% % plot_name = ['plotting_' region '.mat']; % ADD DATE???
% % load(plot_name,'NPL','Sdata','allwells','LayerStructP','region','Well_S','correction_maps')
% % 
% % 
% % krig_str_name = 'kriged_structs12-Sep-2023_Jylland.mat';
% % load([per_drev_mappe,'\',krig_str_name],'kriged_structs','terrain')
% 
% 
% mods_name = 'mods22-Sep-2023_Fyn_seed1_Nreals10.mat';
% load([destination_path,'\',mods_name],'mean_mod','realization_mods')
% 
% plot_name = ['plotting_' region '.mat']; % ADD DATE???
% load(plot_name,'NPL','Sdata','allwells','LayerStructP','region','Well_S','correction_maps')
% XS = LayerStructP{1}.Grid_UTM_X;
% YS = LayerStructP{1}.Grid_UTM_Y;
% 
% 
% krig_str_name = 'kriged_structs22-Sep-2023_Fyn_seed1_Nreals10.mat';
% load([per_drev_mappe,'\',krig_str_name],'kriged_structs','terrain')
% 
% 
% 
% %% Graphically Illustrate the Models
% %laynums = [Npreq-1];
% laynums = [4];
% type = 'East-West';
% ind = 600;
% YS(ind)
% 
% plot_profile_view(laynums,type,ind,kriged_structs,LayerStructP,terrain,Sdata,region,allwells,mean_mod,realization_mods,correction_maps,Well_S,NPL)
% 
% %%
% figure(123);clf;imagesc(XS,YS,Sdata.Wells(:,:,4)); hold on; plot_dk(); set(gca,'Ydir','normal'); colorbar
% caxis([0,20])
% plot([XS(1),XS(end)],[YS(ind),YS(ind)])
% plot(592650,YS(ind),'ko','markersize',10)
% 
% %%
% figure(124); clf;
% % semilogy(1,sqrt(Sdata.PACEP(find(XS == 592650),ind,4)),'.','markersize',10); hold on;
% % plot(1,sqrt(Sdata.PACES(find(XS == 592650),ind,4)),'.','markersize',10);
% % plot(1,sqrt(Sdata.SkyTEM(find(XS == 592650),ind,4)),'.','markersize',10);
% % plot(1,sqrt(Sdata.Wells(find(XS == 592650),ind,4)),'.','markersize',10);
% % plot(1,sqrt(Sdata.MEP(find(XS == 592650),ind,4)),'.','markersize',10);
% % plot(1,sqrt(Sdata.TEM(find(XS == 592650),ind,4)),'.','markersize',10);
% % %plot(1,Sdata.PL(find(XS == 592650),ind,4),'.','markersize',10);
% % plot(1,sqrt(final_variance_themes(find(XS == 592650),ind,4)),'.','markersize',10);
% 
% semilogy(XS,sqrt(Sdata.PACEP(ind,:,4)),'.-','markersize',10); hold on;
% plot(XS,sqrt(Sdata.PACES(ind,:,4)),'.-','markersize',10);
% plot(XS,sqrt(Sdata.SkyTEM(ind,:,4)),'.-','markersize',10);
% plot(XS,sqrt(Sdata.Wells(ind,:,4)),'.-','markersize',10);
% plot(XS,sqrt(Sdata.MEP(ind,:,4)),'.-','markersize',10);
% plot(XS,sqrt(Sdata.TEM(ind,:,4)),'.-','markersize',10);
% plot(XS,sqrt(final_variance_themes(ind,:,4)),'.-','markersize',10);
% 
% 
% %plot(1,Sdata.Wells(find(XS == 592650),ind,4),'.','markersize',10);
% 
% legend([{'PACEP'},{'PACES'},{'SkyTEM'},{'Wells'},{'MEP'},{'TEM'},{'FINAL'}])
% 
% 
% %%
% for i = 1:10
%     plot_reals_profile3(mean_mod,realization_mods,kriged_structs,terrain,0,1,correction_maps,5,1)
%     close all
% end
% plot_reals_profile(mean_mod,realization_mods,kriged_structs,terrain,1:1350,1000,correction_maps,NPL);
% 
% 
% % ???????? HVAD ER DET FOR NOGET ??? - RBM 15/05-2023
% %product = cell(1,2);
% %product{1} = mean_mod;
% %product{2} = realization_mods;
% end

