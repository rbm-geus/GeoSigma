function [SnapList] = ReadWriteTiles(names,layerpath,terrain,tinfo,reg,doplot)
names = names(2:6);

if nargin < 5
    doplot = 0;
end

%Determine Number of Layers to add based on Landsdel
switch reg
    case 'Jylland'
        NL = 5;
    case 'Fyn'
        NL = 3;
    case 'Sjælland'
        NL = 0;
    case 'AnholtLæsø'
        NL = 5;
end

if NL > 0
    if ~strcmp(reg,'AnholtLæsø')
        %Parent Folder
        folder_dir = ['N:\PROJEKTER\N-retentionskortlægning 2022-24\GS3D_Jylland\N_Ret_grids_Tolkede_tiles\', reg,'\'];
    
        %Find Available Tiles
        L = ['A','B','C','D','E','F','G','H','I','J','K','L'];
        N = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15'];
    
        StandardNames = [{'Lag1_Bund_Post_Glacial_Ler_Tørv_Gytje_Adjusted_'},{'Lag2_Bund_Post_Glacial_Sand_Adjusted_'},{'Lag3_Bund_Post_Glacial_Ler_Adjusted_'},{'Lag4_Bund_Sen_Glacial_Sand_Adjusted_'},{'Lag5_Bund_Sen_Glacial_Ler_Adjusted_'}];
    
        NumTiles = size(L,2).*size(N,2)/2;
        c = 0;
    
        TileCoordMatrix = load("TileCoordinates.mat");
        TileCoordMatrix = TileCoordMatrix.TileCoordinates1;
    
        %Read the vendsyssel Mask
        [Vmask,Vinfo] = ReadSurfer7([pwd,'\Vmask.grd']);
        V_xs = Vinfo.UTM_X-50;
        V_ys = Vinfo.UTM_Y+50;
    
    
    
        switch reg
            case 'Jylland'
    
                GlobalXS = (min(TileCoordMatrix(:,1)):100:(max(TileCoordMatrix(:,3))-100));
                GlobalYS = (min(TileCoordMatrix(:,4)):100:(max(TileCoordMatrix(:,2))-100));
    
            case 'Fyn'
    
                GlobalXS = (min(TileCoordMatrix(:,1)):100:(max(TileCoordMatrix(:,3))-100))+50;
                GlobalYS = (min(TileCoordMatrix(:,4)):100:(max(TileCoordMatrix(:,2))-100))+50;
    
            case 'Sjælland'
    
                GlobalXS = (min(TileCoordMatrix(:,1)):100:(max(TileCoordMatrix(:,3))-100))+50;
                GlobalYS = (min(TileCoordMatrix(:,4)):100:(max(TileCoordMatrix(:,2))-100))+50;
    
        end
    
        for i = 1:size(L,2)
            for j = 1:(size(N,2)/2)
                c = c+1;
                tile_name{c} = [L(i),N(2*(j-1)+1:2*j)];
                vec2id(1:3,c) = uint8(tile_name{c});
                vec2id(4,c) = c;
            end
        end
    
        %Get folder names
        list = dir(folder_dir);
        Nfold = size(list,1)-2;
    
        % foldernames = list.name;
        count = 0;
    
        %Check if folder name has tile name
        for i = 1:Nfold
            foldername = list(2+i).name;
            for j = 1:NumTiles
                temp = strfind(foldername,tile_name{j});
    
                if size(temp,1)>0
                    count = count+1;
                    fn_vec{count} = foldername;
                    tn_vec{count} = tile_name{j};
                end
            end
        end
    
        TileToFolderMap = [tn_vec;fn_vec;];
        TileToFolderMap{3,count} = [];
    
        TileSize = 250;
    
        %calculate number of tiles per folder
        dup_mat = zeros(size(TileToFolderMap,2));
    
        for i = 1:size(TileToFolderMap,2)
            for j = (i+1):size(TileToFolderMap,2)
                if size(TileToFolderMap{2,i},2)==size(TileToFolderMap{2,j},2)
                    dup_mat(i,j) = prod(TileToFolderMap{2,i} == TileToFolderMap{2,j});
                end
            end
        end
    
        dup_mat = logical(dup_mat);
        [dx(:,1),dx(:,2)] = find(dup_mat);
    
        %start merging dupes
        for i = 1:size(dx,1)
            ind = 1+size(dx,1)-i;
    
            TileToFolderMap{3,dx(ind,1)} = TileToFolderMap{1,dx(ind,2)};
            TileToFolderMap(:,dx(ind,2)) = [];
        end
    
        layercube = NaN(TileSize*15,TileSize*12,NL);
    
        for i = 1:size(TileToFolderMap,2)
            isdupe = 0;
            local_tile = zeros(250,250,5);
    
            if numel(TileToFolderMap{3,i})>0
                isdupe = true;
                cur_tile = [TileToFolderMap{1,i},'_',TileToFolderMap{3,i}];
            else
                cur_tile = TileToFolderMap{1,i};
            end
            cur_folder = TileToFolderMap{2,i};
    
            co = 0;
            val = 0;
            tile_ID = [];
    
            while val < 2
                co = co+1;
    
    
                if isdupe
                    sv(1,1:3) = uint8(cur_tile(1:3));
                    sv(3,1:3) = uint8(cur_tile(5:7));
    
                    for i = [1,3]
                        if logical(prod(sv(i,:) == (vec2id(1:3,co)')))
                            tile_ID = [tile_ID vec2id(4,co)];
                            val = val+1;
                        end
                    end
                else
                    sv = uint8(cur_tile);
                    if logical(prod(sv == (vec2id(1:3,co)')))
                        tile_ID = vec2id(4,co);
                        val = 2;
                    end
                end
            end
    
            TileCoordMatrix(:,1:4) = TileCoordMatrix(:,1:4);
            tile_coords = TileCoordMatrix(tile_ID,:);
    
            %Tile
            switch reg
                case 'Jylland'
                    tile_xs = (min(tile_coords(:,1)):100:(max(tile_coords(:,3))));
                    tile_ys = (min(tile_coords(:,4)):100:(max(tile_coords(:,2))));
    
                case 'Fyn'
                    tile_xs = (min(tile_coords(:,1)):100:(max(tile_coords(:,3))-100))+50;
                    tile_ys = (min(tile_coords(:,4)):100:(max(tile_coords(:,2))-100))+50;
    
                case 'Sjælland'
                    tile_xs = (min(tile_coords(:,1)):100:(max(tile_coords(:,3))-100))+50;
                    tile_ys = (min(tile_coords(:,4)):100:(max(tile_coords(:,2))-100))+50;
    
            end
    
            %Now load the tile layers
            path_dir = [folder_dir,cur_folder,'\',];
    
            localfiles = dir(path_dir);
            localfilenames = localfiles(3:end).name;
    
            Nfiles = size(localfiles,1)-2;
            ln = [];
    
            for iz = 3:Nfiles+2
                lnt = str2num(localfiles(iz).name(4));
                if sum(size(lnt))>0
                    ln(iz) = str2num(localfiles(iz).name(4));
                end
    
            end
            Nlayers = max(ln);
    
            for is = 1:Nlayers
    
                filename = [StandardNames{is},cur_tile];
    
    
                if isfile([path_dir,filename,'.asc'])
    
                    [grd_xs,grd_ys,tilegrid] = dtm_read_plot([path_dir,filename,'.asc']);
    
                elseif isfile([path_dir,filename,'.grd'])
    
                    [tilegrid,info] = ReadSurfer7([path_dir,filename,'.grd']);
                    grd_xs = info.UTM_X;
                    grd_ys = info.UTM_Y;
    
                else
                    disp('The requested grid file does not exist in .asc or .grd format')
                    keyboard
                end
    
                xmin = find(grd_xs == tile_xs(1));
    
                if prod(size(xmin)) == 0
                    xmin = find(grd_xs == grd_xs(1));
                end
    
                xmax = find(grd_xs == tile_xs(end));
                if prod(size(xmax)) == 0
                    xmax = find(grd_xs == grd_xs(end));
                end
    
                ymin = find(grd_ys == tile_ys(1));
                if prod(size(ymin)) == 0
                    ymin = find(grd_ys == grd_ys(1));
                end
    
                ymax = find(grd_ys == tile_ys(end));
                if prod(size(ymax)) == 0
                    ymax = find(grd_ys == grd_ys(end));
                end
    
                layer = tilegrid(ymin:ymax,xmin:xmax);
                layer_xs = grd_xs(xmin:xmax);
                layer_ys = grd_ys(ymin:ymax);
    
                %find indices
                switch reg
                    case 'Fyn'
                        global_row_start = (layer_ys(1)-((min(TileCoordMatrix(:,4))+50)))/100+1;
                        global_row_end = (layer_ys(end)-((min(TileCoordMatrix(:,4))+50)))/100+1;
                        global_col_start = (layer_xs(1)-((min(TileCoordMatrix(:,1))+50)))/100+1;
                        global_col_end = (layer_xs(end)-((min(TileCoordMatrix(:,1))+50)))/100+1;
    
                    case 'Jylland'
                        global_row_start = (layer_ys(1)-min(TileCoordMatrix(:,4)))/100+1;
                        global_row_end = (layer_ys(end)-min(TileCoordMatrix(:,4)))/100+1;
                        global_col_start = (layer_xs(1)-min(TileCoordMatrix(:,1)))/100+1;
                        global_col_end = (layer_xs(end)-min(TileCoordMatrix(:,1)))/100+1;
    
    
                    case 'Sjælland'
                        global_row_start = (layer_ys(1)-((min(TileCoordMatrix(:,4)))))/100+1;
                        global_row_end = (layer_ys(end)-((min(TileCoordMatrix(:,4)))))/100+1;
                        global_col_start = (layer_xs(1)-((min(TileCoordMatrix(:,1)))))/100+1;
                        global_col_end = (layer_xs(end)-((min(TileCoordMatrix(:,1)))))/100+1;
                end
    
                if global_row_start < 1
                    CutSizeY1 = 1-(global_row_start);
                else
                    CutSizeY1 = 0;
                end
    
                if global_row_end > numel(GlobalYS)
                    CutSizeY2 = global_row_end - numel(GlobalYS);
                else
                    CutSizeY2 = 0;
                end
    
                if global_col_start < 1
                    CutSizeX1 = 1-(global_col_start);
                else
                    CutSizeX1 = 0;
                end
    
                if global_col_end > numel(GlobalXS)
                    CutSizeX2 = global_col_end - numel(GlobalXS);
                else
                    CutSizeX2 = 0;
                end
    
                global_row_start = global_row_start+CutSizeY1;
                global_col_start = global_col_start+CutSizeX1;
                global_row_end = global_row_end-CutSizeY2;
                global_col_end = global_col_end-CutSizeX2;
    
                layer = layer(1+CutSizeY1:end-CutSizeY2,1+CutSizeX1:end-CutSizeX2);
    
                layercube(global_row_start:global_row_end,global_col_start:global_col_end,is) = layer;
            end
        end
    
        %Place the vendsyssel Mask in a copy of the Global Grid
        VGrid = layercube(:,:,1)*0;
        VX = GlobalXS;
        VY = GlobalYS;
    
        [~,indXG,indXV] = intersect(VX,V_xs);
        [~,indYG,indYV] = intersect(VY,V_ys);
    
        VGrid(indYG,indXG) = Vmask(indYV,indXV);
        VGrid = double(VGrid > 0.1);
    
    
        %Use VGrid (the mask) to calculate a weight map for a weighted
        %summation of MHM and LTA
        VGrid2 = VGrid;
        VSZ = size(VGrid2);
        for i = 1:VSZ(1)
            for j = 1:VSZ(2)
    
                XMin = max(j-3,1);
                XMax = min(j+3,VSZ(2));
                YMin = max(i-3,1);
                YMax = min(i+3,VSZ(1));
    
                Minigrid = VGrid(YMin:YMax,XMin:XMax);
                VGrid2(i,j) = sum(Minigrid(:))/numel(Minigrid);
            end
        end
    
        WeightVend = VGrid2;
        WeightHimm = 1-VGrid2;
    
    
        %Load the vendsyssel Model
        VModel = import_vendsyssel(VGrid,VX,VY);
    
        % Lærke and Mette Jylland (Salling + Mors)
        DBname = 'LTA_MHM';
        mainauthor = 1; % Lærke
        points1 = get_points_from_database(DBname,NL,mainauthor);
    
        % Mette Jylland
        DBname = 'MHM';
        mainauthor = 2; % Mette
        points2 = get_points_from_database(DBname,NL,mainauthor);
    
        % Mette Fyn
        DBname = 'FYN';
        mainauthor = 2; % Mette
        points3 = get_points_from_database(DBname,NL,mainauthor);
    
        % Mette Marsk
        DBname = 'RASTE_MHM';
        mainauthor = 1; % Mette
        points4 = get_points_from_database(DBname,NL,mainauthor);
    
        %combine Mette and Lærke's point tables
        points = cell(1,NL);
        for i = 1:NL
            p1 = points1{i};
            p2 = points2{i};
            p3 = points3{i};
            p4 = points4{i};
    
            pz1 = cat(1,p1,p2);
            pz2 = cat(1,pz1,p3);
            pz3 = cat(1,pz2,p4);
    
            points{i} = pz3;
        end
        
        clear pz1
        clear pz2
        clear pz3
        
        %Read Vendsyssel Points
        points_v = get_vpoints();
    
        %Read LOOP2 points
        points_LOOP = get_LOOP2points();
    else % Anholt & Læsø
        points_DK8 = get_DK8_points();
        points = cell(1,NL);
    end

        

    %Gather snapped points in table
    for i = 1:NL
        if ~strcmp(reg,'AnholtLæsø') 
            p1 = points{i};
            SnapIDs = zeros(1,size(p1,1));
            for ii = 1:size(p1,1)
                SnapIDs(ii) = numel(p1.SnapID{ii})>1;
            end
    
            p1 = removevars(p1,"SnapID");
            p1.SnapID = logical(SnapIDs');    

            p2 = points_v{i};
            pz1 = cat(1,p1,p2);
    
            p3 = points_LOOP{i};
            pz2 = cat(1,pz1,p3);
            
            points{i} = pz2;

            filter_positive_point = pz2.PointType == 0;
            filter_snapped_point = pz2.SnapID;
            filter = and(filter_positive_point,filter_snapped_point);
        else
            pz2 = points_DK8{i};
            points{i} = pz2;
            filter = ones(size(pz2,1),1);
        end
        SnapList{i} = pz2(filter,:);

    end

    clear pz1
    clear pz2

    %Write snapped points as table to "tolkningspunkter" folder.
    save([reg,'_peatland_snaps'],"SnapList",'-v7.3')

    if ~strcmp(reg,'AnholtLæsø') 

        for i = 1:NL
    
    
            %Get xs and ys from terrain grid
            curxs = tinfo.xs;
            curys = tinfo.ys;
    
            xs_copy = curxs;
            ys_copy = curys;
    
            switch reg
                case 'Jylland'
                    %NEW LINES: Find area for resampling
                    [~,xcG,xcL] = intersect(GlobalXS+50,curxs);
                    [~,ycG,ycL] = intersect(GlobalYS+50,curys);
    
                    modcube_in = layercube(ycG,xcG,:);
                    Vcube_in = VModel(ycG,xcG,:);
    
                    %NEW LINES : RESAMPLE
    %                keyboard
    %                [modcube_out_L,x_out,y_out,correction_map] = convert2ascii_resample(modcube_in,curxs(xcL),curys(ycL),terrain(ycL,xcL),50);
    %                [modcube_out_V,x_out,y_out,correction_map] = convert2ascii_resample(Vcube_in,curxs(xcL),curys(ycL),terrain(ycL,xcL),50);
    
                    [modcube_out_L,x_out,y_out] = convert2ascii_resample(modcube_in,curxs(xcL),curys(ycL),terrain(ycL,xcL),50);
                    [modcube_out_V,x_out,y_out] = convert2ascii_resample(Vcube_in,curxs(xcL),curys(ycL),terrain(ycL,xcL),50);
    
    
                    %Find the intersection in xs and ys between the entire tile grid (Global) and the
                    %terrain surface
                    [~,xinds_G,xinds_L] = intersect(GlobalXS+50,curxs);
                    [~,yinds_G,yinds_L] = intersect(GlobalYS+50,curys);
    
                    curxs = curxs(xcL);
                    curys = curys(ycL);
    
                    %Cut the layer according to the terrains extent
                    lc2 = modcube_out_L(:,:,i+1);
                    vc2 = modcube_out_V(:,:,i+1);
    
                    WeightVend2 = WeightVend(yinds_G,xinds_G);
                    WeightHimm2 = WeightHimm(yinds_G,xinds_G);
    
                case 'Fyn'
    
                    %NEW LINES: Find area for resampling
                    [~,xcG,xcL] = intersect(GlobalXS,curxs);
                    [~,ycG,ycL] = intersect(GlobalYS,curys);
    
                    modcube_in = layercube(ycG,xcG,:);
    
                    %Fyn requires no resampling
                    modcube_out_L = modcube_in;
    
                    curxs = curxs(xcL);
                    curys = curys(ycL);
    
                    %Cut the layer according to the terrains extent
                    lc2 = modcube_out_L(:,:,i);
    
                    %WeightHimm is named after himmerland as it was designed to
                    %help separate areas in northern Jylland where the vendsyssel grid and the adjacent areas
                    %of himmerland should be. However, the same variable is
                    %used on Fyn but with all 1's.
    
                    WeightHimm2 = WeightHimm(ycG,xcG);
            end
    
            %If this is the first layer there is no previous layer. As such, the
            %terrain surface will form the upper boundary (ceiling).
            if i == 1
                ceiling_l = terrain(ycL,xcL);
                ceiling_v = terrain(ycL,xcL);
            else
                %Uncomment keyboard if debugging is needed after the first layer
                %         keyboard
            end
    
    
            %Outside of the modelling area we set the layer to be
            %equal to the layer above. (0 thickness)
            switch reg
                case 'Jylland'
                    lc2(isnan(lc2)) = ceiling_l(isnan(lc2));
                    vc2(isnan(vc2)) = ceiling_v(isnan(vc2));
                case 'Fyn'
                    lc2(isnan(lc2)) = ceiling_l(isnan(lc2));
            end
    
            switch reg
                case 'Jylland'
                    %Process Vendsyssel
                    vc2pre = vc2;
    
                    current_points = points_v{i};
                    [vc2R,vc2maskR] = remove_artifacts(vc2,ceiling_v,curxs,curys,current_points,0,i,'Vendsyssel');
    
                    vc2 = vc2R;
                    vc2mask = vc2maskR;
            end
    
            %Process the rest of Jylland or Fyn
            lc2pre = lc2;
            current_points = points{i};
            [lc2R,lc2maskR] = remove_artifacts(lc2,ceiling_l,curxs,curys,current_points,0,i,'Other');
    
            lc2 = lc2R;
            lc2mask = lc2maskR;
    
            switch reg
                case 'Jylland'
                    curmask = (lc2mask+vc2mask)>0;
                    curlay = WeightHimm2.*lc2+WeightVend2.*vc2;
                case 'Fyn'
                    curmask = lc2mask>0;
                    curlay = lc2;
            end
    
            % Prepare output
            Nrow_out = size(terrain,1);
            Ncol_out = size(terrain,2);
    
            xs_out = xs_copy;
            ys_out = ys_copy;
    
            outputgrid = zeros(Nrow_out,Ncol_out);
    
            mask_out = outputgrid;
            lay_out = outputgrid;
    
            mask_out(ycL,xcL) = double(curmask);
            lay_out(ycL,xcL) = curlay;
    
            write_grid_ascii([layerpath names{i} '_mask'],mask_out,xs_out,ys_out)
            write_grid_ascii([layerpath names{i}],lay_out,xs_out,ys_out)
    
            ceiling_l = lc2;
        end
    
        if doplot == 1
            for i = 1:size(L,2)
                pL{i} = L(i);
            end
    
            for i = 1:(size(N,2))/2
                pN{i} = N(2*(i-1)+1:2*i);
            end
            pN = fliplr(pN);
    
            figure()
            for i = 1:Nlayers
                subplot(1,5,i)
                imagesc(GlobalXS,GlobalYS,layercube(:,:,i))
                hold on
                plot([TileCoordMatrix(:,1),TileCoordMatrix(:,3)]',[TileCoordMatrix(:,2),TileCoordMatrix(:,2)]','-k')
                plot([TileCoordMatrix(:,1),TileCoordMatrix(:,1)]',[TileCoordMatrix(:,2),TileCoordMatrix(:,4)]','-k')
                plot_dk();
                clim([-20 120])
                set(gca,'ydir','normal')
                set(gca,'Xtick',unique(TileCoordMatrix(:,1))+100*(TileSize/2),'Ytick',unique(TileCoordMatrix(:,4))+100*(TileSize/2),'Yticklabel',pN,'Xticklabel',pL)
                title(StandardNames{i},'Interpreter','none')
            end
        end
    end
end

