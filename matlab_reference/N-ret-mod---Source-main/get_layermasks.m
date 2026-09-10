function [S,probmask] = get_layermasks(layertop,layerbottom,plotresult,plotprocess,save_file,reg,top,peat_log,NPL,itnum,layname,use_calibration_model)
%% DESCRIPTION

%The inputs layertop and layerbottom are ascii files with grids containing the top of a layer and the
%bottom of that same layer. This code produces masks in logical linear indexes for
%gridcells that are offshore (value 1) vs offshore (value 0) and cells
%where the layer is defined (value 1) vs not defined (value 0). A combined
%mask is also produced where the layer is both onshore and defined (value 1) or
%not (value 0). Each of the three masks have two versions - one with a 15
%grid cell buffer zone and one without. The buffer zone is intended for the
%machine learning algorithm's sliding window estimation of sill and range.

% plotresult: enter 1 to produce a map of the final masks, enter 0 to
% disable this feature.

% plotprocess: enter 1 to produce a live animation of the algorithm that
% produces the onshore/offshore mask layer, enter 0 to disable this
% feature. NOTE THAT THIS FEATURE SLOWS DOWN EXECUTION OF THE CODE.

%save_file: enter 1 to save results to a file in the local path

%% THE FUNCTION
if use_calibration_model == 1
    direct = [pwd,'\Flader\',reg,'\Corrected_for_calibration\masks\'];
    Npath = length([pwd,'\Flader\',reg,'\Corrected_for_calibration\']); % Path length
else
    direct = [pwd,'\Flader\',reg,'\masks\'];
    Npath = length([pwd,'\Flader\',reg,'\']); % Path length
end    

disp('Loading Input Layers')
format{1} = layertop(end-3:end);
format{2} = layerbottom(end-3:end);

%Check which file format
bool_asc1 = logical(prod(format{1} == '.asc'));
bool_asc2 = logical(prod(format{2} == '.asc'));

bool_grd1 = logical(prod(format{1} == '.grd'));
bool_grd2 = logical(prod(format{2} == '.grd'));

%Read files
tic
v = isnan(top);
v2 = numel(v) == 1;


switch v2
    case 1
        if bool_asc1

            [x_t,y_t,Top]=dtm_read_plot(layertop);

        elseif bool_grd1

            [Top, infoT]=ReadSurfer7(layertop);

            x_t = double(infoT.UTM_X);
            y_t = double(infoT.UTM_Y);

        else
            disp('Invalid Input File Format')
        end
    case 0
        Top = top;
end

if bool_asc2

    [x_b,y_b,Bottom]=dtm_read_plot(layerbottom);

    switch reg
        case 'Jylland'

            switch peat_log
                case 1

                %keyboard
                lnames = get_layer_names(reg,NPL);
                comb_ma = Top.*0;
                
                % Postglaciale lag
                if contains(layerbottom,'Post_Glacial')
                    startlay = 2;
                elseif contains(layerbottom,'Sen_Glacial')
                    startlay = 5  ;                 
                end

                endlay = startlay-1;
                testlay = 0;
                while  testlay == 0
                    endlay = endlay + 1;
                    testlay = strcmp(layerbottom(end-19:end-4),lnames{endlay}(end-15:end));
                end
                
                disp(['Combining peatland masks: starting layer = ',num2str(startlay) ' ending layer = ',num2str(endlay)])
                for jj = startlay:endlay
                    name = [direct,lnames{jj}];
                    [xvect,yvect,OnMask]=dtm_read_plot([name,'_mask','.asc']);
                    comb_ma = comb_ma+OnMask;
                end
                OnMask = (comb_ma)>0;
%                 figure(10); clf(10)
%                 imagesc(OnMask)
%                 title(lnames{startlay})
%                 caxis([0,1])
%                 colorbar   
%                 keyboard

                case 0
                [xvect,yvect,OnMask]=dtm_read_plot([direct,layerbottom(Npath+1:end-4),'_mask','.asc']);

            end

            probmask = OnMask.*0;


        case 'Fyn'


            switch peat_log
                case 1
                    lnames = get_layer_names(reg,NPL);
                    comb_ma = Top.*0;

                    for jj = 2:(NPL+1)
                        name = [direct,lnames{jj}];
                        [xvect,yvect,OnMask]=dtm_read_plot([name,'_mask','.asc']);
                        comb_ma = comb_ma+OnMask;
                    end
                    OnMask = (comb_ma)>0;
                case 0
                    if strcmp(layname,'ks3b')
                        %keyboard
                        name = 'ks3b_mask';
                        [xvect,yvect,OnMask]=dtm_read_plot([direct,name,'.asc']);
                        probmask = OnMask.*0;
                    else 
                        %keyboard
                        lnames = get_layer_names(reg,NPL);
                        comb_ma = Top.*0;
    
                        for jj = 2:(NPL+1)
                            name = [direct,lnames{jj}];
                            [xvect,yvect,OnMask]=dtm_read_plot([name,'_mask','.asc']);
                            comb_ma = comb_ma+OnMask;
                        end
                        OnMask = (comb_ma)*0+1;
                    end
            end

            probmask = Top.*0;

        case 'Sjælland'

            switch layname

                case 'ks1t'
                    name = 'hub_kl1__mask';
                    [xvect,yvect,OnMask]=dtm_read_plot([direct,name,'.asc']);
                    OnMask = 1-OnMask;
                    probmask = OnMask.*0;

                case 'ks1b'
                    name = 'ub_ks1__mask';
                    [xvect,yvect,OnMask]=dtm_read_plot([direct,name,'.asc']);
                    probmask = OnMask.*0;

                case 'ks2t'
                    name = 'hub_kl2__mask';
                    [xvect,yvect,OnMask]=dtm_read_plot([direct,name,'.asc']);
                    OnMask = 1-OnMask;
                    probmask = OnMask.*0;

                case 'ks2b'
                    name = 'ub_ks2__mask';
                    [xvect,yvect,OnMask]=dtm_read_plot([direct,name,'.asc']);
                    probmask = OnMask.*0;

                case 'ks3t'
                    name = 'hub_kl3__mask';
                    [xvect,yvect,OnMask]=dtm_read_plot([direct,name,'.asc']);
                    OnMask = 1-OnMask;
                    probmask = OnMask.*0;

                case 'ks3b'
                    name = 'ub_ks3__mask';
                    [xvect,yvect,OnMask]=dtm_read_plot([direct,name,'.asc']);
                    probmask = OnMask.*0;

                case 'ks4t'
                    name = 'hub_kl3__mask';
                    [xvect,yvect,OnMask]=dtm_read_plot([direct,name,'.asc']);
                    OnMask = 1-OnMask;
                    probmask = OnMask.*0;

                case 'ks4b'
                    name = 'ub_ks4__mask';
                    [xvect,yvect,OnMask]=dtm_read_plot([direct,name,'.asc']);
                    probmask = OnMask.*0;

                case 'preq'
                    name = 'hlp_preq__mask';
                    [xvect,yvect,OnMask]=dtm_read_plot([direct,name,'.asc']);
                    OnMask = 1-OnMask;
                    probmask = OnMask.*0;

                case 'pl1b'
                    name = 'hlp_pl1b__mask';
                    [xvect,yvect,OnMask]=dtm_read_plot([direct,name,'.asc']);
                    probmask = OnMask;
                    OnMask = (OnMask.*0)+1;

                case 'gk1b'
                    name = 'hlp_gk1b__mask';
                    [xvect,yvect,OnMask]=dtm_read_plot([direct,name,'.asc']);
                    probmask = OnMask;
                    OnMask = (OnMask.*0)+1;

                case 'dk1b'
                    name = 'hlp_dk1b__mask';
                    [xvect,yvect,OnMask]=dtm_read_plot([direct,name,'.asc']);
                    probmask = OnMask;
                    OnMask = (OnMask.*0)+1;

            end

        case 'AnholtLæsø'
            [xvect,yvect,OnMask]=dtm_read_plot([direct,layerbottom(Npath+1:end-4),'_mask','.asc']);
            probmask = OnMask;


        case 'Fyn-MST'
            [xvect,yvect,OnMask]=dtm_read_plot([direct,layerbottom(Npath+1:end-4),'_mask','.asc']);
            probmask = OnMask;

%            switch layname
%                 case '02_Top_KS1_Adjusted0'
%                     name = '02_Top_KS1_Adjusted0_mask';
%                     [xvect,yvect,OnMask]=dtm_read_plot([direct,name,'.asc']);
%                     OnMask = 1-OnMask;
%                     probmask = OnMask.*0;
% 
%                 case '03_Bund_KS1_Adjusted0'
%                     name = '03_Bund_KS1_Adjusted0_mask';
%                     [xvect,yvect,OnMask]=dtm_read_plot([direct,name,'.asc']);
%                     probmask = OnMask.*0;
% 
%                 case '04_Top_KS2_Adjusted0'
%                     name = '04_Top_KS2_Adjusted0_mask';
%                     [xvect,yvect,OnMask]=dtm_read_plot([direct,name,'.asc']);
%                     OnMask = 1-OnMask;
%                     probmask = OnMask.*0;
% 
%                 case '05_Bund_KS2_Adjusted0'
%                     name = '05_Bund_KS2_Adjusted0_mask';
%                     [xvect,yvect,OnMask]=dtm_read_plot([direct,name,'.asc']);
%                     OnMask = 1-OnMask;
%                     probmask = OnMask.*0;
% 
%                 case '06_Top_KS3_Adjusted0'
%                     name = '06_Top_KS3_Adjusted0_mask';
%                     [xvect,yvect,OnMask]=dtm_read_plot([direct,name,'.asc']);
%                     OnMask = 1-OnMask;
%                     probmask = OnMask.*0;
% 
%                 case '07_Bund_KS3_Adjusted0'
%                     name = '07_Bund_KS3_Adjusted0_mask';
%                     [xvect,yvect,OnMask]=dtm_read_plot([direct,name,'.asc']);
%                     OnMask = 1-OnMask;
%                     probmask = OnMask.*0;
% 
%                 case '08_Top_PreQ_Adjusted0'
%                     name = '08_Top_PreQ_Adjusted0_mask';
%                     [xvect,yvect,OnMask]=dtm_read_plot([direct,name,'.asc']);
%                     OnMask = 1-OnMask;
%                     probmask = OnMask.*0;
% 
%                 case '09_Top_Kalk_Adjusted0'
%                     name = '09_Top_Kalk_Adjusted0_mask';
%                     [xvect,yvect,OnMask]=dtm_read_plot([direct,name,'.asc']);
%                     probmask = OnMask.*0;
% 
%             end

    end

end

%keyboard
%%%%%%%%%%%%%% Insert dk9 in specific layers %%%%%%%%%%%%%%%%%%
% Det betyder at du skal have splejset masken for dk9 sammen med dine masker for lagene:
% '0300_Kvartaer_ler_Bund'
% '0400_Kvartaer_sand_Bund'
% '1100_Kvartaer_ler_Bund'
% '1200_Kvartaer_sand_Bund'
% '1300_Kvartaer_ler_Bund'
% '1400_Kvartaer_sand_Bund'
% '1500_Kvartaer_ler_Bund'
% '2100_Kvartaer_sand_Bund'
% '2400_Kvartaer_ler_Bund'
insert_names = [{'0300_Kvartaer_ler_Bund'};...
{'0400_Kvartaer_sand_Bund'};...
{'1100_Kvartaer_ler_Bund'};...
{'1200_Kvartaer_sand_Bund'};...
{'1300_Kvartaer_ler_Bund'};...
{'1400_Kvartaer_sand_Bund'};...
{'1500_Kvartaer_ler_Bund'};...
{'2100_Kvartaer_sand_Bund'};...
{'2400_Kvartaer_ler_Bund'}];

if strcmp(reg,'Jylland')
    for iins = 1:length(insert_names)
        cur_name_str = (insert_names{iins});
        if strcmp(layname,cur_name_str)
           [probmask_temp] =  insert_dk9_mask(xvect,yvect,OnMask);
            OnMask = probmask_temp;
        end
    end
end
clear probmask_temp
clear insert_names

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


switch v2
    case 0
        x_t = x_b;
        y_t = y_b;
end

toc

%Check layers have the same size
topsize = size(Top);
bottomsize = size(Bottom);

xs_same = topsize(2)==bottomsize(2);
ys_same = topsize(1)==bottomsize(1);

size_same = logical(xs_same*ys_same);

%Check layers overlap completely
if size_same == 1
    overlap = logical(prod(x_t == x_b)*prod(y_t == y_b));
    both_same = logical(size_same*overlap);
else
    both_same = 0;
end


if not(both_same)
    [Top, Bottom,x_t,y_t] = match_grid_sizes(Top,Bottom,x_t,y_t,x_b,y_b);

end

% DO LAYER CORRECTIONS TO ACCOMODATE PEAT LANDS

minthick = min(Top(:)-Bottom(:));
check = minthick < 0;

if check
    disp('Negative Thickness found: Performing Layer Corrections')
    correct_function = @(T,B) (abs(T-B)+(T-B))/2;
    Bottom = Top-correct_function(Top,Bottom);
end

next_top = Bottom;
%


thick = Top-Bottom;

tic
%Choose what shapefile to get coastline from
switch reg
    case 'Jylland'
        type = 1;
    case 'Fyn'
        type = 2;
    case 'Fyn-MST'
        type = 2;
    case 'Sjælland'
        type = 3;
    case 'AnholtLæsø'
        type = 1;
end

disp('Looking for Existing National Onshore/Offshore Mask')
%Get shapefile points in UTMX and UTMY
[x,y] = plot_dk(type);

padding = 50;

minx = xvect(1)-(100*padding);
miny = yvect(1)-(100*padding);
maxx = xvect(end)+(100*padding);
maxy = yvect(end)+(100*padding);

xsdk = minx:100:maxx;
ysdk = miny:100:maxy;

dksize = [size(ysdk,2) size(xsdk,2)];
new_grid = zeros(dksize);

% This loop draws the coast line onto the grid by taking each line
% segment, parameterizing it into a number of points and finding out
% which grid cells contain at least one such point. The coast line
% should be without holes, so a large number of points are
% used for each line segment.

for i = 1:size(x,2)-1

    %Number of Parametrized points along line segment
    NP = 200;

    %Parameterization variable t
    t = linspace(0,1,NP);

    %Define end-points for the current line segment
    end_points = [x(i),y(i);x(i+1),y(i+1)];

    %Filters to determine if the line is inside the grid area
    filt1 = minx < x(i);
    filt2 = minx < x(i+1);
    filt3 = miny < y(i);
    filt4 = miny < y(i+1);
    filt5 = maxx > x(i);
    filt6 = maxx > x(i+1);
    filt7 = maxy > y(i);
    filt8 = maxy > y(i+1);

    %finalfilt 0 means that the line is outside or partially outside the
    %grid. finalfilt 1 means that it is inside the grid.
    finalfilt = logical(filt1*filt2*filt3*filt4*filt5*filt6*filt7*filt8);

    %partfilt 1 means that it is partially outside the grid
    partfilt = logical(abs((filt1*filt3*filt5*filt7)-(filt2+filt4+filt6+filt8)));
    
   

    %Skip NaN values. These values play an important role in the shapefile
    %and should not be removed, only skipped.
    if isnan(sum(end_points(:)))
        %         disp('Skipping NaN values')

    elseif or(finalfilt,partfilt)
        %Calculate coordinates for points on the line segment
        line_points = [end_points(1,1)+t*(end_points(2,1)-end_points(1,1));end_points(1,2)+t*(end_points(2,2)-end_points(1,2))];
        
        %Determine which grid cells points on the line fall within
        NX = floor((line_points(1,:)-minx+50)/100)+1;
        NY = floor((line_points(2,:)-miny+50)/100)+1;
        
        if partfilt
            ylog = and(NY>0,NY<=dksize(1));
            xlog = and(NX>0,NX<=dksize(2));

            NX = NX(and(xlog,ylog));
            NY = NY(and(xlog,ylog));
        end

        %sumv is the index vector for the grid cells containing
        %the line segment
        sumv = sub2ind(dksize,NY,NX);

        %If two points or more fall within the same cell we only count
        %the cell once. Any more is a redundant calculation.
        un = unique(sumv);

        %Fill all grid cells that the line segment points fall into with the value 1
        for j = 1:size(un,2)
            unq = find(sumv'==un(j));

            NewCellX = NX(unq(1));
            NewCellY = NY(unq(1));

            new_grid(NewCellY:NewCellY,NewCellX:NewCellX) = 1;

        end
    end
end
toc

prop_grid = zeros(size(new_grid));

points = [1,1];
new_ps = 1;
ps_old = 1;
it = 0;

if plotprocess == 1
    figure()
    imagesc((new_grid))
    set(gca,'ydir','normal')
    hold on
end


disp('Calculating Onshore/Offshore Mask')
tic
while new_ps > 0

    it = it+1;
    lindx = sub2ind(size(new_grid),points(:,1),points(:,2));
    prop_grid(lindx) = 1;

    if plotprocess == 1
        plot(points(:,2),points(:,1),'.r','MarkerSize',3)
    end

    %Propose New Cells
    prop_points = [points(:,1)+1 points(:,2);
        points(:,1)-1 points(:,2);
        points(:,1) points(:,2)+1;
        points(:,1) points(:,2)-1]; %Proposes neighboring points

    %remove points outside grid
    for s = fliplr(1:size(prop_points,1))

        sf1 = prop_points(s,1) > 0;
        sf2 = prop_points(s,2) > 0;
        sf3 = prop_points(s,1) < size(prop_grid,1)+1;
        sf4 = prop_points(s,2) < size(prop_grid,2)+1;

        combf = sf1*sf2*sf3*sf4;

        if combf == 0
            prop_points(s,:) = [];
        end
    end

    if size(prop_points,1) > 0
        %Translate Proposed Points to Linear Indexes
        ps = sub2ind(size(new_grid),prop_points(:,1),prop_points(:,2));
        ps = unique(ps);

        %Remove points that already exist
        if it > 1
            lis = ismember(ps,ps_old_old);
            ps(lis) = [];
        end

        [a,b] = ind2sub(size(new_grid),ps);
        prop_points = [a,b];

        prop_coast = logical(new_grid(ps)'); %Checks if any of the proposed points reach coast line
        no_coast = prop_coast == 0;
        % new_ps = sum(no_coast)

        pgr = logical(prop_grid(ps)'); %Checks if any of the proposed points are already accounted for
        no_account = pgr == 0;

        %accept_points
        point_update = zeros(size(no_coast,2),2);
        for i = fliplr(1:size(no_coast,2))
            if no_coast(i)
                if no_account(i)
                    point_update(i,:) = prop_points(i,:);
                else
                    point_update(i,:) = [];
                end
            else
                point_update(i,:) = [];
            end
        end

    else
        new_ps = 0;
    end

    ps_old_old = ps_old;
    ps_old = ps;

    points = point_update;
    if plotprocess == 1
        pause(0.01)
    end
end

onshoreLayer = prop_grid == 0;

% onshoreLayer = onshoreLayer((padding+1):end-padding,(padding+1):end-padding);
% xsdk = xsdk((padding+1):end-padding);
% ysdk = ysdk((padding+1):end-padding);

%Since onshoreLayer is for all of Denmark we now cut out a piece for the
%current purpose. For this we use the coordinates and make a cutout.

[~, x_ind] = intersect(xsdk,x_t);
[~, y_ind] = intersect(ysdk,y_t);

[~, x_ind2] = intersect(x_t,xsdk);
[~, y_ind2] = intersect(y_t,ysdk);

onshoreLayer = onshoreLayer(y_ind,x_ind);

Bottom = Bottom(y_ind2,x_ind2);
Top = Top(y_ind2,x_ind2);
thick = thick(y_ind2,x_ind2);

x_t = xsdk(x_ind);
y_t = ysdk(y_ind);

toc

if plotresult == 1
    figure()
    imagesc(x_t,y_t,onshoreLayer)
    set(gca,'ydir','normal')
    hold on
    plot_dk();
end

disp('Calculating Layer Mask')
MaskLayerDefined = OnMask;

disp('Adding 100m buffer to Layer Mask')

siz_x = size(MaskLayerDefined,2);
siz_y = size(MaskLayerDefined,1);
smallmask = MaskLayerDefined*0;

for i = 1:siz_y
    for j = 1:siz_x

        val = MaskLayerDefined(i,j);

        if val
            xs = [max(1, j-2) min(siz_x,j+2)];
            ys = [max(1, i-2) min(siz_y,i+2)];

            locg = MaskLayerDefined(ys(1):ys(2),xs(1):xs(2));
            if sum(locg(:)) < numel(locg)
                smallmask(ys(1):ys(2),xs(1):xs(2)) = 1;
            end
        end
    end
end


%Making buffer zones
BufferedMaskLayerDefined = MaskLayerDefined;
BufferedOnshoreLayer = onshoreLayer;

nolayercount = find(BufferedMaskLayerDefined==0);
nolayercount2 = find(BufferedOnshoreLayer==0);

szs = zeros(2,2);

szs(1,1:2) = size(BufferedMaskLayerDefined);
szs(2,1:2) = size(BufferedOnshoreLayer);

[miniy, minix] = ind2sub(szs(1,:),nolayercount);
[miniy2,minix2] = ind2sub(szs(2,:),nolayercount2);

Bs = 16;

%This loop adds a buffer to mask 1 !
BufferedMaskLayerDefined2 = BufferedMaskLayerDefined;
for i = 1:size(nolayercount,1)

    xv = minix(i);
    yv = miniy(i);

    xmin = max(1, xv-Bs);
    xmax = min(szs(2,2), xv+Bs);

    ymin = max(1, yv-Bs);
    ymax = min(szs(2,1), yv+Bs);

    minigrid = BufferedMaskLayerDefined2(ymin:ymax,xmin:xmax);

    check = sum(minigrid(:));

    if check > 0
        BufferedMaskLayerDefined(yv,xv) = 1;
    end
end

%This loop adds a buffer to mask 2 !
BufferedOnshoreLayer2 = BufferedOnshoreLayer;
for i = 1:size(nolayercount2,1)

    xv = minix2(i);
    yv = miniy2(i);

    xmin = max(1, xv-Bs);
    xmax = min(szs(2,2), xv+Bs);

    ymin = max(1, yv-Bs);
    ymax = min(szs(2,1), yv+Bs);

    minigrid = BufferedOnshoreLayer2(ymin:ymax,xmin:xmax);
    check = sum(minigrid(:));

    if check > 0
        BufferedOnshoreLayer(yv,xv) = 1;
    end
end

MaskComp = (smallmask+MaskLayerDefined)>0;

if itnum > NPL
    MaskLayerDefined = (smallmask+MaskLayerDefined)>0;
end

%Gather Ouput
S.LayerTop = Top;
S.LayerBottom = Bottom;
S.LayerThickness = thick;
S.Grid_Size_Rows = size(Top,1);
S.Grid_Size_Columns = size(Top,2);
S.BufferZoneWidth = Bs;

S.Grid_UTM_X = x_t';
S.Grid_UTM_Y = y_t';
S.BufferedGrid_UTM_X = x_t';
S.BufferedGrid_UTM_Y = y_t';
S.Grid_UTM_LowerLeftCorner = [min(x_t) min(y_t)];

S.indexMask_Onshore = logical(onshoreLayer(:));
S.indexMask_LayerExists = logical(MaskLayerDefined(:));
S.indexMask_Combined = logical(onshoreLayer(:).*MaskLayerDefined(:));

S.indexBufferedMask_Onshore = logical(BufferedOnshoreLayer(:));
S.indexBufferedMask_LayerExists = logical(BufferedMaskLayerDefined(:));
S.indexBufferedMask_Combined = logical(BufferedOnshoreLayer(:).*BufferedMaskLayerDefined(:));
S.compositeMask = MaskComp;
S.NextTop = next_top;

if save_file == 1
    n1 = strsplit(layerbottom,'.');
    n2 = strsplit(layertop,'.');
    %Save output to file
    save(strcat('outputmasks','_',n1{1},'_',n2{1}),'S')
end


end