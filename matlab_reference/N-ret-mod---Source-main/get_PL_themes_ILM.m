function [theme,Data_snap,Data_other] = get_PL_themes_ILM(UTM_X,UTM_Y,terrain,complexity,NL,reg,S)
%S is a struct. It is supposed to be "LayerStructsP" (a.k.a. cur_struct) from Main.

%LLc and URc
LLx = min(UTM_X);
LLy = min(UTM_Y);

URx = max(UTM_X);
URy = max(UTM_Y);

nx = max(size(UTM_X));
ny = max(size(UTM_Y));

Grid = zeros(ny,nx,NL);
Grid2 = Grid;

snap_xs = [];
snap_ys = [];
snap_zs = [];

%Read all relevant data
path = 'N:\PROJEKTER\N-retentionskortlægning 2022-24\GS3D_Jylland\Jupiterdatabaser\';

path_ext{1} = 'SQL_Midtjylland\';
path_ext{2} = 'SQL_nordjylland\';
path_ext{3} = 'SQL_Sydjylland\';

name_prefix{1} = 'Midtjyl_';
name_prefix{2} = 'Nordjyl_';
name_prefix{3} = 'Sydjyl_';

layvector = [1 1 2 3 4 5];
%Layer 1
filename{1} = 'Gytje.csv';
filename{2} = 'toerv.csv';

%Layer 2
filename{3} = 'postglac_saltv_sand.csv';

%Layer 3
filename{4} = 'postglac_saltv_ler.csv';

%Layer 4
filename{5} = 'senglac_saltv_sand.csv';

%Layer 5
filename{6} = 'senglac_saltv_ler.csv';

%Vendsyssel wells
vend = [pwd,'\Tolkningspunkter\','vendsyssel_jup_boringer_til_usikkerhed.mat'];
V = load(vend);

jupiter_array = cell(1,5);

%Rearrange Vendsyssel otherwells table to match the tables provided by
%Mette for the rest of the peatlands.
VT0 = V.Jup_bor_data_q(:,["XUTM","YUTM","ELEVATION","MAX_DEPTH"]); %rettet
VT = renamevars(VT0,["MAX_DEPTH"],["BOTTOM"]);
TOP = VT.BOTTOM*0;
VT1 = addvars(VT,TOP,'Before','BOTTOM');             % rettet

%Loop over region
for i = 1:3

    cur_path = [path, path_ext{i}, name_prefix{i}];

    %Loop over names to assign them to the correct peatlands.
    for j = 1:6

        %This if statement is simply because tørv is spelled toerv in the
        %names list.
        if i == 3 && j == 2
            cur_file = [cur_path,'Tørv.csv'];
        else
            cur_file = [cur_path, filename{j}];
        end
        jupiter_array_ind = layvector(j);

        curT = jupiter_array{jupiter_array_ind};

        T = readtable(cur_file);    
        T2 = T(:,["XUTM","YUTM","ELEVATION","TOP","BOTTOM"]);

        %Append T to curT
        Tnew = [curT;T2];

        jupiter_array{jupiter_array_ind} = Tnew;

    end

end


% % points1 = get_points_from_database('LaerkePoints',NL,1);
% % points2 = get_points_from_database('MettePoints',NL,2);
% % points3 = get_points_from_database('MetteFyn',NL,2);

comp2range = 0.5*[100 500 400 250 100];

%combine Mette and Lærke's point tables
points = cell(1,NL);
coords_sna = [];
depths_sna = [];

in = load([reg,'_peatland_snaps']);


for i = 1:NL
    if S{i}.zeromask == 0;
    %Prepare snapped wells
    curTable = in.SnapList{i};
    points{i} = curTable;


    coords_sna = [coords_sna;curTable.X,curTable.Y];
    depths_sna = [depths_sna;curTable.Z];

    %Prepare the rest of the jupiter/other wells for Vendsyssel
    curTableV = jupiter_array{i};
    
    curPeatland = S{i}.LayerBottom;
    curPeatlandDepth = terrain-curPeatland;

    curDepths = VT1.BOTTOM;

    curXS = VT1.XUTM;
    curYS = VT1.YUTM;

    %New filter: xs and ys between LLcorner and URcorner
    filter1 = curXS > (UTM_X(1)-50);
    filter2 = curXS < (UTM_X(end)+50);
    filter3 = curYS > (UTM_Y(1)-50);
    filter4 = curYS < (UTM_Y(end)+50);
    
    filter = logical(filter1.*filter2.*filter3.*filter4);

    xlist = round((curXS-LLx)/100)+1;
    ylist = round((curYS-LLy)/100)+1;
    
    curPeatlandDepths = curPeatlandDepth(sub2ind([ny,nx],ylist(filter),xlist(filter)));

    %If a well is deeper than the layer boundary then I use it (with a
    %margin of error of about 1m)
    Tolerance = 1;
    depth_logical = (curDepths(filter)+Tolerance)>=curPeatlandDepths; % rettet
    Tnew = [curTableV;VT1(depth_logical,:)];
    jupiter_array{i} = Tnew;
    end
end




%Uniques are determined for the snapped wells and the combined list.
%First, determine the number of snapped uniques
[snapped_uniques,snapos] = unique(coords_sna,'rows','stable');
N_un = size(snapped_uniques,1);

all_other_xs = [];
all_other_ys = [];
all_other_elevations = [];
all_other_zs = [];

for i = 1:5
    cur_T = jupiter_array{i};
    xs = cur_T.XUTM;
    ys = cur_T.YUTM;
    zs = cur_T.BOTTOM;
    ele = cur_T.ELEVATION;
    top = cur_T.TOP;

    positions = [coords_sna;xs,ys];

    [~,all_positions] = unique(positions,'rows','stable');
    indices = all_positions(N_un+1:end)-size(coords_sna,1);

    new_xs = xs(indices);
    new_ys = ys(indices);
    new_zs = zs(indices);
    new_ele = ele(indices);
    new_top = top(indices);

    filter1 = new_zs > 2;
    filter2 = new_xs > (UTM_X(1)-50);
    filter3 = new_xs < (UTM_X(end)+50);
    filter4 = new_ys > (UTM_Y(1)-50);
    filter5 = new_ys < (UTM_Y(end)+50);
    
    filter = logical(filter1.*filter2.*filter3.*filter4.*filter5);

    new_T.XUTM = new_xs(filter);
    new_T.YUTM = new_ys(filter);
    new_T.BOTTOM = new_zs(filter);
    new_T.ELEVATION = new_ele(filter);
    new_T.TOP = new_top(filter);
    
    jupiter_array{i} = new_T;

    all_other_xs = [all_other_xs;new_T.XUTM];
    all_other_ys = [all_other_ys;new_T.YUTM];
    all_other_elevations = [all_other_elevations;new_T.ELEVATION];
    all_other_zs = [all_other_zs;new_T.BOTTOM];

end

Data_other.xs = all_other_xs;
Data_other.ys = all_other_ys;
Data_other.zs = all_other_zs;

Data_other.elevations = all_other_elevations;

%next line updated 29-06-2023
var_fun = @(b,z,K) K^2+(b*z).^2;

GS = 0.5*0.02;  %Gradients for depth-to-uncertainty estimate given snapped well (eq. to category 1)
GSj = 0.5*0.09; %Not snapped well (eq. to category 3)

%Constants added 29-06-2023
KS = 0.5;
KSj = 1;

% 5) Jeg skal have en funktion som beskriver hvordan certainty falder med
% afstanden til boringen

cert_fun = @(dist,range,sill) sill*(exp(-(3*power(dist,2))/(power(range,2))));

for i = 1:NL
    if S{i}.zeromask == 0;
        wells = points{i};
    
        point_types = wells.PointType;
        wellxs = wells.X;
        wellys = wells.Y;
        wellzs = wells.Z;
    
        wellsnap = wells.SnapID;
    
        NPoints = numel(wellxs);
    
        %Determine if snap ID is empty or filled
        snap_well = zeros(1,NPoints);
        for j = 1:NPoints
            curid = wellsnap(j);               %%curid = wellsnap{j};
            snap_well(j) = curid==1;     %%snap_well(j) = numel(curid)>1;        
        end
    
        snap_well = logical(snap_well);
    
        snap_well_xs = wellxs(snap_well);
        snap_well_ys = wellys(snap_well);
        snap_well_zs = wellzs(snap_well);
    
        filt_snap = (snap_well_xs>UTM_X(1)) & (snap_well_ys>UTM_Y(1)) & (snap_well_xs<UTM_X(end)) & (snap_well_ys<UTM_Y(end));
    
        nsnap = sum(filt_snap);
    
        wells = zeros([nsnap 5]);
    
        wells(:,1) = snap_well_xs(filt_snap);
    
        wells(:,2) = snap_well_ys(filt_snap);
    
        wells(:,3) = snap_well_zs(filt_snap);
    
        for k = 1:size(wells,1)
            x = wells(k,1);
            y = wells(k,2);
    
            indx = round((x-min(UTM_X))/100)+1;
            indy = round((y-min(UTM_Y))/100)+1;
    
            terrain_kote = terrain(indy(end),indx(end));
            cur_complexity = complexity(indy(end),indx(end));
            wells(k,4) = terrain_kote;
            wells(k,5) = cur_complexity;
    
        end
    
        jwells(:,1) = jupiter_array{i}.XUTM;
        jwells(:,2) = jupiter_array{i}.YUTM;
        jwells(:,3) = jupiter_array{i}.BOTTOM;
        jwells(:,4) = jupiter_array{i}.ELEVATION;
    
        for k = 1:size(jwells,1)
    
            x = jwells(k,1);
            y = jwells(k,2);
    
    
            indx = round((x-min(UTM_X))/100)+1;
            indy = round((y-min(UTM_Y))/100)+1;
    
            cur_complexity = complexity(indy(end),indx(end));
    
            jwells(k,5) = cur_complexity;
    
        end
    
        %%%%%%%%%%%%%%% EDIT
        %%
        %6) Jeg skal besøge hver celle og lægge bidrag for de nærmeste boringer
        SearchRadius = 6;
    
        %%
        for k = 1:ny
            Ymin = max([1 k-SearchRadius]);
            Ymax = min([ny k+SearchRadius]);
    
            ymin = UTM_Y(Ymin);
            ymax = UTM_Y(Ymax);
    
            yv = UTM_Y(k);
    
            Yfilt1 = wells(:,2) < ymax;
            Yfilt2 = wells(:,2) > ymin;
    
            for j = 1:nx
    
                Xmin = max([1 j-SearchRadius]);
                Xmax = min([nx j+SearchRadius]);
    
                xmin = UTM_X(Xmin);
                xmax = UTM_X(Xmax);
    
                xv = UTM_X(j);
    
                Xfilt1 = wells(:,1) < xmax;
                Xfilt2 = wells(:,1) > xmin;
    
                filt = logical(Xfilt1.*Xfilt2.*Yfilt1.*Yfilt2);
                localwells = wells(filt,:);
    
                nwells = size(localwells,1);
    
                if nwells > 0
                    
                    %use max 3 nearest wells
                    nloop = min(nwells,3);
                    
                    %Create array for sorting.
                    sort_array = zeros(nwells,1);
    
                    for s = 1:nwells
                        curwell = localwells(s,:);
    
                        curx = curwell(1);
                        cury = curwell(2);
    
                        curdist = sqrt(power(xv-curx,2)+power(yv-cury,2));
    
                        sort_array(s) = curdist;
                    end
    
                    %Now sort the array by distance and get the indices
                    [~,near_well_indices] = sort(sort_array);
    
                    %Choose the nloop nearest
                    near_well_indices = near_well_indices(1:nloop);
    
                    for a = near_well_indices
    
                        curwell = localwells(a,:);
    
                        curx = curwell(:,1);                   %curx = curwell(1);
                        cury = curwell(:,2);                     %cury = curwell(2);
                        curd = max(curwell(:,4)-curwell(:,3),0);  %curd = max(curwell(4)-curwell(3),0);
                        curc = curwell(:,5);                     %curc = curwell(5); 
    
                        distance = sqrt(power(xv-curx,2)+power(yv-cury,2));
    
                        B = GS;
    
                        if isnan(curc)
                            curc =  0;
                        end
                        
                        for a_i = 1:numel(a)            % ny for løkke til at håndtere hvert element hver for sig
    
                            Range = comp2range(curc(a_i)+1);   %Range = comp2range(curc+1);
                            var0 = var_fun(B,curd(a_i),KS);    %var0 = var_fun(B,curd,KS);
                            certainty =  cert_fun(distance(a_i),Range,1/var0);    %certainty =  cert_fun(distance,Range,1/var0);
        
                            Grid(k,j,i) = Grid(k,j,i)+certainty;
                        end                     % ny
                    end
                end
            end
        end
    
        parfor k = 1:ny
            Ymin = max([1 k-SearchRadius]);
            Ymax = min([ny k+SearchRadius]);
    
            ymin = UTM_Y(Ymin);
            ymax = UTM_Y(Ymax);
    
            yv = UTM_Y(k);
    
            Yfilt1 = jwells(:,2) < ymax;
            Yfilt2 = jwells(:,2) > ymin;
    
            for j = 1:nx
    
                Xmin = max([1 j-SearchRadius]);
                Xmax = min([nx j+SearchRadius]);
    
                xmin = UTM_X(Xmin);
                xmax = UTM_X(Xmax);
    
                xv = UTM_X(j);
    
                Xfilt1 = jwells(:,1) < xmax;
                Xfilt2 = jwells(:,1) > xmin;
    
                filt = logical(Xfilt1.*Xfilt2.*Yfilt1.*Yfilt2);
                localwells = jwells(filt,:);
    
                nwells = size(localwells,1);
    
                if nwells > 0
                    
                    %use max 3 nearest wells
                    nloop = min(nwells,3);
                    
                    %Create array for sorting.
                    sort_array = zeros(nwells,1)
    
                    for s = 1:nwells
                        curwell = localwells(s,:);
    
                        curx = curwell(1);
                        cury = curwell(2);
    
                        curdist = sqrt(power(xv-curx,2)+power(yv-cury,2));
    
                        sort_array(s) = curdist;
                    end
    
                    %Now sort the array by distance and get the indices
                    [~,near_well_indices] = sort(sort_array);
    
                    %Choose the nloop nearest
                    near_well_indices = near_well_indices(1:nloop);
    
                    for a = near_well_indices
    
                        curwell = localwells(a,:);
    
                        curx = curwell(:,1);                   %curx = curwell(1);
                        cury = curwell(:,2);                     %cury = curwell(2);
                        curd = max(curwell(:,4)-curwell(:,3),0);  %curd = max(curwell(4)-curwell(3),0);
                        curc = curwell(:,5);                     %curc = curwell(5);
    
                        distance = sqrt(power(xv-curx,2)+power(yv-cury,2));
    
                        B = GSj;
    
                        if isnan(curc)
                            curc =  0;
                        end
     
                        for a_i=1:numel(a)               %rettet
    
                            Range = comp2range(curc(a_i)+1);  %rettet
                            var0 = var_fun(B,curd(a_i),KSj);  %rettet
                            certainty =  cert_fun(distance(a_i),Range,1/var0);  %rettet
        
                            Grid2(k,j,i) = Grid2(k,j,i)+certainty;  
                        end   %rettet
                    end
                end
            end
        end
        %%%%%%%%%%%%%%%
        snap_xs = [snap_xs;snap_well_xs];
        snap_ys = [snap_ys;snap_well_ys];
        snap_zs = [snap_zs;snap_well_zs];
    
        clear jwells
    else
        Grid(:,:,i) = Grid(:,:,i)+1;
        Grid2(:,:,i) = Grid2(:,:,i)+1; 
    end
end

theme = 1./(Grid+Grid2);

Data_snap.xs = wells(:,1);
Data_snap.ys = wells(:,2);
Data_snap.zs = wells(:,3);
Data_snap.elevations = wells(:,4);


