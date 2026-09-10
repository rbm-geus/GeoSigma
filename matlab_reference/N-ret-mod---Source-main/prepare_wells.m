function [Wells,OtherWells,Thick0s,Well_S,allwells] = prepare_wells(wells,fname,struct,terrain)
Nlay = size(struct,2);

%The desired output of this function is two structs, the Wells struct will
%only be modified in accordance with the peatlands (update when more info
%is available).

for i = 1:Nlay
    if isempty(wells{i})
        nothick_snap{i} = [];   
    else
        wellxs = wells{i}.xutm;
        wellys = wells{i}.yutm;
        zth = wells{i}.zero_layerthk;
    
        zxs = wellxs(zth);
        zys = wellys(zth);
    
        xystruct.x = zxs;
        xystruct.y = zys;
    
        nothick_snap{i} = xystruct;
        clear xystruct
    end
end

Wells = wells;

%The second part constructs a cell-array of length Nlay with the needed
%information from the Jupiter wells for the uncertainty map calculations.

%Load the Jupiter wells
ModelWells = [pwd,'\tolkningspunkter\',fname];
wellstruct = load(ModelWells);

%Determine which are duplicated in the snapped list
OtherWells = cell(Nlay,1);

st = fname(1:3);

% switch st
%     case 'jyl'
% 
%         elevations_jup = table2array([wellstruct.Jup_bor_data(:,"ELEVATION")]);
%         qualities_jup = wellstruct.jup_bhq_l;
%         years_jup = wellstruct.DrillYear;
%         years_mod = wellstruct.model_year_pos;
% 
%         depths_jup = table2array([wellstruct.Jup_bor_data(:,"MAX_DEPTH")]);
%         coords_jup = table2array([wellstruct.Jup_bor_data(:,"XUTM"),wellstruct.Jup_bor_data(:,"YUTM")]);
%         
% 
%         %FILTER OUT WRONG QUALITIES
%         filtered = qualities_jup>2;
%         qualities_jup = qualities_jup(filtered);
%         elevations_jup = elevations_jup(filtered);
%         depths_jup = depths_jup(filtered);
%         coords_jup = coords_jup(filtered,:);
% 
% 
% 
%         %Now, filter outdated models away
%         if numel(years_jup) ~= numel(years_mod)
%             keyboard
%         end
% 
%         %Correct wrong years
%         years_jup(years_jup==2107) = 2017;
% 
%         age_filt = years_jup < years_mod;
%         age_exist = years_jup > 1000; %No medieval wells please (just kidding)
% 
%         combined_logic = and(age_filt,age_exist);
% 
%         qualities_jup = qualities_jup(combined_logic);
% 
%         filt1 = qualities_jup == 3;
%         filt2 = qualities_jup == 4;
%         filt3 = qualities_jup == 5;
%         filt4 = qualities_jup == 6;
%         filt5 = qualities_jup == 7;
% 
%         qualities_jup(filt1) = 4;
%         qualities_jup(filt2) = 3;
%         qualities_jup(filt3) = 3;
%         qualities_jup(filt4) = 2;
%         qualities_jup(filt5) = 1;
% 
%         elevations_jup = elevations_jup(combined_logic);
%         depths_jup = depths_jup(combined_logic);
%         coords_jup = coords_jup(combined_logic);
% 
%     case 'fyn'

elevations_jup = table2array([wellstruct.Jup_bor_data_q(:,"ELEVATION")]);
qualities_jup = table2array([wellstruct.Jup_bor_data_q(:,"borqual_lith")]);
depths_jup = table2array([wellstruct.Jup_bor_data_q(:,"MAX_DEPTH")]);
jup_logical = wellstruct.jup_deeper_than_dkm;

years_jup = table2array([wellstruct.Jup_bor_data_q(:,"DrillYear")]);
years_mod = wellstruct.model_year_pos;

coords_jup = table2array([wellstruct.Jup_bor_data_q(:,"XUTM"),wellstruct.Jup_bor_data_q(:,"YUTM")]);


if numel(years_jup) ~= numel(years_mod)
    keyboard
end

%Correct wrong years
years_jup(years_jup==2107) = 2017;

age_filt = years_jup <= years_mod;
age_exist = years_jup > 1000; %No medieval wells please (just kidding)

combined_logic = and(age_filt,age_exist);

qualities_jup = qualities_jup(combined_logic);

%         filt1 = qualities_jup == 3;
%         filt2 = qualities_jup == 4;
%         filt3 = qualities_jup == 5;
%         filt4 = qualities_jup == 6;
%         filt5 = qualities_jup == 7;
% 
%         qualities_jup(filt1) = 4;
%         qualities_jup(filt2) = 3;
%         qualities_jup(filt3) = 3;
%         qualities_jup(filt4) = 2;
%         qualities_jup(filt5) = 1;

        elevations_jup = elevations_jup(combined_logic);
        depths_jup = depths_jup(combined_logic);
        coords_jup = coords_jup(combined_logic,:);

%     case 'sja'
% 
%         elevations_jup = table2array([wellstruct.Jup_bor_data(:,"ELEVATION")]);
%         qualities_jup = wellstruct.jup_bhq_l;
%         years_jup = wellstruct.DrillYear;
%         years_mod = wellstruct.model_year_pos;
% 
%         depths_jup = table2array([wellstruct.Jup_bor_data(:,"MAX_DEPTH")]);
%         coords_jup = table2array([wellstruct.Jup_bor_data(:,"XUTM"),wellstruct.Jup_bor_data(:,"YUTM")]);
% 
%         %FILTER OUT WRONG QUALITIES
%         filtered = qualities_jup>2;
%         qualities_jup = qualities_jup(filtered);
%         elevations_jup = elevations_jup(filtered);
%         depths_jup = depths_jup(filtered);
%         coords_jup = coords_jup(filtered,:);
% 
% 
% 
%         %Now, filter outdated models away
%         if numel(years_jup) ~= numel(years_mod)
%             keyboard
%         end
% 
%         %Correct wrong years
%         years_jup(years_jup==2107) = 2017;
% 
%         age_filt = years_jup < years_mod;
%         age_exist = years_jup > 1000; %No medieval wells please (just kidding)
% 
%         combined_logic = and(age_filt,age_exist);
% 
%         qualities_jup = qualities_jup(combined_logic);
%         
%         filt1 = qualities_jup == 3;
%         filt2 = qualities_jup == 4;
%         filt3 = qualities_jup == 5;
%         filt4 = qualities_jup == 6;
%         filt5 = qualities_jup == 7;
% 
%         qualities_jup(filt1) = 4;
%         qualities_jup(filt2) = 3;
%         qualities_jup(filt3) = 3;
%         qualities_jup(filt4) = 2;
%         qualities_jup(filt5) = 1;
% 
%         elevations_jup = elevations_jup(combined_logic);
%         depths_jup = depths_jup(combined_logic);
%         coords_jup = coords_jup(combined_logic,:);
% end

Njup = size(coords_jup,1);

LLc = struct{1}.Grid_UTM_LowerLeftCorner;
depths_sna = [];
zeros_sna = [];
kotes_sna = [];
coords_sna = [];

for i = 1:Nlay
    if isempty(wells{i}) == 1

    else
        cw = wells{i};
    
        new_xs_sna = cw.xutm;
        new_ys_sna = cw.yutm;
        new_z_sna = cw.z;
        new_zero_sna = cw.zero_layerthk;


        ind_y = round((new_ys_sna-LLc(2))/100)+1;
        ind_x = round((new_xs_sna-LLc(1))/100)+1;
        lindex = sub2ind(size(terrain),ind_y,ind_x);

        new_kotes =terrain(lindex);
        new_coords = [new_xs_sna,new_ys_sna];
        
        depths_sna = [depths_sna;new_z_sna];
        zeros_sna = [zeros_sna;new_zero_sna];
        kotes_sna = [kotes_sna;new_kotes];
        coords_sna = [coords_sna;new_coords];

    end    
end

coords_sna = flipud(coords_sna);
depths_sna = flipud(depths_sna);
kotes_sna = flipud(kotes_sna);
zeros_sna = flipud(zeros_sna);

kotes_combined = [kotes_sna;elevations_jup];
zeros_combined = [zeros_sna;0*elevations_jup];
depths_combined = [depths_sna;depths_jup];
coords_combined = [coords_sna;coords_jup];

%Uniques are determined for the snapped wells and the combined list.
%First, determine the number of snapped uniques
[snapped_uniques,snapos] = unique(coords_sna,'rows','stable');


%Now determine the combined uniques.
[allwells,all_positions] = unique(coords_combined,'rows','stable');

un_zeros_sna = zeros_sna(snapos);
un_depths_sna = depths_sna(snapos);
un_kotes_sna = kotes_sna(snapos);

%Fill well struct with snap info
Well_S.snap_pos = coords_sna;
Well_S.snap_depth = depths_sna;
Well_S.snap_zerothk = zeros_sna;
Well_S.snap_kote = kotes_sna;

%The number of existing uniques from snapped wells are ignored
NumUniqueSnap = size(snapped_uniques,1);
remaining_positions = all_positions(NumUniqueSnap+1:end)-size(coords_sna,1);

%Turn remaining positions linear index into a logical array
logind = zeros(1,Njup);
logind(remaining_positions) = 1;
logind = logical(logind);

%Take unique wells from Jupiter wells
elevations = elevations_jup(logind);
qualities = qualities_jup(logind);
coordinates = coords_jup(logind,:);
depths = depths_jup(logind);
jup_logical = jup_logical(logind);


Well_S.jup_pos = coordinates;
Well_S.jup_kote = elevations;
Well_S.jup_depth = depths;

%Prepare a list of grid positions for the wells
LLc = struct{1}.Grid_UTM_LowerLeftCorner;
LLx = LLc(1);
LLy = LLc(2);

gridcol_list = round((coordinates(:,1)-LLx)/100)+1;
gridrow_list = round((coordinates(:,2)-LLy)/100)+1;
lin_indices = sub2ind([struct{1}.Grid_Size_Rows,struct{1}.Grid_Size_Columns],gridrow_list,gridcol_list);
terrain = terrain(lin_indices);

%Fix NaN's in elevation by replacing it with terrain elevation
naninds=isnan(elevations);
elevations(naninds) = terrain(naninds);

% depths = (elevations-depths);
temp_terrain = struct{1}.LayerTop;
temp_t_lin = temp_terrain(lin_indices);

for i = 1:Nlay

    curmod = temp_terrain-struct{i}.LayerBottom;
    prevmod = temp_terrain-struct{i}.LayerTop;


    model_depths = curmod(lin_indices);
    model_thicks = curmod(lin_indices)-prevmod(lin_indices);

%   depth_logical = depths > model_depths;
    depth_logical = logical(jup_logical(:,i));

    temp_struct.xutm = coordinates(depth_logical,1);
    temp_struct.yutm = coordinates(depth_logical,2);
    temp_struct.z_true = temp_t_lin(depth_logical)-model_depths(depth_logical);
    temp_struct.bor_qual = qualities(depth_logical);

    new_thicks = model_thicks(depth_logical);
    
    new_xs = temp_struct.xutm;
    new_ys = temp_struct.yutm;

    thick0logical = new_thicks < 0.2;
    if  isempty(nothick_snap{i}) == 1
        thick0s.x = [new_xs(thick0logical)];
        thick0s.y = [new_ys(thick0logical)]; 
    else
        thick0s.x = [new_xs(thick0logical);nothick_snap{i}.x];
        thick0s.y = [new_ys(thick0logical);nothick_snap{i}.y];
    end
    
    Thick0s{i} = thick0s;

    OtherWells{i} = temp_struct;
    clear temp_struct;
end

end