function S = prepare_for_kriging(top_name,bottom_name,gettop,getbottom,doplot,layn,wells,NPL,reg,top,peat_log,Snap_PL,layername,use_calibration_model)

%% DESCRIPTION

% This script produces a struct, S, with fields containing information such as
%   the points drawn
%   the orignal surfaces
%   the layer thickness,
%   masks describing where the layer is found
%   Grid Coordinates
%   Map(s) with estimated effective range
%   Map(s) with estimated sill

% INPUT:
%   top_name (string): filename for local ascii file containing the gridded layer top
%   bottom_name (string): filename for local ascii file containing the gridded layer bottom
%   gettop (logical): draw points and produce range & sill maps for the top
%   getbottom (logical): draw points and produce range & sill maps for the bottom
%   doplot (logical): Produce Figures for Range and Sill

%% THE FUNCTION

if gettop == 0
    if getbottom == 0
        disp('Please specify gettop or getbottom arguments such that one is 0 and the other is 1')
    end
else
    if getbottom == 1
        disp('Please specify gettop or getbottom arguments such that one is 0 and the other is 1')
    end
end


%Get Layer Masks (the function below also reads the files from the directory)
[S,probmask]=get_layermasks(top_name,bottom_name,0,doplot,0,reg,top,peat_log,NPL,layn(1),layername,use_calibration_model);


% figure(); 
% subplot(2,3,1)
% imagesc(reshape(S.indexMask_Onshore,S.Grid_Size_Rows,S.Grid_Size_Columns))
% title('indexMask_Onshore'); set(gca,'Ydir','normal')
% subplot(2,3,2)
% imagesc(reshape(S.indexMask_LayerExists,S.Grid_Size_Rows,S.Grid_Size_Columns))
% title('indexMask_LayerExists'); set(gca,'Ydir','normal')
% subplot(2,3,3)
% imagesc(reshape(S.indexMask_Combined,S.Grid_Size_Rows,S.Grid_Size_Columns))
% title('indexMask_Combined'); set(gca,'Ydir','normal')
% subplot(2,3,4)
% imagesc(reshape(S.indexBufferedMask_Onshore,S.Grid_Size_Rows,S.Grid_Size_Columns))
% title('indexBufferedMask_Onshore'); set(gca,'Ydir','normal')
% subplot(2,3,5)
% imagesc(reshape(S.indexBufferedMask_LayerExists,S.Grid_Size_Rows,S.Grid_Size_Columns))
% title('indexBufferedMask_LayerExists'); set(gca,'Ydir','normal')
% subplot(2,3,6)
% imagesc(reshape(S.indexBufferedMask_Combined,S.Grid_Size_Rows,S.Grid_Size_Columns))
% title('indexBufferedMask_Combined'); set(gca,'Ydir','normal')



% Check if empty mask
if sum(S.indexMask_Combined)>0

    %Extract layer top and bottom from S
    top = S.LayerTop;
    bottom = S.LayerBottom;
    
    %Process the layer and predict ranges and sills.
    %keyboard
    if getbottom == 1
        [ranges, sills, ~, means_bottom, points_bottom,ct] = get_interpolated_minigrids_parallel(bottom,S.indexMask_Combined,[S.Grid_Size_Rows,S.Grid_Size_Columns],wells,layn,S.Grid_UTM_X,S.Grid_UTM_Y,S.indexMask_LayerExists,S.indexMask_Onshore,NPL,Snap_PL,layername,probmask);
    end
    
    
    %Postprocessing of range and sill
    pr = sills;
    newmap = NaN*zeros(S.Grid_Size_Rows,S.Grid_Size_Columns);
    newmap(S.indexMask_Combined) = pr;
    
    newmap2 = newmap;
    cr = [4 100];
    
    SY = size(newmap,1);
    SX = size(newmap,2);
    
    for r = 1:2
        cleanup_radius = cr(r);
    for i = 1:SY
        for j = 1:SX
    
            current_class = newmap(i,j);
    
            if current_class == 0
            is_1 = max(1, i-cleanup_radius);
            is_2 = min(SY, i+cleanup_radius);
            js_1 = max(1, j-cleanup_radius);
            js_2 = min(SX, j+cleanup_radius);
            
            
            neighbor_classes = newmap(is_1:is_2,js_1:js_2);
            neighbor_classes = neighbor_classes(:);
            
            neighbor_classes(neighbor_classes == 0) = [];
            neighbor_classes(isnan(neighbor_classes)) = [];
            
            if numel(neighbor_classes) > 0
                newmap2(i,j) = mean(neighbor_classes(:));
            end
            
            end
    
        end
    end
    end
    
    sills_bottom = newmap2(S.indexMask_Combined);
    
    
    %Get clustering result
    [clas, ~, ~] = get_clusters(sills_bottom,ranges,[S.Grid_Size_Rows,S.Grid_Size_Columns],S.indexMask_Combined,ct);
    
    %Save to struct
    if gettop == 1
        S.ranges_top = ranges(:);
        S.variances_top = sills(:);
        S.points_top = points_top(:);
        S.means_top = means_top(:);
        S.clusters = clas(S.indexMask_Combined);
    end
    
    if getbottom == 1
        S.ranges_bottom = ranges(:);
        S.variances_bottom = sills(:);
        S.points_bottom = points_bottom(:);
        S.means_bottom = means_bottom(:);
        S.clusters = clas(S.indexMask_Combined);
    end
    S.zeromask = 0; % Flag for zeromask
else
    disp('Layer has a zeromask!')
    S.zeromask = 1; % Flag for zeromask
end
S.comment = 'ranges, clusters and variances are for the combined mask without buffer';
