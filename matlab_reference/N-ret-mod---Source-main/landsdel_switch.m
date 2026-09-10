function [region, NPL, Nlay, layernames, Ninterfaces, Npreq, filetype, wellname_interpreted, wellname_not_interpreted, wellname] = landsdel_switch(landsdel, redo_topography, topo_APR2023)

%DESCRIPTION: Datahandler using switch statements. Based on inputs,
% different regions are ran.

switch landsdel
    case 1
        region = 'Jylland';
        NPL = 5;
        [layernames,Ninterfaces,Npreq] = get_layer_names(region,NPL);
        filetype = '.asc';
        wellname_interpreted = 'Jylland_tolkningspunkter_med_boringssnap.mat';
        wellname_not_interpreted = 'Jylland_jup_boringer_til_usikkerhed.mat';
        wellname = {wellname_interpreted, wellname_not_interpreted};
 
        % If jylland is chosen and if redo_topography is 1 we resample jylland
        % to fit with DK model grids given the topography assigned in "get_layer_names".
        % The FOHM topography "0000_Terrain_Bathymetri_100x100m.grd" must exist in
        % the "Flader/Jylland" folder.

        if redo_topography == 1
            % Not implemented
        end
        
        % Redefines layernames accordingly
        if topo_APR2023 == 1
            layernames{1} = 'dk456_topo100m';
            disp('Used topo from DK-model April 2023')
        end

    case 2
        region = 'Fyn';
        NPL = 3;
        [layernames,Ninterfaces,Npreq] = get_layer_names(region,NPL);
        filetype = '.asc';
        wellname_interpreted = 'Fyn_tolkningspunkter_med_boringssnap.mat';
        wellname_not_interpreted = 'fyn_jup_boringer_til_usikkerhed.mat';
        wellname = {wellname_interpreted, wellname_not_interpreted};

        if topo_APR2023 == 1
            layernames{1} = 'dk3_topo100m';
            disp('Used topo from DK-model April 2023')
        end

    case 22
        region = 'Fyn-MST';
        NPL = 0;
        [layernames,Ninterfaces,Npreq] = get_layer_names(region,NPL);
        filetype = '.asc';
        wellname_interpreted = 'Fyn_tolkningspunkter_med_boringssnap.mat'; % Samme som til Alm. Fyn
        wellname_not_interpreted = 'fyn_jup_boringer_til_usikkerhed_MST.mat';
        wellname = {wellname_interpreted, wellname_not_interpreted};

    case 3
        region = 'Sjælland';
        NPL = 0;
        [layernames,Ninterfaces,Npreq] = get_layer_names(region,NPL);
        filetype = '.asc';
        wellname_interpreted = 'Sjaelland_tolkningspunkter_med_boringssnap.mat';
        wellname_not_interpreted = 'sjaelland_jup_boringer_til_usikkerhed.mat';
        wellname = {wellname_interpreted, wellname_not_interpreted};

        if topo_APR2023 == 1
            layernames{1} = 'dk12_topo100m';
            disp('Used topo from DK-model April 2023')
        end

    case 4
        region = 'AnholtLæsø';
         NPL = 5;
         [layernames,Ninterfaces,Npreq] = get_layer_names(region,NPL);
         filetype = '.asc';
         wellname_interpreted = 'DK8_tolkningspunkter_med_boringssnap.mat';
         wellname_not_interpreted = 'DK8_jup_boringer_til_usikkerhed.mat';
         wellname = {wellname_interpreted, wellname_not_interpreted};

        if topo_APR2023 == 1
            disp('Option cannot be set for Anholt and Læsø')
        end
end

% There are Ninterfaces-2 layers to be modelled
% Minus terrain and minus bottom
if ~strcmp(region,'AnholtLæsø')
    Nlay = Ninterfaces-2;

else % For Anholt and Læsø there is no bottom
    Nlay = Ninterfaces-1;
end    

% In Jylland the chalk is not modelled either
if strcmp(region,'Jylland')
    Nlay = Nlay - 1;
end
