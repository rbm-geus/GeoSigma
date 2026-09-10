function well_out = get_wells(fname,reg,Nlay,NPL,local_dir)

% This function will take in the name of the current model
% well-interpretation points and the name of the wells not used
% directly in interpretation, as well as "reg" which is the
% identifier for Fyn, Jylland or Sjaelland.
%
% It will then load the model well points for existing model layers
% as well as the newly modelled peatland well points.
%
% The output is a struct containing first the newly modelled well points and
% then the existing model layer well points.

%Load the wells/points for the original model
ModelWells = [local_dir,'\tolkningspunkter\',fname];
wellstruct = load(ModelWells);

switch reg
    case 'Jylland'

        wells_m = wellstruct.Jylland_TP;

    case 'Fyn'

        wells_m = wellstruct.Fyn_TP;

    case 'Fyn-MST'

        wells_m = wellstruct.Fyn_TP;
        
        % Filter out wells outside study area as file contains wells
        % outside Fyn (eg. Langeland)
        ncols = 805;
        nrows = 676;
        xllcorner = 538550;
        yllcorner = 6098450;
        cellsize = 100.00000;
        
        xbounds = [xllcorner,xllcorner+cellsize*ncols];
        ybounds = [yllcorner,yllcorner+cellsize*nrows];

        for i = 1:length(wells_m)
            idx = wells_m{i}.xutm<xbounds(1) | wells_m{i}.xutm>xbounds(2);
            idy = wells_m{i}.yutm<ybounds(1) | wells_m{i}.yutm>ybounds(2);
            idremove = (idx | idy);

            wells_m{i}.xutm(idremove) = [];
            wells_m{i}.yutm(idremove) = [];
            wells_m{i}.z(idremove) = [];
            wells_m{i}.borsnap_index(idremove) = [];
            wells_m{i}.bor_qual(idremove) = [];
            wells_m{i}.zero_layerthk(idremove) = [];
            wells_m{i}.dguno(idremove) = [];
            wells_m{i}.origin_modelno(idremove) = [];
            wells_m{i}.origin_modelTxt(idremove) = [];
            wells_m{i}.origin_modelyear(idremove) = [];

        end

    case 'Sjælland'

        wells_m = wellstruct.Sjaelland_TP;

    case 'AnholtLæsø'

        wells_m = wellstruct.DK8_TP;


end

% UNCOMMENT THIS WHEN FYN IS READY
% NewFynWells = ['WellPoints',reg];
% wells_p = load(NewFynWells);

% This is a placeholder for the above lines
%wells_p = wells_m(1:NPL);

%Gather cell arrays
%well_out = [wells_p; wells_m]; % OLD VERSION

well_out = [cell(NPL,1); wells_m];

%keyboard

end