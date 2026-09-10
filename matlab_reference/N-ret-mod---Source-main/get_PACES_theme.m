function [Grid, s_cords] = get_PACES_theme(UTM_X,UTM_Y,terrain,complexity,reg,include_peatlands,Npreq,NPL,Nlay,cert_fun_choice)

%Load File based on region
switch reg
    case 'Jylland'
        filename = 'jylland_paces_til_usikkerhed.mat';
    case 'Fyn'
        filename = 'fyn_paces_til_usikkerhed.mat';
    case 'Fyn-MST'
        filename = 'fyn_paces_til_usikkerhed_MST.mat';
    case 'Sjælland'
        filename = 'sjaelland_paces_til_usikkerhed.mat';
end

in = load(['N:\PROJEKTER\N-retentionskortlægning 2022-24\Geostatistisk modellering\N-ret-mod_data_usikkerheder\geofys_usikkerheder\',filename]);

%Rearrange 'in' struct to 'PACEP' struct
switch reg
    case 'Jylland'
        PACES.InfoTable = in.Gmod_pos;
    case {'Fyn','Fyn-MST'}
        PACES.InfoTable = in.fynpacespos;
    case 'Sjælland'
        PACES.InfoTable = in.Gmod_pos;
end

PACES.Model_Depths = in.dkm_d_pos;
PACES.Model_Year = in.model_year_pos;
PACES.DepthInterfaceAboveModel = in.dkm_odvlayertop_pos;
PACES.ThickGeophysModel = in.dkm_odvthk_pos;

Npoints = numel(PACES.Model_Year);

%Determine duplicate points and calculate logical vector-index to ignore them
[~,id,~] = unique(table2array(PACES.InfoTable(:,["xutm_euref89_utm32","yutm_euref89_utm32"])),'rows');
logind = zeros(1,Npoints);
logind(id) = 1;

logind_dupe = logical(logind)';

%Determine if the geophysics is younger than the model. If so, filter it
%out.
ModelYear = PACES.Model_Year;
GeophysYear = table2array(PACES.InfoTable(:,"m_year"));

logind_date = GeophysYear <= ModelYear;

%Combine Logical Indices
logind = logical(logind_date.*logind_dupe);

%Update the struct
PACES.InfoTable = PACES.InfoTable(logind,:);
PACES.Model_Depths = PACES.Model_Depths(logind,:);
PACES.Model_Year = PACES.Model_Year(logind);
PACES.DepthInterfaceAboveModel = PACES.DepthInterfaceAboveModel(logind,:);
PACES.ThickGeophysModel = PACES.ThickGeophysModel(logind,:);


% 1) Determine grid size
nx = max(size(UTM_X));
ny = max(size(UTM_Y));
maxl = numel(PACES.Model_Depths(1,:));

% 2) Produce grid.
Grid = zeros(ny,nx,maxl);

% 3) Jeg definerer effective range for udbredelsen af certainty. Første
% element er 50m og bruges når kompleksiteten er ukendt. de næste 4
% elementer svarer til kompleksitet 1-4, hvor 1 er lav og 4 er høj.

comp2range = [100 500 400 250 100];
comp2rangePreq = [500 500 500 500 500];
NP = sum(logind);

if NP > 0
    
    % Define certainty function
    % Set to FRAFA apr. version for backward compatability
    if nargin < 10
        cert_fun_choice = 'FRAFA_apr2023';
    end

    cert_fun = certainty_function_definer(cert_fun_choice);

    % 6) Jeg skal besøge hver celle og lægge bidrag for de nærmeste boringer
    SearchRadius = 1; %in units of cells

    LLx = min(UTM_X);
    LLy = min(UTM_Y);

    xs = table2array(PACES.InfoTable(:,"xutm_euref89_utm32"));
    ys = table2array(PACES.InfoTable(:,"yutm_euref89_utm32"));
    DOIs = table2array(PACES.InfoTable(:,"doilower"));

    local_dist = NaN([size(terrain),(Nlay-NPL)]);
    local_doi = zeros([size(terrain),(Nlay-NPL)]);
    local_depth = NaN([size(terrain),(Nlay-NPL)]);
    local_model_thick = NaN([size(terrain),(Nlay-NPL)]);


    for s = 1:(Nlay-NPL)
        disp(['Calculating PACES theme for layer ' num2str(s) ' (that is not peatland) using ' cert_fun_choice])

        MDepths = PACES.Model_Depths(:,s);
        MThicks = PACES.ThickGeophysModel(:,s);

        complexities(:,:,s) = complexity;

        parfor i = 1:ny
            Ymin = max([1 i-SearchRadius]);
            Ymax = min([ny i+SearchRadius]);
            
            ymin = UTM_Y(Ymin);
            ymax = UTM_Y(Ymax);
            
            yv = UTM_Y(i);

            filt1 = ys<ymax & ys>ymin;

            for j = 1:nx

                Xmin = max([1 j-SearchRadius]);
                Xmax = min([nx j+SearchRadius]);

                xmin = UTM_X(Xmin);
                xmax = UTM_X(Xmax);

                xv = UTM_X(j);


                filt2 = xs<xmax & xs>xmin;
                filt = filt1 & filt2;

                npoints = sum(filt);

                if npoints > 0
                    local_xs = xs(filt);
                    local_ys = ys(filt);
                    local_depths = MDepths(filt)';
                    local_dois = DOIs(filt);
                    local_thicks = MThicks(filt)';

                    distances = sqrt(power(xv-local_xs,2)+power(yv-local_ys,2));
                    dist = min(distances);

                    filt_mindist = distances == dist;

                    local_dist(i,j,s) = dist;
                    local_doi(i,j,s) = mean(local_dois(filt_mindist));
                    local_depth(i,j,s) = mean(local_depths(filt_mindist));
                    local_model_thick(i,j,s) = mean(local_thicks(filt_mindist));

                end
            end
        end
   end
end

%Fill out missing DOI with mean DOI
local_doi = 18;

complexity2 = complexities;
complexity2(complexity==0) = NaN;
rangemap = complexity2;
rangemap2 = complexity2;

%Calculate var0 map
var0map = (0.5*max(2,0.25*local_depth)).^2;

rangemap(~isnan(complexity2)) = comp2range(complexity2(~isnan(complexity2))+1);
rangemap2(~isnan(complexity2)) = comp2rangePreq(complexity2(~isnan(complexity2))+1);
rangemap(:,:,Npreq-NPL:end) = rangemap2(:,:,Npreq-NPL:end);

kernelmap = local_depth*0+75;
certmap = cert_fun(local_dist,rangemap,kernelmap,1./var0map);

Grid = 1./certmap;
Grid(isnan(Grid)) = 100000;

doi_filter = local_doi < local_depth;
Grid(doi_filter) = 100000;


if include_peatlands == 1
    New_Grid = 1E4*ones(size(Grid,1),size(Grid,2),size(Grid,3)+NPL);
    New_Grid(:,:,(NPL+1):end) = Grid;
    Grid = New_Grid;
end

s_cords.xs = xs;
s_cords.ys = ys;


end