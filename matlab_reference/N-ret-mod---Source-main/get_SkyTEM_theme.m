function [Grid,s_cords] = get_SkyTEM_theme(UTM_X,UTM_Y,terrain,complexity,reg,include_peatlands,Npreq,NPL,Nlay,cert_fun_choice)


%Load File based on region0
switch reg
    case 'Jylland'
        filename = 'jylland_skytem_til_usikkerhed.mat';
    case 'Fyn'
        filename = 'fyn_skytem_til_usikkerhed.mat';
    case 'Fyn-MST'
        filename = 'fyn_skytem_til_usikkerhed_MST.mat';
    case 'Sjælland'
        filename = 'Sjaelland_skytem_til_usikkerhed.mat';
end

in = load(['N:\PROJEKTER\N-retentionskortlægning 2022-24\Geostatistisk modellering\N-ret-mod_data_usikkerheder\geofys_usikkerheder\',filename]);

%Rearrange 'in' struct to 'SkyTEM' struct
switch reg
    case 'Jylland'
    SkyTEM.InfoTable = in.Gmod_pos;
    case 'Fyn'
    SkyTEM.InfoTable = in.fynskytempos;
    case 'Fyn-MST'
    SkyTEM.InfoTable = in.fynskytempos;
    case 'Sjælland'
    SkyTEM.InfoTable = in.Gmod_pos;
end

SkyTEM.Model_Depths = in.dkm_d_pos;
SkyTEM.Model_Year = in.model_year_pos;
SkyTEM.DepthInterfaceAboveModel = in.dkm_odvlayertop_pos;
SkyTEM.ThickGeophysModel = in.dkm_odvthk_pos;

Npoints = numel(SkyTEM.Model_Year);

%Determine duplicate points and calculate logical vector-index to ignore them
[~,id,~] = unique(table2array(SkyTEM.InfoTable(:,["xutm_euref89_utm32","yutm_euref89_utm32"])),'rows');
logind = zeros(1,Npoints);
logind(id) = 1;

logind_dupe = logical(logind)';

%Determine if the geophysics is younger than the model. If so, filter it
%out.
ModelYear = SkyTEM.Model_Year;
GeophysYear = table2array(SkyTEM.InfoTable(:,"m_year"));

logind_date = GeophysYear <= ModelYear;

%Combine Logical Indices
logind = logical(logind_date.*logind_dupe);

%Update the struct
SkyTEM.InfoTable = SkyTEM.InfoTable(logind,:);
SkyTEM.Model_Depths = SkyTEM.Model_Depths(logind,:);
SkyTEM.Model_Year = SkyTEM.Model_Year(logind);
SkyTEM.DepthInterfaceAboveModel = SkyTEM.DepthInterfaceAboveModel(logind,:);
SkyTEM.ThickGeophysModel = SkyTEM.ThickGeophysModel(logind,:);


% 1) Determine grid size
nx = max(size(UTM_X));
ny = max(size(UTM_Y));
maxl = numel(SkyTEM.Model_Depths(1,:));

% 3) Jeg definerer effective range for udbredelsen af certainty. Første
% element er 50m og bruges når kompleksiteten er ukendt. de næste 4
% elementer svarer til kompleksitet 1-4, hvor 1 er lav og 4 er høj.

comp2range = [100 500 400 250 100];
comp2rangePreq = [500 500 500 500 500];

NP = sum(logind);

if NP > 0

    % 4) Jeg skal have et N x M array med N boringer og M variabler

    %%%
    % 1 ) X
    % 2 ) Y
    % 3 ) Z
    % 4 ) Quality
    % 5 ) Homogeneity
    %%%

    % 4) Jeg skal have en funktion som beskriver boringskvalitet/"certainty"
    % 4) I need a function that describes the geophysics uncertainty with
    % depth

    %Her er b = [G N]

    %  - G er en gradient for forholdet mellem dybde og usikkerhed
    %  - N er Nul-variansen (minimumsvarians)

    var_fun = @(b,z) (b(1).*z+b(2)).^2;

    % 5) Jeg skal have en funktion som beskriver hvordan certainty falder med
    % afstanden til boringen

    %cert_fun = @(dist,range,width,sill,d0) (dist<=width).*sill+(dist>width).*sill.*(exp(-(3*power((dist-width),2))./(power(range,2))));
    %cert_fun = @(dist,range,width,sill,d0) (dist<=width).*sill.*(exp(-(3*power((dist-width),2))./(power(range,2))));
    
    % Define certainty function
    % Set to FRAFA apr. version for backward compatability
    if nargin < 10
        cert_fun_choice = 'FRAFA_apr2023';
    end

    cert_fun = certainty_function_definer(cert_fun_choice);

    
    
    % 6) Jeg skal besøge hver celle og lægge bidrag for de nærmeste boringer
    SearchRadius = 6; %in units of cells  - rettet til 6 d 27/6 2023 så alle usikkerhedstemaer bliver ens

    LLx = min(UTM_X);
    LLy = min(UTM_Y);

    %Do correction for ascii grid
    if mod(LLx,100) > 0

        LLx = LLx-50;
        LLy = LLy-50;

    end
    

    xs = table2array(SkyTEM.InfoTable(:,"xutm_euref89_utm32"));
    ys = table2array(SkyTEM.InfoTable(:,"yutm_euref89_utm32"));

    switch reg
        case 'Jylland'
            PalDepth = SkyTEM.Model_Depths(:,Npreq-NPL+27);
            PalThick = SkyTEM.Model_Depths(:,Npreq-NPL+28)-SkyTEM.Model_Depths(:,Npreq-NPL+27);

        case {'Fyn','Fyn-MST'}
            PalDepth = SkyTEM.Model_Depths(:,Npreq-NPL-1);
            PalThick = SkyTEM.Model_Depths(:,Npreq-NPL)-SkyTEM.Model_Depths(:,Npreq-NPL-1);

        case 'Sjælland'
            PalDepth = SkyTEM.Model_Depths(:,Npreq-NPL-1);
            PalThick = SkyTEM.Model_Depths(:,Npreq-NPL)-SkyTEM.Model_Depths(:,Npreq-NPL-1);

    end
    DOIs = table2array(SkyTEM.InfoTable(:,"doilower"));
    
    doimean = nanmean(DOIs);
    
    f_doi = @(x) min(doimean,x+1E3*(PalThick<10));

    DOI_NaN_Filter = isnan(DOIs); 
    estimated_DOIs = f_doi(PalDepth);
    
    DOIs(DOI_NaN_Filter) = estimated_DOIs(DOI_NaN_Filter);

    local_dist = NaN([size(terrain),Nlay-NPL]);
    local_doi = zeros([size(terrain),Nlay-NPL]);
    local_depth = NaN([size(terrain),Nlay-NPL]);
    local_model_thick = NaN([size(terrain),Nlay-NPL]);

    for s = 1:(Nlay-NPL)
        disp(['Calculating SkyTEM theme for layer ' num2str(s) ' (that is not peatland) using ' cert_fun_choice])

        MDepths = SkyTEM.Model_Depths(:,s);
        MThicks = SkyTEM.ThickGeophysModel(:,s);

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
                    local_thicks = MThicks(filt)';
                    local_dois = DOIs(filt);

                    distances = sqrt(power(xv-local_xs,2)+power(yv-local_ys,2));
                    dist = min(distances);

                    filt_mindist = distances == dist;

                    local_dist(i,j,s) = dist;
                    local_doi(i,j,s) = mean(local_dois(filt_mindist));
                    local_depth(i,j,s) = mean(local_depths(filt_mindist));
                    local_model_thick(i,j,s) = min(local_thicks(filt_mindist));
                end
            end
        end
   end
end



complexity2 = complexities;
complexity2(complexities==0) = NaN;

rangemap = complexity2;
rangemap2 = complexity2;

%Calculate var0 map
var0map = (0.5*1.2*local_model_thick).^2;
var0map(local_depth<5) = 100000;

rangemap(~isnan(complexity2)) = comp2range(complexity2(~isnan(complexity2))+1);
rangemap2(~isnan(complexity2)) = comp2rangePreq(complexity2(~isnan(complexity2))+1);
rangemap(:,:,(Npreq-NPL):end) = rangemap2(:,:,(Npreq-NPL):end);

kernelmap = max(2*local_depth, 75);
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