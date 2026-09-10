function [Grid,s_cords] = get_MEP_theme(UTM_X,UTM_Y,terrain,complexity,reg,include_peatlands,Npreq,NPL,Nlay,cert_fun_choice)

%Load File based on region
switch reg
    case 'Jylland'
        filename = 'jylland_mep_til_usikkerhed.mat';
    case 'Fyn'
        filename = 'fyn_mep_til_usikkerhed.mat';
    case 'Fyn-MST'
        filename = 'fyn_mep_til_usikkerhed_MST.mat';
    case 'Sjælland'
        filename = 'Sjaelland_mep_til_usikkerhed.mat';
end

in = load(['N:\PROJEKTER\N-retentionskortlægning 2022-24\Geostatistisk modellering\N-ret-mod_data_usikkerheder\geofys_usikkerheder\',filename]);

%Rearrange 'in' struct to 'MEP' struct
switch reg
    case 'Jylland'
        MEP.InfoTable = in.Gmod_pos;
    case 'Fyn'
        MEP.InfoTable = in.fynmeppos;
    case 'Fyn-MST'
        MEP.InfoTable = in.fynmeppos;
    case 'Sjælland'
        MEP.InfoTable = in.Gmod_pos;
end

MEP.Model_Depths = in.dkm_d_pos;
MEP.Model_Year = in.model_year_pos;
MEP.DepthInterfaceAboveModel = in.dkm_odvlayertop_pos;
MEP.ThickGeophysModel = in.dkm_odvthk_pos;
Npoints = numel(MEP.Model_Year);

typevec = zeros(Npoints,1);
MEP.Type = table2array(MEP.InfoTable(:,"datasubtype"));

for i = 1:Npoints
    cur_string = MEP.Type{i};
    log_str = all(cur_string(1:3) == 'wen');
    typevec(i) = log_str;
end

%1 denotes wennera_2d, 0 denotes NOT wennera_2d
MEP.Type = typevec;

%Determine duplicate points and calculate logical vector-index to ignore them
[~,id,~] = unique(table2array(MEP.InfoTable(:,["xutm_euref89_utm32","yutm_euref89_utm32"])),'rows');
logind = zeros(1,Npoints);
logind(id) = 1;

logind_dupe = logical(logind)';

%Determine if the geophysics is younger than the model. If so, filter it
%out.
ModelYear = MEP.Model_Year;
GeophysYear = table2array(MEP.InfoTable(:,"m_year"));

logind_date = GeophysYear <= ModelYear;

%Combine Logical Indices
logind = logical(logind_date.*logind_dupe);

%Update the struct
MEP.InfoTable = MEP.InfoTable(logind,:);
MEP.Model_Depths = MEP.Model_Depths(logind,:);
MEP.Model_Year = MEP.Model_Year(logind);
MEP.DepthInterfaceAboveModel = MEP.DepthInterfaceAboveModel(logind,:);
MEP.ThickGeophysModel = MEP.ThickGeophysModel(logind,:);
MEP.Type = MEP.Type(logind,:);


% 1) Determine grid size
nx = max(size(UTM_X));
ny = max(size(UTM_Y));
maxl = numel(MEP.Model_Depths(1,:));

% 3) Jeg definerer effective range for udbredelsen af certainty. Første
% element er 50m og bruges når kompleksiteten er ukendt. de næste 4
% elementer svarer til kompleksitet 1-4, hvor 1 er lav og 4 er høj.

comp2range = [100 500 400 250 100];
comp2rangePreq = [500 500 500 500 500];
NP = sum(logind);

if NP > 0

  %  cert_fun = @(dist,range,width,sill) (dist<=width).*sill+(dist>width).*sill.*(exp(-(3*power((dist-width),2))./(power(range,2))));
  %cert_fun = @(dist,range,width,sill) (dist<=width).*sill.*(exp(-(3*power((dist-width),2))./(power(range,2))));
  
    % Define certainty function
    % Set to FRAFA apr. version for backward compatability
    if nargin < 10
        cert_fun_choice = 'FRAFA_apr2023';
    end

    cert_fun = certainty_function_definer(cert_fun_choice);
  
  % 6) Jeg skal besøge hver celle og lægge bidrag for de nærmeste boringer
    SearchRadius = 2; %in units of cells

    LLx = min(UTM_X);
    LLy = min(UTM_Y);

    xs = table2array(MEP.InfoTable(:,"xutm_euref89_utm32"));
    ys = table2array(MEP.InfoTable(:,"yutm_euref89_utm32"));
    DOIs = table2array(MEP.InfoTable(:,"doilower"));

    local_dist = NaN([size(terrain),Nlay-NPL]);
    local_doi = zeros([size(terrain),Nlay-NPL]);
    local_depth = NaN([size(terrain),Nlay-NPL]);
    local_model_thick = NaN([size(terrain),Nlay-NPL]);
    local_MEP_type = NaN([size(terrain),Nlay-NPL]);


    for s = 1:(Nlay-NPL)
        disp(['Calculating MEP theme for layer ' num2str(s) ' (that is not peatland) using ' cert_fun_choice])

        MDepths = MEP.Model_Depths(:,s);
        MThicks = MEP.ThickGeophysModel(:,s);

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
                    local_types = MEP.Type(filt);

                    distances = sqrt(power(xv-local_xs,2)+power(yv-local_ys,2));
                    dist = min(distances);

                    filt_mindist = distances == dist;

                    local_dist(i,j,s) = dist;
                    local_doi(i,j,s) = mean(local_dois(filt_mindist));
                    local_depth(i,j,s) = mean(local_depths(filt_mindist));
                    local_model_thick(i,j,s) = mean(local_thicks(filt_mindist));
                    local_MEP_type(i,j,s) = local_types(1);
                end
            end
        end
   end
end

%Fill out missing DOI with mean DOI
local_doi(isnan(local_doi)) = mean(DOIs(~isnan(DOIs)));

complexity2 = complexities;
complexity2(complexity==0) = NaN;
rangemap = complexity2;
rangemap2 = complexity2;

%Calculate var0 map
WenneraMap = local_MEP_type;
NotWenneraMap = WenneraMap == 0;

WenneraVar0 = 0.5*max(2.4,0.25*local_depth);
NotWenneraVar0 = 0.5*max(1.25,0.15*local_depth);

var0map = (WenneraVar0.*WenneraMap+NotWenneraVar0.*NotWenneraMap).^2;

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