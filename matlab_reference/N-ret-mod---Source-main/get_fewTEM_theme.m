function [Grid,s_cords] = get_fewTEM_theme(UTM_X,UTM_Y,terrain,complexity,reg,include_peatlands,Npreq,NPL,Nlay)

%Load File based on region
switch reg
    case 'Jylland'
        filename = 'jylland_tem_faalag_til_usikkerhed.mat';
    case 'Fyn'
        filename = 'fyn_tem_til_usikkerhed.mat';
    case 'Sjælland'
        filename = 'Sjaelland_tem_faalag_til_usikkerhed.mat';
end

in = load(['N:\PROJEKTER\N-retentionskortlægning 2022-24\Geostatistisk modellering\N-ret-mod_data_usikkerheder\geofys_usikkerheder\',filename]);

%Rearrange 'in' struct to 'PACEP' struct
switch reg
    case 'Jylland'
        TEM.InfoTable = in.Gmod_pos;
    case 'Fyn'
        TEM.InfoTable = in.fynpaceppos;
    case 'Sjælland'
        TEM.InfoTable = in.Gmod_pos;
end

TEM.Model_Depths = in.dkm_d_pos;
TEM.Model_Year = in.model_year_pos;
TEM.DepthInterfaceAboveModel = in.dkm_odvlayertop_pos;
TEM.ThickGeophysModel = in.dkm_odvthk_pos;

Npoints = numel(TEM.Model_Year);

%Determine duplicate points and calculate logical vector-index to ignore them
[~,id,~] = unique(table2array(TEM.InfoTable(:,["xutm_euref89_utm32","yutm_euref89_utm32"])),'rows');
logind = zeros(1,Npoints);
logind(id) = 1;

logind_dupe = logical(logind)';

%Determine if the geophysics is younger than the model. If so, filter it
%out.
ModelYear = TEM.Model_Year;
GeophysYear = table2array(TEM.InfoTable(:,"m_year"));

logind_date = GeophysYear <= ModelYear;

%Combine Logical Indices
logind = logical(logind_date.*logind_dupe);

%Update the struct
TEM.InfoTable = TEM.InfoTable(logind,:);
TEM.Model_Depths = TEM.Model_Depths(logind,:);
TEM.Model_Year = TEM.Model_Year(logind);
TEM.DepthInterfaceAboveModel = TEM.DepthInterfaceAboveModel(logind,:);
TEM.ThickGeophysModel = TEM.ThickGeophysModel(logind,:);


% 1) Determine grid size
nx = max(size(UTM_X));
ny = max(size(UTM_Y));
maxl = numel(TEM.Model_Depths(1,:));

% 2) Produce grid.
Grid = zeros(ny,nx,maxl);

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

    GS = 0.125; %Gradient for depth-to-uncertainty estimate

    % 5) Jeg skal have en funktion som beskriver hvordan certainty falder med
    % afstanden til boringen

%     cert_fun = @(dist,range,width,sill,d0) (dist<=width).*sill+(dist>width).*sill.*(exp(-(3*power((dist-width),2))./(power(range,2))));
    cert_fun = @(dist,range,width,sill,d0) (dist<=width).*sill.*(exp(-(3*power((dist-width),2))./(power(range,2))));

    % 6) Jeg skal besøge hver celle og lægge bidrag for de nærmeste boringer
    SearchRadius = 6; %in units of cells

    LLx = min(UTM_X);
    LLy = min(UTM_Y);

    %Do correction for ascii grid
    if mod(LLx,50) > 0

        LLx = LLx-50;
        LLy = LLy-50;

    end


    xs = table2array(TEM.InfoTable(:,"xutm_euref89_utm32"));
    ys = table2array(TEM.InfoTable(:,"yutm_euref89_utm32"));

        switch reg
        case 'Jylland'
            PalDepth = TEM.Model_Depths(:,Npreq-NPL+27);
            PalThick = TEM.Model_Depths(:,Npreq-NPL+28)-TEM.Model_Depths(:,Npreq-NPL+27);

        case 'Fyn'
            PalDepth = TEM.Model_Depths(:,Npreq-NPL-1);
            PalThick = TEM.Model_Depths(:,Npreq-NPL)-TEM.Model_Depths(:,Npreq-NPL-1);

        case 'Sjælland'
            PalDepth = TEM.Model_Depths(:,Npreq-NPL-1);
            PalThick = TEM.Model_Depths(:,Npreq-NPL)-TEM.Model_Depths(:,Npreq-NPL-1);

    end
    DOIs = table2array(TEM.InfoTable(:,"doilower"));
    doimean = nanmean(DOIs);
    
    f_doi = @(x) min(doimean,x+1E3*(PalThick<10));

    DOI_NaN_Filter = isnan(DOIs); 
    estimated_DOIs = f_doi(PalDepth);

    DOIs(DOI_NaN_Filter) = estimated_DOIs(DOI_NaN_Filter);


    maxl = Nlay-NPL;
    local_dist = NaN([size(terrain),maxl]);
    local_doi = zeros([size(terrain),maxl]);
    local_depth = NaN([size(terrain),maxl]);
    local_model_thick = NaN([size(terrain),maxl]);


    for s = 1:maxl

        MDepths = TEM.Model_Depths(:,s);
        MThicks = TEM.ThickGeophysModel(:,s);

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

complexity2 = complexities;
complexity2(complexity==0) = NaN;
rangemap = complexity2;
rangemap2 = complexity2;

%Calculate var0 map
var0map = (max(1.2,0.5*0.15*local_depth)+2).^2;
var0map(local_depth<10) = 10000;

rangemap(~isnan(complexity2)) = comp2range(complexity2(~isnan(complexity2))+1);
rangemap2(~isnan(complexity2)) = comp2rangePreq(complexity2(~isnan(complexity2))+1);
rangemap(:,:,Npreq-NPL:end) = rangemap2(:,:,Npreq-NPL:end);

kernelmap = max(local_depth,75);
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