function [Grid,s_cords] = get_RESLOG_theme(UTM_X,UTM_Y,terrain,complexity,reg,include_peatlands,Npreq,NPL,Nlay)

%Load File based on region
switch reg
    case 'Jylland'
        filename = 'jylland_resistivitetslog_til_usikkerhed.mat';
    case 'Fyn'
        filename = 'fyn_resistivitetslog_til_usikkerhed.mat';
    case 'Sjælland'
        filename = 'Sjaelland_resistivitetslog_til_usikkerhed.mat';
end

in = load(['N:\PROJEKTER\N-retentionskortlægning 2022-24\Geostatistisk modellering\N-ret-mod_data_usikkerheder\geofys_usikkerheder\',filename]);

%Rearrange 'in' struct to 'MEP' struct
switch reg
    case 'Jylland'
        MEP.InfoTable = in.Gmod_pos;
    case 'Fyn'
        MEP.InfoTable = in.Gmod_pos;
    case 'Sjælland'
        MEP.InfoTable = in.Gmod_pos;
end

MEP.Model_Depths = in.dkm_d_pos;
MEP.Model_Year = in.model_year_pos;
MEP.DepthInterfaceAboveModel = in.dkm_odvlayertop_pos;
MEP.ThickGeophysModel = in.dkm_odvthk_pos;
Npoints = numel(MEP.Model_Year);

%Determine duplicate points and calculate logical vector-index to ignore them
[~,id,~] = unique(table2array(MEP.InfoTable(:,["xutm32euref89","yutm32euref89"])),'rows');
logind = zeros(1,Npoints);
logind(id) = 1;

logind_dupe = logical(logind)';

%Determine if the geophysics is younger than the model. If so, filter it
%out.
ModelYear = MEP.Model_Year;
GeophysYear = table2array(MEP.InfoTable(:,"d_year"));

logind_date = GeophysYear <= ModelYear;

%Combine Logical Indices
logind = logical(logind_date.*logind_dupe);

%Update the struct
MEP.InfoTable = MEP.InfoTable(logind,:);
MEP.Model_Depths = MEP.Model_Depths(logind,:);
MEP.Model_Year = MEP.Model_Year(logind);
MEP.DepthInterfaceAboveModel = MEP.DepthInterfaceAboveModel(logind,:);
MEP.ThickGeophysModel = MEP.ThickGeophysModel(logind,:);

% 1) Determine grid size
nx = max(size(UTM_X));
ny = max(size(UTM_Y));
maxl = numel(MEP.Model_Depths(1,:));

% 2) Produce grid.
Grid = zeros(ny,nx,maxl);

% 3) Jeg definerer effective range for udbredelsen af certainty. Første
% element er 50m og bruges når kompleksiteten er ukendt. de næste 4
% elementer svarer til kompleksitet 1-4, hvor 1 er lav og 4 er høj.

comp2range = [100 500 400 250 100];
comp2rangePreq = [500 500 500 500 500];
NP = sum(logind);

if NP > 0

%     cert_fun = @(dist,range,width,sill) (dist<=width).*sill+(dist>width).*sill.*(exp(-(3*power((dist-width),2))./(power(range,2))));
    cert_fun = @(dist,range,width,sill) (dist<=width).*sill.*(exp(-(3*power((dist-width),2))./(power(range,2))));   %rettet
    % 6) Jeg skal besøge hver celle og lægge bidrag for de nærmeste boringer
    SearchRadius = 6; %in units of cells    %rettet

    LLx = min(UTM_X);
    LLy = min(UTM_Y);

    xs = table2array(MEP.InfoTable(:,"xutm32euref89"));
    ys = table2array(MEP.InfoTable(:,"yutm32euref89"));
    DOIs = table2array(MEP.InfoTable(:,"maxdepth"));
    
    maxl = Nlay-NPL;
    local_dist = NaN([size(terrain),maxl]);
    local_doi = zeros([size(terrain),maxl]);
    local_depth = NaN([size(terrain),maxl]);
    local_model_thick = NaN([size(terrain),maxl]);

    complexities = NaN([size(terrain),maxl]);           %indsæt

    for s = 1:maxl

        MDepths = MEP.Model_Depths(:,s);
        MThicks = MEP.ThickGeophysModel(:,s);    %indsat
                    
        complexities(:,:,s) = complexity;   %indsæt

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
                    local_thicks = MThicks(filt)';      %indsat
                    
                    distances = sqrt(power(xv-local_xs,2)+power(yv-local_ys,2));
                    dist = min(distances);

                    filt_mindist = distances == dist;

                    local_dist(i,j,s) = dist;
                    
                    local_depth(i,j,s) = mean(local_depths(filt_mindist));
                    local_model_thick(i,j,s) = min(local_thicks(filt_mindist));   %indsat

                end
            end
        end
   end
end



complexity2 = complexity;
complexity2(complexity==0) = NaN;
rangemap = complexity2;
rangemap2 = complexity2;

%Calculate var0 map
var0map = 0.25+(0.5*0.09*local_depth).^2;    %rettet
var0map(local_depth<10)=100000;  %indsat

rangemap(~isnan(complexity2)) = comp2range(complexity2(~isnan(complexity2))+1);
rangemap2(~isnan(complexity2)) = comp2rangePreq(complexity2(~isnan(complexity2))+1);
rangemap(:,:,Npreq:end) = rangemap2(:,:,Npreq:end);

kernelmap = local_depth*0+150;           %rettet
certmap = cert_fun(local_dist,rangemap,kernelmap,1./var0map);

Grid = 1./certmap;
Grid(isnan(Grid)) = 100000;

doi_filter = local_model_thick < local_depth;  %rettet
Grid(doi_filter) = 100000;

if include_peatlands == 1
    New_Grid = 1E4*ones(size(Grid,1),size(Grid,2),size(Grid,3)+NPL);
    New_Grid(:,:,(NPL+1):end) = Grid;
    Grid = New_Grid;
end

s_cords.xs = xs;
s_cords.ys = ys;

end