function [Grid,s_cords] = get_REFSEIS_theme(UTM_X,UTM_Y,terrain,complexity,reg,include_peatlands,Npreq,NPL,Nlay)

%Load File based on region
switch reg
    case 'Jylland'
        filename = 'jylland_seismik_til_usikkerhed';
    case 'Fyn'
%         filename = 'fyn_seismik_til_usikkerhed';
    case 'Sjælland'
        filename = 'Sjaelland_seismik_til_usikkerhed.mat';
        
end

local_dir = pwd;
in = load([local_dir,'\Geofysik\',filename]);

%Rearrange 'in' struct to 'MEP' struct
switch reg
    case 'Jylland'
        SEIS.InfoTable = in.Gmod_pos;
    case 'Fyn'
        SEIS.InfoTable = in.Gmod_pos;
    case 'Sjælland'
        SEIS.InfoTable = in.Gmod_pos;
end

SEIS.Model_Depths = in.dkm_d_pos;
SEIS.Model_Year = in.model_year_pos;
SEIS.DepthInterfaceAboveModel = in.dkm_odvlayertop_pos;
SEIS.ThickGeophysModel = in.dkm_odvthk_pos;
layvec = find(SEIS.ThickGeophysModel(1,1:(Nlay-NPL))>1000);

Npoints = numel(SEIS.Model_Year);

%Determine duplicate points and calculate logical vector-index to ignore them
[~,id,~] = unique(table2array(SEIS.InfoTable(:,["xutm_euref89_utm32","yutm_euref89_utm32"])),'rows');
logind = zeros(1,Npoints);
logind(id) = 1;

logind_dupe = logical(logind)';

%Determine if the geophysics is younger than the model. If so, filter it
%out.
ModelYear = SEIS.Model_Year;
GeophysYear = table2array(SEIS.InfoTable(:,"d_year"));

logind_date = GeophysYear <= ModelYear;

%Combine Logical Indices
logind = logical(logind_date.*logind_dupe);

%Update the struct
SEIS.InfoTable = SEIS.InfoTable(logind,:);
SEIS.Model_Depths = SEIS.Model_Depths(logind,:);
SEIS.Model_Year = SEIS.Model_Year(logind);
SEIS.DepthInterfaceAboveModel = SEIS.DepthInterfaceAboveModel(logind,:);
SEIS.ThickGeophysModel = SEIS.ThickGeophysModel(logind,:);

% 1) Determine grid size
nx = max(size(UTM_X));
ny = max(size(UTM_Y));
maxl = numel(SEIS.Model_Depths(1,:));

% % 2) Produce grid.
% Grid = zeros(ny,nx,maxl);

% 3) Jeg definerer effective range for udbredelsen af certainty. Første
% element er 50m og bruges når kompleksiteten er ukendt. de næste 4
% elementer svarer til kompleksitet 1-4, hvor 1 er lav og 4 er høj.

comp2range = [100 500 400 250 100];
comp2rangePreq = [500 500 500 500 500];
NP = sum(logind);

if NP > 0

%     cert_fun = @(dist,range,width,sill) (dist<=width).*sill+(dist>width).*sill.*power((19.*power((dist-width)./range,2)+1),-1);
    cert_fun = @(dist,range,width,sill) (dist<=width).*sill+(dist>width).*sill.*(exp(-(3*power((dist-width),2))./(power(range,2))));
    % 6) Jeg skal besøge hver celle og lægge bidrag for de nærmeste boringer
    SearchRadius = 8; %in units of cells

    LLx = min(UTM_X);
    LLy = min(UTM_Y);

    xs = table2array(SEIS.InfoTable(:,"xutm_euref89_utm32"));
    ys = table2array(SEIS.InfoTable(:,"yutm_euref89_utm32"));
    maxl = Nlay-NPL;
    local_dist = NaN([size(terrain),maxl]);
    complexities = NaN([size(terrain),maxl]);

    for s = layvec

        MThicks = SEIS.ThickGeophysModel(:,s);
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
                    local_thicks = MThicks(filt)';

                    distances = sqrt(power(xv-local_xs,2)+power(yv-local_ys,2));
                    dist = min(distances);

                    filt_mindist = distances == dist;

                    local_dist(i,j,s) = dist;
                    local_model_thick(i,j,s) = max(local_thicks(filt_mindist));
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
var0map = ones(size(local_dist)).*power(0.5*15,2);

rangemap(~isnan(complexity2)) = comp2range(complexity2(~isnan(complexity2))+1);
rangemap2(~isnan(complexity2)) = comp2rangePreq(complexity2(~isnan(complexity2))+1);
rangemap(:,:,Npreq-NPL:end) = rangemap2(:,:,Npreq-NPL:end);

kernelmap = local_dist*0+1;
certmap = cert_fun(local_dist,rangemap,kernelmap,1./var0map);

Grid = 1./certmap;
Grid(isnan(Grid)) = 100000;

toi_filter = local_model_thick < 100;
Grid(toi_filter) = 100000;        

if include_peatlands == 1
    New_Grid = 1E4*ones(size(Grid,1),size(Grid,2),size(Grid,3)+NPL);
    New_Grid(:,:,(NPL+1):end) = Grid;
    Grid = New_Grid;
end

s_cords.xs = xs;
s_cords.ys = ys;

end