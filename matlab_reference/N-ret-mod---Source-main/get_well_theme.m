function Grid_out = get_well_theme(UTM_X,UTM_Y,wellsp,wellsj,terrain,complexity)

% 1) Jeg bestemmer størrelsen af grid
nx = max(size(UTM_X));
ny = max(size(UTM_Y));

% 2) Jeg skal producere griddet og definere en minimum-certainty.
Grid = zeros(ny,nx);
Grid2 = Grid;

nwellspre = size(wellsp.xutm,1);
nwellsj = size(wellsj.xutm,1);

% 3) Jeg definerer effective range for udbredelsen af certainty. Første
% element er 150m og bruges når kompleksiteten er ukendt. de næste 4
% elementer svarer til kompleksitet 1-4, hvor 1 er lav og 4 er høj.

comp2range = [100 500 400 250 100];

if nwellspre > 0
    % 4) Jeg skal have et N x M array med N boringer og M variabler

    %%%
    % 1 ) X
    % 2 ) Y
    % 3 ) Z
    % 4 ) Quality
    % 5 ) Homogeneity
    %%%

    % 4) Jeg skal have en funktion som beskriver boringskvalitet/"certainty"

    %Her er b = [G S H N]

    %  - G er en gradient for forholdet mellem dybde og usikkerhed (afhænger af kvalitet
    %  - S er en vægt som oversætter heterogenitetsfaktoren til en usikkerhed
    %  - H er heterogenitetsfaktoren
    %  - N er Nul-variansen (minimumsvarians)
    
    %OBS I REMOVED A FACTOR 0.25 (0.5 inside brackets)
    var_fun = @(b,z,K) K^2+(b(1)*z).^2;

    GS = 0.5*[0.02 0.03 0.07 0.1]; %Gradients for depth-to-uncertainty estimate - nye værdier 27/6 2023
    GSj = 0.5*[0.025 0.035 0.075 0.105];  %Gradients for depth-to-uncertainty estimate - nye værdier 27/6 2023
    
    KS = [0.15 1 2 2.5];
    KSj = [0.4 1.5 2.5 3];

    % 5) Jeg skal have en funktion som beskriver hvordan certainty falder med
    % afstanden til boringen

    cert_fun = @(dist,range,sill) sill*(exp(-(3*power(dist,2))/(power(range,2))));

    %     cert_fun = @(dist,range,sill) sill*power((19*power(dist/range,2)+1),-1);

    % 6) Jeg skal besøge hver celle og lægge bidrag for de nærmeste boringer
    SearchRadius = 6;      % ændret til 6 d. 27/6 2023 - laves konsistent for alle usikkerhedstemaer

    Qualities = wellsp.bor_qual;
    JQualities = wellsj.bor_qual;

    wellxs = wellsp.xutm;
    jwellxs = wellsj.xutm;

    wellys = wellsp.yutm;
    jwellys = wellsj.yutm;

    wellzs = wellsp.z;
    jwellzs = wellsj.z_true;

    wells = zeros([size(wellxs,1) 6]);
    jwells = zeros([size(jwellxs,1) 6]);

    jwells(:,1) = jwellxs;
    wells(:,1) = wellxs;

    jwells(:,2) = jwellys;
    wells(:,2) = wellys;

    jwells(:,3) = jwellzs;
    wells(:,3) = wellzs;

    jwells(:,4) = JQualities;
    wells(:,4) = Qualities;

    for i = 1:size(wells,1)
        x = wells(i,1);
        y = wells(i,2);

        indx = round((x-min(UTM_X))/100)+1;
        indy = round((y-min(UTM_Y))/100)+1;

        terrain_kote = terrain(indy(end),indx(end));
        cur_complexity = complexity(indy(end),indx(end));
        wells(i,5) = terrain_kote;
        wells(i,6) = cur_complexity;

    end


    for i = 1:size(jwells,1)
        x = jwells(i,1);
        y = jwells(i,2);

      
        indx = round((x-min(UTM_X))/100)+1;
        indy = round((y-min(UTM_Y))/100)+1;

        terrain_kote = terrain(indy(end),indx(end));
        cur_complexity = complexity(indy(end),indx(end));

        jwells(i,5) = terrain_kote;
        jwells(i,6) = cur_complexity;

    end

    parfor i = 1:ny
            Ymin = max([1 i-SearchRadius]);
            Ymax = min([ny i+SearchRadius]);
            
            ymin = UTM_Y(Ymin);
            ymax = UTM_Y(Ymax);

            yv = UTM_Y(i);
            
            Yfilt1 = wells(:,2) < ymax;
            Yfilt2 = wells(:,2) > ymin;
            
        for j = 1:nx

            Xmin = max([1 j-SearchRadius]);
            Xmax = min([nx j+SearchRadius]);

            xmin = UTM_X(Xmin);
            xmax = UTM_X(Xmax);

            xv = UTM_X(j);

            Xfilt1 = wells(:,1) < xmax;
            Xfilt2 = wells(:,1) > xmin;

            filt = logical(Xfilt1.*Xfilt2.*Yfilt1.*Yfilt2);

            localwells = wells(filt,:);
            nwells = size(localwells,1);

            if nwells > 0
                
                %use max 3 nearest wells
                nloop = min(nwells,3);
                
                %Create array for sorting.
                sort_array = zeros(nwells,1);

                for s = 1:nwells
                    curwell = localwells(s,:);

                    curx = curwell(1);
                    cury = curwell(2);

                    curdist = sqrt(power(xv-curx,2)+power(yv-cury,2));

                    sort_array(s) = curdist;
                end
   
                %Now sort the array by distance and get the indices
                [~,near_well_indices] = sort(sort_array);

                %Choose the nloop nearest
                near_well_indices = near_well_indices(1:nloop)';

                for a = near_well_indices

                    curwell = localwells(a,:);

                    curx = curwell(1);
                    cury = curwell(2);
                    curd = max(curwell(5)-curwell(3),0);
                    curq = curwell(4);
                    curc = curwell(6);

                    distance = sqrt(power(xv-curx,2)+power(yv-cury,2));

                    B = GS(curq);
                    K = KS(curq);

                    if isnan(curc)
                        curc =  0;
                    end

                    Range = comp2range(curc+1);
                    var0 = var_fun(B,curd,K);
                    certainty =  cert_fun(distance,Range,1/var0);

                    Grid(i,j) = Grid(i,j)+certainty;
                end
            end
        end
    end
    parfor i = 1:ny
            Ymin = max([1 i-SearchRadius]);
            Ymax = min([ny i+SearchRadius]);
            
            ymin = UTM_Y(Ymin);
            ymax = UTM_Y(Ymax);

            Yfilt1 = jwells(:,2) < ymax;
            Yfilt2 = jwells(:,2) > ymin;

            yv = UTM_Y(i);

        for j = 1:nx

            Xmin = max([1 j-SearchRadius]);
            Xmax = min([nx j+SearchRadius]);

            xmin = UTM_X(Xmin);
            xmax = UTM_X(Xmax);

            xv = UTM_X(j);

            Xfilt1 = jwells(:,1) < xmax;
            Xfilt2 = jwells(:,1) > xmin;

            filt = logical(Xfilt1.*Xfilt2.*Yfilt1.*Yfilt2);
            localwells = jwells(filt,:);

            nwells = size(localwells,1);

            if nwells > 0
                
                %use max 3 nearest wells
                nloop = min(nwells,3);
                
                %Create array for sorting.
                sort_array = zeros(nwells,1);

                for s = 1:nwells
                    curwell = localwells(s,:);

                    curx = curwell(1);
                    cury = curwell(2);

                    curdist = sqrt(power(xv-curx,2)+power(yv-cury,2));

                    sort_array(s) = curdist;
                end

                %Now sort the array by distance and get the indices
                [~,near_well_indices] = sort(sort_array);

                %Choose the nloop nearest
                near_well_indices = near_well_indices(1:nloop)';

                for a = near_well_indices

                    curwell = localwells(a,:);

                    curx = curwell(1);
                    cury = curwell(2);
                    curd = max(curwell(5)-curwell(3),0);
                    curq = curwell(4);
                    curc = curwell(6);

                    distance = sqrt(power(xv-curx,2)+power(yv-cury,2));

                    B = [GSj(curq) 1 0 0];
                    K = KSj(curq);

                    if isnan(curc)
                        curc =  0;
                    end

                    Range = comp2range(curc+1);
                    var0 = var_fun(B,curd,K);
                    certainty =  cert_fun(distance,Range,1/var0);

                    Grid2(i,j) = Grid2(i,j)+certainty;
                    if isnan(Grid2(i,j))
                        keyboard
                    end
                end
            end
        end
    end


    Grid_out = 1./(Grid+Grid2);



end

    
end