function [ranges, sills, variances, means, drawnpoints,ct] = get_interpolated_minigrids_parallel(Layer,mask,buffered_grid_size,wells,layn,Grid_X,Grid_Y,tmask,smask,NPL,Snap_PL,layname,probmask)
mini_grid_size = 33;
halfg = (mini_grid_size-1)/2;
cmask = tmask+smask;

margin_size = ((mini_grid_size-1)/2);

%Feed minigrids to ML Model
disp('Feeding MiniGrids to Model')

ML_model = load('FinalNetworkSillRangeBothNoNormalization.mat');
rangemod = ML_model.net;

[Mx, My] = meshgrid(Grid_X,Grid_Y);

M2 = reshape(mask,size(Layer));
M3 = reshape(tmask,size(Layer));

M3vec = M3==0;

zer = zeros(buffered_grid_size);

Gg = M2;

tm = NaN*zer;

tmap = Layer;
tmap(M3vec) = NaN;

tm = tmap;

mask = Gg(:);

Cell_Size = 100; %meters
G_grid = zeros(buffered_grid_size);
G_grid = Layer;


% Well data
well = wells{layn(1)};
if layn(1)<(NPL+1)
    Snap_well = Snap_PL{layn(1)};
    p0 = 0.5;
else
    Snap_well = [];
    p0 = 0.2;
end

%%

% Choose Random Points
p = drawpoints(p0,G_grid,well,Grid_X,Grid_Y,Snap_well,probmask);

p3 = p;
[psj,psi] = find(p3 == 1);

%Since p3 contains only ones in cells where a point is drawn the
%vector-form of the matrix is a linear logical index
drawnpoints = p3(:);
sz = size(G_grid);

pointlog = find(mask);
[pointsy, pointsx] = ind2sub(sz,pointlog);

disp('Choosing points for Sub-grids')

datadensity = zeros(size(p3));

ct = size(pointlog,1);

xg = linspace(-((margin_size-1)*Cell_Size),((margin_size-1)*Cell_Size),mini_grid_size-2);
yg = linspace(-((margin_size-1)*Cell_Size),((margin_size-1)*Cell_Size),mini_grid_size-2);

[XG,YG] = meshgrid(xg,yg);
ranges = NaN(1,ct);
variances = NaN(1,ct);
means = NaN(1,ct);
sills = NaN(1,ct);

disp(['Number of Elements to Process: ',num2str(ct),' in Layer ',num2str(layn(1))])
disp(['Processing Ranges for Layer ',num2str(layn(1)),' out of ',num2str(layn(2))])
GridEls = power(mini_grid_size,2);

% wb = parwaitbar(ct,'WaitMessage','Predicting Ranges');

tic
parfor i = 1:ct
    x = pointsx(i);
    y = pointsy(i);

    %Local Grid Coords
    x_min = max([1 x-margin_size]);
    x_max = min([sz(2) x+margin_size]);
    y_min = max([1 y-margin_size]);
    y_max = min([sz(1) y+margin_size]);


    mini_grid = tm(y_min:y_max,x_min:x_max);
    
    %Check if a full 33x33 grid is defined
    if numel(mini_grid) == GridEls
        test = (sum(isnan((mini_grid(:)))))/numel(mini_grid);


        %Find points with xy-values in local grid
        k1 = (x_min<=psi).*(psi<=x_max);
        k2 = (y_min<=psj).*(psj<=y_max);

        PS = logical(k1.*k2);

        points = [psj(PS) psi(PS)];

        points(:,2) = x-psi(PS)+margin_size+1;
        points(:,1) = y-psj(PS)+margin_size+1;

        linp = sub2ind(mini_grid_size*[1 1],points(:,1),points(:,2));

        pointvalues = mini_grid(linp);
        pointvalues(isnan(pointvalues)) = mean(pointvalues(~isnan(pointvalues)));

        means(i) = mean(pointvalues);
        if layn(1) < (NPL+1)
            test = 0;
        end

        if test < 0.1
            variances(i) = var(pointvalues);

            %Count Data Points Within Sliding Window
            depths = pointvalues-means(i);
            depths_new = depths(:);

            xs = 100*(points(:,1)-(margin_size+1));
            ys = 100*(points(:,2)-(margin_size+1));

            F1 = scatteredInterpolant(xs,ys,depths_new,'natural','linear');
            TG = F1(XG,YG);

            val = predict(rangemod, TG);

            sills(i) = var(depths_new);
            ranges(i) = val(2);


        end

    else
    
    means(i) = mean(mini_grid(~isnan(mini_grid)));

    end




end
toc

sills(sills<0) = 1;
ranges(ranges<0) = 100;

end