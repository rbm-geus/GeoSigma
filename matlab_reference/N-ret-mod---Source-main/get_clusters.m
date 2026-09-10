function [classes,rangeout,sillout] = get_clusters(sill,range,mapsize,mask,ct,maxclustsize)
%% DESCRIPTION

%This script preprocesses range and sill, clusters them with a self
%organizing map (a type of neural network) and postprocesses the result.
%The output is the clustering classes.

%INPUT:
% sill: vector with sill values
% range: vector with range values
% mapsize: dimensions of the grid
% mask: vector indices for the position of sill and range values within the grid
% ct: number of elements to cluster

if nargin < 6
    maxclustsize = 5000;
    disp('No maxclustsize set: Default of 5000 is used!')
end

%% THE FUNCTION
%Normalize Ranges to the interval [0,1]
frange = max(range(:));
range = range./frange;

%Transform Sill (Variance) to Standard Deviation
sill = sqrt(abs(sill));

%Normalize Standard Deviations to the interval [0,1]
fsill = max(sill);
sill = sill./fsill;

%Determine which range and sill values are NOT NaN (when one is NaN the
%other should be too, so one index array is sufficient).
indzs = ~isnan(range);

%Prepare a 2D array for the clustering classes. Initially as NaN.
preclasses = NaN(size(indzs));

%Set number of nodes in sill and range for the self-organizing map.
if ct < 20000
    NN = 2;
else
    NN = 3;
end

%The number of clusters will be NN^2
Nclust = NN.^2;

%Set up the variable clus for holding the clustering variables. The array
%index "indzs" is used here in order to only cluster actual numbers.
clus = [range(indzs)' sill(indzs)]';

%Define clustering network with 250 training iterations for each dimension
netz = selforgmap(NN*[1 1],250);

%Train the self organizing map.
netz2 = train(netz,clus);

%Predict clusters with self organizing map
y = netz2(clus);
cla = vec2ind(y);

%Prepare post-processing by first filling clustering result into a 2D map
preclasses(indzs) = cla;

classes = zeros(mapsize);
classes(mask) = preclasses;

newclasses = zeros(mapsize);
cleanup_radius = 4;

nn3 = reshape(classes,mapsize);

SY = mapsize(1);
SX = mapsize(2);

ss4 = nn3*0;
rr4 = nn3*0;
ss4(mask) = sill;
rr4(mask) = range;

%Post-processing loop to remove very small or single-cell clusters
for i = 1:mapsize(1)
    for j = 1:mapsize(2)

        current_class = nn3(i,j);

        is_1 = max(1, i-cleanup_radius);
        is_2 = min(SY, i+cleanup_radius);
        js_1 = max(1, j-cleanup_radius);
        js_2 = min(SX, j+cleanup_radius);

        neighbor_classes = nn3(is_1:is_2,js_1:js_2);
        neighbor_classes = neighbor_classes(:);

        neighbor_classes(neighbor_classes == 0) = [];
        neighbor_classes(isnan(neighbor_classes)) = [];

        if numel(neighbor_classes) > 10
            [mode_class,Nmode]= mode(neighbor_classes(:));
            
            if Nmode > 0.2*numel(neighbor_classes)
                if mode_class ~= current_class
                    current_class = mode_class;
   
                end
            end
        end
        newclasses(i,j) = current_class;
    end
end

nn4 = newclasses;

mean_list = NaN(2,Nclust);

for i = 1:Nclust 
    ins = nn4==i;
    ins2 = ~isnan(rr4);
    ins3 = ~isnan(ss4);

mean_list(1,i) = mean(rr4(logical(ins.*ins2)));
mean_list(2,i) = mean(ss4(logical(ins.*ins3)));

end

cleanup_radius_0 = 32;
for i = 1:mapsize(1)
    for j = 1:mapsize(2)

        current_class = nn4(i,j);

        if isnan(current_class)
            vari = 0;
            cleanup_radius = cleanup_radius_0;

            while vari == 0
                cleanup_radius = cleanup_radius+1;

                is_1 = max(1, i-cleanup_radius);
                is_2 = min(SY, i+cleanup_radius);
                js_1 = max(1, j-cleanup_radius);
                js_2 = min(SX, j+cleanup_radius);

                neighbor_classes = nn4(is_1:is_2,js_1:js_2);
                neighbor_classes = neighbor_classes(:);

                neighbor_classes(neighbor_classes == 0) = [];
                neighbor_classes(isnan(neighbor_classes)) = [];
                vari = numel(neighbor_classes);
            end

            [mode_class,~]= mode(neighbor_classes(:));
            current_class = mode_class;
            
            rr4(i,j) = mean_list(1,current_class);
            ss4(i,j) = mean_list(2,current_class);


        end
        newclasses(i,j) = current_class;

    end
end

rangeout = frange*rr4(:);
sillout = fsill*ss4(:);

classes(mask) = newclasses(mask);

%This following line calls a function that checks all subclusters
%if they are larger than 10.000, if they are then they are split in two.
%For this reason the output classes might have more than 9 total clusters.

%The output clusters may be slightly larger than 10.000 but not by much.
%For example, if a subcluster has size 18000 and is split into two clusters 
%of size 10500 and 7500.

classes = post_process_clusters(classes,maxclustsize);

end