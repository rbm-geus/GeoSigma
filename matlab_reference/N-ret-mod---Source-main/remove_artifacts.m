function [newbottom,mask_out] = remove_artifacts(bottom_in,top,xs,ys,points,doplot,itnum,region)
bottom = bottom_in;
t1 = (top-bottom_in);

px = table2array(points(:,1));
py = table2array(points(:,2));
pz = table2array(points(:,3));

ptype = table2array(points(:,4));
pID = table2array(points(:,5));

sz = size(top);
npoints = size(points,1);

LLx = min(xs);
LLy = min(ys);

%Separate negative points and positive points
posind = ptype == 0;
negind = ptype == 1;

pos_x = px(posind);
pos_y = py(posind);
pos_z = pz(posind);

pos_ID = pID(posind);

neg_x = px(negind);
neg_y = py(negind);
neg_z = pz(negind);

neg_ID = pID(negind);

%Produce a map of distance to points
itc = itnum == 1;

switch itc
    case 1
        SearchRadius = 1000;
    case 0
        SearchRadius = 2500;
end

SR1 = SearchRadius/100;
pos_map = zeros(sz);
neg_map = zeros(sz);

for i = 1:sz(1)
    Ymin = max([1 i-SR1]);
    Ymax = min([sz(1) i+SR1]);

    ymin = ys(Ymin);
    ymax = ys(Ymax);

    yv = ys(i);

    %Determine which negative and positive point y values fall within the sliding
    %window
    filt_pos1 = pos_y < ymax & pos_y > ymin;
    filt_neg1 = neg_y < ymax & neg_y > ymin;

    for j = 1:sz(2)

        Xmin = max([1 j-SR1]);
        Xmax = min([sz(2) j+SR1]);

        xmin = xs(Xmin);
        xmax = xs(Xmax);

        xv = xs(j);

        filt_pos2 = pos_x < xmax & pos_x > xmin;
        filt_neg2 = neg_x < xmax & neg_x > xmin;

        filt_pos = filt_pos1 & filt_pos2;
        filt_neg = filt_neg1 & filt_neg2;

        cur_n_pos = sum(filt_pos);
        cur_n_neg = sum(filt_neg);

        if cur_n_pos > 0
            local_xs = pos_x(filt_pos);
            local_ys = pos_y(filt_pos);
            local_ID = pos_ID(filt_pos);

            distances = sqrt(power(xv-local_xs,2)+power(yv-local_ys,2));
            dist_pos = min(distances);
            pos_map(i,j) = dist_pos;
        else

            %Outside search radius
            pos_map(i,j) = NaN;
        end

        if cur_n_neg > 0
            local_xs = neg_x(filt_neg);
            local_ys = neg_y(filt_neg);
            local_ID = neg_ID(filt_neg);

            distances = sqrt(power(xv-local_xs,2)+power(yv-local_ys,2));
            dist_neg = min(distances);
            neg_map(i,j) = dist_neg;
        else

            %Outside search radius
            neg_map(i,j) = NaN;
        end

    end
end

comb_map = zeros(sz(1),sz(2),2);
comb_map(:,:,1) = pos_map;
comb_map(:,:,2) = neg_map;

mask_pos_neg = nanmin(comb_map,[],3)==comb_map(:,:,1) & nanmin(comb_map,[],3)<SearchRadius;

smooth_mask = zeros(size(mask_pos_neg));
sy = size(smooth_mask,1);
sx = size(smooth_mask,2);
parfor ii = 1:sy
    Ymin = max([1 ii-25]);
    Ymax = min([sz(1) ii+25]);

    ymin = ys(Ymin);
    ymax = ys(Ymax);

    for jj = 1:sx

        Xmin = max([1 jj-50]);
        Xmax = min([sz(2) jj+50]);
        xmin = xs(Xmin);
        xmax = xs(Xmax);

        local_window = bottom(Ymin:Ymax,Xmin:Xmax);
        mean_value = mean(local_window(:));

        smooth_mask(ii,jj) = mean_value;
    end
end

smooth_mask2 = (smooth_mask+1)>bottom;
[dX,dY] = gradient(top);
grad_magnitude = sqrt(power(dX,2)+power(dY,2));

grad_mask = grad_magnitude < 2;
SearchRadius = 100;
SR = SearchRadius/100;

% imagesc(xs,ys,mask_pos_neg)
% hold on
% scatter(pos_x,pos_y,'r','filled')
% scatter(neg_x,neg_y,'b','filled')
% plot_dk();
% set(gca,'ydir','normal')

%Calculate Thickness Map
Thick = top-bottom;

%Only allow positive thickness
Thick = (abs(Thick)+Thick)/2;

%Make a binary logical map of thick > 0
LogMap = Thick>0;
LogMap2 = Thick<0.1;

LogMap3 = logical(LogMap.*LogMap2);

%Remove all the thicknesses between 0 and 0.1
bottom(LogMap3) = top(LogMap3);

%Now find the remaining thicknesses
LogMap4=Thick>0.1;
Subs = bwlabel(LogMap4,4);


%Extract all interpretation points of type 0 and with a snap ID
npos = numel(pos_ID);
pmap = zeros(sz);

for i = 1:npos
    switch region
        case 'Vendsyssel'
            curid = pos_ID(1);
            snap_well(i) = curid == 1;
        case 'Other'
            curid = pos_ID(i);
            snap_well(i) = numel(curid)>1;
    end

    CX = round((pos_x(i)-LLx)/100)+1;
    CY = round((pos_y(i)-LLy)/100)+1;

    if (CX > 0) & (CY > 0)
        if CX < sz(2) & CY < sz(1)
            pmap(CY,CX) = 1;
        end
    end
end

hmask = zeros(size(pmap));
Nsub = max(Subs(:));
for i = 1:Nsub

    inds = Subs==i;
    summap = (inds+pmap)>1;
    Filt = max(summap(:))<1;

    if Filt
        bottom(inds) = top(inds);
    else

        %Determine which the terrain kote for all interpretation points in
        %the body.

        %Top-Koter (Top Heights) determine max kote
        TKs = top(summap);

        %Set the maximum allowed bounding box size
        max_kote = median(TKs)+2;
        
       
        MaxKoteMask = top < max_kote;

        FILT = MaxKoteMask.*inds;
        
        
        F2 = bwlabel(FILT,4);
        
        switch region
            case 'Other'
        for ii = 1:max(F2(:))

            indii = F2 == ii;
            test = (indii+pmap)>1;
            test2 = max(test(:)) == 0;

            if test2
                FILT(indii) = 0;
            end
        end

        newfilt = logical(~FILT+inds-1);
        newfilt(find(ys == 6306050):find(ys == 6330050),find(xs == 563050):find(xs == 590050)) = 0; % HOTFIX for Eastern part of Himmerland
        bottom(newfilt) = top(newfilt);
            case 'Vendsyssel'
                
        end

        %Determine the max kote of the bottom of the body
        max_height = max(bottom(logical(FILT)));
        %Make a mask based on the max kote
        tempmask = bottom < (max_height+1);
        %Make a vicinity mask
        vicinmask = calculate_buffer_zones(inds,size(pmap,2),size(pmap,1));
        vm = vicinmask>0;
        vicinmask = calculate_buffer_zones(vm,size(pmap,2),size(pmap,1));
        vm = vicinmask>0;
        vicinmask = calculate_buffer_zones(vm,size(pmap,2),size(pmap,1));
        vm = vicinmask>0;
        vicinmask = calculate_buffer_zones(vm,size(pmap,2),size(pmap,1));
        vm = vicinmask>0;

        ve = itnum > 1;
        switch ve
            case 0

            case 1
                vicinmask = calculate_buffer_zones(vm,size(pmap,2),size(pmap,1));
                vm = vicinmask>0;
                vicinmask = calculate_buffer_zones(vm,size(pmap,2),size(pmap,1));
                vm = vicinmask>0;
                vicinmask = calculate_buffer_zones(vm,size(pmap,2),size(pmap,1));
                vm = vicinmask>0;
                vicinmask = calculate_buffer_zones(vm,size(pmap,2),size(pmap,1));
                vm = vicinmask>0;
                vicinmask = calculate_buffer_zones(vm,size(pmap,2),size(pmap,1));
                vm = vicinmask>0;
        end

        combmas = logical(vm.*tempmask);
        hmask(combmas) = hmask(combmas)+1;
    end

end
mask_hill = hmask > 0;

%snap_well x and y coords
snap_x = pos_x(snap_well);
snap_y = pos_y(snap_well);
snap_z = pos_z(snap_well);
% snap_z = pos_z(snap_well);

snap_map = zeros(sz);
snap_d_map = zeros(sz);

loga_map = zeros(sz);

for i = 1:sz(1)
    Ymin = max([1 i-SR]);
    Ymax = min([sz(1) i+SR]);

    ymin = ys(Ymin);
    ymax = ys(Ymax);

    yv = ys(i);

    %Determine which negative and positive point y values fall within the sliding
    %window
    filt_snap1 = snap_y < ymax & snap_y > ymin;

    for j = 1:sz(2)

        Xmin = max([1 j-SR]);
        Xmax = min([sz(2) j+SR]);

        xmin = xs(Xmin);
        xmax = xs(Xmax);

        xv = xs(j);

        filt_snap2 = snap_x < xmax & snap_x > xmin;

        filt_snap = filt_snap1 & filt_snap2;

        cur_n = sum(filt_snap);

        if cur_n > 0
            local_xs = snap_x(filt_snap);
            local_ys = snap_y(filt_snap);
            local_zs = snap_z(filt_snap);

            distances = sqrt(power(xv-local_xs,2)+power(yv-local_ys,2));
            sumdi = sum(1./distances);

            local_weights = (1./distances)/sumdi;

            dist_snap = min(distances);
            snap_map(i,j) = dist_snap;
            snap_d_map(i,j) = sum(local_zs.*local_weights);

            loga_map(i,j) = 1;
        else

            %Outside search radius
            snap_map(i,j) = NaN;
            snap_d_map(i,j) = NaN;
        end

    end
end

loga_map = logical(loga_map);
dmax = 70.7;

snap_map2 = (snap_map<dmax).*snap_map;
snap_map2(isnan(snap_map2)) = dmax;
snap_map2(snap_map2==0) = dmax;

weight_map = 1-(snap_map2/dmax);
weight_map(isnan(weight_map)) = 0;

snap_d_map(isnan(snap_d_map)) = 0;

bottom2 = (weight_map.*snap_d_map)+((1-weight_map).*bottom);

if doplot == 1
    ear = 1243;
    xv = xs(ear);

    Sind = abs(snap_x-xv)<dmax;
    snap_y_profile = snap_y(Sind);
    snap_d_profile = snap_z(Sind);

    plot(snap_y_profile,snap_d_profile,'.r','DisplayName','Point with Snap ID','MarkerSize',15)
    hold on
    plot(ys,bottom(:,ear),'-r','DisplayName','Surface')
    plot(ys,bottom2(:,ear),'--b','DisplayName','Inverse Distance Corrected Surface')
    grid on
    xlabel('Position UTM Y [m]','FontSize',16)
    ylabel('Depth [m]','FontSize',16)
    legend

    ear2 = 2043;
    yv = ys(ear2);

    Sind2 = abs(snap_y-yv)<dmax;
    snap_x_profile = snap_x(Sind2);
    snap_d_profile = snap_z(Sind2);

    plot(snap_x_profile,snap_d_profile,'.r','DisplayName','Point with Snap ID')
    hold on
    plot(xs,bottom(ear2,:),'-r','DisplayName','Surface')
    plot(xs,bottom2(ear2,:),'--b','DisplayName','Inverse Distance Corrected Surface')
    grid on
    xlabel('Position UTM X [m]','FontSize',16)
    ylabel('Depth [m]','FontSize',16)
    legend
end

tfinal = top-bottom;
mask3 = (logical(mask_hill).*mask_pos_neg);
mask4 = mask3.*smooth_mask2.*grad_mask;
mask_out = (mask4+(tfinal>0))>0;

newbottom =  bottom;
end