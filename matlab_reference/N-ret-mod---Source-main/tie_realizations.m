function new_real2 = tie_realizations(real1,real2,ties)
%Real 2 is the lower layer
%Real 1 is the upper layer

%Real 1 will remain unchanged
%Real 2 will be forced to terminate on Real 1 at ties

nx = size(real1,2);
ny = size(real2,1);

weightfun =  @(dx) exp(-dx.^2/3);
weights = zeros(size(real1));

combine_function = @(weight,v1,v2) (1-weight)*v2+(weight)*v1;

for i = 1:ny
    for j = 1:nx

        %Look around with radius 4
        sr = 4;

        locx = [max(1,j-sr),min(nx,j+sr)];
        locy = [max(1,i-sr),min(ny,i+sr)];

        loc_ties = ties(locy(1):locy(2),locx(1):locx(2));
        test = logical(sum(loc_ties(:)));

        if test

            %Compute distances to all local ties
            [lx,ly] = find(loc_ties);
            if j < sr+1
                mx = size(loc_ties,2)-sr;
            else
                mx = sr+1;
            end

            if i < sr+1
                my = size(loc_ties,1)-sr;
            else
                my = sr+1;
            end

            dists = sqrt(power(lx-mx,2)+power(ly-my,2));
            mindist = min(dists);

            new_weight = weightfun(mindist);
            weights(i,j) = new_weight;
        end

    end
end

weights = weights.*(weights>0);

new_real2 = real2;

for i = 1:ny
    for j = 1:nx

        cur_weight = weights(i,j);
        cur_1 = real1(i,j);
        cur_2 = real2(i,j);

        tes = ~isnan(cur_1+cur_2);
        if tes
            new_real2(i,j) = combine_function(cur_weight,cur_1,cur_2);
        else
%             keyboard
        end
    end
end

end