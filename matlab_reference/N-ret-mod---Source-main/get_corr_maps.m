function Grid = get_corr_maps(UTM_X,UTM_Y,wellsp)
% 1) Jeg bestemmer størrelsen af grid
nx = max(size(UTM_X));
ny = max(size(UTM_Y));

% 2) Jeg skal producere griddet hvor alle celler er "slukket" (0)
Grid = NaN(ny,nx);

nwells = size(wellsp.xutm,1);
zt_logs = wellsp.zero_layerthk;
LLx = min(UTM_X);
LLy = min(UTM_Y);

if nwells > 0
    %Take only wells with a defined 0-thicknessm
    zt_inds = zt_logs == 1;

    zwells_x = wellsp.xutm(zt_inds); %Zero-thickness well x coord
    zwells_y = wellsp.yutm(zt_inds); %Zero-thickness well y coord
    zwells_q = wellsp.bor_qual; %Zero-thickness well quality rating
    
    n_zwells = numel(zwells_x);

    for a = 1:n_zwells

        curx = zwells_x(a);
        cury = zwells_y(a);
        curq = zwells_q(a);

        %Now determine which grid cell this is in!
        indx = ceil((curx-(LLx))/100);
        indy = ceil((cury-(LLy))/100);

        Grid(indy,indx) = curq;
    end

end


end