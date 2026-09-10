function points = drawpoints(p0,surface,wells,Grid_X,Grid_Y,PL_wells,increase_prob)
%Wells er en N x M array med N boringer og M variabler. første variabel er
%UTM X og anden variabel er UTM Y
%p0 = 0.2;


LLx = min(Grid_X);
LLy = min(Grid_Y);

% Check if any wells exist for the current layer
if isempty(wells)
    Nwells = 0;
else
    Nwells = size(wells.xutm,1);
end
NPLwells = size(PL_wells,1);

maxp = numel(surface)*1;

lapl = abs(del2(surface));

fun3 = @(l,lim,c) (l<lim).*10.*(0.3+(0.3.*(tanh(2*l-3))));

lim = 12;

f1 = @(l,c,mp) 0.5*(tanh((l-mp)*c)+1);
f2 = @(l,lim,pun,scale) (scale+pun).*power(power((l-lim)/0.3,2)+1,-1);

fun3 = @(l,lim,c,pun,scale) (l<lim).*scale.*f1(l,c,0.20.*lim)+(1-(l<lim)).*(f2(l,lim,pun,scale)-pun);

PlapExtra = fun3(lapl,lim,0.8,0.25*p0,1.25*p0);
GridExtra = zeros(size(PlapExtra));
GridExtra(1:10:end,1:10:end) = 1;

p1 = p0.*ones(size(surface));
plap = p1+PlapExtra+GridExtra;

expp = sum(plap(:));

if expp > maxp
    factor = expp/maxp;
    plap = plap/factor;
end

if Nwells > 0
    wellxs = wells.xutm;
    wellys = wells.yutm;
    
    for i = 1:Nwells
    
        cur_x = wellxs(i);
        cur_y = wellys(i);
        
        indx = round((cur_x-LLx)/100)+1;
        indy = round((cur_y-LLy)/100)+1;
    
        plap(indy,indx) = plap(indy,indx)+1;
    end
end



if NPLwells > 0
    wellxs = PL_wells.X;
    wellys = PL_wells.Y;
    
    for i = 1:NPLwells
    
        cur_x = wellxs(i);
        cur_y = wellys(i);
        
        filt1 = cur_x > LLx & cur_y > LLy;
        filt2 = cur_x < Grid_X(end) & cur_y < Grid_Y(end);
        filt = filt1 & filt2;
        
        if filt
        indx = round((cur_x-(LLx))/100)+1;
        indy = round((cur_y-(LLy))/100)+1;
    
        plap(indy,indx) = plap(indy,indx)+1;
        end
    end

    plap = plap+0.15*increase_prob;
end

points = rand(size(surface))<plap;


end