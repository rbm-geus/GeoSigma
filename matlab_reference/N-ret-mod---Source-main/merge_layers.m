function output_model = merge_layers(model,bg_model,mask)

%This script takes a model and merges it into a background model at
%locations specified by a mask. The default buffer distance is set to be
%5 grid cells (2 in each direction across the mask boundary)
csize = 3;

fun = @(d,cs) (d<cs).*(d./cs)+(d>=cs);
dfun = @(curx,cury,xlist,ylist) sqrt(power(curx-xlist,2)+power(cury-ylist,2));
mapsize = size(model);

sz = size(mask);

map = find_100_m_buffer(mask,'negative');
map2 = find_100_m_buffer(mask-map,'negative');
map3 = find_100_m_buffer(mask-map2,'negative');

map_out = mask(:)*0;
map_out(logical(mask)) = 1;
map_out(logical(map3)) = 0.75;
map_out(logical(map2)) = 0.50;
map_out(logical(map)) = 0.25;

map_out = reshape(map_out,sz);

weights = map_out;
weights0 = 1-map_out;

nonan_model = model;
nonan_model(isnan(model)) = 0;

output_model = (weights.*nonan_model)+(weights0.*bg_model);

end