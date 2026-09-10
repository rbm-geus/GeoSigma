function map = get_minimum_map(terrain,model)
Nlay = numel(model);
[NY,NX] = size(terrain);

depth_model = zeros(NY,NX,numel(model));
for i = 1:Nlay
    curmodel = model{i}.LayerBottom;
    depth_model(:,:,i) = terrain-curmodel;   
end

map = power(0.01*depth_model,2);

end