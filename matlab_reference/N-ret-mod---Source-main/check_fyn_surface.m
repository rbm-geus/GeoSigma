clear all; close all; clc

% Check surfaces on fyn before input


laynames = get_layer_names('Fyn',3)
laynames{1} = 'dk3_topo100m';


figure(1); clf(1)
for i = 1:14
    [xvec,yvec,img_calib] = dtm_read_plot(['C:\N-ret-mod---Source\Flader\Fyn\Corrected_for_calibration\' laynames{i} '.asc']);
    [xvec,yvec,img_old] = dtm_read_plot(['C:\N-ret-mod---Source\Flader\Fyn\'  laynames{i} '.asc']);
    caxis([0,20])
    subplot(3,5,i)
    imagesc(xvec,yvec,img_calib-img_old)
    title(laynames{i})
end