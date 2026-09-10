function [m_org] =  insert_dk9_mask(x_org,y_org,m_org)

[xutm,yutm,m_grid_dk9]= dtm_read_plot(['C:\N-ret-mod---Source\Flader\Jylland\','dk9_grid.asc']);

xmin = find(x_org==min([xutm]));
ymin = find(y_org==min([yutm]));

xmax = find(x_org==max([xutm]));
ymax = find(y_org==max([yutm]));



m_org(ymin:1:ymax,xmin:1:xmax) = m_org(ymin:1:ymax,xmin:1:xmax)+m_grid_dk9;
m_org = m_org>0;


