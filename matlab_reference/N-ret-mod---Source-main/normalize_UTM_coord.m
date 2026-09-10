function [X_NORM,Y_NORM] = normalize_UTM_coord(XUTM,YUTM,dx,dy,minX,minY)

if nargin < 5
    minX = min(XUTM(:));
    minY = min(YUTM(:));
end

% minX = double(minX);
% minY = double(minY);

X_NORM = (XUTM-minX)/dx+1;
Y_NORM = (YUTM-minY)/dy+1;