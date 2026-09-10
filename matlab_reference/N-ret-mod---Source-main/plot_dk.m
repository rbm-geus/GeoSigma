function [x,y]=plot_dk(type,plotting,linewidth)

if nargin < 3
    linewidth = 2;
end

if nargin < 2
    plotting = 1;
end

if nargin < 1
    type = 0;
end


switch type
    case 0
        filename_1 = 'DNK_adm0.shx';
    case 1
        filename_1 = 'JyllandNew.shx';
    case 2
        filename_1 = 'FynNew2.shx';
    case 3
        filename_1 = 'SjællandNew.shx';
end

filename_2 = 'N:\PROJEKTER\N-retentionskortlægning 2022-24\Geostatistisk modellering\N-ret-mod\DK_Shapefile\';
filename = strcat(filename_2,filename_1);

S = shaperead(filename);

XS = S.X;
YS = S.Y;

% z1 = utmzone(YS(1),XS(1));
z1 = '32U';
[ellipsoid,~] = utmgeoid(z1);
utmstruct = defaultm('utm');
utmstruct.zone = z1;
utmstruct.geoid = ellipsoid(1,:);
utmstruct = defaultm(utmstruct);


[x,y] = projfwd(utmstruct,YS,XS);

if plotting == 1
    plot(x,y,'-k','LineWidth',linewidth,'MarkerSize',5)
end