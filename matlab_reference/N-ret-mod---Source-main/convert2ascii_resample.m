function [modcube,x_out,y_out] = convert2ascii_resample(LayerCube,xs,ys,TopoNew,shift)

Nsurf = size(LayerCube,3)+1;

SurfaceGrids = cell(Nsurf,1);


%SurfaceGrids er et cell array som indeholder alle grids som skal
%resamples (inkl. topo). De ascii grids som ligger i jyllandsmappen ER resamplede, men
%det er grids som indlæses direkte fra Mette og Lærke's tolkede tiles ikke.
%Derfor kaldes denne funktion fra "ReadWriteTiles" efter grids er indlæst
%og "syet" sammen, så de passer med topo og de øvrige grids, før
%ReadWriteTiles fjerner artifakter osv.

%Denne funktion bruges også til at resample Jylland i Main.m

info.Grid = TopoNew;
SurfaceGrids{1} = info;

%Udgangspunkt topografi
%try
    [Topo_old,infoT] = ReadSurfer7([pwd,'\Flader\Jylland\','0000_Terrain_Bathymetri_100x100m.grd']);
%catch
%    [Topo_old,infoT] = dtm_read_plot([pwd,'\Flader\Jylland\','0000_Terrain_Bathymetri_100x100m.asc']);
%end

x_topo_old = infoT.UTM_X;
y_topo_old = infoT.UTM_Y;

for i = 1:Nsurf-1

    info.UTM_X = xs;
    info.UTM_Y = ys;
    info.Grid = LayerCube(:,:,i);
    SurfaceGrids{i+1} = info;

end

OldX = xs-shift;
OldY = ys-shift;

[~,indx,~] = intersect(x_topo_old,OldX);
[~,indy,~] = intersect(y_topo_old,OldY);

Topo_old = Topo_old(indy,indx);

%Start correcting topo
[xsM,ysM] = meshgrid(xs,ys);
F = interp2(OldX,OldY,Topo_old,xsM,ysM,'cubic');

CorrectionMap = F-TopoNew;

%Now loop over the other layers
SurfaceGrids0 = SurfaceGrids;

for i = 1:Nsurf-1
    info = SurfaceGrids{i+1};
    Grid = info.Grid;
    
    NewG = interp2(OldX,OldY,Grid,xsM,ysM,'cubic');
    NewG = NewG-CorrectionMap;

    info.Grid = NewG;
    info.UTM_X = xs;
    info.UTM_Y = ys;

    SurfaceGrids{i+1} = info;
end

nx = size(NewG,2);
ny = size(NewG,1);

modcube0 = zeros(ny,nx,Nsurf);
modcube = zeros(ny,nx,Nsurf);

for i = 1:Nsurf

    if i == 1
        modcube(:,:,i) = TopoNew;
        modcube0(:,:,i) = Topo_old;
    else

        modcube0(:,:,i) = SurfaceGrids0{i}.Grid;

        blay = modcube0(:,:,i);
        tlay = modcube0(:,:,i-1);

        thick = tlay-blay;
        logt = thick < 0.1;

        bottom = SurfaceGrids{i}.Grid;
        top = SurfaceGrids{i-1}.Grid;


        bottom(logt) = top(logt);

        logt2 = top<bottom;
        bottom(logt2) = top(logt2);

        SurfaceGrids{i}.Grid = bottom;

        modcube(:,:,i) = bottom;
    end
end

x_out = xs;
y_out = ys;

end