function write_grid_ascii(file_n,Grid,xx,yy)
    %% Determine Grid Geometry
    % (xll and yll are lower left corner but the values are always pertaining to the center of the cell)
    sz = size(Grid);
    ncol = sz(2);
    nrow = sz(1);
    res = xx(2)-xx(1);
    xllcorner = xx(1)-0.5*res;
    yllcorner = yy(1)-0.5*res;

    %% Replace NaN with a no-value number
    noval = -9999;
    Grid(isnan(Grid)) = noval;

    %% Flip Grid Up/Down to comply with the ASCII formar
    Grid = flipud(Grid);

    %% Write the ASCII grid file
    fileID = fopen([file_n,'.asc'],'w','n','ascii');
    fwrite(fileID,['ncols ',num2str(ncol), newline]);
    fwrite(fileID,['nrows ',num2str(nrow), newline]);
    fwrite(fileID,['xllcorner ',num2str(xllcorner), newline]);
    fwrite(fileID,['yllcorner ',num2str(yllcorner), newline]);
    fwrite(fileID,['cellsize ',num2str(res), newline]);
    fwrite(fileID,['nodata_value ',num2str(noval), newline]);

    for i = 1:nrow
    rowstr = mat2str(Grid(i,:));
    fwrite(fileID,[rowstr(2:end-1) newline]);
    end

    fclose(fileID);

end