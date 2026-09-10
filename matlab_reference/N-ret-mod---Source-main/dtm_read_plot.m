function [xx,yy,G_grid]=dtm_read_plot(file_n)

fileID = fopen(file_n);

formatSpec ='%s %f';
N = 6;
G_text = textscan(fileID,formatSpec,N);
G_data = textscan(fileID,'%f');
fclose(fileID);

G_t=[G_text{1,2}];
N_col=G_t(1);
N_row=G_t(2);
xllcorner=G_t(3);
yllcorner=G_t(4);
c_size=G_t(5);
G_d=cell2mat(G_data);
G_grid=flipud(reshape(G_d,N_col,N_row)');
xx=(xllcorner:c_size:xllcorner+c_size*N_col-c_size)+0.5*c_size;
yy=(yllcorner:c_size:yllcorner+c_size*N_row-c_size)+0.5*c_size;

end
% figure
% pcolor(xx,yy,G_grid),shading flat
