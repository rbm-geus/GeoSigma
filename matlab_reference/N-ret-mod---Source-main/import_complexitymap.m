function map = import_complexitymap(X,Y,reg)
disp('Loading Geological Complexity Map')
[M,info] = ReadSurfer7('Kompleksitetskort.grd');

%Place M within a larger grid (equivalent to padding)
Nrow = size(M,1);
Ncol = size(M,2);

PadSize = 400;

BigM = zeros(Nrow+2*PadSize,Ncol+2*PadSize);
BigM(PadSize+1:end-PadSize,PadSize+1:end-PadSize) = M;
BigM(BigM<-1) = 0;

M = BigM;

switch reg

    case 'Jylland'
        Xg = info.UTM_X+50;
        Yg = info.UTM_Y+50;

    case 'Fyn'
        Xg = info.UTM_X+50;
        Yg = info.UTM_Y+50;

    case 'Fyn-MST'
        Xg = info.UTM_X;
        Yg = info.UTM_Y;
        
    case 'Sjælland'
        Xg = info.UTM_X+50;
        Yg = info.UTM_Y+50;

    case 'AnholtLæsø'
        Xg = info.UTM_X+50;
        Yg = info.UTM_Y+50;        

end
        Xg = [int32(((-PadSize):(-1))*100)+Xg(1) Xg int32((1:PadSize)*100)+Xg(end)];
        Yg = [int32(((-PadSize):(-1))*100)+Yg(1) Yg int32((1:PadSize)*100)+Yg(end)];

[~,IX1,~] = intersect(Xg,X);
[~,IY1,~] = intersect(Yg,Y);

map = M(IY1,IX1);
end