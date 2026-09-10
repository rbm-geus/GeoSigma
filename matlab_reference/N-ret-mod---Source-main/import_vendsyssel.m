function VG = import_vendsyssel(Grid_in,X,Y)

    %Produce Empty Grid

    names = {'0100_Postglacial_toerv_Bund_Adjusted','0110_Holocæn_Sand_Litt_Adjusted','0120_Holocæn_Ler_Litt_Adjusted','0130_Yoldia_Sand_Adjusted','0140_Yoldia_Ler_Adjusted'};
    path = [pwd,'\Vendsyssel\'];
    extension = '.grd';

    Nlay = numel(names);

    VG = NaN([size(Grid_in,1),size(Grid_in,2),Nlay]); 

    for i = 1:Nlay
        curname = [path,names{i},extension];
        [VL,Vinfo] = ReadSurfer7(curname);
        V_xs = Vinfo.UTM_X;
        V_ys = Vinfo.UTM_Y;

        [~,indXG,indXV] = intersect(X,V_xs);
        [~,indYG,indYV] = intersect(Y,V_ys);
        
        VG(indYG,indXG,i) = VL(indYV,indXV);
    end

end