function points = get_vpoints()

in = load([pwd,'\Tolkningspunkter\Vendsyssel_tolkningspunkter_med_boringssnap.mat']);
in2 = in.Vendsyssel_TP;

for i = 1:5
    curtab = in2{i};
    tabel{i} = [curtab(:,1:3) curtab(:,5) curtab(:,4)];
    
end

points = tabel;
end