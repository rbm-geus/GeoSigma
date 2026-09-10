function points = get_LOOP2points()

in = load([pwd,'\Tolkningspunkter\LOOP2_PL_tolkningspunkter.mat']);
in2 = in.LOOP2_PL_TP;

for i = 1:5
    curtab = in2{i};
    tabel{i} = [curtab(:,1:3) curtab(:,5) curtab(:,4)];
    
end

points = tabel;
end