function points = get_DK8_points()

in = load([pwd,'\Tolkningspunkter\DK8_PL_tolkningspunkter.mat']);
in2 = in.DK8_PL_TP;


for i = 1:5
    curtab = in2{i};
    if isempty(curtab) % Fix for double instead of table in ILM matfile
       curtab = array2table(zeros(0,5));
    end
    tabel{i} = [curtab(:,1:3) curtab(:,5) curtab(:,4)];
end

points = tabel;
end