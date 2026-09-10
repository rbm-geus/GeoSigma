function theme = get_modeltheme(native_dir,reg,s,terrain,NPL)

% GRADIENTS
grad_less50 = 0.3; 
grad_abv50 = 0.15;



switch reg
    case 'Jylland'
        navn = 'Jylland'
        modeltheme1 = load([native_dir,'\','Kriging Mappe\UsikkerhedsTemaer\',navn,'_modelleringsomraede_grid_klasser.mat']);
        modeltheme2 = modeltheme1.grid_ma_class_eval;

        modeltheme = modeltheme2*0;
        m0 = modeltheme2 == 0;
        m1 = modeltheme2 == 1;
        m2 = modeltheme2 == 2;
        m3 = modeltheme2 == 3;
        m4 = modeltheme2 == 4;
        m5 = modeltheme2 == 5;


        theme = zeros(size(m0,1),size(m0,2),size(s,2));

        % MIN VARIANCE AT SURFACE
        f0min = 20; 
        f1min = 20; % old: 15
        f2min = 12.5;
        f3min = 12.5; 
        f4min = 11.25; 
        f5min = 10;

        f0 = @(x) (0.5*((x<50).*(grad_less50.*x+f0min)+(x>=50).*(grad_abv50*x+(grad_less50-grad_abv50)*50+f0min))).^2;
        f1 = @(x) (0.5*((x<50).*(grad_less50.*x+f1min)+(x>=50).*(grad_abv50*x+(grad_less50-grad_abv50)*50+f1min))).^2;
        f2 = @(x) (0.5*((x<50).*(grad_less50.*x+f2min)+(x>=50).*(grad_abv50*x+(grad_less50-grad_abv50)*50+f2min))).^2;
        f3 = @(x) (0.5*((x<50).*(grad_less50.*x+f3min)+(x>=50).*(grad_abv50*x+(grad_less50-grad_abv50)*50+f3min))).^2;
        f4 = @(x) (0.5*((x<50).*(grad_less50.*x+f4min)+(x>=50).*(grad_abv50*x+(grad_less50-grad_abv50)*50+f4min))).^2;
        f5 = @(x) (0.5*((x<50).*(grad_less50.*x+f5min)+(x>=50).*(grad_abv50*x+(grad_less50-grad_abv50)*50+f5min))).^2;


        for i = 1:size(s,2)
            curlay = s{i}.LayerBottom;
            cl = terrain-curlay;

            f0map = f0(cl);
            f1map = f1(cl);
            f2map = f2(cl);
            f3map = f3(cl);
            f4map = f4(cl);
            f5map = f5(cl);


            if i <= NPL
                modeltheme = f5map;
            else
                modeltheme(m0) = f0map(m0);
                modeltheme(m1) = f1map(m1);
                modeltheme(m2) = f2map(m2);
                modeltheme(m3) = f3map(m3);
                modeltheme(m4) = f4map(m4);
                modeltheme(m5) = f5map(m5);
            end
            theme(:,:,i) = modeltheme;

        end

    case {'Fyn','Fyn-MST'}
        navn = 'fyn';
        if strcmp(reg,'Fyn')
            modeltheme1 = load([native_dir,'\','Kriging Mappe\UsikkerhedsTemaer\',navn,'_modelleringsomraede_grid_klasser.mat']);
        elseif  strcmp(reg,'Fyn-MST')
            modeltheme1 = load([native_dir,'\','Kriging Mappe\UsikkerhedsTemaer\',navn,'_modelleringsomraede_grid_klasser_MST.mat']);
        end
        modeltheme2 = modeltheme1.grid_ma_class;

        modeltheme = modeltheme2*0;
        m0 = modeltheme2 == 0;
        m1 = modeltheme2 == 1;
        m2 = modeltheme2 == 2;
        m3 = modeltheme2 == 3;


        theme = zeros(size(m0,1),size(m0,2),size(s,2));

        % MIN VARIANCE AT SURFACE
        f0min = 20; 
        f1min = 20; % old: 15
        f2min = 12.5;
        f3min = 10; 


        f0 = @(x) (0.5*((x<50).*(grad_less50.*x+f0min)+(x>=50).*(grad_abv50*x+(grad_less50-grad_abv50)*50+f0min))).^2;
        f1 = @(x) (0.5*((x<50).*(grad_less50.*x+f1min)+(x>=50).*(grad_abv50*x+(grad_less50-grad_abv50)*50+f1min))).^2;
        f2 = @(x) (0.5*((x<50).*(grad_less50.*x+f2min)+(x>=50).*(grad_abv50*x+(grad_less50-grad_abv50)*50+f2min))).^2;
        f3 = @(x) (0.5*((x<50).*(grad_less50.*x+f3min)+(x>=50).*(grad_abv50*x+(grad_less50-grad_abv50)*50+f3min))).^2;


        for i = 1:size(s,2)
            curlay = s{i}.LayerBottom;
            cl = terrain-curlay;

            f0map = f0(cl);
            f1map = f1(cl);
            f2map = f2(cl);
            f3map = f3(cl);

            
            if i <= NPL
                modeltheme = f3map;
            else
                modeltheme(m0) = f0map(m0);
                modeltheme(m1) = f1map(m1);
                modeltheme(m2) = f2map(m2);
                modeltheme(m3) = f3map(m3);

            end
            theme(:,:,i) = modeltheme;
            
        end

    case 'Sjælland'
        navn = 'sjaelland';
        modeltheme1 = load([native_dir,'\','Kriging Mappe\UsikkerhedsTemaer\',navn,'_modelleringsomraede_grid_klasser.mat']);
        modeltheme2 = modeltheme1.grid_ma_class;

        modeltheme = modeltheme2*0;
        m0 = modeltheme2 == 0;
        m1 = modeltheme2 == 1;
        m2 = modeltheme2 == 2;
        m3 = modeltheme2 == 3;


        theme = zeros(size(m0,1),size(m0,2),size(s,2));

        % MIN VARIANCE AT SURFACE
        f0min = 20; 
        f1min = 20; % old: 15
        f2min = 12.5;
        f3min = 10; 

        f0 = @(x) (0.5*((x<50).*(grad_less50.*x+f0min)+(x>=50).*(grad_abv50*x+(grad_less50-grad_abv50)*50+f0min))).^2;
        f1 = @(x) (0.5*((x<50).*(grad_less50.*x+f1min)+(x>=50).*(grad_abv50*x+(grad_less50-grad_abv50)*50+f1min))).^2;
        f2 = @(x) (0.5*((x<50).*(grad_less50.*x+f2min)+(x>=50).*(grad_abv50*x+(grad_less50-grad_abv50)*50+f2min))).^2;
        f3 = @(x) (0.5*((x<50).*(grad_less50.*x+f3min)+(x>=50).*(grad_abv50*x+(grad_less50-grad_abv50)*50+f3min))).^2;

        for i = 1:size(s,2)
            curlay = s{i}.LayerBottom;
            cl = terrain-curlay;

            f0map = f0(cl);
            f1map = f1(cl);
            f2map = f2(cl);
            f3map = f3(cl);

            modeltheme(m0) = f0map(m0);
            modeltheme(m1) = f1map(m1);
            modeltheme(m2) = f2map(m2);
            modeltheme(m3) = f3map(m3);

            theme(:,:,i) = modeltheme;
        end
    case 'AnholtLæsø'
        navn = 'DK8';
        modeltheme1 = load([native_dir,'\','Kriging Mappe\UsikkerhedsTemaer\',navn,'_modelleringsomraede_grid_klasser.mat']);
        modeltheme2 = modeltheme1.grid_ma_class;

        modeltheme = modeltheme2*0;
        m0 = modeltheme2 == 0;
        m1 = modeltheme2 == 1;
        m2 = modeltheme2 == 2;
        m3 = modeltheme2 == 3;


        theme = zeros(size(m0,1),size(m0,2),size(s,2));

        % MIN VARIANCE AT SURFACE
        f0min = 20; 
        f1min = 20; % old: 15
        f2min = 12.5;
        f3min = 10; 

        f0 = @(x) (0.5*((x<50).*(grad_less50.*x+f0min)+(x>=50).*(grad_abv50*x+(grad_less50-grad_abv50)*50+f0min))).^2;
        f1 = @(x) (0.5*((x<50).*(grad_less50.*x+f1min)+(x>=50).*(grad_abv50*x+(grad_less50-grad_abv50)*50+f1min))).^2;
        f2 = @(x) (0.5*((x<50).*(grad_less50.*x+f2min)+(x>=50).*(grad_abv50*x+(grad_less50-grad_abv50)*50+f2min))).^2;
        f3 = @(x) (0.5*((x<50).*(grad_less50.*x+f3min)+(x>=50).*(grad_abv50*x+(grad_less50-grad_abv50)*50+f3min))).^2;

        for i = 1:size(s,2)
            curlay = s{i}.LayerBottom;
            cl = terrain-curlay;

            f0map = f0(cl);
            f1map = f1(cl);
            f2map = f2(cl);
            f3map = f3(cl);

            modeltheme(m0) = f0map(m0);
            modeltheme(m1) = f1map(m1);
            modeltheme(m2) = f2map(m2);
            modeltheme(m3) = f3map(m3);

            theme(:,:,i) = modeltheme;
        end
end

end
