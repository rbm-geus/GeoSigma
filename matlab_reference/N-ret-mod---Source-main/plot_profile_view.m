function plot_profile_view(laynums,type,ind,S,S2,terrain,SData,reg,wells,mean_mod,real_mods,cmap,Well_S,NPL)

%Define model for icon sizes
dist_model = @(x) exp(-0.5*x.^2./(70^2));
dist_model = @(x) (200-x)/200;
s0 = 35;

NL = numel(laynums);

xs = S{1}.UTMX;
ys = S{1}.UTMY;

SY = S{1}.Ny;
SX = S{1}.Nx;

switch type

    case 'North-South'

        xvec = xs(ind);
        yvec = ys;

    case 'East-West'

        xvec = xs;
        yvec = ys(ind);

    case ''

        xvec = xs;
        yvec = ind;
end

%Prepare models
Nlay = size(S,2);

ref_mod = zeros(SY,SX,Nlay);
% mean_mod = zeros(SY,SX,Nlay);
point_mod = zeros(SY,SX,Nlay);
posterior = zeros(SY,SX,Nlay);

% real_mods = zeros(SY,SX,Nlay,5);

ref_mod(:,:,1) = terrain;
% mean_mod(:,:,1) = terrain;

for k = 1:Nlay

    c = S{k};
    c2 = S2{k};

    cur_ref_mod = c.img_FOHM;
    model_est = c.img_m_est;
    points = c.img_point;
    layer_posterior = c.img_post_std;
    localmean = c.img_means;

    point_mod(:,:,k+1) = points;
    %     mean_mod(:,:,k+1) = localmean+model_est;
    posterior(:,:,k+1) = layer_posterior;
    ref_mod(:,:,k+1) = cur_ref_mod;
    %
    %     for i = 1:5
    %         real_i = localmean+model_est+c.img_reals{i};
    %         real_mods(:,:,1,i) = terrain;
    %         real_mods(:,:,k+1,i) = real_i;
    %     end

end

%Function for plotting uncertainties
Nsub = (NL-1)+NL*5;
axcount = 0;
geophys_colors = ['r','g','b','c','m','k'];

%Load geophysics

if strcmp(reg,'Jylland')
    cur_TEM_xs = [SData.cfewTEM.xs;SData.cmanyTEM.xs];
    cur_TEM_ys = [SData.cfewTEM.ys;SData.cmanyTEM.ys];

else
    cur_TEM_xs = SData.cTEM.xs;
    cur_TEM_ys = SData.cTEM.ys;
end

cur_SkyTEM_xs = SData.cSkyTEM.xs;
cur_SkyTEM_ys = SData.cSkyTEM.ys;

cur_PACES_xs = SData.cPACES.xs;
cur_PACES_ys = SData.cPACES.ys;

cur_wells_xs = wells(:,1);
cur_wells_ys = wells(:,2);

jup_wells_xs = Well_S.jup_pos(:,1);
jup_wells_ys = Well_S.jup_pos(:,2);
jup_wells_zs = Well_S.jup_depth;
jup_wells_kote = Well_S.jup_kote;

sna_wells_xs = Well_S.snap_pos(:,1);
sna_wells_ys = Well_S.snap_pos(:,2);
sna_wells_zs = Well_S.snap_depth;
sna_wells_ztk = Well_S.snap_zerothk;
sna_wells_kote = Well_S.snap_kote;

cur_PACEP_xs = SData.cPACEP.xs;
cur_PACEP_ys = SData.cPACEP.ys;

cur_MEP_xs = SData.cMEP.xs;
cur_MEP_ys = SData.cMEP.ys;


figure();
for i = 1:NL
    switch type
        case 'North-South'
            curdists = abs(cur_TEM_xs-xvec);
            near_TEM_inds = find(abs(cur_TEM_xs-xvec)<200);
            TEM_xs = cur_TEM_xs(near_TEM_inds);
            TEM_ys = cur_TEM_ys(near_TEM_inds);
            TEMdists = curdists(near_TEM_inds);
            TEM_sz = dist_model(TEMdists)*s0;

            curdists = abs(cur_SkyTEM_xs-xvec);
            near_SkyTEM_inds = find(abs(cur_SkyTEM_xs-xvec)<200);
            SkyTEM_xs = cur_SkyTEM_xs(near_SkyTEM_inds);
            SkyTEM_ys = cur_SkyTEM_ys(near_SkyTEM_inds);
            SkyTEMdists = curdists(near_SkyTEM_inds);
            SkyTEM_sz = dist_model(SkyTEMdists)*s0;

            curdists = abs(cur_wells_xs-xvec);
            near_well_inds = find(curdists<200);
            well_xs = cur_wells_xs(near_well_inds);
            well_ys = cur_wells_ys(near_well_inds);
            welldists = curdists(near_well_inds);
            Well_sz = dist_model(welldists)*s0;

            local_jup_inds = find(abs(jup_wells_xs-xvec)<50);
            local_jup_xs = jup_wells_xs(local_jup_inds);
            local_jup_ys = jup_wells_ys(local_jup_inds);
            local_jup_zs = jup_wells_zs(local_jup_inds);
            local_jup_kote = jup_wells_kote(local_jup_inds);

            curdists = abs(jup_wells_xs-xvec);
            near_jup_inds = find(abs(jup_wells_xs-xvec)<200);
            near_jup_xs = jup_wells_xs(near_jup_inds);
            near_jup_ys = jup_wells_ys(near_jup_inds);
            near_jup_zs = jup_wells_zs(near_jup_inds);
            near_jup_kote = jup_wells_kote(near_jup_inds);
            jupdists = curdists(near_jup_inds);
            jup_sz = dist_model(jupdists)*s0;
            

            local_sna_inds = find(abs(sna_wells_xs-xvec)<50);
            local_sna_xs = sna_wells_xs(local_sna_inds);
            local_sna_ys = sna_wells_ys(local_sna_inds);
            local_sna_zs = sna_wells_zs(local_sna_inds);
            local_sna_kote = sna_wells_kote(local_sna_inds);
            local_sna_ztk = sna_wells_ztk(local_sna_inds);
            
            curdists = abs(sna_wells_xs-xvec);
            near_sna_inds = find(abs(sna_wells_xs-xvec)<200);
            near_sna_xs = sna_wells_xs(near_sna_inds);
            near_sna_ys = sna_wells_ys(near_sna_inds);
            near_sna_zs = sna_wells_zs(near_sna_inds);
            near_sna_kote = sna_wells_kote(near_sna_inds);
            near_sna_ztk = sna_wells_ztk(near_sna_inds);
            snadists = curdists(near_sna_inds);
            sna_sz = dist_model(snadists)*s0;

            curdists = abs(cur_PACES_xs-xvec);
            near_PACES_inds = find(abs(cur_PACES_xs-xvec)<200);
            PACES_xs = cur_PACES_xs(near_PACES_inds);
            PACES_ys = cur_PACES_ys(near_PACES_inds);
            PACESdists = curdists(near_PACES_inds);
            PACES_sz = dist_model(PACESdists)*s0;

            curdists = abs(cur_PACEP_xs-xvec);
            near_PACEP_inds = find(abs(cur_PACEP_xs-xvec)<200);
            PACEP_xs = cur_PACEP_xs(near_PACEP_inds);
            PACEP_ys = cur_PACEP_ys(near_PACEP_inds);
            PACEPdists = curdists(near_PACEP_inds);
            PACEP_sz = dist_model(PACEPdists)*s0;

            curdists = abs(cur_MEP_xs-xvec);
            near_MEP_inds = find(abs(cur_MEP_xs-xvec)<200);
            MEP_xs = cur_MEP_xs(near_MEP_inds);
            MEP_ys = cur_MEP_ys(near_MEP_inds);
            MEPdists = curdists(near_MEP_inds);
            MEP_sz = dist_model(MEPdists)*s0;

        case 'East-West'
            curdists = abs(cur_TEM_ys-yvec);
            near_TEM_inds = find(abs(cur_TEM_ys-yvec)<200);
            TEM_xs = cur_TEM_xs(near_TEM_inds);
            TEM_ys = cur_TEM_ys(near_TEM_inds);
            TEMdists = curdists(near_TEM_inds);
            TEM_sz = dist_model(TEMdists)*s0;

            curdists = abs(cur_SkyTEM_ys-yvec);
            near_SkyTEM_inds = find(abs(cur_SkyTEM_ys-yvec)<200);
            SkyTEM_xs = cur_SkyTEM_xs(near_SkyTEM_inds);
            SkyTEM_ys = cur_SkyTEM_ys(near_SkyTEM_inds);
            SkyTEMdists = curdists(near_SkyTEM_inds);
            SkyTEM_sz = dist_model(SkyTEMdists)*s0;

            curdists = abs(cur_wells_ys-yvec);
            near_well_inds = find(curdists<200);
            well_xs = cur_wells_xs(near_well_inds);
            well_ys = cur_wells_ys(near_well_inds);
            welldists = curdists(near_well_inds);
            Well_sz = dist_model(welldists)*s0;

            local_jup_inds = find(abs(jup_wells_ys-yvec)<50);
            local_jup_xs = jup_wells_xs(local_jup_inds);
            local_jup_ys = jup_wells_ys(local_jup_inds);
            local_jup_zs = jup_wells_zs(local_jup_inds);
            local_jup_kote = jup_wells_kote(local_jup_inds);

            curdists = abs(jup_wells_ys-yvec);
            near_jup_inds = find(abs(jup_wells_ys-yvec)<200);
            near_jup_xs = jup_wells_xs(near_jup_inds);
            near_jup_ys = jup_wells_ys(near_jup_inds);
            near_jup_zs = jup_wells_zs(near_jup_inds);
            near_jup_kote = jup_wells_kote(near_jup_inds);
            jupdists = curdists(near_jup_inds);
            jup_sz = dist_model(jupdists)*s0;

            local_sna_inds = find(abs(sna_wells_ys-yvec)<50);
            local_sna_xs = sna_wells_xs(local_sna_inds);
            local_sna_ys = sna_wells_ys(local_sna_inds);
            local_sna_zs = sna_wells_zs(local_sna_inds);
            local_sna_kote = sna_wells_kote(local_sna_inds);
            local_sna_ztk = sna_wells_ztk(local_sna_inds);
            
            curdists = abs(sna_wells_ys-yvec);
            near_sna_inds = find(abs(sna_wells_ys-yvec)<200);
            near_sna_xs = sna_wells_xs(near_sna_inds);
            near_sna_ys = sna_wells_ys(near_sna_inds);
            near_sna_zs = sna_wells_zs(near_sna_inds);
            near_sna_kote = sna_wells_kote(near_sna_inds);
            near_sna_ztk = sna_wells_ztk(near_sna_inds);
            snadists = curdists(near_sna_inds);
            sna_sz = dist_model(snadists)*s0;

            curdists = abs(cur_PACES_ys-yvec);
            near_PACES_inds = find(abs(cur_PACES_ys-yvec)<200);
            PACES_xs = cur_PACES_xs(near_PACES_inds);
            PACES_ys = cur_PACES_ys(near_PACES_inds);
            PACESdists = curdists(near_PACES_inds);
            PACES_sz = dist_model(PACESdists)*s0;

            curdists = abs(cur_PACEP_ys-yvec);
            near_PACEP_inds = find(abs(cur_PACEP_ys-yvec)<200);
            PACEP_xs = cur_PACEP_xs(near_PACEP_inds);
            PACEP_ys = cur_PACEP_ys(near_PACEP_inds);
            PACEPdists = curdists(near_PACEP_inds);
            PACEP_sz = dist_model(PACEPdists)*s0;

            curdists = abs(cur_MEP_ys-yvec);
            near_MEP_inds = find(abs(cur_MEP_ys-yvec)<200);
            MEP_xs = cur_MEP_xs(near_MEP_inds);
            MEP_ys = cur_MEP_ys(near_MEP_inds);
            MEPdists = curdists(near_MEP_inds);
            MEP_sz = dist_model(MEPdists)*s0;
    end


    axcount = axcount+1;
    switch type
        case 'North-South'

            ax(axcount)=subplot(Nsub,1,1+(i-1)*6);
            %             plot(well_ys,well_ys*0+1,'.','Color',geophys_colors(1),'displayname','Wells','MarkerSize',10)
            %             hold on
            scatter(well_ys,well_ys*0+2,Well_sz,'filled',geophys_colors(1),'displayname','Wells')
            hold on
            scatter(SkyTEM_ys,SkyTEM_ys*0+4,SkyTEM_sz,'filled',geophys_colors(2),'displayname','SkyTEM')
            scatter(TEM_ys,TEM_ys*0+6,TEM_sz,'filled',geophys_colors(3),'displayname','TEM')
            scatter(PACES_ys,PACES_ys*0+8,PACES_sz,'filled',geophys_colors(4),'displayname','PACES')
            scatter(PACEP_ys,PACEP_ys*0+10,PACEP_sz,'filled',geophys_colors(5),'displayname','PACEP')
            scatter(MEP_ys,MEP_ys*0+12,MEP_sz,'filled',geophys_colors(6),'displayname','MEP')
            %
            %             plot(TEM_ys,TEM_ys*0+3,'.','Color',geophys_colors(3),'displayname','TEM','MarkerSize',10)
            %             plot(PACES_ys,PACES_ys*0+4,'.','Color',geophys_colors(4),'displayname','PACES','MarkerSize',10)
            %             plot(PACEP_ys,PACEP_ys*0+5,'.','Color',geophys_colors(5),'displayname','PACEP','MarkerSize',10)
            %             plot(MEP_ys,MEP_ys*0+6,'.','Color',geophys_colors(6),'displayname','MEP','MarkerSize',10)
            %
            legend('location','northoutside','Orientation','horizontal','FontSize',16)
            ylim([0,14])

        case 'East-West'

            ax(axcount)=subplot(Nsub,1,1+(i-1)*6);
            scatter(well_xs,well_xs*0+2,Well_sz,'filled',geophys_colors(1),'displayname','Wells')
            hold on
            scatter(SkyTEM_xs,SkyTEM_xs*0+4,SkyTEM_sz,'filled',geophys_colors(2),'displayname','SkyTEM')
            scatter(TEM_xs,TEM_xs*0+6,TEM_sz,'filled',geophys_colors(3),'displayname','TEM')
            scatter(PACES_xs,PACES_xs*0+8,PACES_sz,'filled',geophys_colors(4),'displayname','PACES')
            scatter(PACEP_xs,PACEP_xs*0+10,PACEP_sz,'filled',geophys_colors(5),'displayname','PACEP')
            scatter(MEP_xs,MEP_xs*0+12,MEP_sz,'filled',geophys_colors(6),'displayname','MEP')

            legend('location','northoutside','Orientation','horizontal','FontSize',16)
            ylim([0,14])

    end
    axcount = axcount+1;
    ax(axcount)=subplot(Nsub,1,[2,3,4,5]+(i-1)*6);
    hold on
    box on


    cols = [0.4 0.2 0;1 1 0];
    NPLcols = ['r','g','b','c','m'];
    NPLnames = {'Post_Glacial_Ler_Tørv_Gytje_','Post_Glacial_Sand_','Post_Glacial_Ler_','Sen_Glacial_Sand_','Sen_Glacial_Ler_'};
    fact = [0.8 0.8];

    for z = 1:NPL
        switch type
            case 'North-South'
                fill([yvec;flipud(yvec)],[ref_mod(:,ind,z);flipud(ref_mod(:,ind,z+1))],NPLcols(z),'FaceAlpha',0.2,'DisplayName',NPLnames{z})

            
            case 'East-West'
                fill([xvec;flipud(xvec)],[ref_mod(ind,:,z),fliplr(ref_mod(ind,:,z+1))]',NPLcols(z),'FaceAlpha',0.2,'DisplayName',NPLnames{z})

        end
    end

    for z = (NPL+1):Nlay
        switch mod(z,2)
            case 1
                ltype = 1;
            case 0
                ltype = 2;
        end
        switch type
            case 'North-South'
                %                 fill([yvec;flipud(yvec)],[ref_mod(:,ind,z);flipud(ref_mod(:,ind,z+1))],'y','FaceAlpha',0.1,'HandleVisibility','off')
                fill([yvec;flipud(yvec)],[ref_mod(:,ind,z);flipud(ref_mod(:,ind,z+1))],cols(ltype,:),'FaceAlpha',0.2,'HandleVisibility','off')
                cols(ltype,:) = cols(ltype,:)*fact(ltype);
                fact(ltype) = 1/fact(ltype);
            case 'East-West'
              %                 fill([yvec;flipud(yvec)],[ref_mod(:,ind,z);flipud(ref_mod(:,ind,z+1))],'y','FaceAlpha',0.1,'HandleVisibility','off')
                fill([xvec;flipud(xvec)],[ref_mod(ind,:,z),fliplr(ref_mod(ind,:,z+1))]',cols(ltype,:),'FaceAlpha',0.2,'HandleVisibility','off')
                cols(ltype,:) = cols(ltype,:)*fact(ltype);
                fact(ltype) = 1/fact(ltype);
        end
    end

    for z = 1:Nlay
        switch type
            case 'North-South'
                plot(yvec,ref_mod(:,ind,z+1),'-k','HandleVisibility','off')

            case 'East-West'
                plot(xvec,ref_mod(ind,:,z+1),'-k','HandleVisibility','off')
        end
    end

    switch type
        case 'North-South'

            mv=mean_mod(:,ind,laynums(i)+1);
            xp = point_mod(:,ind,laynums(i)+1)>0;
            confin = 2*S{laynums(i)}.img_post_std; %Posterior uncertainty
            confind = 2*sqrt(S{laynums(i)}.img_unc); %Data uncertainty
            m = S2{laynums(i)}.indexMask_Combined;
            mask = reshape(m,size(confin));
            mask_v = mask(:,ind);
            mask_v(isnan(mask_v)) = 0;
            mask_d = [false;logical(abs(diff(mask_v)))];
            mask_vertices = yvec(mask_d);
            nmask = numel(mask_vertices)/2;

            for q = 1:nmask
                xmin = mask_vertices((q-1)*2+1);
                xmax = mask_vertices(q*2);
                if q == 1
                fill([xmin xmax-100 xmax-100 xmin],[-500 -500 200 200],'k','FaceAlpha',0.1,'DisplayName','Modelling Area')
                else
                fill([xmin xmax-100 xmax-100 xmin],[-500 -500 200 200],'k','FaceAlpha',0.1,'HandleVisibility','off')
                end
            end
            conf_d_vec = confind(:,ind);
            conf_vec = confin(:,ind);

            plot(yvec,mv+conf_vec,'--r','LineWidth',2,'HandleVisibility','off')
            plot(yvec,mv-conf_vec,'--r','LineWidth',2,'HandleVisibility','off')
            plot(yvec,mv,'-r','LineWidth',3,'HandleVisibility','off')

            errorbar(yvec(xp),ref_mod(xp,ind,laynums(i)+1),conf_d_vec(xp),'.k','LineWidth',2,'HandleVisibility','off')

            for j = 1:5
                curreal = real_mods(:,ind,laynums(i)+1,j);
                plot(yvec,curreal,'-r','LineWidth',1,'HandleVisibility','off')
            end

            plot(yvec,ref_mod(:,ind,laynums(i)+1),'-k','LineWidth',2,'HandleVisibility','off')
            plot(yvec,ref_mod(:,ind,1),'-b','LineWidth',2)
            plot([local_sna_ys,local_sna_ys]',[local_sna_kote,local_sna_zs]','Color',[0 0 0],'LineWidth',2,'HandleVisibility','off')
            plot([local_jup_ys,local_jup_ys]',[local_jup_kote,local_jup_zs]','Color',[0 0 0],'LineWidth',2,'HandleVisibility','off')
            plot(local_sna_ys,local_sna_zs,'or','MarkerSize',8,'LineWidth',2,'DisplayName','Snapped Point')
            plot(local_sna_ys(logical(local_sna_ztk)),local_sna_zs(logical(local_sna_ztk)),'xm','MarkerSize',8,'LineWidth',2,'DisplayName','Snapped Point - 0 thickness.')
            
                        %Plot nearby wells
            Nsna = numel(near_sna_zs);
            Njup = numel(near_jup_zs);

            for l = 1:Nsna
                plot([near_sna_ys(l),near_sna_ys(l)]',[near_sna_kote(l),near_sna_zs(l)]','Color',[1 0 0 sna_sz(l)/s0],'LineWidth',2,'HandleVisibility','off')
            end

            for l = 1:Njup
                plot([near_jup_ys(l),near_jup_ys(l)]',[near_jup_kote(l),near_jup_zs(l)]','Color',[1 0 0 jup_sz(l)/s0],'LineWidth',2,'HandleVisibility','off')
            end
                    
            legend
            grid on

        case 'East-West'

            mv=mean_mod(ind,:,laynums(i)+1);
            xp = point_mod(ind,:,laynums(i)+1)>0;
            confin = 2*S{laynums(i)}.img_post_std; %Posterior uncertainty
            confind = 2*(S{laynums(i)}.img_unc); %Data uncertainty            
            m = S2{laynums(i)}.indexMask_Combined;
            mask = reshape(m,size(confin));
            mask = mask(ind,:);
            mask(isnan(mask)) = 0;
            mask_d = [false logical(abs(diff(mask)))];
            mask_vertices = xvec(mask_d);
            nmask = numel(mask_vertices)/2;

            for q = 1:nmask
                xmin = mask_vertices((q-1)*2+1);
                xmax = mask_vertices(q*2);
                if q == 1
                fill([xmin xmax-100 xmax-100 xmin],[-500 -500 200 200],'k','FaceAlpha',0.1,'DisplayName','Modelling Area')
                else
                fill([xmin xmax-100 xmax-100 xmin],[-500 -500 200 200],'k','FaceAlpha',0.1,'HandleVisibility','off')
                end
            end

            conf_d_vec = confind(ind,:);
            conf_vec = confin(ind,:);

            plot(xvec,mv+conf_vec,'--r','LineWidth',2,'HandleVisibility','off')
            plot(xvec,mv-conf_vec,'--r','LineWidth',2,'HandleVisibility','off')
            plot(xvec,mv,'-r','LineWidth',3,'HandleVisibility','off')

            errorbar(xvec(xp),ref_mod(ind,xp,laynums(i)+1),conf_d_vec(xp),'.k','LineWidth',2,'HandleVisibility','off')

            for j = 1:5
                curreal = real_mods(ind,:,laynums(i)+1,j);
                plot(xvec,curreal,'-r','LineWidth',1,'HandleVisibility','off')
            end

            plot(xvec,ref_mod(ind,:,laynums(i)+1),'-k','LineWidth',2,'HandleVisibility','off')

            plot([local_sna_xs,local_sna_xs]',[local_sna_kote,local_sna_zs]','Color',[1 0 0],'LineWidth',2,'HandleVisibility','off')
            plot([local_jup_xs,local_jup_xs]',[local_jup_kote,local_jup_zs]','Color',[1 0 0],'LineWidth',2,'HandleVisibility','off')
            plot(local_sna_xs,local_sna_zs,'or','MarkerSize',8,'LineWidth',2,'DisplayName','Snapped Point')
            plot(local_sna_xs(logical(local_sna_ztk)),local_sna_zs(logical(local_sna_ztk)),'xm','MarkerSize',8,'LineWidth',2,'DisplayName','Snapped Point - 0 thickness.')
            
            %Plot nearby wells
            Nsna = numel(near_sna_zs);
            Njup = numel(near_jup_zs);

            for l = 1:Nsna
                plot([near_sna_xs(l),near_sna_xs(l)]',[near_sna_kote(l),near_sna_zs(l)]','Color',[1 0 0 sna_sz(l)/s0],'LineWidth',2,'HandleVisibility','off')
            end

            for l = 1:Njup
                plot([near_jup_xs(l),near_jup_xs(l)]',[near_jup_kote(l),near_jup_zs(l)]','Color',[1 0 0 jup_sz(l)/s0],'LineWidth',2,'HandleVisibility','off')
            end
            
            legend
            grid on
    end


    ylim([min(mv)-25,max(mv+25)])

end
linkaxes(ax,'x')
linkaxes(ax(2:2:end),'xy')
hold off
end