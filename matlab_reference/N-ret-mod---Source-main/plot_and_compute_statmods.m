function [mode_statmods] = plot_and_compute_statmods(statmodspoints2,use_stds,Nclust,variances_use,stds_use,ranges_use)


    clear mode_statmods
    
    colbarplot = [flipud(bone(10));bone(10)];
    subplot(2,1,1)
    if use_stds == 1
        for istat = 0:Nclust-1
            for i = 1:20
                plot([istat+1,istat+1],[prctile(stds_use(statmodspoints2==istat+1),0+(i-1)*5),prctile(stds_use(statmodspoints2==istat+1),i*5)],'-','LineWidth',15,'color',colbarplot(i,:)); hold on
            end
            mode_statmods(istat+1,1) = prctile(stds_use(statmodspoints2==istat+1),50);
        end
        plot(1:Nclust,mode_statmods(:,1),':k')
        ylim([0,max(stds_use)])
        grid on
        title('Standard deviations')
        ylabel('Standard deviations')
        xlabel('Statmodel')
    else
        for istat = 0:Nclust-1
            for i = 1:20
                plot([istat+1,istat+1],[prctile(variances_use(statmodspoints2==istat+1),0+(i-1)*5),prctile(variances_use(statmodspoints2==istat+1),i*5)],'-','LineWidth',15,'color',colbarplot(i,:)); hold on
            end
            mode_statmods(istat+1,1) = prctile(variances_use(statmodspoints2==istat+1),50);
        end
        plot(1:Nclust,mode_statmods(:,1),':k')
        ylim([0,max(variances_use)])
        grid on
        title('Variances')
        ylabel('Variances')
        xlabel('Statmodel')
    end
    
    
    subplot(2,1,2)
    for istat = 0:Nclust-1
        for i = 1:20
                plot([istat+1,istat+1],[prctile(ranges_use(statmodspoints2==istat+1),0+(i-1)*5),prctile(ranges_use(statmodspoints2==istat+1),i*5)],'-','LineWidth',15,'color',colbarplot(i,:)); hold on
    
        end
        mode_statmods(istat+1,2) = prctile(ranges_use(statmodspoints2==istat+1),50);
    end
    plot(1:Nclust,mode_statmods(:,2),':k')
    grid on
    title('Ranges')
    ylabel('Ranges')
    xlabel('Statmodel')