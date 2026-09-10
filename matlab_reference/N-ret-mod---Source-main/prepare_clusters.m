function [out,rout,sout] = prepare_clusters(Sp,reg,perc,cert_fun_choice)

if nargin < 4 % Compatability with old ModeValues
    cert_fun_choice = [];
end

Number_of_Layers = numel(Sp);

try
    for k = 1:Number_of_Layers
        S = Sp{k};
        if S.zeromask==1
            disp('Zeromask detected, skipping cluster preparation')
        else
            mask = S.indexMask_Combined;
                    
            rm = S.ranges_bottom;
            sm = S.variances_bottom;
            mm = S.means_bottom;
            
            clust = S.clusters;
            
            %M er antal clusters
            M = numel(unique(clust));
            N = numel(rm);
            
            out = zeros(M,3);
            out(:,1) = unique(clust);
            
            
            for i = 1:M
                index = clust == i;
            
            
                ranges = rm(index);
                sills = sm(index);
                
                % if all cells in cluster is nan in ranges and sills, use
                % average range and sill as substitute
                if sum(~isnan(ranges))==0
                    disp(['Fixed cluster ' num2str(i) ' in layer ' num2str(k) ' as all ' num2str(length(ranges)) ' entries are NAN.'])
                    disp('Replaced with sill and range from average value')
%                   index2 = clust == 1;
                    index2 = ~isnan(rm) & ~isnan(sm);
                    ranges = rm(index2);
                    sills = sm(index2);
                end
                clear index2
            
            
                sortrange = sort(ranges);
                sortsill = sort(sills);
            
                sortsill(isnan(sortsill)) = [];
                sortrange(isnan(sortrange)) = [];
                
                sillindx = ceil(perc*max(size(sortsill)));
                sill_percentile = sortsill(sillindx);
            
                    
                rangeindx = ceil(0.5*max(size(sortrange)));
                range_percentile = sortrange(rangeindx);
                
                out(i,2) = range_percentile;
                out(i,3) = sill_percentile;
            end
            
            rind = isnan(rm);
            sind = isnan(sm);
            
            cr = clust(rind);
            rm(rind) = out(cr,2);
            rout{k} = rm;
            
            cs = clust(sind);
            sm(sind) = out(cs,3);
            sout{k} = sm;
            
            save(['ModeValues\ClusterindexModeSillRange_',reg,cert_fun_choice,'_Layer_',num2str(k)],"out")
        end
    end
catch
    keyboard
end