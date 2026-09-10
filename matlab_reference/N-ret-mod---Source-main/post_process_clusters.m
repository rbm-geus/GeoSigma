function clust = post_process_clusters(clust,maxsize)
%clust is an M by N matrix where M represents the UTM Y direction and N
%represents the UTM X direction.

Nclust = size(unique(clust(:)),1)-1;

extraclusts = 0;
count = 0;
NC = Nclust;
for i = 1:Nclust
    curclust = clust == i;
    subclusts = bwlabel(curclust);
    Nsub(i) = max(subclusts(:));

    for j = 1:Nsub(i)
        count = count+1;
        localinds = subclusts == j;
        subclustsize = sum(localinds(:));
        subclustersizes(count) = subclustsize;

        if subclustsize > maxsize
            [locy, locx] = find(localinds);
            Nsegments = ceil(subclustsize/maxsize);
            
            clusts = kmeans([locy locx],Nsegments);
            clust(localinds) = clusts+NC;

            extraclusts = extraclusts+(Nsegments-1);
            NC = NC+Nsegments;
        end
    end
end

end