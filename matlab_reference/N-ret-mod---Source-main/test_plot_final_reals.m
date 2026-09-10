figure(1); clf(1)
for ireal = 1:10
    for i = 2:14
        subplot(4,4,i-1)
        imagesc(final_reals(:,:,i-1,ireal)-final_reals(:,:,i,ireal))
        set(gca,'ydir','normal')
        clim([0,10])
    end
    pause(1)
end