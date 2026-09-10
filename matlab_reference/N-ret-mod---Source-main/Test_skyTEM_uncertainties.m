% Test
clear all; close all;

comp2range = [100 500 400 250 100];
comp2rangePreq = [500 500 500 500 500];

cert_fun_frafa = @(dist,range,width,sill) (dist<=width).*sill+(dist>width).*sill.*(exp(-(3*power((dist-width),2))./(power(range,2))));
cert_fun_ilm_sep = @(dist,range,width,sill) (dist<=width).*sill.*(exp(-(3*power((dist-width),2))./(power(range,2))));
cert_fun_ilm_oct = @(dist,range,width,sill) (dist<=width).*sill.*(exp(-(3*power((dist),2))./(power(range,2))));
cert_fun_rbm_oct_ins = @(dist,range,width,sill,damp) sill.*(exp(-(damp*power((dist),2))./(power(range,2))));
cert_fun_rbm_oct_out = @(dist,range,width,sill) sill.*(exp(-(3*power((dist),2))./(power(range,2))));

%cert_fun_rbm_oct = @(dist,range,width,sill,damp) (dist<=width).*cert_fun_rbm_oct_pol(dist,range,width,sill,damp)...
%    +(dist>width).*cert_fun_rbm_oct_pol(dist,range,width,sill+sill+0.05-cert_fun_rbm_oct_pol(width,range,width,sill,damp),3);

cert_fun_rbm_oct = @(dist,range,width,sill,damp) (dist<=width).*cert_fun_rbm_oct_ins(dist,range,width,sill,damp)+...
                                                 (dist>width).*cert_fun_rbm_oct_out(dist,range,width,sill+(sill-cert_fun_rbm_oct_ins(width,range,width,sill,damp)));


%cert_fun_rbm_oct = @(dist,range,width,sill,damp) (dist<=2*width).*((1-dist/(width*2)).*cert_fun_rbm_oct_ins(dist,range,width,sill,damp)+...
%                                                 dist/(width*2).*cert_fun_rbm_oct_out(dist,range,width,sill))+...
%                                                 (dist>2*width).*cert_fun_rbm_oct_out(dist,range,width,sill);


%cert_fun_rbm_oct = @(dist,range,width,sill,damp) (dist<=width).*(sill-0.1*dist/width)+(dist>width)*sill*-0.2.*(dist-width)/width;

dist = 80;
damp = 1.5;

cert_fun_rbm_oct(75,500,75,1,damp)


%test= (exp(-(damp*power((75),2))./(power(500,2))))
sill = 1;
distvec = 0:5:1000;

local_depth = [20,50,100,200];


%cer_frafa = cert_fun_ilm(distvec,range,kernelmap,sill);

figure(1); clf(1)
set(gcf,'units','normalized','Position',[0,0,1,1])
set(gcf,'color','white')

for i = 1:4
    subplot(2,2,i)
    for j = 1:1
        range = comp2range(j+1);
        kernelmap = max(2*local_depth(i), 75);
        damp = 1.5;
        cer_frafa = cert_fun_frafa(distvec,range,kernelmap,sill);
        cer_ilm_sep = cert_fun_ilm_sep(distvec,range,kernelmap,sill);
        cer_ilm_oct = cert_fun_ilm_oct(distvec,range,kernelmap,sill);
        cer_rbm_oct = cert_fun_rbm_oct(distvec,range,kernelmap,sill,damp);

        
        plot(distvec,cer_frafa,'linewidth',1); hold on
        plot(distvec,cer_ilm_sep,'linewidth',1); hold on
        plot(distvec,cer_ilm_oct,'linewidth',2); hold on
        plot(distvec,cer_rbm_oct,'linewidth',2); hold on
        title(['depth: ',num2str(local_depth(i)),' m'])
        legend('FRAFA - Apr 2023','ILM - Sep 2023','ILM - Oct 2023','RBM - Oct 2023')
    end
end

%%
figure(2); clf(2)
set(gcf,'units','normalized','Position',[0,0,1,1])
set(gcf,'color','white')
legendstr = [];

for i = 1:4
    subplot(2,2,i)
    for j = 1:4
        range = comp2range(j+1);
        kernelmap = max(2*local_depth(i), 75);
        damp = 1.5;
        %cer_frafa = cert_fun_frafa(distvec,range,kernelmap,sill);
        %cer_ilm_sep = cert_fun_ilm_sep(distvec,range,kernelmap,sill);
        cer_ilm_oct = cert_fun_ilm_oct(distvec,range,kernelmap,sill);
        cer_rbm_oct = cert_fun_rbm_oct(distvec,range,kernelmap,sill,damp);

        
        %plot(distvec,cer_frafa,'linewidth',1); hold on
        %plot(distvec,cer_ilm_sep,'linewidth',1); hold on
        plot(distvec,cer_ilm_oct,'linewidth',j,'color','b'); hold on
        plot(distvec,cer_rbm_oct,'linewidth',j,'color','r'); hold on
        title(['depth: ',num2str(local_depth(i)),' m'])
        legendstr = [legendstr,{['ILM: comp=' num2str(j)]},{['RBM: comp=' num2str(j)]}];
    end
    legend(legendstr)
end

