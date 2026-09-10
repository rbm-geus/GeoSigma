function cert_fun = certainty_function_definer(cert_fun_choice)

    % FRAFA APRIL 2023
    if strcmp(cert_fun_choice,'FRAFA_apr2023')
        cert_fun = @(dist,range,width,sill) (dist<=width).*sill+(dist>width).*sill.*(exp(-(3*power((dist-width),2))./(power(range,2))));
 
    % ILM SEP 2023
    elseif strcmp(cert_fun_choice,'ILM_sep2023')
        cert_fun = @(dist,range,width,sill) (dist<=width).*sill.*(exp(-(3*power((dist-width),2))./(power(range,2))));

    % ILM OCT 2023
    elseif strcmp(cert_fun_choice,'ILM_oct2023')
        cert_fun = @(dist,range,width,sill) (dist<=width).*sill.*(exp(-(3*power((dist),2))./(power(range,2))));

    % RBM OCT 2023
    elseif strcmp(cert_fun_choice,'RBM_oct2023')
        cert_fun_rbm_oct_ins = @(dist,range,width,sill,damp) sill.*(exp(-(damp*power((dist),2))./(power(range,2))));
        cert_fun_rbm_oct_out = @(dist,range,width,sill) sill.*(exp(-(3*power((dist),2))./(power(range,2))));

        cert_fun = @(dist,range,width,sill) (dist<=width).*cert_fun_rbm_oct_ins(dist,range,width,sill,1.5)+...
                                                 (dist>width).*cert_fun_rbm_oct_out(dist,range,width,sill+(sill-cert_fun_rbm_oct_ins(width,range,width,sill,1.5)));
    end
