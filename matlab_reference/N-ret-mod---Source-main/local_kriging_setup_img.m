function [G,Cm,Cd,d_obs,m0] = local_kriging_setup_img(GLOBAL,i_buf,ip_buf,curvariance,currange,cor_noise_rat)

% Function that sets up input for local kriging

i_buf_local = i_buf(i_buf == 1);
ip_buf_local = ip_buf(i_buf == 1);
%ip_buf_local_lin = find(ip_buf_local==1);

% G
G = eye(numel(i_buf_local),numel(i_buf_local));
G = G(ip_buf_local,:);

% CM
statmod_var = sprintf('%g %s(%g)',curvariance,'Gau',currange);
Cm = precal_cov([GLOBAL.xx_norm(i_buf)*100,GLOBAL.yy_norm(i_buf)*100],[GLOBAL.xx_norm(i_buf)*100,GLOBAL.yy_norm(i_buf)*100],statmod_var);

% Cd
statmod_noi = sprintf('%g %s(%g)',1,'Gau',currange);
Cd_shape = precal_cov([GLOBAL.xx_norm(ip_buf)*100,GLOBAL.yy_norm(ip_buf)*100],[GLOBAL.xx_norm(ip_buf)*100,GLOBAL.yy_norm(ip_buf)*100],statmod_noi);
Cd_diag = eye(size(Cd_shape)).*GLOBAL.img_unc(ip_buf);
Cd = Cd_diag*(cor_noise_rat*Cd_shape+(1-cor_noise_rat)*eye(size(Cd_shape)))*Cd_diag;

% d_obs
d_obs = GLOBAL.img_dobs(ip_buf);

% m0
m0 = 0;
