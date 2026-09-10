function [GLOBAL] = setup_GLOBAL_structure(BOT)


% UTM
GLOBAL.UTMX = BOT.Grid_UTM_X;
GLOBAL.UTMY = BOT.Grid_UTM_Y;
GLOBAL.Nx = numel(GLOBAL.UTMX);
GLOBAL.Ny = numel(GLOBAL.UTMY);

% Global grid
[GLOBAL.xx,GLOBAL.yy] = meshgrid(GLOBAL.UTMX,GLOBAL.UTMY);
[GLOBAL.xx_buf,GLOBAL.yy_buf] = meshgrid(BOT.BufferedGrid_UTM_X,BOT.BufferedGrid_UTM_Y);

% Points (remove from outside innergrid)
GLOBAL.xp = GLOBAL.xx_buf(logical(BOT.points_bottom));
GLOBAL.yp = GLOBAL.yy_buf(logical(BOT.points_bottom));

% Only points in inner grid
indrmv = GLOBAL.xp<=max(GLOBAL.UTMX) & GLOBAL.xp>=min(GLOBAL.UTMX) & GLOBAL.yp<=max(GLOBAL.UTMY) & GLOBAL.yp>=min(GLOBAL.UTMY);
GLOBAL.xp = GLOBAL.xp(indrmv);
GLOBAL.yp = GLOBAL.yp(indrmv);

% Original surface
GLOBAL.img_FOHM = BOT.LayerBottom;
GLOBAL.img_FOHM_mask = GLOBAL.img_FOHM;
GLOBAL.img_FOHM_mask(~BOT.indexMask_Combined) = nan;

% Normalize point values
[GLOBAL.xp_norm,GLOBAL.yp_norm] = normalize_UTM_coord(GLOBAL.xp,GLOBAL.yp,100,100,min(GLOBAL.UTMX),min(GLOBAL.UTMY));

% Normalize grid values
[GLOBAL.xx_norm,GLOBAL.yy_norm] = normalize_UTM_coord(GLOBAL.xx,GLOBAL.yy,100,100);

% Normalize coordinate values
[GLOBAL.X_norm,GLOBAL.Y_norm] = normalize_UTM_coord(GLOBAL.UTMX,GLOBAL.UTMY,100,100);


% Inferred ranges, means and variances
GLOBAL.ranges = BOT.ranges_bottom;
GLOBAL.variances = BOT.ModelPrior(BOT.indexMask_Combined);
GLOBAL.stds = sqrt(GLOBAL.variances);
GLOBAL.log_ranges = log10(GLOBAL.ranges); % log-transformed
GLOBAL.log_variances = log10(GLOBAL.variances); % log-transformed
GLOBAL.means = BOT.means_bottom';

if isfield(BOT,'clusters')
    GLOBAL.clust = BOT.clusters;
end

% Setting up images
GLOBAL.img_ranges = nan(size(GLOBAL.img_FOHM));
GLOBAL.img_ranges(BOT.indexMask_Combined) = GLOBAL.ranges; 

GLOBAL.img_variances = nan(size(GLOBAL.img_FOHM));
GLOBAL.img_variances(BOT.indexMask_Combined) = GLOBAL.variances; 

GLOBAL.img_stds = nan(size(GLOBAL.img_FOHM));
GLOBAL.img_stds(BOT.indexMask_Combined) = GLOBAL.stds; 

GLOBAL.img_log_ranges = nan(size(GLOBAL.img_FOHM));
GLOBAL.img_log_ranges(BOT.indexMask_Combined) = GLOBAL.log_ranges; 

GLOBAL.img_log_variances = nan(size(GLOBAL.img_FOHM));
GLOBAL.img_log_variances(BOT.indexMask_Combined) = GLOBAL.log_variances; 

GLOBAL.img_means = nan(size(GLOBAL.img_FOHM));
GLOBAL.img_means(BOT.indexMask_Combined) = GLOBAL.means; 

if isfield(BOT,'clusters')
    GLOBAL.img_clust = nan(size(GLOBAL.img_FOHM));
    GLOBAL.img_clust(BOT.indexMask_Combined) = GLOBAL.clust; 
end

% Observed data for kriging image
GLOBAL.img_dobs = GLOBAL.img_FOHM-GLOBAL.img_means;



% Points to gridindex
GLOBAL.img_point = zeros(size(GLOBAL.img_FOHM));

for k = 1:length(GLOBAL.xp)
    i = GLOBAL.xp_norm(k);
    j = GLOBAL.yp_norm(k);
%    if j > 1900
%        keyboard
%    end
    GLOBAL.img_point(j,i) = 1;
end

% Initialize result image - m
GLOBAL.img_m_est = zeros(size(GLOBAL.img_FOHM));
% Initialize result image - std
GLOBAL.img_std = zeros(size(GLOBAL.img_FOHM));
