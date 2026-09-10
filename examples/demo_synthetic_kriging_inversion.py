import numpy as np
import matplotlib.pyplot as plt
from geosigma import local_kriging_setup_img
from utils import matlab_meshgrid
from plotting import add_in_between_gridlines
from geosigma import precal_cov
from geosigma import get_reals_cholesky
from geosigma import least_squares_inversion

np.random.seed(42)

# --- Grid definition ---
nx, ny = 12, 15
dx, dy = 100, 100
x = np.arange(nx) * dx
y = np.arange(ny) * dy
xx, yy = matlab_meshgrid(x, y)
n_cells = nx * ny

# GLOBAL dictionary with grid and reference model
var_ref = 5
ran_ref = 250
     


statmod_var_ref = f"{var_ref} Gau({ran_ref})"
Cm_ref, _ = precal_cov(
    np.column_stack((xx.ravel(), yy.ravel())),
    np.column_stack((xx.ravel(), yy.ravel())),
    statmod_var_ref
)


#Generate a synthetic observation model ---
m_ref = get_reals_cholesky(Cm_ref, 1).ravel()

#Generate a synthetic variance model ---
std_ref = get_reals_cholesky(Cm_ref/5, 1).ravel()


GLOBAL = {
    "xx_norm": xx.ravel() / dx,  # normalized for kriging input
    "yy_norm": yy.ravel() / dy,
    "img_unc": abs(std_ref),  # uncertainty
    "img_dobs": m_ref,  # observed values
}


# --- Simple mask: bottom-left 5x5 cells inactive ---
mask = np.ones(n_cells, dtype=bool)
mask_idx = [
    (np.arange(5)[:, None] * ny + np.arange(3)[None, :]).ravel(),
    (np.arange(2)[:, None] * ny + np.arange(8)[None, :]).ravel(),
    (np.arange(8,10)[:, None] * ny + np.arange(10,12)[None, :]).ravel()
    ]
mask[mask_idx[0].ravel()] = False
mask[mask_idx[1].ravel()] = False
mask[mask_idx[2].ravel()] = False
i_buf = mask  # active cells

# --- Random observations within mask ---
ip_buf = np.zeros(n_cells, dtype=bool)
ip_buf[mask & (np.random.rand(n_cells) < 0.2)] = True

# --- Setup kriging ---
curvariance = var_ref
currange = ran_ref
cor_noise_rat = 0.7

G, Cm, Cd, d_obs, m0 = local_kriging_setup_img(
    GLOBAL, i_buf, ip_buf, curvariance, currange, cor_noise_rat, dx=dx, dy=dy, verbose=True
)

# --- Run linear least-squares inversion ---
m_est, Cm_est = least_squares_inversion(G, Cm, Cd, m0, d_obs, type=2)

# --- Reshape for plotting ---
img_unc_plot = GLOBAL["img_unc"].reshape((nx, ny)).T
img_dobs_plot = GLOBAL["img_dobs"].reshape((nx, ny)).T
i_buf_plot = i_buf.reshape((nx, ny)).T
ip_buf_plot = ip_buf.reshape((nx, ny)).T

# --- Coordinates for scatter plot ---
obs_x = xx.ravel()[ip_buf]
obs_y = yy.ravel()[ip_buf]

# --- Plots ---
fig, axs = plt.subplots(2, 3, figsize=(15, 10))


# Observed data with observations
im0 = axs[0, 0].imshow(img_dobs_plot, origin='lower', cmap='viridis', vmin=-var_ref, vmax=var_ref,
                       extent=[x.min()-dx/2, x.max()+dx/2, y.min()-dy/2, y.max()+dy/2])
axs[0, 0].scatter(obs_x, obs_y, color='red', s=50, label='Observation points')
axs[0, 0].set_title("Observed data img_dobs")
axs[0, 0].set_xlabel("X")
axs[0, 0].set_ylabel("Y")
axs[0, 0].set_xticks(x)  
axs[0, 0].set_yticks(y)  
add_in_between_gridlines(axs[0, 0], x, y, color='white', lw=0.5, ls='-', alpha=0.8, axisbelow=False)
cbar = fig.colorbar(im0, ax=axs[0, 0]);


# Uncertainty with observations
im1 = axs[0, 1].imshow(img_unc_plot, origin='lower', cmap='viridis', vmin=0, vmax=var_ref,
                       extent=[x.min()-dx/2, x.max()+dx/2, y.min()-dy/2, y.max()+dy/2])
axs[0, 1].scatter(obs_x, obs_y, color='red', s=50, label='Observation points')
axs[0, 1].set_title("Uncertainty img_unc")
axs[0, 1].set_xlabel("X")
axs[0, 1].set_ylabel("Y")
axs[0, 1].set_xticks(x)  
axs[0, 1].set_yticks(y)
add_in_between_gridlines(axs[0, 1], x, y, color='white', lw=0.5, ls='-', alpha=0.8, axisbelow=False)  
fig.colorbar(im1, ax=axs[0, 1])






# Mask
im2 = axs[0, 2].imshow(i_buf_plot, origin='lower', cmap='gray',
                       extent=[x.min()-dx/2, x.max()+dx/2, y.min()-dy/2, y.max()+dy/2])
axs[0, 2].set_title("Mask (active/inactive)")
axs[0, 2].set_xlabel("X")
axs[0, 2].set_ylabel("Y")
axs[0, 2].set_xticks(x)  
axs[0, 2].set_yticks(y)  
add_in_between_gridlines(axs[0, 2], x, y, color='black', lw=0.5, ls='-', alpha=0.8, axisbelow=False)

#ax.set_yticks([-1, 0, 1])   
fig.colorbar(im2, ax=axs[0, 2])



# Covariance matrices
im3 = axs[1, 0].imshow(Cm, origin='lower', cmap='viridis')
axs[1, 0].set_title("Model covariance Cm")
fig.colorbar(im3, ax=axs[1, 0])

im4 = axs[1, 1].imshow(Cd, origin='lower', cmap='viridis')
axs[1, 1].set_title("Uncertainty covariance Cd")
fig.colorbar(im4, ax=axs[1, 1])

# G matrix
im5 = axs[1, 2].imshow(G, origin='lower', cmap='gray')
axs[1, 2].set_title("Selection matrix G")
axs[1, 2].set_xlabel("Grid cells")
axs[1, 2].set_ylabel("Observation points")
axs[1, 2].set_aspect('auto')  # <-- This allows non-square pixels
fig.colorbar(im5, ax=axs[1, 2])

plt.tight_layout()
plt.show()



fig, axs = plt.subplots(2, 3, figsize=(15, 10))


# Observed data with observations
im0 = axs[0, 0].imshow(img_dobs_plot, origin='lower', cmap='viridis', vmin=-var_ref, vmax=var_ref,
                       extent=[x.min()-dx/2, x.max()+dx/2, y.min()-dy/2, y.max()+dy/2])
axs[0, 0].scatter(obs_x, obs_y, color='red', s=50, label='Observation points')
axs[0, 0].set_title("Observed data img_dobs")
axs[0, 0].set_xlabel("X")
axs[0, 0].set_ylabel("Y")
axs[0, 0].set_xticks(x)  
axs[0, 0].set_yticks(y)  
add_in_between_gridlines(axs[0, 0], x, y, color='white', lw=0.5, ls='-', alpha=0.8, axisbelow=False)
cbar = fig.colorbar(im0, ax=axs[0, 0]);



# --- [0,1] Observed data at observation points only ---
d_obs_plot = np.full(n_cells, np.nan)
#d_obs_plot[i_buf] = 0.0  # initialize
d_obs_plot[ip_buf] = d_obs  # place only observed values
d_obs_plot_img = d_obs_plot.reshape((nx, ny)).T

im1 = axs[0, 1].imshow(d_obs_plot_img, origin='lower', cmap='viridis', vmin=-var_ref, vmax=var_ref,
                       extent=[x.min()-dx/2, x.max()+dx/2, y.min()-dy/2, y.max()+dy/2])
axs[0, 1].set_title("Observed points d_obs")
axs[0, 1].set_xlabel("X")
axs[0, 1].set_ylabel("Y")
axs[0, 1].set_xticks(x)
axs[0, 1].set_yticks(y)
add_in_between_gridlines(axs[0, 1], x, y, color='white', lw=0.5, ls='-', alpha=0.8, axisbelow=False)
fig.colorbar(im1, ax=axs[0, 1])

# --- [0,2] Inversion result: posterior mean ---
m_est_full = np.full(n_cells, np.nan)
m_est_full[i_buf] = m_est  # put estimated values only in active cells
m_est_img = m_est_full.reshape((nx, ny)).T

im2 = axs[0, 2].imshow(m_est_img, origin='lower', cmap='viridis', vmin=-var_ref, vmax=var_ref,
                       extent=[x.min()-dx/2, x.max()+dx/2, y.min()-dy/2, y.max()+dy/2])
axs[0, 2].set_title("Inversion: posterior mean")
axs[0, 2].set_xlabel("X")
axs[0, 2].set_ylabel("Y")
axs[0, 2].set_xticks(x)
axs[0, 2].set_yticks(y)
add_in_between_gridlines(axs[0, 2], x, y, color='white', lw=0.5, ls='-', alpha=0.8, axisbelow=False)
fig.colorbar(im2, ax=axs[0, 2])

# --- [1,0] Posterior covariance (model uncertainty) ---
im3 = axs[1, 0].imshow(Cm_est, origin='lower', cmap='viridis')
axs[1, 0].set_title("Posterior covariance Cm_est")
fig.colorbar(im3, ax=axs[1, 0])

# --- [1,1] Posterior standard deviation ---
std_est = np.sqrt(np.diag(Cm_est))
std_est_full = np.full(n_cells, np.nan)
std_est_full[i_buf] = std_est
std_est_img = std_est_full.reshape((nx, ny)).T

im4 = axs[1, 1].imshow(std_est_img, origin='lower', cmap='viridis', vmin=0, vmax=var_ref,
                       extent=[x.min()-dx/2, x.max()+dx/2, y.min()-dy/2, y.max()+dy/2])
axs[1, 1].set_title("Posterior std (sqrt diag Cm_est)")
axs[1, 1].set_xlabel("X")
axs[1, 1].set_ylabel("Y")
fig.colorbar(im4, ax=axs[1, 1])

# --- [1,2] Difference between posterior mean and ref model ---
diff = np.full(n_cells, np.nan)
diff[i_buf] = m_est - m_ref[i_buf]
diff_img = diff.reshape((nx, ny)).T

im5 = axs[1, 2].imshow(diff_img, origin='lower', cmap='seismic', vmin=-var_ref, vmax=var_ref,
                       extent=[x.min()-dx/2, x.max()+dx/2, y.min()-dy/2, y.max()+dy/2])
axs[1, 2].set_title("Posterior mean - true model")
axs[1, 2].set_xlabel("X")
axs[1, 2].set_ylabel("Y")
fig.colorbar(im5, ax=axs[1, 2])

plt.tight_layout()
plt.show()

plt.tight_layout()
plt.show()



# --- Generate posterior realizations ---
Nreals = 3
reals_post = get_reals_cholesky(Cm_est, Nreals, m_est,verbose=True)  # shape: (n_active, Nreals)

# Map to full grid including inactive cells
reals_full = np.full((n_cells, Nreals), np.nan)
i_buf_idx = np.where(i_buf)[0]
for k in range(Nreals):
    reals_full[i_buf_idx, k] = reals_post[:, k]

# Reshape for plotting
reals_imgs = [reals_full[:, k].reshape((nx, ny)).T for k in range(Nreals)]

# --- Plot 3 realizations ---
fig, axs = plt.subplots(1, Nreals, figsize=(15, 5))

for k in range(Nreals):
    im = axs[k].imshow(reals_imgs[k], origin='lower', cmap='viridis',
                       vmin=m_est.min(), vmax=m_est.max(),
                       extent=[x.min()-dx/2, x.max()+dx/2, y.min()-dy/2, y.max()+dy/2])
    axs[k].set_title(f"Posterior realization {k+1}")
    axs[k].set_xlabel("X")
    axs[k].set_ylabel("Y")
    axs[k].set_xticks(x)
    axs[k].set_yticks(y)
    add_in_between_gridlines(axs[k], x, y, color='white', lw=0.5, ls='-', alpha=0.8, axisbelow=False)
    fig.colorbar(im, ax=axs[k])

plt.tight_layout()
plt.show()

