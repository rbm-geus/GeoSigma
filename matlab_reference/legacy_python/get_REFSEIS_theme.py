import numpy as np
import pandas as pd
import scipy.io as sio
from matio import load_from_mat
def get_REFSEIS_theme(UTM_X, UTM_Y, terrain, complexity, reg, include_peatlands, Npreq, NPL, Nlay):
    # Load data based on region
    region_files = {
        'Jylland': 'Jylland_seismik_til_usikkerhed.mat',
        'Fyn': None,  # No file specified for Fyn in MATLAB code
        'Sjælland': 'Sjaelland_seismik_til_usikkerhed.mat'
    }
    filename = region_files.get(reg)
    if not filename:
        raise ValueError("Invalid region or missing file for Fyn")

    in_data = sio.loadmat(f'data/{filename}')
    Gmod = load_from_mat(f'data/{filename}')
    Gmod = Gmod['Gmod_pos']

    # Rearrange data
    SEIS = {
        'InfoTable': Gmod,
        'Model_Depths': in_data['dkm_d_pos'],
        'Model_Year': in_data['model_year_pos'].flatten(),
        'DepthInterfaceAboveModel': in_data['dkm_odvlayertop_pos'],
        'ThickGeophysModel': in_data['dkm_odvthk_pos']
    }

    layvec = np.where(SEIS['ThickGeophysModel'][0, :Nlay - NPL] > 1000)[0]
    coords = SEIS['InfoTable'][['xutm_euref89_utm32', 'yutm_euref89_utm32']].values
    _, idx = np.unique(coords, axis=0, return_index=True)
    logind_dupe = np.zeros(len(coords), dtype=bool)
    logind_dupe[idx] = True

    model_year = SEIS['Model_Year']
    geophys_year = SEIS['InfoTable']['d_year'].values
    logind_date = geophys_year <= model_year
    logind = logind_dupe & logind_date

    for key in ['InfoTable', 'Model_Depths', 'Model_Year', 'DepthInterfaceAboveModel', 'ThickGeophysModel']:
        SEIS[key] = SEIS[key][logind]

    nx, ny = len(UTM_X), len(UTM_Y)
    maxl = Nlay - NPL
    Grid = np.zeros((ny, nx, maxl))

    xs = SEIS['InfoTable']['xutm_euref89_utm32'].values
    ys = SEIS['InfoTable']['yutm_euref89_utm32'].values

    local_dist = np.full((ny, nx, maxl), np.nan)
    local_model_thick = np.full((ny, nx, maxl), np.nan)
    complexities = np.full((ny, nx, maxl), np.nan)

    SearchRadius = 8
    comp2range = [100, 500, 400, 250, 100]
    comp2rangePreq = [500] * 5

    # Certainty function for REFSEIS
    def cert_fun(dist, range_, width, sill):
        return np.where(dist <= width, sill, sill * np.exp(-3 * (dist - width)**2 / range_**2))

    for s in layvec:
        MThicks = SEIS['ThickGeophysModel'][:, s]
        complexities[:, :, s] = complexity

        for i in range(ny):
            yv = UTM_Y[i]
            ymin = UTM_Y[max(0, i - SearchRadius)]
            ymax = UTM_Y[min(ny - 1, i + SearchRadius)]
            filt1 = (ys < ymax) & (ys > ymin)

            for j in range(nx):
                xv = UTM_X[j]
                xmin = UTM_X[max(0, j - SearchRadius)]
                xmax = UTM_X[min(nx - 1, j + SearchRadius)]
                filt2 = (xs < xmax) & (xs > xmin)
                filt = filt1 & filt2

                if np.any(filt):
                    
                    local_xs = xs[filt]
                    local_ys = ys[filt]
                    local_thicks = MThicks[filt]

                    distances = np.sqrt((xv - local_xs)**2 + (yv - local_ys)**2)
                    dist = np.min(distances)
                    filt_mindist = distances == dist

                    local_dist[i, j, s] = dist
                    local_model_thick[i, j, s] = np.max(local_thicks[filt_mindist])

    complexity2 = np.where(complexities == 0, np.nan, complexities)
    rangemap = np.copy(complexity2)
    rangemap2 = np.copy(complexity2)

    for i in range(1, 5):
        rangemap[complexity2 == i] = comp2range[i]
        rangemap2[complexity2 == i] = comp2rangePreq[i]

    rangemap[:, :, Npreq - NPL:] = rangemap2[:, :, Npreq - NPL:]
    var0map = np.ones_like(local_dist) * (0.5 * 15)**2
    kernelmap = np.full_like(local_dist, 1)

    certmap = cert_fun(local_dist, rangemap, kernelmap, 1.0 / var0map)
    Grid = 1.0 / certmap
    Grid[np.isnan(Grid)] = 100000

    toi_filter = local_model_thick < 100
    Grid[toi_filter] = 100000

    if include_peatlands:
        New_Grid = np.full((Grid.shape[0], Grid.shape[1], Grid.shape[2] + NPL), 1E4)
        New_Grid[:, :, NPL:] = Grid
        Grid = New_Grid

    s_cords = {'xs': xs, 'ys': ys}
    return Grid, s_cords