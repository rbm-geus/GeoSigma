import numpy as np
import pandas as pd
import scipy.io as sio
from matio import load_from_mat
def get_RESLOG_theme(UTM_X, UTM_Y, terrain, complexity, reg, include_peatlands, Npreq, NPL, Nlay):
    # Load data based on region
    region_files = {
        'Jylland': 'Jylland_resistivitetslog_til_usikkerhed.mat',
        'Fyn': 'fyn_resistivitetslog_til_usikkerhed.mat',
        'Sjælland': 'Sjaelland_resistivitetslog_til_usikkerhed.mat'
    }
    filename = region_files.get(reg)
    if not filename:
        raise ValueError("Invalid region specified")

    in_data = sio.loadmat(f'data/{filename}')
    Gmod = load_from_mat(f'data/{filename}')
    Gmod = Gmod['Gmod_pos']
    # Rearrange data
    info_table = Gmod
    MEP = {
        'InfoTable': info_table,
        'Model_Depths': in_data['dkm_d_pos'],
        'Model_Year': in_data['model_year_pos'].flatten(),
        'DepthInterfaceAboveModel': in_data['dkm_odvlayertop_pos'],
        'ThickGeophysModel': in_data['dkm_odvthk_pos']
    }

    coords = MEP['InfoTable'][['xutm32euref89', 'yutm32euref89']].values
    _, idx = np.unique(coords, axis=0, return_index=True)
    logind_dupe = np.zeros(len(coords), dtype=bool)
    logind_dupe[idx] = True

    model_year = MEP['Model_Year']
    geophys_year = MEP['InfoTable']['d_year'].values
    logind_date = geophys_year <= model_year
    logind = logind_dupe & logind_date
    for key in ['InfoTable', 'Model_Depths', 'Model_Year', 'DepthInterfaceAboveModel', 'ThickGeophysModel']:
        MEP[key] = MEP[key][logind]

    nx, ny = len(UTM_X), len(UTM_Y)
    maxl = Nlay - NPL
    Grid = np.zeros((ny, nx, maxl))

    xs = MEP['InfoTable']['xutm32euref89'].values
    ys = MEP['InfoTable']['yutm32euref89'].values
    DOIs = MEP['InfoTable']['maxdepth'].values

    local_dist = np.full((ny, nx, maxl), np.nan)
    local_doi = np.zeros((ny, nx, maxl))
    local_depth = np.full((ny, nx, maxl), np.nan)
    local_model_thick = np.full((ny, nx, maxl), np.nan)
    complexities = np.full((ny, nx, maxl), np.nan)

    SearchRadius = 6
    comp2range = [100, 500, 400, 250, 100]
    comp2rangePreq = [500] * 5

    # Certainty function for RESLOG
    def cert_fun(dist, range_, width, sill):
        return np.where(dist <= width, sill * np.exp(-3 * (dist - width)**2 / range_**2), 0)

    for s in range(maxl):
        MDepths = MEP['Model_Depths'][:, s]
        MThicks = MEP['ThickGeophysModel'][:, s]
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
                    local_depths = MDepths[filt]
                    local_thicks = MThicks[filt]

                    distances = np.sqrt((xv - local_xs)**2 + (yv - local_ys)**2)
                    dist = np.min(distances)
                    filt_mindist = distances == dist

                    local_dist[i, j, s] = dist
                    #local_doi[i, j, s] = np.mean(DOIs[filt_mindist])
                    local_depth[i, j, s] = np.mean(local_depths[filt_mindist])
                    local_model_thick[i, j, s] = np.min(local_thicks[filt_mindist])

    complexity2 = np.where(complexities == 0, np.nan, complexities)
    rangemap = np.copy(complexity2)
    rangemap2 = np.copy(complexity2)

    for i in range(1, 5):
        rangemap[complexity2 == i] = comp2range[i]
        rangemap2[complexity2 == i] = comp2rangePreq[i]

    rangemap[:, :, Npreq:] = rangemap2[:, :, Npreq:]
    var0map = 0.25 + (0.5 * 0.09 * local_depth)**2
    var0map[local_depth < 10] = 100000
    kernelmap = np.full_like(local_depth, 150)

    certmap = cert_fun(local_dist, rangemap, kernelmap, 1.0 / var0map)
    Grid = 1.0 / certmap
    Grid[np.isnan(Grid)] = 100000

    doi_filter = local_model_thick < local_depth
    Grid[doi_filter] = 100000

    if include_peatlands:
        New_Grid = np.full((Grid.shape[0], Grid.shape[1], Grid.shape[2] + NPL), 1E4)
        New_Grid[:, :, NPL:] = Grid
        Grid = New_Grid

    s_cords = {'xs': xs, 'ys': ys}
    return Grid, s_cords