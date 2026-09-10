import numpy as np
import pandas as pd
import scipy.io as sio
from certainty_function_definer import certainty_function_definer
def get_PACES_theme(UTM_X, UTM_Y, terrain, complexity, reg, include_peatlands, Npreq, NPL, Nlay, cert_fun_choice='FRAFA_apr2023'):
    # Load data based on region
    region_files = {
        'Jylland': 'Jylland_paces_til_usikkerhed.mat',
        'Fyn': 'fyn_paces_til_usikkerhed.mat',
        'Fyn-MST': 'fyn_paces_til_usikkerhed_MST.mat',
        'Sjælland': 'sjaelland_paces_til_usikkerhed.mat'
    }
    filename = region_files.get(reg)
    if not filename:
        raise ValueError("Invalid region specified")

    # TODO: Replace with actual loading using scipy.io.loadmat
    in_data = sio.loadmat(f'data\{filename}')
    Gmod = pd.read_csv(f'data\Gmod_pos.csv')
    # Rearranging data
    if reg in ['Jylland', 'Sjælland']:
        info_table = Gmod 
    else:
        info_table = in_data['fynpacespos']

    PACES = {
        'InfoTable': info_table,
        'Model_Depths': in_data['dkm_d_pos'],
        'Model_Year': in_data['model_year_pos'],
        'DepthInterfaceAboveModel': in_data['dkm_odvlayertop_pos'],
        'ThickGeophysModel': in_data['dkm_odvthk_pos']
    }

    coords = PACES['InfoTable'][['xutm_euref89_utm32', 'yutm_euref89_utm32']].values
    _, idx = np.unique(coords, axis=0, return_index=True)
    logind_dupe = np.zeros(len(coords), dtype=bool)
    logind_dupe[idx] = True

    model_year = PACES['Model_Year']
    geophys_year = PACES['InfoTable']['m_year'].values
    logind_date = geophys_year <= model_year.flatten()
 
    logind = logind_dupe & logind_date

    for key in ['InfoTable', 'Model_Depths', 'Model_Year', 'DepthInterfaceAboveModel', 'ThickGeophysModel']:
        PACES[key] = PACES[key][logind]

    nx, ny = len(UTM_X), len(UTM_Y)
    maxl = PACES['Model_Depths'].shape[1]
    Grid = np.zeros((ny, nx, maxl))

    # Placeholder certainty function
    cert_fun = certainty_function_definer(cert_fun_choice)


    xs = PACES['InfoTable']['xutm_euref89_utm32'].values
    ys = PACES['InfoTable']['yutm_euref89_utm32'].values
    DOIs = PACES['InfoTable']['doilower'].values

    local_dist = np.full((ny, nx, Nlay - NPL), np.nan)
    local_doi = np.zeros((ny, nx, Nlay - NPL))
    local_depth = np.full((ny, nx, Nlay - NPL), np.nan)
    local_model_thick = np.full((ny, nx, Nlay - NPL), np.nan)
    complexities = np.zeros((ny, nx, Nlay - NPL))

    for s in range(Nlay - NPL):
        MDepths = PACES['Model_Depths'][:, s]
        MThicks = PACES['ThickGeophysModel'][:, s]
        complexities[:, :, s] = complexity

        for i in range(ny):
            yv = UTM_Y[i]
            ymin = UTM_Y[max(0, i - 1)]
            ymax = UTM_Y[min(ny - 1, i + 1)]
            filt1 = (ys < ymax) & (ys > ymin)

            for j in range(nx):
                xv = UTM_X[j]
                xmin = UTM_X[max(0, j - 1)]
                xmax = UTM_X[min(nx - 1, j + 1)]
                filt2 = (xs < xmax) & (xs > xmin)
                filt = filt1 & filt2

                if np.any(filt):
                    local_xs = xs[filt]
                    local_ys = ys[filt]
                    local_depths = MDepths[filt]
                    local_dois = DOIs[filt]
                    local_thicks = MThicks[filt]

                    distances = np.sqrt((xv - local_xs)**2 + (yv - local_ys)**2)
                    dist = np.min(distances)
                    filt_mindist = distances == dist

                    local_dist[i, j, s] = dist
                    local_doi[i, j, s] = np.mean(local_dois[filt_mindist])
                    local_depth[i, j, s] = np.mean(local_depths[filt_mindist])
                    local_model_thick[i, j, s] = np.mean(local_thicks[filt_mindist])

    local_doi[:] = 18


    complexity2 = np.where(complexities == 0, np.nan, complexities)

    comp2range = [100, 500, 400, 250, 100]
    comp2rangePreq = [500] * 5

    rangemap = np.copy(complexity2)
    rangemap2 = np.copy(complexity2)

    for i in range(1, 5):
        rangemap[complexity2 == i] = comp2range[i]
        rangemap2[complexity2 == i] = comp2rangePreq[i]

    rangemap[:, :, Npreq - NPL:] = rangemap2[:, :, Npreq - NPL:]
    var0map = (0.5 * np.maximum(2, 0.25 * local_depth)) ** 2
    kernelmap = np.full_like(local_depth, 75)

    certmap = cert_fun(local_dist, rangemap, kernelmap, 1.0 / var0map)
    Grid = 1.0 / certmap
    Grid[np.isnan(Grid)] = 100000
    
    doi_filter = local_doi < local_depth
    Grid[doi_filter] = 100000


    if include_peatlands:
        New_Grid = np.full((Grid.shape[0], Grid.shape[1], Grid.shape[2] + NPL), 1E4)
        New_Grid[:, :, NPL:] = Grid
        Grid = New_Grid

    s_cords = {
        'xs': xs,
        'ys': ys
    }

    return Grid, s_cords