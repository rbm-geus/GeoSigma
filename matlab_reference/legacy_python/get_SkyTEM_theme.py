
import numpy as np
import pandas as pd
import scipy.io as sio
from certainty_function_definer import certainty_function_definer
from matio import load_from_mat
def get_SkyTEM_theme(UTM_X, UTM_Y, terrain, complexity, reg, include_peatlands, Npreq, NPL, Nlay, cert_fun_choice='FRAFA_apr2023'):
    # Load data based on region
    region_files = {
        'Jylland': 'jylland_skytem_til_usikkerhed.mat',
        'Fyn': 'fyn_skytem_til_usikkerhed.mat',
        'Fyn-MST': 'fyn_skytem_til_usikkerhed_MST.mat',
        'Sjælland': 'Sjaelland_skytem_til_usikkerhed.mat'
    }
    filename = region_files.get(reg)
    if not filename:
        raise ValueError("Invalid region specified")

    in_data = sio.loadmat(f'data/{filename}')
    Gmod = load_from_mat(f'data/{filename}')
    Gmod = Gmod['Gmod_pos']

    # Rearrange data
    if reg in ['Fyn', 'Fyn-MST']:
        info_table = pd.DataFrame(in_data['fynskytempos'])
    else:
        info_table = Gmod

    SkyTEM = {
        'InfoTable': info_table,
        'Model_Depths': in_data['dkm_d_pos'],
        'Model_Year': in_data['model_year_pos'].flatten(),
        'DepthInterfaceAboveModel': in_data['dkm_odvlayertop_pos'],
        'ThickGeophysModel': in_data['dkm_odvthk_pos']
    }

    Npoints = len(SkyTEM['Model_Year'])

    # Remove duplicates
    coords = SkyTEM['InfoTable'][['xutm_euref89_utm32', 'yutm_euref89_utm32']].values
    _, idx = np.unique(coords, axis=0, return_index=True)
    logind_dupe = np.zeros(Npoints, dtype=bool)
    logind_dupe[idx] = True

    # Filter by date
    model_year = SkyTEM['Model_Year']
    geophys_year = SkyTEM['InfoTable']['m_year'].values
    logind_date = geophys_year <= model_year
    logind = logind_dupe & logind_date

    # Apply filter
    for key in ['InfoTable', 'Model_Depths', 'Model_Year', 'DepthInterfaceAboveModel', 'ThickGeophysModel']:
        SkyTEM[key] = SkyTEM[key][logind]

    nx, ny = len(UTM_X), len(UTM_Y)
    maxl = SkyTEM['Model_Depths'].shape[1]

    comp2range = [100, 500, 400, 250, 100]
    comp2rangePreq = [500] * 5
    NP = np.sum(logind)

    if NP > 0:
        cert_fun = certainty_function_definer(cert_fun_choice)
        SearchRadius = 6

        xs = SkyTEM['InfoTable']['xutm_euref89_utm32'].values
        ys = SkyTEM['InfoTable']['yutm_euref89_utm32'].values
        DOIs = SkyTEM['InfoTable']['doilower'].values

        # DOI handling
        if reg == 'Jylland':
            PalDepth = SkyTEM['Model_Depths'][:, Npreq - NPL + 27]
            PalThick = SkyTEM['Model_Depths'][:, Npreq - NPL + 28] - SkyTEM['Model_Depths'][:, Npreq - NPL + 27]
        else:
            PalDepth = SkyTEM['Model_Depths'][:, Npreq - NPL - 1]
            PalThick = SkyTEM['Model_Depths'][:, Npreq - NPL] - SkyTEM['Model_Depths'][:, Npreq - NPL - 1]

        doimean = np.nanmean(DOIs)
        estimated_DOIs = np.minimum(doimean, PalDepth + 1e3 * (PalThick < 10))
        DOIs[np.isnan(DOIs)] = estimated_DOIs[np.isnan(DOIs)]

        local_dist = np.full((ny, nx, Nlay - NPL), np.nan)
        local_doi = np.zeros((ny, nx, Nlay - NPL))
        local_depth = np.full((ny, nx, Nlay - NPL), np.nan)
        local_model_thick = np.full((ny, nx, Nlay - NPL), np.nan)
        complexities = np.full((ny, nx, Nlay - NPL), np.nan)

        for s in range(Nlay - NPL):
            MDepths = SkyTEM['Model_Depths'][:, s]
            MThicks = SkyTEM['ThickGeophysModel'][:, s]
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
                        local_dois = DOIs[filt]

                        distances = np.sqrt((xv - local_xs)**2 + (yv - local_ys)**2)
                        dist = np.min(distances)
                        filt_mindist = distances == dist

                        local_dist[i, j, s] = dist
                        local_doi[i, j, s] = np.mean(local_dois[filt_mindist])
                        local_depth[i, j, s] = np.mean(local_depths[filt_mindist])
                        local_model_thick[i, j, s] = np.min(local_thicks[filt_mindist])

        complexity2 = np.where(complexities == 0, np.nan, complexities)
        rangemap = np.copy(complexity2)
        rangemap2 = np.copy(complexity2)

        for i in range(1, 5):
            rangemap[complexity2 == i] = comp2range[i]
            rangemap2[complexity2 == i] = comp2rangePreq[i]

        rangemap[:, :, Npreq - NPL:] = rangemap2[:, :, Npreq - NPL:]
        var0map = (0.5 * 1.2 * local_model_thick) ** 2
        var0map[local_depth < 5] = 100000
        kernelmap = np.maximum(2 * local_depth, 75)

        certmap = cert_fun(local_dist, rangemap, kernelmap, 1.0 / var0map)
        Grid = 1.0 / certmap
        Grid[np.isnan(Grid)] = 100000

        doi_filter = local_doi < local_depth
        Grid[doi_filter] = 100000

        if include_peatlands:
            New_Grid = np.full((Grid.shape[0], Grid.shape[1], Grid.shape[2] + NPL), 1E4)
            New_Grid[:, :, NPL:] = Grid
            Grid = New_Grid

        s_cords = {'xs': xs, 'ys': ys}
        return Grid, s_cords
