
import numpy as np
import pandas as pd
import scipy.io as sio
from certainty_function_definer import certainty_function_definer
from matio import load_from_mat
def get_MEP_theme(UTM_X, UTM_Y, terrain, complexity, reg, include_peatlands, Npreq, NPL, Nlay, cert_fun_choice='FRAFA_apr2023'):
    # Load data based on region
    region_files = {
        'Jylland': 'jylland_mep_til_usikkerhed.mat',
        'Fyn': 'fyn_mep_til_usikkerhed.mat',
        'Fyn-MST': 'fyn_mep_til_usikkerhed_MST.mat',
        'Sjælland': 'Sjaelland_mep_til_usikkerhed.mat'
    }
    filename = region_files.get(reg)
    if not filename:
        raise ValueError("Invalid region specified")

    in_data = sio.loadmat(f'data/{filename}')
    Gmod = load_from_mat(f'data/{filename}')
    Gmod = Gmod['Gmod_pos']

    # Rearrange data
    if reg in ['Fyn', 'Fyn-MST']:
        info_table = pd.DataFrame(in_data['fynmeppos'])
    else:
        info_table = Gmod

    MEP = {
        'InfoTable': info_table,
        'Model_Depths': in_data['dkm_d_pos'],
        'Model_Year': in_data['model_year_pos'].flatten(),
        'DepthInterfaceAboveModel': in_data['dkm_odvlayertop_pos'],
        'ThickGeophysModel': in_data['dkm_odvthk_pos']
    }

    Npoints = len(MEP['Model_Year'])

    # Determine type (wennera_2d)
    typevec = np.zeros(Npoints, dtype=int)
    types = MEP['InfoTable']['datasubtype'].values
    for i, cur_string in enumerate(types):
        cur_string = str(cur_string).lower()
        typevec[i] = int(cur_string.startswith('wen'))
    MEP['Type'] = typevec

    # Remove duplicates
    coords = MEP['InfoTable'][['xutm_euref89_utm32', 'yutm_euref89_utm32']].values
    _, idx = np.unique(coords, axis=0, return_index=True)
    logind_dupe = np.zeros(Npoints, dtype=bool)
    logind_dupe[idx] = True

    # Filter by date
    model_year = MEP['Model_Year']
    geophys_year = MEP['InfoTable']['m_year'].values
    logind_date = geophys_year <= model_year
    logind = logind_dupe & logind_date

    # Apply filter
    for key in ['InfoTable', 'Model_Depths', 'Model_Year', 'DepthInterfaceAboveModel', 'ThickGeophysModel']:
        MEP[key] = MEP[key][logind]
    MEP['Type'] = MEP['Type'][logind]

    nx, ny = len(UTM_X), len(UTM_Y)
    maxl = MEP['Model_Depths'].shape[1]

    comp2range = [100, 500, 400, 250, 100]
    comp2rangePreq = [500] * 5
    NP = np.sum(logind)

    if NP > 0:
        cert_fun = certainty_function_definer(cert_fun_choice)
        SearchRadius = 2

        xs = MEP['InfoTable']['xutm_euref89_utm32'].values
        ys = MEP['InfoTable']['yutm_euref89_utm32'].values
        DOIs = MEP['InfoTable']['doilower'].values

        local_dist = np.full((ny, nx, Nlay - NPL), np.nan)
        local_doi = np.zeros((ny, nx, Nlay - NPL))
        local_depth = np.full((ny, nx, Nlay - NPL), np.nan)
        local_model_thick = np.full((ny, nx, Nlay - NPL), np.nan)
        local_MEP_type = np.full((ny, nx, Nlay - NPL), np.nan)
        complexities = np.full((ny, nx, Nlay - NPL), np.nan)

        for s in range(Nlay - NPL):
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
                        local_dois = DOIs[filt]
                        local_thicks = MThicks[filt]
                        local_types = MEP['Type'][filt]

                        distances = np.sqrt((xv - local_xs)**2 + (yv - local_ys)**2)
                        dist = np.min(distances)
                        filt_mindist = distances == dist

                        local_dist[i, j, s] = dist
                        local_doi[i, j, s] = np.mean(local_dois[filt_mindist])
                        local_depth[i, j, s] = np.mean(local_depths[filt_mindist])
                        local_model_thick[i, j, s] = np.mean(local_thicks[filt_mindist])
                        local_MEP_type[i, j, s] = local_types[0]

        local_doi[np.isnan(local_doi)] = np.nanmean(DOIs)

        complexity2 = np.where(complexities == 0, np.nan, complexities)
        rangemap = np.copy(complexity2)
        rangemap2 = np.copy(complexity2)

        WenneraMap = local_MEP_type
        NotWenneraMap = WenneraMap == 0
        WenneraVar0 = 0.5 * np.maximum(2.4, 0.25 * local_depth)
        NotWenneraVar0 = 0.5 * np.maximum(1.25, 0.15 * local_depth)
        var0map = (WenneraVar0 * WenneraMap + NotWenneraVar0 * NotWenneraMap) ** 2

        for i in range(1, 5):
            rangemap[complexity2 == i] = comp2range[i]
            rangemap2[complexity2 == i] = comp2rangePreq[i]

        rangemap[:, :, Npreq - NPL:] = rangemap2[:, :, Npreq - NPL:]
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

        s_cords = {'xs': xs, 'ys': ys}
        return Grid, s_cords
