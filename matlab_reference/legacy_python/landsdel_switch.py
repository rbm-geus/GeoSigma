
def landsdel_switch(landsdel, redo_topography=False, topo_APR2023=False):
    """
    Python equivalent of MATLAB landsdel_switch.
    Returns region, NPL, Nlay, layernames, Ninterfaces, Npreq, filetype, wellname_interpreted, wellname_not_interpreted, wellname.
    """

    # Initialize variables
    region = None
    NPL = None
    layernames = None
    Ninterfaces = None
    Npreq = None
    filetype = '.asc'
    wellname_interpreted = None
    wellname_not_interpreted = None
    wellname = None

    # Switch logic
    if landsdel == 1:
        region = 'Jylland'
        NPL = 5
        layernames, Ninterfaces, Npreq = get_layer_names(region, NPL)
        wellname_interpreted = 'Jylland_tolkningspunkter_med_boringssnap.mat'
        wellname_not_interpreted = 'Jylland_jup_boringer_til_usikkerhed.mat'
        wellname = [wellname_interpreted, wellname_not_interpreted]

        if topo_APR2023:
            layernames[0] = 'dk456_topo100m'
            print('Used topo from DK-model April 2023')

    elif landsdel == 2:
        region = 'Fyn'
        NPL = 3
        layernames, Ninterfaces, Npreq = get_layer_names(region, NPL)
        wellname_interpreted = 'Fyn_tolkningspunkter_med_boringssnap.mat'
        wellname_not_interpreted = 'fyn_jup_boringer_til_usikkerhed.mat'
        wellname = [wellname_interpreted, wellname_not_interpreted]

        if topo_APR2023:
            layernames[0] = 'dk3_topo100m'
            print('Used topo from DK-model April 2023')

    elif landsdel == 22:
        region = 'Fyn-MST'
        NPL = 0
        layernames, Ninterfaces, Npreq = get_layer_names(region, NPL)
        wellname_interpreted = 'Fyn_tolkningspunkter_med_boringssnap.mat'
        wellname_not_interpreted = 'fyn_jup_boringer_til_usikkerhed_MST.mat'
        wellname = [wellname_interpreted, wellname_not_interpreted]

    elif landsdel == 3:
        region = 'Sjælland'
        NPL = 0
        layernames, Ninterfaces, Npreq = get_layer_names(region, NPL)
        wellname_interpreted = 'Sjaelland_tolkningspunkter_med_boringssnap.mat'
        wellname_not_interpreted = 'sjaelland_jup_boringer_til_usikkerhed.mat'
        wellname = [wellname_interpreted, wellname_not_interpreted]

        if topo_APR2023:
            layernames[0] = 'dk12_topo100m'
            print('Used topo from DK-model April 2023')

    elif landsdel == 4:
        region = 'AnholtLæsø'
        NPL = 5
        layernames, Ninterfaces, Npreq = get_layer_names(region, NPL)
        wellname_interpreted = 'DK8_tolkningspunkter_med_boringssnap.mat'
        wellname_not_interpreted = 'DK8_jup_boringer_til_usikkerhed.mat'
        wellname = [wellname_interpreted, wellname_not_interpreted]

        if topo_APR2023:
            print('Option cannot be set for Anholt and Læsø')

    # Calculate Nlay
    if region != 'AnholtLæsø':
        Nlay = Ninterfaces - 2  # Minus terrain and bottom
    else:
        Nlay = Ninterfaces - 1  # No bottom for AnholtLæsø

    if region == 'Jylland':
        Nlay -= 1  # Chalk not modeled

    return region, NPL, Nlay, layernames, Ninterfaces, Npreq, filetype, wellname_interpreted, wellname_not_interpreted, wellname
