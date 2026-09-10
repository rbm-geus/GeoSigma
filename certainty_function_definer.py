import numpy as np

def certainty_function_definer(cert_fun_choice):
    if cert_fun_choice == 'FRAFA_apr2023':
        def cert_fun(dist, range_, width, sill):
            return np.where(dist <= width, sill, sill * np.exp(-3 * (dist - width)**2 / range_**2))
    
    elif cert_fun_choice == 'ILM_sep2023':
        # Multiplicative MATLAB form (dist<=width).*sill.*exp(...), NOT
        # np.where(..., 0): at a no-data cell dist (or range) is NaN, and
        # 0*NaN=NaN propagates to the NODATA sentinel just like MATLAB's
        # Grid(isnan)=100000 — whereas np.where would leak a literal 0 -> 1/0=+inf.
        def cert_fun(dist, range_, width, sill):
            return (dist <= width) * sill * np.exp(-3 * (dist - width)**2 / range_**2)

    elif cert_fun_choice == 'ILM_oct2023':
        def cert_fun(dist, range_, width, sill):
            return (dist <= width) * sill * np.exp(-3 * dist**2 / range_**2)
    
    elif cert_fun_choice == 'RBM_oct2023':
        def cert_fun_rbm_oct_ins(dist, range_, width, sill, damp):
            return sill * np.exp(-damp * dist**2 / range_**2)

        def cert_fun_rbm_oct_out(dist, range_, width, sill):
            return sill * np.exp(-3 * dist**2 / range_**2)

        def cert_fun(dist, range_, width, sill):
            inside = dist <= width
            outside = dist > width
            sill_adjusted = sill + (sill - cert_fun_rbm_oct_ins(width, range_, width, sill, 1.5))
            return np.where(inside,
                            cert_fun_rbm_oct_ins(dist, range_, width, sill, 1.5),
                            cert_fun_rbm_oct_out(dist, range_, width, sill_adjusted))
    else:
        raise ValueError(f"Unknown certainty function choice: {cert_fun_choice}")

    return cert_fun