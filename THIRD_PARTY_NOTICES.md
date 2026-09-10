# Third-Party Notices

GeoSigma itself is distributed under the ISC License (see `LICENSE`). Parts of
it are derived from third-party software; this file reproduces the required
copyright notices and license texts for those parts.

## mGstat

Upstream source: <https://github.com/AUProbGeo/mGstat>
Author: Thomas Mejer Hansen
License: MIT

Several of GeoSigma's low-level geostatistical routines are Python
translations of MATLAB functions from mGstat. The MIT License requires that
the copyright notice and permission notice be retained in all copies and
substantial portions of the software, which is the purpose of this file.

### License text

```
MIT License

Copyright (c) 2024 Thomas Mejer Hansen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Which GeoSigma modules derive from mGstat

| GeoSigma module | mGstat routine | Relationship |
|---|---|---|
| `geosigma/core/deformat_variogram.py` | `deformat_variogram.m` | Direct translation |
| `geosigma/core/edist.py` | `edist.m` | Direct translation (isotropic case only) |
| `geosigma/core/semivar_synth.py` | `semivar_synth.m` | Direct translation |
| `geosigma/core/precal_cov.py` | `precal_cov.m` | Direct translation |
| `geosigma/core/least_squares_inversion.py` | `least_squares_inversion.m` | Direct translation (Tarantola 2005 Type 1/2) |

Each of those files carries the mGstat copyright and MIT notice in its own
header, so the notice travels with the file if it is copied out of this
repository.

Calling one of the routines above does not make the caller a derived work, so
modules that merely use them — `geosigma/core/local_kriging_setup_img.py`, for
instance — are original GeoSigma code and carry no mGstat attribution.

**For readers comparing the two libraries:** `geosigma/core/get_reals_cholesky.py`
generates realizations by Cholesky decomposition of the covariance matrix, the
same well-known approach that mGstat implements in
`gaussian_simulation_cholesky.m`. It is not derived from it. It is translated
from the N-ret MATLAB codebase's own `get_reals_cholesky.m`, and the method
itself is textbook material rather than anyone's invention. This note is here
to answer the resemblance, not to claim a debt.

The remainder of GeoSigma — the theme engine, preprocessing, clustering,
kriging driver, assembly and I/O — is original work or a translation of the
N-ret hydrostratigraphic MATLAB codebase, not of mGstat. GeoSigma is not a
fork or a wholesale translation of mGstat.

### A note on the SourceForge copy of mGstat

An older copy of mGstat is still published on SourceForge under GPL-2.0. That
copy is **outdated and its license statement is incorrect**; it should have
been removed. The current and authoritative source is
<https://github.com/AUProbGeo/mGstat>, which is MIT-licensed. This was
confirmed directly with the author, Thomas Mejer Hansen, and verified against
the `LICENSE` file in the GitHub repository. MIT is compatible with ISC, so
no copyleft obligation attaches to GeoSigma.
