# -*- coding: utf-8 -*-
#
# GeoSigma — original work.
# Author:     Rasmus Bødker Madsen (rbm@geus.dk)
# Co-authors: Frederik Falk; Claude (Anthropic)
"""geosigma.kriging — local kriging driver and simulation.

Working-state packing (``setup_GLOBAL_structure``), buffer-zone logic
(``calculate_buffer_zones``), and the cluster-element loop that sets up and
solves each local kriging system, draws realizations, and blends overlapping
windows (``local_kriger``).

Consumes a variance-map stack (from ``geosigma.themes`` or user-supplied) plus
clusters/range-sill from ``preprocess``; never requires the built-in theme
builder.

Scaffold only (Phase 0); populated in **Phase 6**.
"""
