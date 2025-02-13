#!/usr/bin/env python3
"""
fully_dynamic_schema.py

Refactored with a simpler main() that:
1) Iterates over *all* schemas, each with a unique name + columns + generator_func.
2) Generates a small table for each schema (e.g., 4 rows).
3) Prints both the DataFrame representation (so you can see columns & random data)
   and a JSON representation (so you can see how you'd store or export each row).

Usage:
  python fully_dynamic_schema.py
"""

import numpy as np
import pandas as pd
import json
from typing import Any, Dict, List, Optional, Set, Union

# =============================================================================
# MASTER_OPTIONS: The central dictionary for filler/matrix/treatment terms
# =============================================================================
MASTER_OPTIONS = {
    "unfilled": ["unfilled", "none", "no filler", "unloaded"],
    "size_descriptors": [
        "nano", "micro", "fine", "coarse",
        "giga", "pico", "normal", "hybrid",
        "surface-modified"
    ],
    "surface_treatments": [
        "untreated", "vinyl silane", "KH-550",
        "silanized", "amine-functionalized", "none",
        "treated", "VT"
    ],
    "filler_bases": [
        "silica", "alumina", "titania", "ZnO", "MMT",
        "nanosilica", "nanoclay", "BaTiO3", "Graphene",
        "SiO2", "carbon nanotubes", "clay", "mica",
        "boron nitride", "montmorillonite"
    ],
    "matrix_materials": [
        "epoxy resin", "unfilled epoxy resin", "polyimide", "LDPE",
        "XLPE", "HDPE", "PE", "PP", "polypropylene",
        "bio-based epoxy", "polymer A", "resin B",
        "polycarbonate", "ABS", "PA6", "PI", "LCP"
    ],
}

# Additional sets for certain specialized schemas
TREATED_NANOCOMPOSITES = [
    "Micro",
    "Untreated nano",
    "Vinylsilane treated nano",
    "Untreated nanocomposite",
    "Microcomposite",
    "Vinylsilane-treated nanocomposite",
    "Aminosilane-treated nanocomposite",
    "HMDS-treated nanocomposite",
]

class SynonymPool:
    """
    Optional helper for synonyms, though mostly we use MASTER_OPTIONS directly.
    """
    def __init__(self, data: Dict[str, List[str]]):
        self.data = data

    def get_synonym(self, concept: str, rng: np.random.Generator) -> str:
        if concept not in self.data or not self.data[concept]:
            return concept
        idx = rng.integers(len(self.data[concept]))
        return self.data[concept][idx]


def unify_unfilled(rng: np.random.Generator, p_unfilled=0.2) -> Optional[str]:
    """
    With probability p_unfilled, pick a random "unfilled" synonym 
    from MASTER_OPTIONS["unfilled"].
    """
    if rng.random() < p_unfilled:
        return rng.choice(MASTER_OPTIONS["unfilled"])
    return None


def pick_unique_label(rng: np.random.Generator, used: Optional[Set[str]], 
                      label_generator) -> str:
    """
    Attempts up to 100 times to pick a label that's not in 'used'.
    Returns "MaxAttempts_Exceeded" if unable to find one.
    """
    if used is None:
        used = set()

    for _ in range(100):
        candidate = label_generator()
        if candidate not in used:
            used.add(candidate)
            return candidate

    return "MaxAttempts_Exceeded"


###############################################################################
# SCHEMA GENERATOR FUNCTIONS (all accept `used: Optional[Set[str]] = None`)
###############################################################################

def generate_weibull_epoxy_tio2_row(rng: np.random.Generator, used: Optional[Set[str]] = None) -> List[str]:
    def label_generator():
        if rng.random() < 0.2:
            return rng.choice(MASTER_OPTIONS["unfilled"])
        c = round(float(rng.uniform(0.1,10.0)),1)
        desc = rng.choice(MASTER_OPTIONS["size_descriptors"])
        base = rng.choice(MASTER_OPTIONS["filler_bases"])
        matrix = rng.choice(MASTER_OPTIONS["matrix_materials"])
        if rng.random() < 0.3:
            treat = rng.choice(MASTER_OPTIONS["surface_treatments"])
            return f"{c}% {desc} {base} ({treat}) {matrix}"
        else:
            return f"{c}% {desc} {base} {matrix}"

    comp = pick_unique_label(rng, used, label_generator)
    shape = round(float(rng.uniform(5,30)),3)
    scale = round(float(rng.uniform(10,60)),2)
    return [comp, str(shape), str(scale)]


def generate_weibull_epoxy_al2o3_row(rng: np.random.Generator, used: Optional[Set[str]] = None) -> List[str]:
    def label_generator():
        if rng.random()<0.2:
            return rng.choice(MASTER_OPTIONS["unfilled"])
        c = round(float(rng.uniform(0.1,10.0)),1)
        desc = rng.choice(MASTER_OPTIONS["size_descriptors"])
        base = rng.choice(MASTER_OPTIONS["filler_bases"])
        matrix = rng.choice(MASTER_OPTIONS["matrix_materials"])
        return f"{c}% {desc} {base} {matrix}"

    comp = pick_unique_label(rng, used, label_generator)
    shape = round(float(rng.uniform(5,20)),3)
    scale = round(float(rng.uniform(25,55)),2)
    return [comp, str(shape), str(scale)]


def generate_weibull_epoxy_zno_row(rng: np.random.Generator, used: Optional[Set[str]] = None) -> List[str]:
    def label_generator():
        if rng.random() < 0.2:
            return rng.choice(MASTER_OPTIONS["unfilled"])
        c_amt = round(float(rng.uniform(0.1,20.0)),1)
        desc = rng.choice(MASTER_OPTIONS["size_descriptors"])
        base = rng.choice(MASTER_OPTIONS["filler_bases"])
        matrix = rng.choice(MASTER_OPTIONS["matrix_materials"])
        return f"{c_amt}% {desc} {base} {matrix}"

    comp = pick_unique_label(rng, used, label_generator)

    def maybe_na():
        if rng.random() < 0.2:
            return "not available"
        return str(round(float(rng.uniform(5,40)),2))

    nano_val = maybe_na()
    micro_val = maybe_na()
    return [comp, nano_val, micro_val]


def generate_mech_eva_row(rng: np.random.Generator, used: Optional[Set[str]] = None) -> List[str]:
    def label_generator():
        step = int(rng.integers(0,6))
        load = step*2
        if load == 0:
            return "Pure " + rng.choice(MASTER_OPTIONS["matrix_materials"])
        else:
            desc = rng.choice(MASTER_OPTIONS["size_descriptors"])
            base = rng.choice(MASTER_OPTIONS["filler_bases"])
            matrix = rng.choice(MASTER_OPTIONS["matrix_materials"])
            return f"{load} wt% {desc} {base} {matrix}"

    s_str = pick_unique_label(rng, used, label_generator)
    tens = round(float(rng.uniform(3,10)),1)
    mod_300 = round(float(rng.uniform(0.3,2.0)),2)
    elong = rng.integers(500,2001)
    return [s_str, str(tens), str(mod_300), str(elong)]


def generate_dynamic_mech_eva_row(rng: np.random.Generator, used: Optional[Set[str]] = None) -> List[str]:
    def label_generator():
        step = int(rng.integers(0,6))
        load = step*2
        if load == 0:
            return "Pure " + rng.choice(MASTER_OPTIONS["matrix_materials"])
        else:
            desc = rng.choice(MASTER_OPTIONS["size_descriptors"])
            base = rng.choice(MASTER_OPTIONS["filler_bases"])
            matrix = rng.choice(MASTER_OPTIONS["matrix_materials"])
            return f"{load} wt% {desc} {base} {matrix}"

    s_str = pick_unique_label(rng, used, label_generator)
    T_c = rng.integers(-35,-25)
    E_t8 = f"{round(rng.uniform(0.05,3.5),2)} x 10^{rng.integers(6,9)}"
    E_30 = f"{round(rng.uniform(0.1,4.0),2)} x 10^{rng.integers(5,8)}"
    tan_tg = round(float(rng.uniform(0.3,1.2)),2)
    tan_30 = round(float(rng.uniform(0.05,0.4)),2)
    return [s_str, str(T_c), E_t8, E_30, str(tan_tg), str(tan_30)]


def generate_activation_energies_tio2_row(rng: np.random.Generator, used: Optional[Set[str]] = None) -> List[str]:
    below_nano = str(round(float(rng.uniform(0.5,2.5)),1))
    above_micro = str(round(float(rng.uniform(0.5,2.5)),1))
    return [below_nano, above_micro]


def generate_breakdown_strength_xlpe_row(rng: np.random.Generator, used: Optional[Set[str]] = None) -> List[str]:
    def label_generator():
        if rng.random()<0.3:
            return rng.choice(MASTER_OPTIONS["unfilled"]) + " " + rng.choice(MASTER_OPTIONS["matrix_materials"])
        load = round(float(rng.uniform(2,15)),1)
        desc = rng.choice(MASTER_OPTIONS["size_descriptors"])
        base = rng.choice(MASTER_OPTIONS["filler_bases"])
        treat = rng.choice(MASTER_OPTIONS["surface_treatments"])
        matrix = rng.choice(MASTER_OPTIONS["matrix_materials"])
        return f"{load}% {desc}{base}({treat}) {matrix}"

    mat_str = pick_unique_label(rng, used, label_generator)

    def combo():
        val = rng.integers(80,501)
        shape = round(float(rng.uniform(1,8)),1)
        return f"{val} ({shape})"
    return [mat_str, combo(), combo()]


def generate_lichtenecker_rother_row(rng: np.random.Generator, used: Optional[Set[str]] = None) -> List[str]:
    def label_generator():
        c = round(float(rng.uniform(1,15)),1)
        desc = rng.choice(MASTER_OPTIONS["size_descriptors"])
        base = rng.choice(MASTER_OPTIONS["filler_bases"])
        matrix = rng.choice(MASTER_OPTIONS["matrix_materials"])
        if rng.random() < 0.2:
            return rng.choice(MASTER_OPTIONS["unfilled"]) + " " + matrix
        return f"{c} wt% {desc} {base} {matrix}"

    mat_str = pick_unique_label(rng, used, label_generator)
    freq_val = rng.choice([1e3,1e4,1e5,1e6])
    lr_val = str(round(float(rng.uniform(2,10)),2)) if rng.random()<0.7 else ""
    e_val = str(round(float(rng.uniform(2,20)),1)) if rng.random()<0.8 else ""
    return [mat_str, str(freq_val), lr_val, e_val]


def generate_current_decay_exponent_row(rng: np.random.Generator, used: Optional[Set[str]] = None) -> List[str]:
    def label_generator():
        if rng.random()<0.4:
            return "unfilled " + rng.choice(MASTER_OPTIONS["matrix_materials"])
        load_val = round(float(rng.uniform(2,15)),1)
        desc = rng.choice(MASTER_OPTIONS["size_descriptors"])
        base = rng.choice(MASTER_OPTIONS["filler_bases"])
        treat = rng.choice(MASTER_OPTIONS["surface_treatments"])
        mat_ = rng.choice(MASTER_OPTIONS["matrix_materials"])
        return f"{load_val}% {desc} {base}({treat}) {mat_}"

    mat_str = pick_unique_label(rng, used, label_generator)
    c_val = round(float(rng.uniform(0.5,2.0)),2)
    return [mat_str, str(c_val)]


def generate_impulse_test_breakdown_row(rng: np.random.Generator, used: Optional[Set[str]] = None) -> List[str]:
    def label_generator():
        unfilled_label = unify_unfilled(rng, p_unfilled=0.3)
        if unfilled_label:
            return f"{unfilled_label} {rng.choice(MASTER_OPTIONS['matrix_materials'])}"
        else:
            load_val = round(float(rng.uniform(5,20)),1)
            fill_desc = rng.choice(MASTER_OPTIONS["size_descriptors"])
            fill_base = rng.choice(MASTER_OPTIONS["filler_bases"])
            treat = rng.choice(MASTER_OPTIONS["surface_treatments"])
            base = rng.choice(MASTER_OPTIONS["matrix_materials"])
            return f"{load_val}% {fill_desc} {fill_base} ({treat}) {base}"

    mat_str = pick_unique_label(rng, used, label_generator)
    main_val = int(rng.integers(200,401))
    shape_val = round(float(rng.uniform(2,6)),1)
    return [mat_str, f"{main_val} ({shape_val})"]


def generate_polarization_space_charge_row(rng: np.random.Generator, used: Optional[Set[str]] = None) -> List[str]:
    def label_generator():
        unfilled_label = unify_unfilled(rng, p_unfilled=0.3)
        if unfilled_label:
            return f"{unfilled_label} {rng.choice(MASTER_OPTIONS['matrix_materials'])}"
        else:
            load_val = round(float(rng.uniform(5,20)),1)
            fill_desc = rng.choice(MASTER_OPTIONS["size_descriptors"])
            fill_base = rng.choice(MASTER_OPTIONS["filler_bases"])
            base_mat = rng.choice(MASTER_OPTIONS["matrix_materials"])
            return f"{load_val}% {fill_desc} {fill_base}-filled {base_mat}"

    mat_str = pick_unique_label(rng, used, label_generator)
    pol_val = int(rng.integers(10,120))
    sc_val = int(rng.integers(500,10001))
    return [mat_str, str(pol_val), str(sc_val)]


def generate_ac_breakdown_scale_params_row(rng: np.random.Generator, used: Optional[Set[str]] = None) -> List[str]:
    def label_generator():
        temps = [200,250,300,350]
        t_c = int(rng.choice(temps))
        unfilled_label = unify_unfilled(rng, p_unfilled=0.3)
        if unfilled_label:
            return f"{unfilled_label} at {t_c}°C"
        else:
            load_val = round(float(rng.uniform(1,15)),1)
            treat = rng.choice(["untr","tr","none","VT"])
            fill_base = rng.choice(MASTER_OPTIONS["filler_bases"])
            return f"{load_val} wt% {treat} {fill_base} at {t_c}°C"

    matc = pick_unique_label(rng, used, label_generator)
    n_val = int(rng.integers(50,300))
    shape_val = int(rng.integers(1,8))
    pct_dec = int(rng.integers(1,70))
    return [matc, str(n_val), str(shape_val), str(pct_dec)]


def generate_dc_breakdown_scale_params_row(rng: np.random.Generator, used: Optional[Set[str]] = None) -> List[str]:
    def label_generator():
        temps = [200,250,300,350]
        t_c = int(rng.choice(temps))
        unfilled_label = unify_unfilled(rng, p_unfilled=0.3)
        if unfilled_label:
            return f"{unfilled_label} at {t_c}°C"
        else:
            load_val = round(float(rng.uniform(1,15)),1)
            treat = rng.choice(["untr","tr","none","VT"])
            fill_base = rng.choice(MASTER_OPTIONS["filler_bases"])
            return f"{load_val} wt% {treat} {fill_base} at {t_c}°C"

    matc = pick_unique_label(rng, used, label_generator)
    scale_val = int(rng.integers(50,400))
    shape_val = int(rng.integers(1,15))
    pct_dec = int(rng.integers(1,80))
    return [matc, str(scale_val), str(shape_val), str(pct_dec)]


def generate_dc_breakdown_scale_params_dual_row(rng: np.random.Generator, used: Optional[Set[str]] = None) -> List[str]:
    def label_generator():
        unfilled_label = unify_unfilled(rng, p_unfilled=0.3)
        if unfilled_label:
            return unfilled_label
        else:
            load_val = round(float(rng.uniform(1,15)),1)
            treat = rng.choice(["untr","tr","none","VT"])
            fill_base = rng.choice(MASTER_OPTIONS["filler_bases"])
            return f"{load_val} wt% {treat} {fill_base}"

    matc = pick_unique_label(rng, used, label_generator)
    scale1 = int(rng.integers(50,400))
    shape1 = int(rng.integers(1,15))
    dec1 = int(rng.integers(1,80))
    scale2 = int(rng.integers(50,400))
    shape2 = int(rng.integers(1,15))
    dec2 = int(rng.integers(1,80))
    return [matc, str(scale1), str(shape1), str(dec1), str(scale2), str(shape2), str(dec2)]


def generate_pi_ofg_elemental_analysis_row(rng: np.random.Generator, used: Optional[Set[str]] = None) -> List[str]:
    load_choices = [0,3,7,10,15]
    feed = f"{rng.choice(load_choices)}%"
    n_val = round(float(rng.uniform(5.5,8.0)),2)
    c_val = round(float(rng.uniform(60,70)),2)
    h_val = round(float(rng.uniform(2.8,3.5)),2)
    return [feed, str(n_val), str(c_val), str(h_val)]


def generate_pi_ofg_dielectric_density_row(rng: np.random.Generator, used: Optional[Set[str]] = None) -> List[str]:
    load_choices = [0,3,7,10,15]
    feed = f"{rng.choice(load_choices)}%"
    diel = round(float(rng.uniform(2.0,3.5)),2)
    th_dens = round(float(rng.uniform(1.2,1.4)),3)
    meas_dens = round(float(rng.uniform(1.0,1.4)),3)
    rp_inc = round(float(rng.uniform(0,25)),2)
    free_vol = round(float(rng.uniform(0,0.35)),3)
    return [feed, str(diel), str(th_dens), str(meas_dens), str(rp_inc), str(free_vol)]


def generate_pi_ofg_thermal_mech_row(rng: np.random.Generator, used: Optional[Set[str]] = None) -> List[str]:
    load_choices = [0,3,7,10,15]
    feed = f"{rng.choice(load_choices)}%"
    tg_val = round(float(rng.uniform(330,380)),1)
    tcb = round(float(rng.uniform(400,550)),1)
    cte = round(float(rng.uniform(20,60)),1)
    e_mod = round(float(rng.uniform(2.3,3.2)),2)
    max_stress = round(float(rng.uniform(150,350)),1)
    elong = round(float(rng.uniform(0.5,2.5)),2)
    return [feed, str(tg_val), str(tcb), str(cte), str(e_mod), str(max_stress), str(elong)]


def generate_pi_ofg_surface_props_row(rng: np.random.Generator, used: Optional[Set[str]] = None) -> List[str]:
    load_choices = [0,3,7,10,15]
    feed = f"{rng.choice(load_choices)}%"
    h2o = round(float(rng.uniform(50,80)),1)
    gly = round(float(rng.uniform(60,85)),1)
    ys = round(float(rng.uniform(40,60)),1)
    vd = round(float(rng.uniform(3,8)),2)
    y_s = round(float(rng.uniform(25,60)),1)
    return [feed, str(h2o), str(gly), str(ys), str(vd), str(y_s)]


def generate_nano_concentration_tg_ips_row(rng: np.random.Generator, used: Optional[Set[str]] = None) -> List[str]:
    wt_val = round(float(rng.uniform(0,60)),1)
    vol_val = round(float(rng.uniform(0,30)),1)
    tg_tmdsc = int(rng.integers(350,391))
    tg_bds = int(rng.integers(350,391))
    ips = round(float(rng.uniform(5,70)),1)
    saxs = round(float(rng.uniform(2,7)),1)
    bds = round(float(rng.uniform(3,8)),1)
    return [str(wt_val), str(vol_val), str(tg_tmdsc), str(tg_bds), str(ips), str(saxs), str(bds)]


def generate_uv_cured_props_row(rng: np.random.Generator, used: Optional[Set[str]] = None) -> List[str]:
    load_choices = [0,5,10,20]
    filler = "AgSbF6"
    load_val = rng.choice(load_choices)
    if load_val==0:
        sample_str = "CE"
    else:
        sample_str = f"CE + {load_val} wt.-% {filler}"
    conv = int(rng.integers(70,101))
    gel = int(rng.integers(90,101))
    tg_val = int(rng.integers(150,200))
    abs_max = int(rng.integers(350,431))
    return [sample_str,str(conv),str(gel),str(tg_val),str(abs_max)]


def generate_np_content_tga_row(rng: np.random.Generator, used: Optional[Set[str]] = None) -> List[str]:
    load_choices = [0,3,5,7,10,15,20]
    filler = "AgSbF6"
    load_val = rng.choice(load_choices)
    if load_val==0:
        sample_str = "CE"
    else:
        sample_str = f"CE + {load_val} wt.-% {filler}"
    exp_char = round(float(rng.uniform(0.5,6.0)),1)
    approx_np = round(exp_char*rng.uniform(0,1),1)
    return [sample_str,str(exp_char),str(approx_np)]


def generate_dsc_params_pvdf_bt_row(rng: np.random.Generator, used: Optional[Set[str]] = None) -> List[str]:
    possible_samples = [
        "PVDF", "D-BT/PVDF", "D-h-BT/PVDF", "P-BT/PVDF", "M-BT/PVDF"
    ]
    sample_str = pick_unique_label(rng, used, lambda: rng.choice(possible_samples))
    jg_val = round(float(rng.uniform(15,30)),2)
    tg_val = round(float(rng.uniform(150,170)),2)
    xc_val = round(float(rng.uniform(20,35)),2)
    return [sample_str,str(jg_val),str(tg_val),str(xc_val)]


def generate_saturated_moisture_content_v2_row(rng: np.random.Generator, used: Optional[Set[str]] = None) -> List[str]:
    def label_generator():
        unfilled_label = unify_unfilled(rng, p_unfilled=0.4)
        if unfilled_label:
            return unfilled_label
        else:
            load_val = round(float(rng.uniform(2,15)),1)
            fill_desc = rng.choice(MASTER_OPTIONS["size_descriptors"])
            fill_base = rng.choice(MASTER_OPTIONS["filler_bases"])
            return f"{load_val}% {fill_desc} {fill_base}"

    mat_str = pick_unique_label(rng, used, label_generator)
    def rand_val():
        return round(float(rng.uniform(0,5)),2)
    c25 = rand_val()
    c50 = rand_val()
    c80 = rand_val()
    c50_75 = round(float(rng.uniform(0,1)),2)
    return [mat_str, str(c25), str(c50), str(c80), str(c50_75)]


def generate_freezable_nonfrozen_water_v2_row(rng: np.random.Generator, used: Optional[Set[str]] = None) -> List[str]:
    def label_generator():
        base = unify_unfilled(rng, p_unfilled=0.3)
        if base:
            return base
        else:
            load_val = round(float(rng.uniform(2,15)),1)
            fill_desc = rng.choice(MASTER_OPTIONS["size_descriptors"])
            fill_base = rng.choice(MASTER_OPTIONS["filler_bases"])
            return f"{load_val}% {fill_desc} {fill_base}"

    mat_str = pick_unique_label(rng, used, label_generator)
    def rfloat(a,b):
        return round(float(rng.uniform(a,b)),1)
    fw = rfloat(0,6)
    nfw = rfloat(0,10)
    tot = round(fw+nfw,1)
    return [mat_str, str(fw), str(nfw), str(tot)]


def generate_free_space_length_weibull_v2_row(rng: np.random.Generator, used: Optional[Set[str]] = None) -> List[str]:
    possible_labels = [
        "neat epoxy", "1 wt% bare SiO2", "2 wt% SiO2- PGMA",
        "2 wt% SiO2- PGMA-ferro", "2 wt% SiO2- PGMA-thio"
    ]
    mat_str = pick_unique_label(rng, used, lambda: rng.choice(possible_labels))
    def rint(a,b):
        return int(rng.integers(a,b))
    Lval = rng.choice(["NA",str(rint(100,1001))])
    c63 = str(rint(150,300))
    plus = str(rint(10,40))
    a_3p = str(rint(50,150))
    shape_3p = str(round(float(rng.uniform(1,2.5)),1))
    t_3p = str(rint(50,200))
    a_2p = str(rint(150,300))
    shape_2p = str(round(float(rng.uniform(2,5)),1))
    return [mat_str,Lval,c63,plus,a_3p,shape_3p,t_3p,a_2p,shape_2p]


def generate_real_relative_permittivity_v2_row(rng: np.random.Generator, used: Optional[Set[str]] = None) -> List[str]:
    possible_fields = ["Low field","2","5","7.5","10","AC field","High field"]
    field_label = pick_unique_label(rng, used, lambda: rng.choice(possible_fields))
    def rflt(a,b):
        return round(float(rng.uniform(a,b)),1)
    val1 = rflt(10,20)
    if rng.random()<0.2:
        val2 = "-"
    else:
        val2 = str(rflt(10,25))
    return [field_label, str(val1), val2]


def generate_crystallinity_melting_xlpe_v2_row(rng: np.random.Generator, used: Optional[Set[str]] = None) -> List[str]:
    possible_samples = [
        "XLPE Only","5% untreated nanosilica + XLPE",
        "5% vinylsilane treated nanosilica + XLPE",
        "5% untreated microsilica + XLPE"
    ]
    st = pick_unique_label(rng, used, lambda: rng.choice(possible_samples))
    cryst = str(rng.integers(40,61))
    melt = str(round(float(rng.uniform(100,120)),1))
    return [st, cryst, melt]


def generate_low_freq_activation_energies_silica_v2_row(rng: np.random.Generator, used: Optional[Set[str]] = None) -> List[str]:
    possible_samples = [
        "XLPE + 5% untreated nanosilica","XLPE + 5% vinylsilane treated nanosilica",
        "XLPE + 5% microsilica"
    ]
    st = pick_unique_label(rng, used, lambda: rng.choice(possible_samples))
    val = round(float(rng.uniform(0.15,0.35)),2)
    return [st, str(val)]


def generate_peak_epr_signal_particulates_v2_row(rng: np.random.Generator, used: Optional[Set[str]] = None) -> List[str]:
    st = pick_unique_label(rng, used, lambda: rng.choice(TREATED_NANOCOMPOSITES))
    p_sig = rng.integers(500,7001)
    c_sig = rng.integers(600,8001)
    return [st,str(p_sig),str(c_sig)]


def generate_trap_depths_tsc_v2_row(rng: np.random.Generator, used: Optional[Set[str]] = None) -> List[str]:
    st = pick_unique_label(rng, used, lambda: rng.choice(TREATED_NANOCOMPOSITES))
    c5v = round(float(rng.uniform(0.3,2.5)),2)
    c4v = round(float(rng.uniform(1.0,1.2)),2)
    return [st,str(c5v),str(c4v)]


def generate_peak_epr_signal_particulates_v3_row(rng: np.random.Generator, used: Optional[Set[str]] = None) -> List[str]:
    some_other_list = [
        "Micron Scale silica","Nanoscale silica","AEAPS treated nanosilica",
        "HMDS treated nanosilica","TES treated nanosilica"
    ]
    combined_list = TREATED_NANOCOMPOSITES + some_other_list
    st = pick_unique_label(rng, used, lambda: rng.choice(combined_list))
    pow_s = rng.integers(250,6001)
    comp_s = rng.integers(600,8001)
    return [st,str(pow_s),str(comp_s)]


def generate_activation_energies_dielectric_spectroscopy_v2_row(rng: np.random.Generator, used: Optional[Set[str]] = None) -> List[str]:
    possible_samples = [
        "XLPE + 5% untreated nanosilica","XLPE + 5% AEAPS treated nanosilica",
        "XLPE + 5% HMDS-treated nanosilica","XLPE + 5% TES treated nanosilica",
        "XLPE + 5% microsilica"
    ]
    st = pick_unique_label(rng, used, lambda: rng.choice(possible_samples))
    val = round(float(rng.uniform(0.15,0.35)),2)
    return [st,str(val)]


def generate_threshold_field_charge_accum_v2_row(rng: np.random.Generator, used: Optional[Set[str]] = None) -> List[str]:
    possible_samples = [
        "XLPE","XLPE + Micron Scale silica","XLPE + untreated nanoscale silica",
        "XLPE + AEAPS nanoscale silica","XLPE + HMDS nanoscale silica","XLPE + TES nanoscale silica"
    ]
    st = pick_unique_label(rng, used, lambda: rng.choice(possible_samples))
    def rint(a,b):
        return int(rng.integers(a,b))
    if rng.random()<0.15:
        pos = "-"
        neg = "-"
    else:
        pos = str(rint(10,30))
        neg = str(-rint(10,30))
    return [st,pos,neg]


def generate_characteristic_breakdown_voltage_v2_row(rng: np.random.Generator, used: Optional[Set[str]] = None) -> List[str]:
    possible_samples = [
        "XLPE","XLPE + untreated nanosilica","XLPE + AEAPS nanosilica",
        "XLPE + HMDS nanosilica","XLPE + TES nanosilica"
    ]
    st = pick_unique_label(rng, used, lambda: rng.choice(possible_samples))
    def rint(a,b):
        return int(rng.integers(a,b))
    def rshp():
        return round(float(rng.uniform(1.5,3.0)),2)
    c25 = f"{rint(100,450)} ({rshp()})"
    c60 = f"{rint(50,420)} ({rshp()})"
    c70 = f"{rint(40,400)} ({rshp()})"
    c80 = f"{rint(30,300)} ({rshp()})"
    return [st,c25,c60,c70,c80]


def generate_peak_epr_signal_particulates_v4_row(rng: np.random.Generator, used: Optional[Set[str]] = None) -> List[str]:
    possible_labels = [
        "Micron scale silica","Nanoscale silica","AEAPS treated nanosilica",
        "HMDS treated nanosilica","TES treated nanosilica"
    ]
    combined_list = TREATED_NANOCOMPOSITES + possible_labels
    st = pick_unique_label(rng, used, lambda: rng.choice(combined_list))
    si_val = rng.integers(250,6001)
    comp_val = rng.integers(500,8001)
    return [st,str(si_val),str(comp_val)]


def generate_activation_energies_dielectric_spectroscopy_v3_row(rng: np.random.Generator, used: Optional[Set[str]] = None) -> List[str]:
    possible_samples = [
        "XLPE + 5% untreated nanosilica","XLPE + 5% AEAPS treated nanosilica",
        "XLPE + 5% HMDS-treated nanosilica","XLPE + 5% TES treated nanosilica",
        "XLPE + 5% microsilica"
    ]
    st = pick_unique_label(rng, used, lambda: rng.choice(possible_samples))
    val = round(float(rng.uniform(0.15,0.35)),2)
    return [st,str(val)]


def generate_bt_ag_pvdf_composites_table37_row(rng: np.random.Generator, used: Optional[Set[str]] = None) -> List[str]:
    mat = "PVDF"
    fill = "BT-Ag"
    comp_wt = rng.integers(0,81)
    dk1 = round(rng.uniform(5,200),1)
    lt1 = round(rng.uniform(0.01,0.15),2)
    dk2 = round(dk1*rng.uniform(0.8,0.9),1)
    lt2 = round(lt1*rng.uniform(0.8,1.2),2)
    return [mat, fill, str(comp_wt), str(dk1), str(lt1), str(dk2), str(lt2)]


###############################################################################
# "Schemas" dictionary for demonstration in main()
###############################################################################
# Each entry: "schema_name": (columns, generator_func)
ALL_SCHEMAS = {
    "schema01_weibull_epoxy_tio2": (
        ["Composition", "Shape Param (B)", "Scale Param (a)"],
        generate_weibull_epoxy_tio2_row
    ),
    "schema02_weibull_epoxy_al2o3": (
        ["Composition", "Shape Param (B)", "Scale Param (a)"],
        generate_weibull_epoxy_al2o3_row
    ),
    "schema03_weibull_epoxy_zno": (
        ["Composition", "Nanocomposites", "Microcomposites"],
        generate_weibull_epoxy_zno_row
    ),
    "schema04_mech_eva": (
        ["Sample", "Tensile Strength (MPa)", "Modulus at 300% Elongation (MPa)", "Elongation Break at (%)"],
        generate_mech_eva_row
    ),
    "schema05_dynamic_mech_eva": (
        ["Sample", "T.(C)", "E'(Pa) at T8", "E'(Pa) at 30C", "tanδ at Tg", "tanδ at 30C"],
        generate_dynamic_mech_eva_row
    ),
    "schema06_activation_energies_tio2": (
        ["TiO2 nanocomposite", "TiO2 microcomposite"],
        generate_activation_energies_tio2_row
    ),
    "schema07_breakdown_strength_xlpe": (
        ["Material", "dc Breakdown (B)", "ac Breakdown (B)"],
        generate_breakdown_strength_xlpe_row
    ),
    "schema08_lichtenecker_rother": (
        ["Material", "f(Hz)", "(L-R)", "Measured E'"],
        generate_lichtenecker_rother_row
    ),
    "schema09_current_decay_exponent": (
        ["Material", "Current Decay Exponent n"],
        generate_current_decay_exponent_row
    ),
    "schema10_impulse_test_breakdown": (
        ["Material", "Impulse Strength(1.2x50 µs)"],
        generate_impulse_test_breakdown_row
    ),
    "schema11_polarization_space_charge": (
        ["Material", "Polarization Decay(s)", "Space Charge Decay(s)"],
        generate_polarization_space_charge_row
    ),
    "schema12_ac_breakdown_scale_params": (
        ["MaterialCondition", "n_value", "ShapeParam", "% Decrease"],
        generate_ac_breakdown_scale_params_row
    ),
    "schema13_dc_breakdown_scale_params": (
        ["MaterialCondition", "Scale", "Shape", "% Decrease"],
        generate_dc_breakdown_scale_params_row
    ),
    "schema14_dc_breakdown_scale_params_dual": (
        ["MaterialCondition", "Scale1", "Shape1", "Dec1", "Scale2", "Shape2", "Dec2"],
        generate_dc_breakdown_scale_params_dual_row
    ),
    "schema15_pi_ofg_elemental": (
        ["OFG Feed (%)", "N (%)", "C (%)", "H (%)"],
        generate_pi_ofg_elemental_analysis_row
    ),
    "schema16_pi_ofg_dielectric_density": (
        ["OFG Feed (%)", "Diel Const", "Theoretical Density", "Measured Density", "Rel Porosity Inc", "Free Vol"],
        generate_pi_ofg_dielectric_density_row
    ),
    "schema17_pi_ofg_thermal_mech": (
        ["OFG Feed (%)", "Tg (C)", "Tcb (C)", "CTE (ppm/C)", "E_mod(GPa)", "Max_Stress(MPa)", "Elong(%)"],
        generate_pi_ofg_thermal_mech_row
    ),
    "schema18_pi_ofg_surface_props": (
        ["OFG Feed (%)", "H2O", "Glycerol", "Ys", "vd", "y_s"],
        generate_pi_ofg_surface_props_row
    ),
    "schema19_nano_concentration_tg_ips": (
        ["wt%", "vol%", "Tg(TMDSC)", "Tg(BDS)", "IPS", "SAXS", "BDS?"],
        generate_nano_concentration_tg_ips_row
    ),
    "schema20_uv_cured_props": (
        ["Sample", "EpoxyConv(%)", "GelContent(%)", "Tg(C)", "Abs Max(nm)"],
        generate_uv_cured_props_row
    ),
    "schema21_np_content_tga": (
        ["Sample", "Exp Char (wt%)", "Approx NP (wt%)"],
        generate_np_content_tga_row
    ),
    "schema22_dsc_params_pvdf_bt": (
        ["Sample", "Jg", "Tg (C)", "Xc (%)"],
        generate_dsc_params_pvdf_bt_row
    ),
    "schema23_saturated_moisture_v2": (
        ["Material", "c25", "c50", "c80", "c50_75"],
        generate_saturated_moisture_content_v2_row
    ),
    "schema24_freezable_nonfrozen_water_v2": (
        ["Material", "Freezable water", "Nonfrozen water", "Total water"],
        generate_freezable_nonfrozen_water_v2_row
    ),
    "schema25_free_space_length_weibull_v2": (
        ["Material", "Length", "c63", "plus", "a_3p", "shape_3p", "t_3p", "a_2p", "shape_2p"],
        generate_free_space_length_weibull_v2_row
    ),
    "schema26_real_relative_permittivity_v2": (
        ["FieldLabel", "Val1", "Val2"],
        generate_real_relative_permittivity_v2_row
    ),
    "schema27_crystallinity_melting_xlpe_v2": (
        ["SampleType", "Crystallinity(%)", "Melting(C)"],
        generate_crystallinity_melting_xlpe_v2_row
    ),
    "schema28_low_freq_activation_energies_silica_v2": (
        ["Sample", "ActivationEnergy(eV)"],
        generate_low_freq_activation_energies_silica_v2_row
    ),
    "schema29_peak_epr_signal_particulates_v2": (
        ["Treatment", "PeakSignal(a.u.)", "CompositeSignal(a.u.)"],
        generate_peak_epr_signal_particulates_v2_row
    ),
    "schema30_trap_depths_tsc_v2": (
        ["Treatment", "C5(eV)", "C4(eV)"],
        generate_trap_depths_tsc_v2_row
    ),
    "schema31_peak_epr_signal_particulates_v3": (
        ["Treatment", "PowderSignal", "CompositeSignal"],
        generate_peak_epr_signal_particulates_v3_row
    ),
    "schema32_activation_energies_dielectric_spectroscopy_v2": (
        ["Sample", "Eact(eV)"],
        generate_activation_energies_dielectric_spectroscopy_v2_row
    ),
    "schema33_threshold_field_charge_accum_v2": (
        ["Sample", "PosThresh", "NegThresh"],
        generate_threshold_field_charge_accum_v2_row
    ),
    "schema34_characteristic_breakdown_voltage_v2": (
        ["Sample", "25C", "60C", "70C", "80C"],
        generate_characteristic_breakdown_voltage_v2_row
    ),
    "schema35_peak_epr_signal_particulates_v4": (
        ["MaterialLabel", "SiO2signal", "CompositeSignal"],
        generate_peak_epr_signal_particulates_v4_row
    ),
    "schema36_activation_energies_dielectric_spectroscopy_v3": (
        ["Sample", "Eact(eV)"],
        generate_activation_energies_dielectric_spectroscopy_v3_row
    ),
    "schema37_bt_ag_pvdf_composites": (
        ["Matrix Name", "Filler Name", "BT-Ag (wt%)", "Dk(1kHz)", "Tan(1kHz)", "Dk(100kHz)", "Tan(100kHz)"],
        generate_bt_ag_pvdf_composites_table37_row
    ),
}


def ensure_schema_title(schema_name: str) -> str:
    """
    A simple fallback that returns 'Table: {schema_name}'
    You can expand or randomize further as you like.
    """
    return f"Table: {schema_name}"


def main():
    """
    A simpler interface that:
    1) Iterates over ALL_SCHEMAS
    2) For each schema, generates a small table (4 rows)
    3) Prints the DataFrame
    4) Prints JSON for each row
    """
    rng = np.random.default_rng(seed=42)

    for schema_name, (columns, gen_func) in ALL_SCHEMAS.items():
        print(f"\n=== {ensure_schema_title(schema_name)} ===")
        used = set()
        rows = []
        for _ in range(4):  # generate 4 rows per schema
            row_data = gen_func(rng, used)
            rows.append(row_data)

        df = pd.DataFrame(rows, columns=columns)
        print("\nDataFrame Output:")
        print(df.to_string(index=False))

        # Show JSON as well
        print("\nJSON Rows:")
        json_rows = []
        for i, row in df.iterrows():
            # minimal conversion to dict
            row_dict = {"row_id": i+1}
            for col in columns:
                row_dict[col] = row[col]
            json_rows.append(row_dict)

        print(json.dumps(json_rows, indent=2))


if __name__ == "__main__":
    main()
