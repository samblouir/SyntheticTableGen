#!/usr/bin/env python3
"""
fully_dynamic_schema.py

A single-file script containing:
1) All 37 schema generator functions with a "while-based" uniqueness approach.
2) A dictionary ALL_SCHEMAS mapping each schema_name -> (columns, generator_func).
3) A main() function that iterates through ALL_SCHEMAS, generates sample rows
   for each schema, and prints them in both DataFrame and JSON formats.

**Major Changes for More Realistic Outputs:**
- Refined numeric ranges for each schema to approximate plausible real-world data
  (e.g., Tensile Strength in 10..100 MPa, not random 0..500).
- Reduced extremely exotic combinations ("pico", "giga", etc. occur less often).
- Avoid obviously contradictory combos like "unfilled" + "none" repeated.
- Implemented consistent logic so that each filler or matrix combination looks
  more "coherent" (though still random).

**Additionally**, we include parsing logic that:
- Splits columns named "Sample", "Composition", or "Material" into subfields
  (base, percentage, filler).
- Creates a `<column_name>_raw` field so you can see the original string.
- Minimally handles "Treatment" columns to store a "treatment_label."

Usage:
  python fully_dynamic_schema.py

You can adjust the `NUM_ROWS` or the random `seed` in `main()` to see more variety.
"""

import numpy as np
import pandas as pd
import json
import re
from typing import Any, Dict, List, Optional, Set, Union

###############################################################################
# MASTER_OPTIONS - consolidated filler, matrix, etc., plus advanced synonyms
###############################################################################
MASTER_OPTIONS = {
    "unfilled": [
        "unfilled",
        "no filler",
        "none"
    ],
    "size_descriptors": [
        "nano",
        "micro",
        "fine",
        "coarse",
        "normal",
    ],
    "surface_treatments": [
        "untreated",
        "vinyl silane",
        "KH-550",
        "amine-functionalized",
        "treated",
        "VT"
    ],
    "filler_bases": [
        "alumina",
        "silica",
        "titania",
        "montmorillonite",
        "BaTiO3",
        "Graphene",
        "SiO2",
        "boron nitride",
        "carbon nanotubes",
        "nanoclay",
    ],
    "matrix_materials": [
        "epoxy resin", "polyimide", "LDPE", "XLPE", "HDPE",
        "PP", "polypropylene", "bio-based epoxy", "polymer A",
        "resin B", "polycarbonate", "ABS", "PA6", "PI", "LCP"
    ],
}

# Some specialized sets for certain schemas referencing "treated" nanocomposites
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


###############################################################################
# unify_unfilled + pick_unique_label for uniqueness
###############################################################################
def unify_unfilled(rng: np.random.Generator, p_unfilled=0.2) -> Optional[str]:
    """Pick from MASTER_OPTIONS["unfilled"] with probability p_unfilled."""
    if rng.random() < p_unfilled:
        return rng.choice(MASTER_OPTIONS["unfilled"])
    return None


def pick_unique_label(rng: np.random.Generator,
                      used: Optional[Set[str]],
                      label_generator) -> str:
    """
    Attempt up to 100 times to get a label not in 'used'.
    Return "MaxAttempts_Exceeded" if no success.
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
# A naive parser that splits text fields into subfields
###############################################################################
def parse_base_percentage_filler(raw: str) -> Dict[str, str]:
    """
    Attempt to parse e.g. "XLPE + 5% nano alumina" 
    => { 'base': 'XLPE', 'percentage': '5%', 'filler': 'nano alumina' }.

    Also removes parentheses like (none) or (untreated).
    """
    out = {}
    # minimal cleanup
    parsed = re.sub(r"\(none\)", "", raw, flags=re.IGNORECASE)
    parsed = re.sub(r"\(untreated\)", "", parsed, flags=re.IGNORECASE)
    parsed = parsed.strip()

    # if there's a ' + ' => split
    if " + " in parsed:
        left, right = parsed.split(" + ", 1)
        out["base"] = left.strip()
        # try to capture leading digits + % as 'percentage'
        m = re.match(r"^([\d.]+%?)\s+(.*)$", right.strip())
        if m:
            out["percentage"] = m.group(1)
            out["filler"] = m.group(2)
        else:
            out["filler"] = right.strip()
    else:
        # look for leading digits% ...
        m = re.match(r"^([\d.]+%?)\s+(.*)$", parsed)
        if m:
            # e.g. "4.4% coarse alumina epoxy resin"
            first = m.group(1)
            rest = m.group(2).strip()
            # see if rest ends with a known matrix
            found_matrix = None
            for mat in MASTER_OPTIONS["matrix_materials"]:
                if rest.endswith(mat):
                    idx = rest.rfind(mat)
                    left_part = rest[:idx].strip()
                    base_part = rest[idx:].strip()
                    out["percentage"] = first
                    out["filler"] = left_part
                    out["base"] = base_part
                    found_matrix = True
                    break
            if not found_matrix:
                # fallback
                out["percentage"] = first
                out["filler"] = rest
        else:
            # fallback => see if ends with known matrix
            found_mat = False
            for mat in MASTER_OPTIONS["matrix_materials"]:
                if parsed.endswith(mat):
                    idx = parsed.rfind(mat)
                    out["filler"] = parsed[:idx].strip()
                    out["base"] = mat
                    found_mat = True
                    break
            if not found_mat:
                # store entire as filler
                out["filler"] = parsed

    for k in list(out.keys()):
        if not out[k]:
            out.pop(k)
    return out


def parse_special_column(col_name: str, raw_value: str) -> Dict[str, str]:
    """
    If col_name includes 'sample','composition','material', parse with parse_base_percentage_filler().
    If col_name is 'treatment', store a 'treatment_label'.
    Also store a <col_name>_raw field.
    """
    out = {}
    out[f"{col_name}_raw"] = raw_value

    if raw_value == "MaxAttempts_Exceeded":
        return out

    lower = col_name.lower()
    if any(x in lower for x in ("composition", "sample", "material")):
        # parse
        sub = parse_base_percentage_filler(raw_value)
        for k, v in sub.items():
            out[f"{col_name}_{k}"] = v
    elif "treatment" in lower:
        out[f"{col_name}_treatment_label"] = raw_value.strip()

    return out


###############################################################################
# Generator Functions (#1..#37)
###############################################################################
def generate_weibull_epoxy_tio2_row(rng: np.random.Generator, used: Optional[Set[str]] = None) -> List[str]:
    """
    columns: [Composition, Shape Param (B), Scale Param (a)]
    Numeric ranges:
    - shape param in 2..20
    - scale param in 20..80
    """
    def label_generator():
        # pick a random composition
        if rng.random() < 0.2:
            return rng.choice(MASTER_OPTIONS["unfilled"])
        # 0.1..15% 
        c = round(rng.uniform(0.1,15.0),1)
        desc = rng.choice(MASTER_OPTIONS["size_descriptors"])
        base = rng.choice(MASTER_OPTIONS["filler_bases"])
        mat = rng.choice(MASTER_OPTIONS["matrix_materials"])
        if rng.random()<0.3:
            treat = rng.choice(MASTER_OPTIONS["surface_treatments"])
            return f"{c}% {desc} {base} ({treat}) {mat}"
        return f"{c}% {desc} {base} {mat}"

    comp = pick_unique_label(rng, used, label_generator)
    shape = round(rng.uniform(2,20),3)
    scale = round(rng.uniform(20,80),2)
    return [comp, str(shape), str(scale)]


def generate_weibull_epoxy_al2o3_row(rng: np.random.Generator, used: Optional[Set[str]] = None) -> List[str]:
    """
    columns: [Composition, Shape Param (B), Scale Param (a)]
    shape param in 2..20, scale in 20..80
    """
    def label_generator():
        if rng.random()<0.2:
            return rng.choice(MASTER_OPTIONS["unfilled"])
        c = round(rng.uniform(0.1,15.0),1)
        desc = rng.choice(MASTER_OPTIONS["size_descriptors"])
        base = rng.choice(MASTER_OPTIONS["filler_bases"])
        mat = rng.choice(MASTER_OPTIONS["matrix_materials"])
        return f"{c}% {desc} {base} {mat}"

    comp = pick_unique_label(rng, used, label_generator)
    shape = round(rng.uniform(2,20),3)
    scale = round(rng.uniform(20,80),2)
    return [comp, str(shape), str(scale)]


def generate_weibull_epoxy_zno_row(rng: np.random.Generator, used: Optional[Set[str]] = None) -> List[str]:
    """
    columns: [Composition, Nanocomposites, Microcomposites]
    Possibly "not available".
    """
    def label_generator():
        if rng.random() < 0.2:
            return rng.choice(MASTER_OPTIONS["unfilled"])
        c_amt = round(rng.uniform(0.1,15.0),1)
        desc = rng.choice(MASTER_OPTIONS["size_descriptors"])
        base = rng.choice(MASTER_OPTIONS["filler_bases"])
        mat = rng.choice(MASTER_OPTIONS["matrix_materials"])
        return f"{c_amt}% {desc} {base} {mat}"

    comp = pick_unique_label(rng, used, label_generator)
    def maybe_na():
        if rng.random() < 0.3:
            return "not available"
        return str(round(rng.uniform(10,40),2))

    nano_val = maybe_na()
    micro_val = maybe_na()
    return [comp, nano_val, micro_val]


def generate_mech_eva_row(rng: np.random.Generator, used: Optional[Set[str]] = None) -> List[str]:
    """
    columns: [Sample, Tensile Strength (MPa), Modulus at 300% Elongation (MPa), Elongation Break at (%)]
    Tensile strength => ~3..50 MPa
    Modulus => ~0.4..2.0
    Elongation break => 500..2000
    """
    def label_generator():
        step = rng.integers(0,8)
        load = step*2
        if load==0:
            return f"Pure {rng.choice(MASTER_OPTIONS['matrix_materials'])}"
        desc = rng.choice(MASTER_OPTIONS["size_descriptors"])
        base = rng.choice(MASTER_OPTIONS["filler_bases"])
        mat = rng.choice(MASTER_OPTIONS["matrix_materials"])
        return f"{load} wt% {desc} {base} {mat}"

    s_str = pick_unique_label(rng, used, label_generator)
    ts = round(rng.uniform(3,50),1)
    mod_300 = round(rng.uniform(0.2,2.0),2)
    elong = rng.integers(500,2001)
    return [s_str, str(ts), str(mod_300), str(elong)]


def generate_dynamic_mech_eva_row(rng: np.random.Generator, used: Optional[Set[str]] = None) -> List[str]:
    """
    columns: [Sample, T.(C), E'(Pa) at T8, E'(Pa) at 30C, tanδ at Tg, tanδ at 30C]
    T: -30..+30
    E': random x10^...
    tanδ: 0.05..1.2
    """
    def label_generator():
        step = rng.integers(0,6)
        load = step*2
        if load==0:
            return f"Pure {rng.choice(MASTER_OPTIONS['matrix_materials'])}"
        desc = rng.choice(MASTER_OPTIONS["size_descriptors"])
        base = rng.choice(MASTER_OPTIONS["filler_bases"])
        mat = rng.choice(MASTER_OPTIONS["matrix_materials"])
        return f"{load} wt% {desc} {base} {mat}"

    s_str = pick_unique_label(rng, used, label_generator)
    T_c = rng.integers(-30, 31)
    E_t8 = f"{round(rng.uniform(0.1,2.5),2)} x 10^{rng.integers(5,9)}"
    E_30 = f"{round(rng.uniform(0.1,2.5),2)} x 10^{rng.integers(5,9)}"
    tan_tg = round(rng.uniform(0.05,1.2),2)
    tan_30 = round(rng.uniform(0.05,0.4),2)
    return [s_str, str(T_c), E_t8, E_30, str(tan_tg), str(tan_30)]


def generate_activation_energies_tio2_row(rng: np.random.Generator, used: Optional[Set[str]] = None) -> List[str]:
    """
    columns: [TiO2 nanocomposite, TiO2 microcomposite]
    We'll pick random ~0.6..2.0 eV
    """
    below_nano = str(round(rng.uniform(0.6,2.0),1))
    above_micro = str(round(rng.uniform(0.5,2.0),1))
    return [below_nano, above_micro]


def generate_breakdown_strength_xlpe_row(rng: np.random.Generator, used: Optional[Set[str]] = None) -> List[str]:
    """
    columns: [Material, dc Breakdown (B), ac Breakdown (B)]
    We'll produce random composition + breakdown in range 80..400, shape ~1..8
    """
    def label_generator():
        if rng.random()<0.3:
            return f"{rng.choice(MASTER_OPTIONS['unfilled'])} {rng.choice(MASTER_OPTIONS['matrix_materials'])}"
        load = round(rng.uniform(1,15),1)
        desc = rng.choice(MASTER_OPTIONS["size_descriptors"])
        base = rng.choice(MASTER_OPTIONS["filler_bases"])
        treat = rng.choice(MASTER_OPTIONS["surface_treatments"])
        mat = rng.choice(MASTER_OPTIONS["matrix_materials"])
        return f"{load}% {desc}{base}({treat}) {mat}"

    mat_str = pick_unique_label(rng, used, label_generator)
    def combo():
        val = rng.integers(80,401)
        shape = round(rng.uniform(1,8),1)
        return f"{val} ({shape})"

    return [mat_str, combo(), combo()]


def generate_lichtenecker_rother_row(rng: np.random.Generator, used: Optional[Set[str]] = None) -> List[str]:
    """
    columns: [Material, f(Hz), (L-R), Measured E']
    freq ~ [1e3,1e4,1e5,1e6]
    L-R ~ 2..10
    E' ~ 2..20
    """
    def label_generator():
        c = round(rng.uniform(1,15),1)
        desc = rng.choice(MASTER_OPTIONS["size_descriptors"])
        base = rng.choice(MASTER_OPTIONS["filler_bases"])
        mat = rng.choice(MASTER_OPTIONS["matrix_materials"])
        if rng.random() < 0.2:
            return f"{rng.choice(MASTER_OPTIONS['unfilled'])} {mat}"
        return f"{c} wt% {desc} {base} {mat}"

    mat_str = pick_unique_label(rng, used, label_generator)
    freq_val = rng.choice([1e3,1e4,1e5,1e6])
    lr_val = str(round(rng.uniform(2,10),2)) if rng.random()<0.8 else ""
    e_val = str(round(rng.uniform(2,20),1)) if rng.random()<0.8 else ""
    return [mat_str, str(freq_val), lr_val, e_val]


def generate_current_decay_exponent_row(rng: np.random.Generator, used: Optional[Set[str]] = None) -> List[str]:
    """
    columns: [Material, Current Decay Exponent n]
    n ~ 0.5..2.0
    """
    def label_generator():
        if rng.random()<0.4:
            return f"{rng.choice(MASTER_OPTIONS['unfilled'])} {rng.choice(MASTER_OPTIONS['matrix_materials'])}"
        load = round(rng.uniform(1,15),1)
        desc = rng.choice(MASTER_OPTIONS["size_descriptors"])
        base = rng.choice(MASTER_OPTIONS["filler_bases"])
        treat = rng.choice(MASTER_OPTIONS["surface_treatments"])
        mat_ = rng.choice(MASTER_OPTIONS["matrix_materials"])
        return f"{load}% {desc} {base}({treat}) {mat_}"

    mat_str = pick_unique_label(rng, used, label_generator)
    c_val = round(rng.uniform(0.5,2.0),2)
    return [mat_str, str(c_val)]


def generate_impulse_test_breakdown_row(rng: np.random.Generator, used: Optional[Set[str]] = None) -> List[str]:
    """
    columns: [Material, Impulse Strength(1.2x50 µs)]
    range ~200..400, shape ~1..6
    """
    def label_generator():
        unl = unify_unfilled(rng, p_unfilled=0.3)
        if unl:
            return f"{unl} {rng.choice(MASTER_OPTIONS['matrix_materials'])}"
        load_val = round(rng.uniform(2,15),1)
        desc = rng.choice(MASTER_OPTIONS["size_descriptors"])
        fill_base = rng.choice(MASTER_OPTIONS["filler_bases"])
        treat = rng.choice(MASTER_OPTIONS["surface_treatments"])
        base = rng.choice(MASTER_OPTIONS["matrix_materials"])
        return f"{load_val}% {desc} {fill_base} ({treat}) {base}"

    mat_str = pick_unique_label(rng, used, label_generator)
    val = rng.integers(200,401)
    shape_val = round(rng.uniform(1,6),1)
    return [mat_str, f"{val} ({shape_val})"]


def generate_polarization_space_charge_row(rng: np.random.Generator, used: Optional[Set[str]] = None) -> List[str]:
    """
    columns: [Material, PolDecay, SCDecay]
    PolDecay: 1..100
    SCDecay: up to 10000
    """
    def label_generator():
        unl = unify_unfilled(rng, p_unfilled=0.3)
        if unl:
            return f"{unl} {rng.choice(MASTER_OPTIONS['matrix_materials'])}"
        load_val = round(rng.uniform(5,20),1)
        desc = rng.choice(MASTER_OPTIONS["size_descriptors"])
        fill_base = rng.choice(MASTER_OPTIONS["filler_bases"])
        base_mat = rng.choice(MASTER_OPTIONS["matrix_materials"])
        return f"{load_val}% {desc} {fill_base}-filled {base_mat}"

    mat_str = pick_unique_label(rng, used, label_generator)
    pol_val = int(rng.integers(1,101))
    sc_val = int(rng.integers(200,10001))
    return [mat_str, str(pol_val), str(sc_val)]


def generate_ac_breakdown_scale_params_row(rng: np.random.Generator, used: Optional[Set[str]] = None) -> List[str]:
    """
    columns: [MaterialCondition, n_value, ShapeParam, % Decrease]
    n_value ~50..300
    shape param ~1..8
    % decrease ~1..70
    """
    def label_generator():
        temps = [200,250,300,350]
        t_c = int(rng.choice(temps))
        unl = unify_unfilled(rng, p_unfilled=0.3)
        if unl:
            return f"{unl} at {t_c}°C"
        load_val = round(rng.uniform(1,15),1)
        treat = rng.choice(["untr","tr","none","VT"])
        fill_base = rng.choice(MASTER_OPTIONS["filler_bases"])
        return f"{load_val} wt% {treat} {fill_base} at {t_c}°C"

    matc = pick_unique_label(rng, used, label_generator)
    n_val = int(rng.integers(50,301))
    shape_val = int(rng.integers(1,9))
    pct_dec = int(rng.integers(1,71))
    return [matc, str(n_val), str(shape_val), str(pct_dec)]


def generate_dc_breakdown_scale_params_row(rng: np.random.Generator, used: Optional[Set[str]] = None) -> List[str]:
    """
    columns: [MaterialCondition, Scale, Shape, % Decrease]
    scale 50..400
    shape 1..12
    % decrease 0..80
    """
    def label_generator():
        temps = [200,250,300,350]
        t_c = int(rng.choice(temps))
        unl = unify_unfilled(rng, p_unfilled=0.3)
        if unl:
            return f"{unl} at {t_c}°C"
        load_val = round(rng.uniform(1,15),1)
        treat = rng.choice(["untr","tr","none","VT"])
        fill_base = rng.choice(MASTER_OPTIONS["filler_bases"])
        return f"{load_val} wt% {treat} {fill_base} at {t_c}°C"

    matc = pick_unique_label(rng, used, label_generator)
    scale_val = int(rng.integers(50,401))
    shape_val = int(rng.integers(1,13))
    pct_dec = int(rng.integers(0,81))
    return [matc, str(scale_val), str(shape_val), str(pct_dec)]


def generate_dc_breakdown_scale_params_dual_row(rng: np.random.Generator, used: Optional[Set[str]] = None) -> List[str]:
    """
    columns: [MaterialCondition, Scale1, Shape1, Dec1, Scale2, Shape2, Dec2]
    """
    def label_generator():
        unl = unify_unfilled(rng, p_unfilled=0.3)
        if unl:
            return unl
        load_val = round(rng.uniform(1,15),1)
        treat = rng.choice(["untr","tr","none","VT"])
        fill_base = rng.choice(MASTER_OPTIONS["filler_bases"])
        return f"{load_val} wt% {treat} {fill_base}"

    matc = pick_unique_label(rng, used, label_generator)

    scale1 = int(rng.integers(50,400))
    shape1 = int(rng.integers(1,13))
    dec1 = int(rng.integers(0,80))
    scale2 = int(rng.integers(50,400))
    shape2 = int(rng.integers(1,13))
    dec2 = int(rng.integers(0,80))
    return [matc, str(scale1), str(shape1), str(dec1), str(scale2), str(shape2), str(dec2)]


def generate_pi_ofg_elemental_analysis_row(rng: np.random.Generator, used: Optional[Set[str]] = None) -> List[str]:
    """
    columns: [OFG Feed (%), N (%), C (%), H (%)]
    Ranges:
      feed => 0,3,5,7,10,15
      N => ~5..9
      C => ~60..70
      H => ~2..4
    """
    feed_choices = [0,3,5,7,10,15]
    feed = f"{rng.choice(feed_choices)}%"
    n_val = round(rng.uniform(5,9),2)
    c_val = round(rng.uniform(60,70),2)
    h_val = round(rng.uniform(2,4),2)
    return [feed, str(n_val), str(c_val), str(h_val)]


def generate_pi_ofg_dielectric_density_row(rng: np.random.Generator, used: Optional[Set[str]] = None) -> List[str]:
    """
    columns: [OFG Feed (%), Diel Const, Theoretical Density, Measured Density, Rel Porosity Inc, Free Vol]
    feed => same
    diel => 2..4
    dens => 1.0..1.4
    """
    feed_choices = [0,3,7,10,15]
    feed = f"{rng.choice(feed_choices)}%"
    diel = round(rng.uniform(2.0,4.0),2)
    th_dens = round(rng.uniform(1.2,1.4),3)
    meas_dens = round(rng.uniform(1.0,1.4),3)
    rp_inc = round(rng.uniform(0,25),2)
    free_vol = round(rng.uniform(0,0.35),3)
    return [feed, str(diel), str(th_dens), str(meas_dens), str(rp_inc), str(free_vol)]


def generate_pi_ofg_thermal_mech_row(rng: np.random.Generator, used: Optional[Set[str]] = None) -> List[str]:
    """
    columns: [OFG Feed (%), Tg (C), Tcb (C), CTE (ppm/C), E_mod(GPa), Max_Stress(MPa), Elong(%)]
    """
    feed_choices = [0,3,7,10,15]
    feed = f"{rng.choice(feed_choices)}%"
    tg_val = round(rng.uniform(300,400),1)
    tcb = round(rng.uniform(400,550),1)
    cte = round(rng.uniform(20,60),1)
    e_mod = round(rng.uniform(2.2,3.2),2)
    max_stress = round(rng.uniform(100,350),1)
    elong = round(rng.uniform(0.5,2.5),2)
    return [feed, str(tg_val), str(tcb), str(cte), str(e_mod), str(max_stress), str(elong)]


def generate_pi_ofg_surface_props_row(rng: np.random.Generator, used: Optional[Set[str]] = None) -> List[str]:
    """
    columns: [OFG Feed (%), H2O, Glycerol, Ys, vd, y_s]
    """
    feed_choices = [0,3,7,10,15]
    feed = f"{rng.choice(feed_choices)}%"
    h2o = round(rng.uniform(50,80),1)
    gly = round(rng.uniform(60,85),1)
    ys = round(rng.uniform(20,60),1)
    vd = round(rng.uniform(2,7),2)
    y_s = round(rng.uniform(25,60),1)
    return [feed,str(h2o),str(gly),str(ys),str(vd),str(y_s)]


def generate_nano_concentration_tg_ips_row(rng: np.random.Generator, used: Optional[Set[str]] = None) -> List[str]:
    """
    columns: [wt%, vol%, Tg(TMDSC), Tg(BDS), IPS, SAXS, BDS?]
    typical ranges
    """
    wt_val = round(rng.uniform(0,60),1)
    vol_val = round(rng.uniform(0,40),1)
    tg_tmdsc = int(rng.integers(350,391))
    tg_bds = int(rng.integers(350,391))
    ips = round(rng.uniform(5,70),1)
    saxs = round(rng.uniform(2,7),1)
    bds = round(rng.uniform(2,8),1)
    return [str(wt_val), str(vol_val), str(tg_tmdsc), str(tg_bds), str(ips), str(saxs), str(bds)]


def generate_uv_cured_props_row(rng: np.random.Generator, used: Optional[Set[str]] = None) -> List[str]:
    """
    columns: [Sample, EpoxyConv(%), GelContent(%), Tg(C), Abs Max(nm)]
    """
    load_choices = [0,5,7,10,20]
    filler = "AgSbF6"
    load_val = rng.choice(load_choices)
    if load_val==0:
        sample_str = "CE"
    else:
        sample_str = f"CE + {load_val} wt.-% {filler}"
    conv = int(rng.integers(70,101))
    gel = int(rng.integers(90,101))
    tg_val = int(rng.integers(130,220))
    abs_max = int(rng.integers(350,431))
    return [sample_str,str(conv),str(gel),str(tg_val),str(abs_max)]


def generate_np_content_tga_row(rng: np.random.Generator, used: Optional[Set[str]] = None) -> List[str]:
    """
    columns: [Sample, Exp Char (wt%), Approx NP (wt%)]
    """
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
    """
    columns: [Sample, Jg, Tg(C), Xc(%)]
    """
    possible_samples = ["PVDF","D-BT/PVDF","D-h-BT/PVDF","P-BT/PVDF","M-BT/PVDF"]
    sample_str = pick_unique_label(rng, used, lambda: rng.choice(possible_samples))
    jg_val = round(rng.uniform(10,30),2)
    tg_val = round(rng.uniform(150,170),2)
    xc_val = round(rng.uniform(15,40),2)
    return [sample_str,str(jg_val),str(tg_val),str(xc_val)]


def generate_saturated_moisture_content_v2_row(rng: np.random.Generator, used: Optional[Set[str]] = None) -> List[str]:
    """
    columns: [Material, c25, c50, c80, c50_75]
    typical ~0..5
    """
    def label_generator():
        unl = unify_unfilled(rng, 0.3)
        if unl:
            return unl
        load_val = round(rng.uniform(1,15),1)
        desc = rng.choice(MASTER_OPTIONS["size_descriptors"])
        base = rng.choice(MASTER_OPTIONS["filler_bases"])
        return f"{load_val}% {desc} {base}"

    mat_str = pick_unique_label(rng, used, label_generator)
    def rand_val():
        return round(rng.uniform(0,5),2)
    c25 = rand_val()
    c50 = rand_val()
    c80 = rand_val()
    c50_75 = rand_val()
    return [mat_str, str(c25), str(c50), str(c80), str(c50_75)]


def generate_freezable_nonfrozen_water_v2_row(rng: np.random.Generator, used: Optional[Set[str]] = None) -> List[str]:
    """
    columns: [Material, Freezable water, Nonfrozen water, Total water]
    typical ~0..6
    """
    def label_generator():
        unl = unify_unfilled(rng, 0.3)
        if unl:
            return unl
        load_val = round(rng.uniform(1,15),1)
        desc = rng.choice(MASTER_OPTIONS["size_descriptors"])
        base = rng.choice(MASTER_OPTIONS["filler_bases"])
        return f"{load_val}% {desc} {base}"

    mat_str = pick_unique_label(rng, used, label_generator)
    def rfloat(a,b):
        return round(rng.uniform(a,b),1)
    fw = rfloat(0,6)
    nfw = rfloat(0,6)
    tot = round(fw+nfw,1)
    return [mat_str, str(fw), str(nfw), str(tot)]


def generate_free_space_length_weibull_v2_row(rng: np.random.Generator, used: Optional[Set[str]] = None) -> List[str]:
    """
    columns: [Material, Length, c63, plus, a_3p, shape_3p, t_3p, a_2p, shape_2p]
    We'll choose from possible labels
    """
    possible_labels = [
        "neat epoxy","1 wt% bare SiO2","2 wt% SiO2- PGMA",
        "2 wt% SiO2- PGMA-ferro","2 wt% SiO2- PGMA-thio"
    ]
    mat_str = pick_unique_label(rng, used, lambda: rng.choice(possible_labels))

    def rint(a,b):
        return int(rng.integers(a,b))
    Lval = rng.choice(["NA",str(rint(100,1001))])
    c63 = str(rint(150,300))
    plus = str(rint(10,40))
    a_3p = str(rint(50,150))
    shape_3p = str(round(rng.uniform(1,2.5),1))
    t_3p = str(rint(50,200))
    a_2p = str(rint(150,300))
    shape_2p = str(round(rng.uniform(1,4),1))
    return [mat_str,Lval,c63,plus,a_3p,shape_3p,t_3p,a_2p,shape_2p]


def generate_real_relative_permittivity_v2_row(rng: np.random.Generator, used: Optional[Set[str]] = None) -> List[str]:
    """
    columns: [FieldLabel, Val1, Val2]
    typical ~10..25
    """
    possible_fields = ["Low field","AC field","High field"]
    field_label = pick_unique_label(rng, used, lambda: rng.choice(possible_fields))
    def rflt(a,b):
        return round(rng.uniform(a,b),1)
    val1 = rflt(10,25)
    if rng.random()<0.3:
        val2 = "-"
    else:
        val2 = str(rflt(10,25))
    return [field_label, str(val1), val2]


def generate_crystallinity_melting_xlpe_v2_row(rng: np.random.Generator, used: Optional[Set[str]] = None) -> List[str]:
    """
    columns: [SampleType, Crystallinity(%), Melting(C)]
    """
    possible_samples = [
        "XLPE Only",
        "5% untreated nanosilica + XLPE",
        "5% vinylsilane treated nanosilica + XLPE",
        "5% untreated microsilica + XLPE"
    ]
    st = pick_unique_label(rng, used, lambda: rng.choice(possible_samples))
    cryst = str(rng.integers(40,61))
    melt = str(round(rng.uniform(100,120),1))
    return [st, cryst, melt]


def generate_low_freq_activation_energies_silica_v2_row(rng: np.random.Generator, used: Optional[Set[str]] = None) -> List[str]:
    """
    columns: [Sample, ActivationEnergy(eV)]
    ~0.15..0.35
    """
    possible_samples = [
        "XLPE + 5% untreated nanosilica",
        "XLPE + 5% vinylsilane treated nanosilica",
        "XLPE + 5% microsilica"
    ]
    st = pick_unique_label(rng, used, lambda: rng.choice(possible_samples))
    val = round(rng.uniform(0.15,0.35),2)
    return [st, str(val)]


def generate_peak_epr_signal_particulates_v2_row(rng: np.random.Generator, used: Optional[Set[str]] = None) -> List[str]:
    """
    columns: [Treatment, Powder Signal, Composite_Signal]
    signals ~500..8000
    """
    st = pick_unique_label(rng, used, lambda: rng.choice(TREATED_NANOCOMPOSITES))
    p_sig = rng.integers(500,8001)
    c_sig = rng.integers(500,8001)
    return [st,str(p_sig),str(c_sig)]


def generate_trap_depths_tsc_v2_row(rng: np.random.Generator, used: Optional[Set[str]] = None) -> List[str]:
    """
    columns: [Treatment, C5(eV), C4(eV)]
    eV => 0.8..2.5
    """
    st = pick_unique_label(rng, used, lambda: rng.choice(TREATED_NANOCOMPOSITES))
    c5v = round(rng.uniform(0.8,2.5),2)
    c4v = round(rng.uniform(1.0,1.2),2)
    return [st,str(c5v),str(c4v)]


def generate_peak_epr_signal_particulates_v3_row(rng: np.random.Generator, used: Optional[Set[str]] = None) -> List[str]:
    """
    columns: [Treatment, Powder Signal, Composite_Signal]
    merges TREATED_NANOCOMPOSITES + some other
    """
    some_other_list = [
        "Micron Scale silica","Nanoscale silica","AEAPS treated nanosilica",
        "HMDS treated nanosilica","TES treated nanosilica"
    ]
    combined_list = TREATED_NANOCOMPOSITES + some_other_list
    st = pick_unique_label(rng, used, lambda: rng.choice(combined_list))
    pow_s = rng.integers(500,8001)
    comp_s = rng.integers(500,8001)
    return [st,str(pow_s),str(comp_s)]


def generate_activation_energies_dielectric_spectroscopy_v2_row(rng: np.random.Generator, used: Optional[Set[str]] = None) -> List[str]:
    """
    columns: [Sample, Eact(eV)]
    """
    possible_samples = [
        "XLPE + 5% untreated nanosilica","XLPE + 5% AEAPS treated nanosilica",
        "XLPE + 5% HMDS-treated nanosilica","XLPE + 5% TES treated nanosilica",
        "XLPE + 5% microsilica"
    ]
    st = pick_unique_label(rng, used, lambda: rng.choice(possible_samples))
    val = round(rng.uniform(0.15,0.35),2)
    return [st,str(val)]


def generate_threshold_field_charge_accum_v2_row(rng: np.random.Generator, used: Optional[Set[str]] = None) -> List[str]:
    """
    columns: [Sample, PosThresh, NegThresh]
    """
    possible_samples = [
        "XLPE","XLPE + Micron Scale silica","XLPE + untreated nanoscale silica",
        "XLPE + AEAPS nanoscale silica","XLPE + HMDS nanoscale silica","XLPE + TES nanoscale silica"
    ]
    st = pick_unique_label(rng, used, lambda: rng.choice(possible_samples))
    if rng.random()<0.2:
        pos = "-"
        neg = "-"
    else:
        pos_val = rng.integers(10,31)
        neg_val = -rng.integers(10,31)
        pos = str(pos_val)
        neg = str(neg_val)
    return [st,pos,neg]


def generate_characteristic_breakdown_voltage_v2_row(rng: np.random.Generator, used: Optional[Set[str]] = None) -> List[str]:
    """
    columns: [Sample, 25C, 60C, 70C, 80C]
    typical ~ 40..450 + shape param
    """
    possible_samples = [
        "XLPE","XLPE + untreated nanosilica","XLPE + AEAPS nanosilica",
        "XLPE + HMDS nanosilica","XLPE + TES nanosilica"
    ]
    st = pick_unique_label(rng, used, lambda: rng.choice(possible_samples))
    def rint(a,b):
        return int(rng.integers(a,b))
    def rshp():
        return round(rng.uniform(1.5,3.0),2)
    c25 = f"{rint(40,450)} ({rshp()})"
    c60 = f"{rint(40,420)} ({rshp()})"
    c70 = f"{rint(40,400)} ({rshp()})"
    c80 = f"{rint(30,300)} ({rshp()})"
    return [st,c25,c60,c70,c80]


def generate_peak_epr_signal_particulates_v4_row(rng: np.random.Generator, used: Optional[Set[str]] = None) -> List[str]:
    """
    columns: [Materia_lLabel, SiO2_signal, Composite_Signal]
    signals ~500..8000
    """
    possible_labels = [
        "Micron scale silica","Nanoscale silica","AEAPS treated nanosilica",
        "HMDS treated nanosilica","TES treated nanosilica"
    ]
    combined_list = TREATED_NANOCOMPOSITES + possible_labels
    st = pick_unique_label(rng, used, lambda: rng.choice(combined_list))
    si_val = rng.integers(500,8001)
    comp_val = rng.integers(500,8001)
    return [st,str(si_val),str(comp_val)]


def generate_activation_energies_dielectric_spectroscopy_v3_row(rng: np.random.Generator, used: Optional[Set[str]] = None) -> List[str]:
    """
    columns: [Sample, Eact(eV)]
    """
    possible_samples = [
        "XLPE + 5% untreated nanosilica","XLPE + 5% AEAPS treated nanosilica",
        "XLPE + 5% HMDS-treated nanosilica","XLPE + 5% TES treated nanosilica",
        "XLPE + 5% microsilica"
    ]
    st = pick_unique_label(rng, used, lambda: rng.choice(possible_samples))
    val = round(rng.uniform(0.15,0.35),2)
    return [st,str(val)]


def generate_bt_ag_pvdf_composites_table37_row(rng: np.random.Generator, used: Optional[Set[str]] = None) -> List[str]:
    """
    columns: [Matrix Name, Filler Name, BT-Ag (wt%), Dk(1kHz), Tan(1kHz), Dk(100kHz), Tan(100kHz)]
    typical => Dk(1kHz) ~30..200, Tan(1kHz) ~0.01..0.2
    """
    mat = "PVDF"
    fill = "BT-Ag"
    comp_wt = rng.integers(0,81)
    dk1 = round(rng.uniform(30,200),1)
    lt1 = round(rng.uniform(0.01,0.2),2)
    dk2 = round(dk1*rng.uniform(0.8,0.9),1)
    lt2 = round(lt1*rng.uniform(0.8,1.2),2)
    return [mat, fill, str(comp_wt), str(dk1), str(lt1), str(dk2), str(lt2)]


###############################################################################
# ALL_SCHEMAS dictionary
###############################################################################
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
        ["Treatment", "Powder Signal", "Composite_Signal"],
        generate_peak_epr_signal_particulates_v2_row
    ),
    "schema30_trap_depths_tsc_v2": (
        ["Treatment", "C5(eV)", "C4(eV)"],
        generate_trap_depths_tsc_v2_row
    ),
    "schema31_peak_epr_signal_particulates_v3": (
        ["Treatment", "Powder Signal", "Composite_Signal"],
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
        ["MaterialLabel", "SiO2_signal", "Composite_Signal"],
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
    """Simple fallback title function."""
    return f"Table: {schema_name}"


def safe_is_numeric(x):
    try:
        return x.isnumeric()
    except:
        return False

def cast_dict_values(x):
    for x_idx, (key, value) in enumerate(x.items()):
        if isinstance(value, str):
            if "." in value:
                try:
                    x[key] = float(value)
                except:
                    pass
            elif safe_is_numeric(value):
                x[key] = int(value)
    return x

def table_cleanup(x:pd.DataFrame):
    if "Material" in x.columns:
        Material_column = x["Material"]
        # where names start with "None ", remove the "None " prefix
        x["Material"] = Material_column.apply(lambda x: x[5:].strip() if x.strip().startswith("None ") or x.strip().startswith("none ") else x)
    return x

def json_cleanup(x):

    x = cast_dict_values(x)

    keys_to_drop = []
    for (k) in x.keys():
        if k.endswith("_raw"):
            non_raw_key = k[:-4]
            non_raw_value = x[non_raw_key]
            raw_value = x[k]

            if non_raw_value == raw_value:
                keys_to_drop.append(non_raw_key)
                continue
    
    equals_keys = [
        "Material",
        "SampleType",
    ]
    for eq_key in equals_keys:
        if f"{eq_key}_raw" in x and f"{eq_key}_filler" in x:
            if x[f"{eq_key}_raw"] == x[f"{eq_key}_filler"]:
                keys_to_drop.append(eq_key + "_raw")

    for k in keys_to_drop:
        x.pop(k, None)
        
    return x



def main():
    rng = np.random.default_rng(seed=42)
    NUM_ROWS = 4

    for schema_name, (columns, gen_func) in ALL_SCHEMAS.items():
        
        print(f"\n" * 3, end='',)
        print(f"---")
        print(f"\n ## {ensure_schema_title(schema_name)} ")

        used = set()
        rows = []
        for _ in range(NUM_ROWS):
            row_data = gen_func(rng, used)
            rows.append(row_data)

        df = pd.DataFrame(rows, columns=columns)
        df = table_cleanup(df)
        print("\nDataFrame:")
        print(df.to_string(index=False))
        print(f"---")
        print("\nJSON Rows:")
        json_rows = []
        for i, row in df.iterrows():
            row_dict = {"row_id": i+1}
            for col in columns:
                cell_val = str(row[col])
                row_dict[col] = cell_val
                # parse & subfields
                parsed_dict = parse_special_column(col, cell_val)
                row_dict.update(parsed_dict)

            row_dict = json_cleanup(row_dict)
            json_rows.append(row_dict)

        fj = json.dumps(json_rows, indent=2)
        print(fj)


if __name__ == "__main__":
    main()
