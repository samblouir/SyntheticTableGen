# generators.py
# -----------------------------------------------------------------------------
# This module defines row-generation functions for synthetic composite data. 
# Each function returns a single row (list) matching a specific table schema.
#
# Contains functions for generating random synthetic data rows for 
# various table schemas. Each function is meant to return a Python 
# list corresponding to the columns in that schema.
# -----------------------------------------------------------------------------


import random
import numpy as np

# Import from data_lists for filler, matrix lists, etc.
from synthetic_table_gen.data_lists import FILLERS, MATRICES

# Import utility functions for optional missing data or random numeric values
from synthetic_table_gen.utils import (
    maybe_missing_str,
    maybe_missing_float,
    get_random_property_value,
    get_random_property_unit
)

# -----------------------------------------------------------------------------
# Utility random label generators
# -----------------------------------------------------------------------------
def random_percent_label(rng: np.random.Generator) -> str:
    """
    Generate a random percentage label in increments of 0.5 or 2.5.
    Example outputs: "12%", "12.5%", "25.0%".

    :param rng: A numpy Random Generator instance.
    :return: Random percentage as a string.
    """
    step = rng.choice([0.5, 2.5])
    max_multiple = int(100 / step)
    multiple = rng.integers(0, max_multiple + 1)
    val = step * multiple
    if val.is_integer():
        if rng.random() < 0.5:
            return f"{int(val)}%"
        else:
            return f"{val:.1f}%"
    else:
        return f"{val:.1f}%"


def random_filler_label(rng: np.random.Generator) -> str:
    """
    Generate a random filler label from a small set, 
    sometimes adding a modifier like 'calcined' or 'treated'.

    :param rng: A numpy Random Generator instance.
    :return: Random filler label string.
    """
    fillers = [
        "nano silica", "micron Al2O3", "boehmite", "graphene",
        "untreated nanosilica", "vinyl silane-treated nanosilica",
        "montmorillonite", "fumed silica", "carbon nanotubes", "micro SiC"
    ]
    if rng.random() < 0.5:
        mod = rng.choice(["treated", "calcined", "surface-modified", "hybrid"])
        return f"{mod} {rng.choice(fillers)}"
    return rng.choice(fillers)


def random_matrix_label(rng: np.random.Generator) -> str:
    """
    Generate a random matrix (polymer) label, 
    possibly with an adjective like 'high-temp'.

    :param rng: A numpy Random Generator instance.
    :return: Random matrix label string.
    """
    matrices = [
        "XLPE", "EVA", "epoxy", "polyimide", "PET", "polypropylene",
        "Nylon 6", "ABS", "polycarbonate", "PPS"
    ]
    base = rng.choice(matrices)
    if rng.random() < 0.3:
        adjective = rng.choice(["high-temp", "UV-curable", "thermoset", "bio-based"])
        return f"{adjective} {base}"
    return base


def random_composition_label(rng: np.random.Generator, p_unfilled=0.15) -> str:
    """
    Generate a label describing filler composition, e.g. "22.5% carbon nanotubes".
    May return 'Unfilled' with probability p_unfilled.

    :param rng: A numpy Random Generator.
    :param p_unfilled: Probability of returning "Unfilled".
    :return: String describing the composition.
    """
    if rng.random() < p_unfilled:
        return "Unfilled"
    pct = random_percent_label(rng)
    filler = random_filler_label(rng)
    return f"{pct} {filler}"


def random_material_label(rng: np.random.Generator, p_unfilled=0.2) -> str:
    """
    Generate a label combining filler and matrix, e.g. '30% fumed silica epoxy'.
    With probability p_unfilled, yield e.g. 'Unfilled EVA'.

    :param rng: A numpy Random Generator.
    :param p_unfilled: Probability of 'Unfilled' label.
    :return: Composite or unfilled label as a string.
    """
    mat = random_matrix_label(rng)
    if rng.random() < p_unfilled:
        return f"Unfilled {mat}"
    pct = random_percent_label(rng)
    filler = random_filler_label(rng)
    return f"{pct} {filler} {mat}"


def random_condition_label(rng: np.random.Generator) -> str:
    """
    Generate a random label for a processing condition, e.g. 
    'Reactive doping at 200°C with 12.5%'.

    :param rng: A numpy Random Generator.
    :return: Condition label string.
    """
    templates = [
        "Tritherm at {temp}°C",
        "Reactive doping at {temp}°C with {wt}",
        "{wt} additive at {temp}°C",
        "{wt} doping at {temp}°C, vacuum environment",
        "Conditioning at {temp}°C with {wt}",
    ]
    template = rng.choice(templates)
    temp = rng.integers(150, 401)
    wt_str = random_percent_label(rng)
    return template.format(temp=temp, wt=wt_str)


def maybe_na_str(rng: np.random.Generator, base_str: str, p=0.15) -> str:
    """
    Return 'not available' with probability p, else base_str.

    :param rng: A numpy Random Generator.
    :param base_str: The string to possibly replace.
    :param p: Probability of 'not available'.
    :return: base_str or 'not available'.
    """
    if rng.random() < p:
        return "not available"
    return base_str

# ---------------------------------------------------------------------------
# 1) The new fixed-property composite generator
# ---------------------------------------------------------------------------
def generate_fixed_property_composite_row(rng: np.random.Generator):
    """
    Generates a single row with:
      - Matrix Name
      - Filler Name
      - Filler Loading
      - Weibull shape param (β)
      - Weibull scale param (α)
      - Interfacial Tension (value + unit)
      - Compression Stress At Break (value + unit)
      - Poisson’s Ratio (value + unit)

    :param rng: A numpy Random Generator instance
    :return: A list of 11 values
    """
    # 1) Matrix
    matrix = rng.choice(MATRICES)

    # 2) Filler
    filler = rng.choice(FILLERS)

    # 3) Filler Loading (like "22.0 wt%")
    loading_value = rng.uniform(0.1, 50.0)
    loading_unit = rng.choice(["wt%", "vol%", "phr"])
    filler_loading = f"{loading_value:.2f} {loading_unit}"

    # 4) Weibull shape & scale
    shape = round(rng.uniform(5, 30), 3)
    scale = round(rng.uniform(10, 60), 2)

    # 5) Interfacial Tension
    it_value = round(rng.uniform(0, 1e6), 3)
    it_unit = rng.choice(["mN/m", "dyne/cm", "degrees"])

    # 6) Compression Stress at Break
    cs_value = round(rng.uniform(0, 1e6), 3)
    cs_unit = rng.choice(["MPa", "kPa", "°R", "psi"])

    # 7) Poisson’s Ratio
    pr_value = round(rng.uniform(0.1, 0.5), 3)
    pr_unit = "dimensionless"

    return [
        matrix,
        filler,
        filler_loading,
        shape,
        scale,
        it_value,
        it_unit,
        cs_value,
        cs_unit,
        pr_value,
        pr_unit
    ]

# ---------------------------------------------------------------------------
# 2) Specialized row generators
# ---------------------------------------------------------------------------

def generate_weibull_epoxy_tio2_row(rng: np.random.Generator):
    """
    Generate a row with composition, shape, and scale parameters 
    for Weibull epoxy-tio2 data.

    :param rng: Numpy Random Generator.
    :return: [composition_label, shape, scale]
    """
    comp = random_composition_label(rng, p_unfilled=0.2)
    shape = round(rng.uniform(3, 45), 2)
    scale = round(rng.uniform(10, 100), 2)
    return [comp, shape, scale]


def generate_weibull_epoxy_al2o3_row(rng: np.random.Generator):
    """
    Generate a row with composition, shape, and scale for 
    Weibull epoxy-al2o3 data.

    :param rng: Numpy Random Generator.
    :return: [composition_label, shape, scale]
    """
    comp = random_composition_label(rng, p_unfilled=0.15)
    shape = round(rng.uniform(3, 40), 2)
    scale = round(rng.uniform(15, 90), 2)
    return [comp, shape, scale]


def generate_weibull_epoxy_zno_row(rng: np.random.Generator):
    """
    Generate a row with composition, nano, and micro filler data
    for epoxy-zno composites.

    :param rng: Numpy Random Generator.
    :return: [composition_label, nano_val, micro_val]
    """
    comp = random_composition_label(rng, p_unfilled=0.1)

    def maybe_na():
        if rng.random() < 0.15:
            return "not available"
        else:
            return round(rng.uniform(3, 80), 2)

    nano_val = maybe_na()
    micro_val = maybe_na()
    return [comp, nano_val, micro_val]


def generate_mech_eva_row(rng: np.random.Generator):
    """
    Generate mechanical property data for EVA composites, 
    including tensile strength, modulus, and elongation.

    :param rng: Numpy Random Generator.
    :return: [sample, tensile_strength, modulus_300, elongation]
    """
    s = random_material_label(rng, p_unfilled=0.3)
    t_str = round(rng.uniform(3, 15), 1)
    mod_300 = round(rng.uniform(0.3, 3.0), 2)
    elong = rng.integers(500, 2001)
    return [s, t_str, mod_300, elong]


def generate_dynamic_mech_eva_row(rng: np.random.Generator):
    """
    Generate dynamic mechanical analysis data for EVA composites.

    :param rng: Numpy Random Generator.
    :return: [sample, temperature, E'(T8), E'(30C), tanδ(Tg), tanδ(30C)]
    """
    s = random_material_label(rng, p_unfilled=0.25)
    temp_c = round(rng.uniform(-60, 60), 1)
    e_t8 = f"{round(rng.uniform(0.05, 2.5), 2)} x 10^{rng.integers(6, 9)}"
    e_30c = f"{round(rng.uniform(0.05, 2.5), 2)} x 10^{rng.integers(5, 8)}"
    tan_tg = round(rng.uniform(0.3, 1.2), 2)
    tan_30c = round(rng.uniform(0.05, 0.5), 2)
    return [s, temp_c, e_t8, e_30c, tan_tg, tan_30c]


def generate_activation_energies_tio2_row(rng: np.random.Generator):
    """
    Generate data representing activation energies below and above Tg.

    :param rng: Numpy Random Generator.
    :return: [below, above]
    """
    def random_text(prefix):
        base = round(rng.uniform(0.5, 3.0), 1)
        return f"{prefix} {base} ± 0.1 eV"

    below = random_text("Below T")
    above = random_text("Above Tg")
    return [below, above]


def generate_breakdown_strength_xlpe_row(rng: np.random.Generator):
    """
    Generate DC and AC characteristic breakdown data for XLPE.

    :param rng: Numpy Random Generator.
    :return: [material_label, dc_break, ac_break]
    """
    mat = random_material_label(rng, p_unfilled=0.25)

    def rand_with_parenthesis():
        val = round(rng.uniform(80, 300), 1)
        shape = round(rng.uniform(1, 8), 1)
        return f"{val} ({shape})"

    dc_break = rand_with_parenthesis()
    ac_break = rand_with_parenthesis()
    return [mat, dc_break, ac_break]


def generate_lichtenecker_rother_row(rng: np.random.Generator):
    """
    Generate data for Lichtenecker-Rother formula evaluations.

    :param rng: Numpy Random Generator.
    :return: [material_label, freq, LR_pred, meas_E']
    """
    mat = random_material_label(rng, p_unfilled=0.2)
    freq_options = ["1k", "10k", "100k", "1M", "2M", "500k"]
    freq = rng.choice(freq_options)
    lr_pred = maybe_missing_float(rng, 1, 200)
    meas_e = maybe_missing_float(rng, 1, 250)
    return [mat, freq, lr_pred, meas_e]


def generate_current_decay_exponent_row(rng: np.random.Generator):
    """
    Generate a single row for current decay exponent data.

    :param rng: Numpy Random Generator.
    :return: [material_label, exponent]
    """
    mat = random_material_label(rng, p_unfilled=0.3)
    n_exp = round(rng.uniform(0.3, 2.5), 2)
    return [mat, n_exp]


def generate_breakdown_strength_unfilled_nano_resins(rng: np.random.Generator):
    """
    Generate DC breakdown strengths at 25C and 80C for unfilled/nano resins.

    :param rng: Numpy Random Generator.
    :return: [material_label, DC_25, DC_80]
    """
    material = random_material_label(rng, p_unfilled=0.4)

    def random_breakdown_str():
        main_val = round(rng.uniform(30, 600), 1)
        shape = round(rng.uniform(1, 12), 2)
        return f"{main_val} ({shape})"

    dc_25 = random_breakdown_str()
    dc_80 = random_breakdown_str()
    return [material, dc_25, dc_80]


def generate_impulse_test_breakdown_row(rng: np.random.Generator):
    """
    Generate impulse breakdown data for composites at 25C.

    :param rng: Numpy Random Generator.
    :return: [material_label, impulse_strength(B)]
    """
    mat = random_material_label(rng, p_unfilled=0.3)
    val = round(rng.uniform(100, 500), 1)
    shape = round(rng.uniform(1, 7), 1)
    combo = f"{val} ({shape})"
    return [mat, combo]


def generate_polarization_space_charge_decay_row(rng: np.random.Generator):
    """
    Generate polarization and space charge decay times for materials.

    :param rng: Numpy Random Generator.
    :return: [material_label, pol_decay, sc_decay]
    """
    mat = random_material_label(rng, p_unfilled=0.15)
    pol = round(rng.uniform(5, 150), 2)
    sc = round(rng.uniform(500, 10000), 2)
    return [mat, pol, sc]


def generate_ac_breakdown_scale_params_row(rng: np.random.Generator):
    """
    Generate AC breakdown scale parameters.

    :param rng: Numpy Random Generator.
    :return: [condition, nval, shape, pctdec]
    """
    condition = random_condition_label(rng)
    nval = maybe_missing_float(rng, 40, 450)
    shape = maybe_missing_float(rng, 2, 12)
    pctdec = maybe_missing_float(rng, 0, 70)
    return [condition, nval, shape, pctdec]


def generate_dc_breakdown_scale_params_row(rng: np.random.Generator):
    """
    Generate DC breakdown scale parameters.

    :param rng: Numpy Random Generator.
    :return: [condition, nval, shape, pctdec]
    """
    condition = random_condition_label(rng)
    nval = maybe_missing_float(rng, 50, 600)
    shape = maybe_missing_float(rng, 2, 16)
    pctdec = maybe_missing_float(rng, 0, 120)
    return [condition, nval, shape, pctdec]


def generate_pi_ofg_elemental_analysis_row(rng: np.random.Generator):
    """
    Generate elemental analysis data for PI with OFG (O-functional group).

    :param rng: Numpy Random Generator.
    :return: [feed, N%, C%, H%, measured_wt, measured_mol, vol]
    """
    feed = random_percent_label(rng)
    n_pct = maybe_missing_float(rng, 4, 10)
    c_pct = maybe_missing_float(rng, 50, 75)
    h_pct = maybe_missing_float(rng, 1, 5)
    measured_wt = maybe_missing_float(rng, 0, 25)
    measured_mol = maybe_missing_float(rng, 0, 8)
    vol = maybe_missing_float(rng, 0, 40)
    return [feed, n_pct, c_pct, h_pct, measured_wt, measured_mol, vol]


def generate_pi_ofg_dielectric_density_row(rng: np.random.Generator):
    """
    Generate dielectric constant & density data for PI with OFG.

    :param rng: Numpy Random Generator.
    :return: [feed, diel, th_dens, meas_dens, rp_inc, free_vol]
    """
    feed = random_percent_label(rng)
    diel = f"{round(rng.uniform(2.0,4.0),2)} ± 0.04"
    th_dens = maybe_missing_float(rng, 1.1, 1.5)
    meas_dens = f"{round(rng.uniform(1.0,1.4),2)} ± 0.03"
    rp_inc = maybe_missing_float(rng, 0, 35)
    free_vol = maybe_missing_float(rng, 0, 0.8)
    return [feed, diel, th_dens, meas_dens, rp_inc, free_vol]


def generate_pi_ofg_thermal_mech_row(rng: np.random.Generator):
    """
    Generate thermal/mechanical data for PI with OFG.

    :param rng: Numpy Random Generator.
    :return: [feed, tg, tcb, cte, e_mod, max_stress, elong]
    """
    feed = random_percent_label(rng)
    tg = round(rng.uniform(330, 380), 1)
    tcb = round(rng.uniform(390, 550), 1)
    cte = round(rng.uniform(20, 60), 1)
    e_mod = f"{round(rng.uniform(1.5,3.2),2)} ± 0.03"
    max_stress = f"{round(rng.uniform(100,400),1)} ± 15"
    elong = f"{round(rng.uniform(0.3,2.5),2)} ± 0.3"
    return [feed, tg, tcb, cte, e_mod, max_stress, elong]


def generate_pi_ofg_surface_props_row(rng: np.random.Generator):
    """
    Generate surface property data for PI with OFG, 
    e.g. contact angles, surface energies.

    :param rng: Numpy Random Generator.
    :return: [feed, H2O, glycerol, ys, vd, y_s]
    """
    feed = random_percent_label(rng)
    h2o = f"{round(rng.uniform(50,80),1)} ± 1.0"
    gly = f"{round(rng.uniform(60,85),1)} ± 1.0"
    ys = round(rng.uniform(20,60),1)
    vd = round(rng.uniform(2,10),2)
    y_s = round(rng.uniform(15,60),1)
    return [feed, h2o, gly, ys, vd, y_s]


def generate_nano_concentration_tg_ips_row(rng: np.random.Generator):
    """
    Generate nano concentration, Tg, and IPS data for a table.

    :param rng: Numpy Random Generator.
    :return: [wt_pct, vol, tg_tmdsc, tg_bds, ips_nm, saxs, bds]
    """
    wt_pct = maybe_missing_float(rng, 0, 80)
    vol = maybe_missing_float(rng, 0, 40)
    tg_tmdsc = rng.integers(350, 391)
    tg_bds = rng.integers(350, 391)
    ips_nm = maybe_missing_float(rng, 1, 100)
    saxs = maybe_missing_float(rng, 0, 15)
    bds = maybe_missing_float(rng, 0, 15)
    return [wt_pct, vol, tg_tmdsc, tg_bds, ips_nm, saxs, bds]


def generate_uv_cured_props_row(rng: np.random.Generator):
    """
    Generate UV-cured sample properties: conversion, gel content, Tg, etc.

    :param rng: Numpy Random Generator.
    :return: [sample, conversion, gel, tg, abs_max]
    """
    s = random_material_label(rng, p_unfilled=0.1)
    conv = maybe_missing_float(rng, 60, 95)
    gel = maybe_missing_float(rng, 80, 100)
    tg = maybe_missing_float(rng, 130, 200)
    abs_max = rng.integers(350, 431)
    return [s, conv, gel, tg, abs_max]


def generate_np_content_tga_row(rng: np.random.Generator):
    """
    Generate approximate nanoparticle content and char content from TGA data.

    :param rng: Numpy Random Generator.
    :return: [sample, exp_char, approx_np]
    """
    s = random_material_label(rng, p_unfilled=0.2)
    exp_char = maybe_missing_float(rng, 0, 15)
    approx_np = maybe_missing_float(rng, 0, 15)
    return [s, exp_char, approx_np]


def generate_dsc_params_pvdf_bt_row(rng: np.random.Generator):
    """
    Generate DSC parameters for PVDF/BT samples.

    :param rng: Numpy Random Generator.
    :return: [sample, jg, tg, xc]
    """
    s = random_material_label(rng, p_unfilled=0.2)
    jg = round(rng.uniform(10, 40), 2)
    tg = round(rng.uniform(140, 180), 2)
    xc = round(rng.uniform(15, 40), 2)
    return [s, jg, tg, xc]


def generate_saturated_moisture_content_row(rng: np.random.Generator):
    """
    Generate data for saturated moisture content at different conditions.

    :param rng: Numpy Random Generator.
    :return: [sample, c25, c50, c80, c50_75]
    """
    s = random_material_label(rng, p_unfilled=0.3)
    c25 = f"{round(rng.uniform(0,1),2)} ± 0.01"
    c50 = f"{round(rng.uniform(0,2),2)} ± 0.03"
    c80 = f"{round(rng.uniform(0,5),2)} ± 0.05"
    c50_75 = f"{round(rng.uniform(0,1),2)} ± 0.02"
    return [s, c25, c50, c80, c50_75]


def generate_freezable_nonfrozen_water_row(rng: np.random.Generator):
    """
    Generate data for freezable vs. non-frozen water in a sample.

    :param rng: Numpy Random Generator.
    :return: [sample, fw, nfw, total]
    """
    s = random_material_label(rng, p_unfilled=0.2)
    fw = maybe_missing_float(rng, 0, 15)
    nfw = maybe_missing_float(rng, 0, 15)
    tot = (fw or 0) + (nfw or 0)
    return [s, fw, nfw, round(tot,2)]


def generate_free_space_length_weibull_row(rng: np.random.Generator):
    """
    Generate free space length Weibull data.

    :param rng: Numpy Random Generator.
    :return: [sample, L, 63%, +, a, B, t, a_2p]
    """
    s = random_material_label(rng, p_unfilled=0.25)

    def mm_str(x):
        if rng.random() > 0.15:
            return x
        else:
            return "not available"

    lval = mm_str(str(rng.integers(0, 1001)))
    three_63 = mm_str(str(rng.integers(100, 301)))
    three_plus = mm_str(str(rng.integers(5, 50)))
    three_a = mm_str(str(round(rng.uniform(0.5, 3),1)))
    three_b = mm_str(str(round(rng.uniform(30, 200),2)))
    three_t = mm_str(str(round(rng.uniform(80, 300),1)))
    two_a = mm_str(str(round(rng.uniform(100, 400),2)))
    return [s, lval, three_63, three_plus, three_a, three_b, three_t, two_a]


def generate_real_relative_permittivity_row(rng: np.random.Generator):
    """
    Generate real relative permittivity data under different field conditions.

    :param rng: Numpy Random Generator.
    :return: [field_label, vol1, vol2]
    """
    field_labels = ["Low field", "2kV", "5kV", "7.5kV", "10kV", "AC field", "High field"]
    f = rng.choice(field_labels)
    vol1 = maybe_missing_float(rng, 5, 30)
    vol2 = maybe_missing_float(rng, 10, 40)
    return [f, vol1, vol2]


def generate_crystallinity_melting_xlpe_row(rng: np.random.Generator):
    """
    Generate crystallinity and melting point data for XLPE.

    :param rng: Numpy Random Generator.
    :return: [sample_type, crystallinity, melting_point]
    """
    s = random_material_label(rng, p_unfilled=0.2)
    crys = f"{rng.integers(30, 71)} {rng.integers(0, 2)}"
    melt = f"{round(rng.uniform(95,115),1)} ± 1.0"
    return [s, crys, melt]


def generate_low_freq_activation_energies_silica_row(rng: np.random.Generator):
    """
    Generate low-frequency activation energies for silica-based composites.

    :param rng: Numpy Random Generator.
    :return: [sample_name, activation_energy]
    """
    s = random_material_label(rng, p_unfilled=0.2)
    val = f"{round(rng.uniform(0.05,0.4),2)} ± {round(rng.uniform(0.03,0.12),2)}"
    return [s, val]


def generate_peak_epr_signal_row(rng: np.random.Generator):
    """
    Generate peak EPR signal data for a powder vs. composite.

    :param rng: Numpy Random Generator.
    :return: [material_label, powder_signal, composite_signal]
    """
    m = random_material_label(rng, p_unfilled=0.1)
    p = rng.integers(200,8001)
    c = rng.integers(500,9001)
    return [m, p, c]


def generate_trap_depths_tsc_row(rng: np.random.Generator):
    """
    Generate TSC trap depths for two peaks: C5 and C4.

    :param rng: Numpy Random Generator.
    :return: [samples, c5_peak, c4_peak]
    """
    s = random_material_label(rng, p_unfilled=0.2)
    c5 = f"{round(rng.uniform(0.2,3.0),2)} ± 0.1"
    c4 = f"{round(rng.uniform(0.8,1.5),2)} ± 0.1"
    return [s, c5, c4]


def generate_peak_epr_signal_particulates_row(rng: np.random.Generator):
    """
    Generate EPR signal data for base material plus particulates.

    :param rng: Numpy Random Generator.
    :return: [material, SiO2, composite]
    """
    m = random_material_label(rng, p_unfilled=0.1)
    si = str(rng.integers(50,10001))
    comp = str(rng.integers(300,10001))
    return [m, si, comp]


def generate_activation_energies_dielectric_spectroscopy_row(rng: np.random.Generator):
    """
    Generate activation energies from dielectric spectroscopy.

    :param rng: Numpy Random Generator.
    :return: [sample_name, activation_energy]
    """
    s = random_material_label(rng, p_unfilled=0.2)
    val = f"{round(rng.uniform(0.05,0.5),2)} ± {round(rng.uniform(0.01,0.1),2)}"
    return [s, val]


def generate_threshold_field_charge_accum_row(rng: np.random.Generator):
    """
    Generate threshold field data for charge accumulation (positive/negative).

    :param rng: Numpy Random Generator.
    :return: [sample, threshold_field_pos, threshold_field_neg]
    """
    s = random_material_label(rng, p_unfilled=0.3)
    pos_val = rng.integers(10,31)
    neg_val = rng.integers(-30, -9)
    if rng.random() < 0.1:
        pos = "-"
    else:
        pos = str(pos_val)
    if rng.random() < 0.1:
        neg = "-"
    else:
        neg = str(neg_val)
    return [s, pos, neg]


def generate_characteristic_breakdown_voltage_row(rng: np.random.Generator):
    """
    Generate characteristic breakdown voltages at various temperatures.

    :param rng: Numpy Random Generator.
    :return: [materials, 25C, 60C, 70C, 80C]
    """
    s = random_material_label(rng, p_unfilled=0.2)

    def rand_bdv():
        main = rng.integers(40, 451)
        shape = round(rng.uniform(1, 6),2)
        return f"{main} ({shape})"

    c25 = rand_bdv()
    c60 = rand_bdv()
    c70 = rand_bdv()
    c80 = rand_bdv()
    return [s, c25, c60, c70, c80]


def generate_bt_ag_pvdf_dielectric_row(rng: np.random.Generator):
    """
    Generate data for BT-Ag-PVDF dielectric properties.

    :param rng: Numpy Random Generator.
    :return: [sample, bt_ag_wt, bt_ag_vol, bt_vol, ag_vol, dk_1k, dk_100k]
    """
    s = random_material_label(rng, p_unfilled=0.1)
    bt_ag_wt = rng.integers(0,101)
    bt_ag_vol = rng.integers(0,81)
    bt_vol = round(bt_ag_vol * rng.uniform(0.5, 0.9),1)
    ag_vol = round(bt_ag_vol - bt_vol,1)

    def random_dk_tan():
        dk = round(rng.uniform(3, 200), 1)
        tan = round(rng.uniform(0.005, 0.15), 3)
        return f"{dk}/{tan}"

    dk_1k = random_dk_tan()
    dk_100k = random_dk_tan()
    return [s, bt_ag_wt, bt_ag_vol, bt_vol, ag_vol, dk_1k, dk_100k]


def generate_pp_graphite_therm_mech_row(rng: np.random.Generator):
    """
    Generate thermal/mechanical data for PP with graphite reinforcement.

    :param rng: Numpy Random Generator.
    :return: [sample, y_mod, y_str, elong, absorbed, cryst_temp, half_time]
    """
    s = random_material_label(rng, p_unfilled=0.25)
    y_mod = f"{rng.integers(700,2001)} + {rng.integers(10,60)}"
    y_str = rng.choice([str(rng.integers(15,50)), "N/A"])
    elong = f"{rng.integers(3,1001)} + {rng.integers(1,50)}"
    absorbed = f"{round(rng.uniform(0.3,4.5),2)} + {round(rng.uniform(0.1,1.0),2)}"
    cryst_temp = rng.integers(370, 431)
    half_time = rng.choice([f">{rng.integers(40,160)}", str(round(rng.uniform(0.5,15),1))])
    return [s, y_mod, y_str, elong, absorbed, cryst_temp, half_time]


def generate_pp_graphite_therm_mech_repeat_row(rng: np.random.Generator):
    """
    Generate the same row format as generate_pp_graphite_therm_mech_row.

    :param rng: Numpy Random Generator.
    :return: A repeated format row for test duplication.
    """
    return generate_pp_graphite_therm_mech_row(rng)


def generate_xrd_patterns_row(rng: np.random.Generator):
    """
    Generate XRD pattern data: peaks with 2-theta and intensity.

    :param rng: Numpy Random Generator.
    :return: [sample, peak_str, phase]
    """
    s = random_material_label(rng, p_unfilled=0.2)
    peaks = []
    num_peaks = rng.integers(2,5)
    for _ in range(num_peaks):
        two_theta = round(rng.uniform(5,80),1)
        intensity = rng.integers(50,1001)
        peaks.append(f"{two_theta}({intensity})")
    peak_str = "; ".join(peaks)
    phase = rng.choice(["Alpha-phase", "Beta-phase", "Mixed phases", "Gamma-phase", "Delta-phase"])
    return [s, peak_str, phase]


def generate_rheological_props_row(rng: np.random.Generator):
    """
    Generate rheological properties: shear rate, viscosity, storage/loss mod, freq.

    :param rng: Numpy Random Generator.
    :return: [material, shear_rate, viscosity, storage_mod, loss_mod, freq_hz]
    """
    m = random_material_label(rng, p_unfilled=0.2)
    shear_rate = round(rng.uniform(0.5,2000),2)
    viscosity = round(rng.uniform(5,20000),2)
    storage_mod = round(rng.uniform(50,5000),2)
    loss_mod = round(rng.uniform(20,3000),2)
    freq_hz = round(rng.uniform(0.01,200),2)
    return [m, shear_rate, viscosity, storage_mod, loss_mod, freq_hz]


def generate_sem_eds_row(rng: np.random.Generator):
    """
    Generate SEM/EDS data including grain size and elemental composition.

    :param rng: Numpy Random Generator.
    :return: [sample, grain_size_nm, composition_str]
    """
    s = random_material_label(rng, p_unfilled=0.15)
    grain_size_nm = round(rng.uniform(40,3000),2)

    elements = ["Fe", "Cr", "Ni", "C", "Al", "Si", "Mg", "Ti", "Cu", "Zr", "Zn"]
    rng.shuffle(elements)
    composition_str = []
    left_pct = 100
    how_many = rng.integers(3,6)
    for elem in elements[:how_many]:
        portion = rng.integers(5, max(6, left_pct//2 + 1))
        composition_str.append(f"{elem}={portion}%")
        left_pct -= portion
        if left_pct <= 0:
            break
    if left_pct > 0:
        composition_str.append(f"Other={left_pct}%")

    comp = "; ".join(composition_str)
    return [s, grain_size_nm, comp]


def generate_hpc_simulation_summary_row(rng: np.random.Generator):
    """
    Generate HPC simulation summary data.

    :param rng: Numpy Random Generator.
    :return: [sim_name, model_type, final_energy, cpu_hours, note]
    """
    sim_list = ["MD run A", "DFT calc B", "FEM model C", "KMC sim D", "Phase-Field run E", "Monte Carlo run F"]
    sim = rng.choice(sim_list)
    model = rng.choice([
        "Molecular Dynamics", "Density Functional Theory", "Finite Element",
        "Kinetic Monte Carlo", "Phase Field", "Coarse-grain MD"
    ])
    final_energy = round(rng.uniform(-600, 600), 2)
    cpu_hours = round(rng.uniform(5,3000),1)
    note = rng.choice([
        "Converged", "Divergent in final step", "Partially converged",
        "Converged with warnings", "Exceeded max steps", "Insufficient memory"
    ])
    return [sim, model, final_energy, cpu_hours, note]


def generate_flammability_test_row(rng: np.random.Generator):
    """
    Generate flammability test data: LOI, UL94 rating, ignition time, residue.

    :param rng: Numpy Random Generator.
    :return: [material, LOI, UL94, time_ign, char_res]
    """
    mat = random_material_label(rng, p_unfilled=0.1)
    loi = round(rng.uniform(15,40),1)
    ul94 = rng.choice(["V-0", "V-1", "V-2", "HB", "5VB", "5VA"])
    time_ign = round(rng.uniform(1,60),1)
    char_res = round(rng.uniform(0,50),1)
    return [mat, loi, ul94, time_ign, char_res]


def generate_oit_test_row(rng: np.random.Generator):
    """
    Generate OIT test data: OIT, test temperature, outcome note.

    :param rng: Numpy Random Generator.
    :return: [material, OIT, test_temp, note]
    """
    m = random_material_label(rng, p_unfilled=0.3)
    oit_val = round(rng.uniform(1,60),1)
    temp = round(rng.uniform(150,280),1)
    note = rng.choice(["Pass", "Fail", "Needs retest", "Conditional", "Deferred"])
    return [m, oit_val, temp, note]

