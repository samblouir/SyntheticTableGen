# schemas.py
# -----------------------------------------------------------------------------
# Defines a big TABLE_SCHEMAS dictionary linking table names to columns 
# and generator functions. Also includes logic to assign random table numbers 
# and ensure each schema has a title.
#
# Also, declares the TABLE_SCHEMAS for all synthetic tables, mapping columns 
# to row generator functions. Also provides helper functions to 
# automatically set or ensure a table title (with random numbering).
# -----------------------------------------------------------------------------


import numpy as np

# Import all row generator functions from generators
from synthetic_table_gen.generators import (
    generate_fixed_property_composite_row,
    generate_weibull_epoxy_tio2_row,
    generate_weibull_epoxy_al2o3_row,
    generate_weibull_epoxy_zno_row,
    generate_mech_eva_row,
    generate_dynamic_mech_eva_row,
    generate_activation_energies_tio2_row,
    generate_breakdown_strength_xlpe_row,
    generate_lichtenecker_rother_row,
    generate_current_decay_exponent_row,
    generate_breakdown_strength_unfilled_nano_resins,
    generate_impulse_test_breakdown_row,
    generate_polarization_space_charge_decay_row,
    generate_ac_breakdown_scale_params_row,
    generate_dc_breakdown_scale_params_row,
    generate_pi_ofg_elemental_analysis_row,
    generate_pi_ofg_dielectric_density_row,
    generate_pi_ofg_thermal_mech_row,
    generate_pi_ofg_surface_props_row,
    generate_nano_concentration_tg_ips_row,
    generate_uv_cured_props_row,
    generate_np_content_tga_row,
    generate_dsc_params_pvdf_bt_row,
    generate_saturated_moisture_content_row,
    generate_freezable_nonfrozen_water_row,
    generate_free_space_length_weibull_row,
    generate_real_relative_permittivity_row,
    generate_crystallinity_melting_xlpe_row,
    generate_low_freq_activation_energies_silica_row,
    generate_peak_epr_signal_row,
    generate_trap_depths_tsc_row,
    generate_peak_epr_signal_particulates_row,
    generate_activation_energies_dielectric_spectroscopy_row,
    generate_threshold_field_charge_accum_row,
    generate_characteristic_breakdown_voltage_row,
    generate_bt_ag_pvdf_dielectric_row,
    generate_pp_graphite_therm_mech_row,
    generate_pp_graphite_therm_mech_repeat_row,
    generate_xrd_patterns_row,
    generate_rheological_props_row,
    generate_sem_eds_row,
    generate_hpc_simulation_summary_row,
    generate_flammability_test_row,
    generate_oit_test_row,
)

def auto_table_number(rng: np.random.Generator) -> int:
    """
    Return a random table number (1..999).

    :param rng: Numpy Random Generator.
    :return: Random int in [1..999].
    """
    return rng.integers(1, 1000)


def auto_table_title(table_name: str, columns: list) -> str:
    """
    Create a fallback table title. The columns list is not used except 
    for possible debugging or extension.

    :param table_name: The schema key name (e.g. 'weibull_epoxy_tio2').
    :param columns: Column names for that schema.
    :return: "Table X. <title>" with X placeholder for random number.
    """
    nice_name = table_name.replace("_", " ").title()
    return f"Table X. {nice_name}"


TABLE_SCHEMAS = {
    "composite_fixed_props": {
        "columns": [
            "Matrix Name",
            "Filler Name",
            "Filler Loading",
            "Weibull Shape Param (β)",
            "Weibull Scale Param (α)",
            "Interfacial Tension",
            "IT Unit",
            "Compression Stress at Break",
            "CS Unit",
            "Poissons Ratio",
            "PR Unit"
        ],
        "generator_func": generate_fixed_property_composite_row
    },
    "weibull_epoxy_tio2": {
        "columns": ["Composition", "Shape Parameter (B)", "Scale Parameter (a)"],
        "generator_func": generate_weibull_epoxy_tio2_row
    },
    "weibull_epoxy_al2o3": {
        "columns": ["Composition", "Shape Parameter (B)", "Scale Parameter (a)"],
        "generator_func": generate_weibull_epoxy_al2o3_row
    },
    "weibull_epoxy_zno": {
        "columns": ["Composition", "Nanocomposites", "Microcomposites"],
        "generator_func": generate_weibull_epoxy_zno_row
    },
    "eva_mechanical": {
        "columns": [
            "Sample",
            "Tensile Strength (MPa)",
            "Modulus at 300% Elongation (MPa)",
            "Elongation Break at (%)"
        ],
        "generator_func": generate_mech_eva_row
    },
    "eva_dynamic_mech": {
        "columns": [
            "Sample",
            "T.(C)",
            "E'(Pa) at T8",
            "E'(Pa) at 30C",
            "tanδ at Tg",
            "tanδ at 30C"
        ],
        "generator_func": generate_dynamic_mech_eva_row
    },
    "activation_energies_tio2": {
        "columns": ["TiO2 nanocomposite", "TiO2 microcomposite"],
        "generator_func": generate_activation_energies_tio2_row
    },
    "breakdown_strength_xlpe": {
        "columns": [
            "Material",
            "dc Characteristic Breakdown (B)",
            "ac Characteristic Breakdown (B)"
        ],
        "generator_func": generate_breakdown_strength_xlpe_row
    },
    "lichtenecker_rother": {
        "columns": ["Material", "f(Hz)", "(L-R)", "Measured E'"],
        "generator_func": generate_lichtenecker_rother_row
    },
    "current_decay_exponent": {
        "columns": ["Material", "Current Decay Exponent n"],
        "generator_func": generate_current_decay_exponent_row
    },
    "breakdown_strength_unfilled_nano_resins": {
        "columns": [
            "Material [Ref]",
            "dc Breakdown @25C (B)",
            "dc Breakdown @80C (B)"
        ],
        "generator_func": generate_breakdown_strength_unfilled_nano_resins
    },
    "impulse_test_breakdown": {
        "columns": ["Material", "1.2x50 us Impulse strength @25C (B)"],
        "generator_func": generate_impulse_test_breakdown_row
    },
    "polarization_space_charge": {
        "columns": ["Material", "Polarization Decay (s)", "Space Charge Decay (s)"],
        "generator_func": generate_polarization_space_charge_decay_row
    },
    "ac_breakdown_scale_params": {
        "columns": ["MaterialCondition", "n_value", "ShapeParam", "% decrease"],
        "generator_func": generate_ac_breakdown_scale_params_row
    },
    "dc_breakdown_scale_params": {
        "columns": ["MaterialCondition", "n_value", "ShapeParam", "% decrease"],
        "generator_func": generate_dc_breakdown_scale_params_row
    },
    "pi_ofg_elemental_analysis": {
        "columns": [
            "OFG in PI (Feed wt %)",
            "N (%)",
            "C (%)",
            "H (%)",
            "OFG in PI (Measured wt %)",
            "OFG in PI (Measured mol %)",
            "OFG in PI (vol %)"
        ],
        "generator_func": generate_pi_ofg_elemental_analysis_row
    },
    "pi_ofg_dielectric_density": {
        "columns": [
            "OFG in PI (Feed wt %)",
            "Dielectric Constant (at 100 KHz)",
            "Theoretical Density (g/cm^3)",
            "Measured Density (g/cm^3)",
            "Relative Porosity Increase (%)",
            "Total Free Volume Fraction"
        ],
        "generator_func": generate_pi_ofg_dielectric_density_row
    },
    "pi_ofg_thermal_mech": {
        "columns": [
            "OFG in PI (Feed wt %)",
            "Tg (C)",
            "T (C)b",
            "CTE (ppm/°C)c",
            "Young's Modulus (GPa)",
            "Maximum Stress (MPa)",
            "Elongation at Break (%)"
        ],
        "generator_func": generate_pi_ofg_thermal_mech_row
    },
    "pi_ofg_surface_props": {
        "columns": [
            "OFG in PI (Feed wt %)",
            "H2O",
            "Glycerol",
            "Y's (mN/m)^2",
            "vd (mN/m)",
            "y's (mN/m)^2"
        ],
        "generator_func": generate_pi_ofg_surface_props_row
    },
    "nanoparticle_concentration_tg_ips": {
        "columns": [
            "SiO (wt %)",
            "SiO (vol)",
            "Tg(TMDSC) (K)",
            "Tg(BDS) (K)",
            "IPS (nm)",
            "SAXS",
            "BDS"
        ],
        "generator_func": generate_nano_concentration_tg_ips_row
    },
    "uv_cured_sample_props": {
        "columns": [
            "Cured sample",
            "Epoxy group conversion (%)",
            "Gel content (%)",
            "Tg (C)",
            "Absorption maximum (nm)"
        ],
        "generator_func": generate_uv_cured_props_row
    },
    "np_content_tga": {
        "columns": [
            "Cured sample",
            "Experimental char content (wt.-%)",
            "Approximate NP content (wt.-%)"
        ],
        "generator_func": generate_np_content_tga_row
    },
    "dsc_params_pvdf_bt": {
        "columns": [
            "sample",
            "[J/g]",
            "Tg , [°C]",
            "X-Subscript(c) [%]"
        ],
        "generator_func": generate_dsc_params_pvdf_bt_row
    },
    "saturated_moisture_content": {
        "columns": [
            "Sample",
            "25 °C 100% rh",
            "50 °C 100% rh",
            "80 °C 100% rh",
            "50 °C 75% rh"
        ],
        "generator_func": generate_saturated_moisture_content_row
    },
    "freezable_nonfrozen_water": {
        "columns": [
            "Sample",
            "Freezable water (mg/g)",
            "Non-frozen water (mg/g)",
            "Total water (mg/g)"
        ],
        "generator_func": generate_freezable_nonfrozen_water_row
    },
    "free_space_length_weibull": {
        "columns": ["Sample", "L", "63%", "+", "a", "B", "t", "a_2p"],
        "generator_func": generate_free_space_length_weibull_row
    },
    "real_relative_permittivity": {
        "columns": [
            "ac field (kV/mm)",
            "20 vol% BaTiO3 fibers",
            "20 vol% BaTiO3 fibers +0.43 vol% GPLs"
        ],
        "generator_func": generate_real_relative_permittivity_row
    },
    "crystallinity_melting_xlpe": {
        "columns": [
            "Sample Type",
            "Degree of crystallinity (%)",
            "Melting Point (C)"
        ],
        "generator_func": generate_crystallinity_melting_xlpe_row
    },
    "low_freq_activation_energies_silica": {
        "columns": ["Sample Name", "Activation Energy (eV)"],
        "generator_func": generate_low_freq_activation_energies_silica_row
    },
    "peak_epr_signal": {
        "columns": ["MaterialLabel", "PowderSignal(a.u.)", "CompositeSignal(a.u.)"],
        "generator_func": generate_peak_epr_signal_row
    },
    "trap_depths_tsc": {
        "columns": ["Samples", "C5 peak(eV)", "C4 peak(eV)"],
        "generator_func": generate_trap_depths_tsc_row
    },
    "peak_epr_signal_particulates": {
        "columns": ["Material", "SiO2", "Composite"],
        "generator_func": generate_peak_epr_signal_particulates_row
    },
    "activation_energies_dielectric_spectroscopy": {
        "columns": ["Sample name", "Activation energy (eV)"],
        "generator_func": generate_activation_energies_dielectric_spectroscopy_row
    },
    "threshold_field_charge_accum": {
        "columns": ["Sample", "Threshold field (positive) kV/mm", "Threshold field (negative) kV/mm"],
        "generator_func": generate_threshold_field_charge_accum_row
    },
    "characteristic_breakdown_voltage": {
        "columns": ["Materials", "25 °C", "60 °C", "70 °C", "80 °C"],
        "generator_func": generate_characteristic_breakdown_voltage_row
    },
    "bt_ag_pvdf_dielectric": {
        "columns": [
            "sample",
            "BT-Ag (wt %)",
            "BT-Ag (vol %)",
            "BT (vol %)",
            "Ag (vol %)",
            "Dk/tan 8 1 kHz",
            "Dk/tan 8 100 kHz"
        ],
        "generator_func": generate_bt_ag_pvdf_dielectric_row
    },
    "pp_graphite_therm_mech": {
        "columns": [
            "samples",
            "Young's modulus, E (MPa)",
            "yield strength, Oy (MPa)",
            "elongation at break, EB (%)",
            "absorbed energy per thickness, W (J/cm)",
            "crystallization temp, Tconset at -10K/min (K)",
            "isothermal crystallization half-time, T1/2, at 413 K (min)"
        ],
        "generator_func": generate_pp_graphite_therm_mech_row
    },
    "pp_graphite_therm_mech_repeat": {
        "columns": [
            "samples",
            "Young's modulus, E (MPa)",
            "yield strength, Oy (MPa)",
            "elongation at break, EB (%)",
            "absorbed energy per thickness, W (J/cm)",
            "crystallization temp, Tconset at -10K/min (K)",
            "isothermal crystallization half-time, T1/2, at 413 K (min)"
        ],
        "generator_func": generate_pp_graphite_therm_mech_repeat_row
    },
    "xrd_patterns": {
        "columns": ["Sample", "Peaks_2theta_Intensity", "PhaseIdentification"],
        "generator_func": generate_xrd_patterns_row
    },
    "rheological_props": {
        "columns": [
            "Material",
            "Shear Rate (1/s)",
            "Viscosity (Pa.s)",
            "G' (Pa)",
            "G'' (Pa)",
            "Frequency (Hz)"
        ],
        "generator_func": generate_rheological_props_row
    },
    "sem_eds": {
        "columns": ["Sample", "GrainSize (nm)", "ElementalComposition"],
        "generator_func": generate_sem_eds_row
    },
    "hpc_simulation_summary": {
        "columns": [
            "SimulationName",
            "ModelType",
            "FinalEnergy (kJ/mol)",
            "CPU_hours",
            "ResultNote"
        ],
        "generator_func": generate_hpc_simulation_summary_row
    },
    "flammability_test": {
        "columns": [
            "Material",
            "LOI (%)",
            "UL94 Rating",
            "TimeToIgnition (s)",
            "CharResidue (%)"
        ],
        "generator_func": generate_flammability_test_row
    },
    "oit_test": {
        "columns": [
            "Material",
            "OIT (min)",
            "TestTemperature (C)",
            "OutcomeNote"
        ],
        "generator_func": generate_oit_test_row
    }
}


def ensure_schema_title(schema_name: str, rng: np.random.Generator):
    """
    If no 'title' is present in TABLE_SCHEMAS[schema_name], build one with 
    auto_table_title() and insert a random table number.

    :param schema_name: The table schema key.
    :param rng: Numpy Random Generator for random table number.
    """
    schema = TABLE_SCHEMAS[schema_name]
    if "title" not in schema:
        columns = schema["columns"]
        fallback = auto_table_title(schema_name, columns)
        tbl_num = auto_table_number(rng)
        new_title = fallback.replace("Table X.", f"Table {tbl_num}.", 1)
        schema["title"] = new_title

