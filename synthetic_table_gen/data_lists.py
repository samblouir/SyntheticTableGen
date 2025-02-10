# data_lists.py
# -----------------------------------------------------------------------------
# This module holds large lists of fillers, matrices, properties, and units, 
# plus optional dictionaries (PROPERTY_UNIT_MAP, PROPERTY_RANGES) to restrict 
# random values based on the property name. It also provides helper functions
# for mapping synonyms to canonical property names.
#
# Provides global data structures for filler names, matrix polymers,
# properties, and units. Also defines mapping dictionaries for property
# synonyms, units, and typical numeric ranges, along with helper functions
# to retrieve canonical property data.
# -----------------------------------------------------------------------------
# 1) Lists of FILLERS, MATRICES, PROPERTIES, UNITS
# 2) Synonyms for each
# 3) Dictionaries mapping synonyms -> canonical forms, property -> unit lists,
#    property -> numeric ranges, etc.
# 4) Helper functions for property name lookups
# -----------------------------------------------------------------------------

# ============== FILLERS ==============
# Declare the large list of fillers, plus synonyms, then merge them.
FILLERS = [
    "(3-Chloropropyl)(ethoxy)dimethylsilane",
    "9-Anthracenemethanol",
    "Aluminium",
    "Aluminium oxide",
    "Aminopropyldimethylethoxysilane",
    "Barium strontium titanate",
    "Barium titanate",
    "Bentonite",
    "Buckminsterfullerene",
    "Calcium carbonate",
    "Calcium copper titanate",
    "Carbon black",
    "Carbon nanofibers",
    "Cellulose",
    "Clay",
    "Copper(II) phenyl phosphate",
    "Dodecylbenzenesulfonic acid",
    "Fluorine mica",
    "Gold",
    "Graphene",
    "Graphene oxide",
    "Graphite",
    "Graphite oxide",
    "Iron oxide - barium titanate core-shell nanoparticles",
    "Iron(II,III) oxide",
    "Lead(II) sulfide",
    "Magnesium oxide",
    "Methoxy(dimethyl)octylsilane",
    "Mineral oil",
    "Molybdenum Disulfide",
    "Molybdenum disulfide",
    "Montmorillonite",
    "Multi-wall carbon nanotubes",
    "Octa(aminophenyl) polyhedral oligomeric silsesquioxane",
    "Octakis(dimethylsiloxyhexafluoropropylglycidyl ether)silsesquioxane",
    "Organo-modified layered silicates",
    "Poly(acrylonitrile)",
    "Poly(vinyl alcohol)",
    "Polyaniline",
    "Reactive fluorine polyhedral oligomeric silsesquioxane",
    "Reduced graphene oxide",
    "Sepiolite",
    "Silicon carbide",
    "Silicon dioxide",
    "Silver",
    "Silver hexafluoroantimonate",
    "Silver nitrate",
    "Single-wall carbon nanotubes",
    "Sodium titanate nanotubes",
    "Talcum",
    "Titanium dioxide",
    "Triphenyl phosphate",
    "Zinc oxide",
    "Zirconium dioxide",
    "Boron nitride",
    "Boron carbide",
    "Zirconium carbide",
    "Hexagonal boron nitride",
    "Cellulose nanocrystals",
    "Halloysite nanotubes",
    "Graphitic carbon nitride",
    "MXene (Ti3C2Tx)",
    "Bismuth telluride",
    "Vanadium pentoxide",
    "Indium tin oxide",
    "Phosphorene",
    "Silicon nitride",
    "Tungsten disulfide",
    "Halloysite clay",
    "Titanium carbide",
    "Boron nitride nanotubes",
]

# FILLER SYNONYMS & HISTORICAL TERMS
FILLER_SYNONYMS = [
    "Reinforcement",
    "Reinforcing agent",
    "Filling agent",
    "Particulate reinforcement",
    "Nano-additive",
    "Micro-additive",
    "Extender",
    "Silica",
    "SiO2",
    "Kieselguhr",
    "Silicic anhydride",
    "Alumina",
    "Al2O3",
    "Corundum",
    "Magnesia",
    "Moly Disulfide",
    "MoS2",
    "White graphite",
    "Barium titanate ceramic",
    "Graphitic oxide",
    "Graphitic whiskers",
    "Zinc white",
    "Fuller’s earth",
]

# Merge synonyms into FILLERS, removing duplicates
FILLERS.extend(FILLER_SYNONYMS)
FILLERS = list(set(FILLERS))

# ============== MATRICES ==============
MATRICES = [
    "Acrylonitrile butadiene styrene",
    "Bisphenol E cyanate ester resin",
    "bisphenol-A phthalonitrile resin",
    "Bisphenol-A-epoxy vinyl ester resin",
    "Butanediol-alt-Diphenylmethane Diisocyanate",
    "C11H8",
    "C17H16O4N2",
    "C2H1Cl3",
    "C3H2O1Cl4",
    "C3O1S1Cl4",
    "C4H2O1Cl4",
    "C5H8O2",
    "C6H4-C4H2S-C4H2S-C4H2S",
    "C6H4-C4H2S-C6H4-C4H2S",
    "C6H4-C6H4-C4H2S-C4H2S",
    "C6H4-C6H4-C6H4-C4H2S",
    "C8H4F4",
    "carboxylated acrylonitrile butadiene rubber",
    "Cellulose",
    "Cellulose acetate butyrate",
    "CH2-C4H2S-C4H2S-C4H2S",
    "CH2-C4H2S-C6H4-C4H2S",
    "CH2-C4H2S-CH2-C4H2S",
    "CH2-C4H2S-NH-C4H2S",
    "CH2-C4H2S-NH-CS",
    "CH2-C4H2S-NH-O",
    "CH2-C6H4-C4H2S-C4H2S",
    "CH2-C6H4-C4H2S-C6H4",
    "CH2-C6H4-C6H4-C4H2S",
    "CH2-C6H4-C6H4-C6H4",
    "CH2-C6H4-CH2-C4H2S",
    "CH2-C6H4-CH2-C6H4",
    "CH2-C6H4-NH-C4H2S",
    "CH2-C6H4-NH-C6H4",
    "CH2-CH2-C4H2S-C4H2S",
    "CH2-CH2-C6H4-C4H2S",
    "CH2-CH2-C6H4-C6H4",
    "CH2-CH2-CH2-C4H2S",
    "CH2-CH2-CH2-C6H4",
    "CH2-CH2-CH2-NH",
    "CH2-CH2-NH-C4H2S",
    "CH2-CH2-NH-C6H4",
    "CH2-NH-C4H2S-C6H4",
    "CH2-NH-C4H2S-NH",
    "CH2-NH-C6H4-C4H2S",
    "CH2-NH-C6H4-NH",
    "CH2-NH-CH2-C4H2S",
    "CH2-NH-CH2-C6H4",
    "CH2-NH-CH2-NH",
    "Cpoly(bisphenol F carbonate)",
    "Cyclic olefin copolymer",
    "Cycloaliphatic Epoxy Resin",
    "DGEBA Epoxy Resin",
    "DGEBA/Edamine Epoxy",
    "DGEBA/Hexamine Epoxy",
    "DGEBA/Octanediamine Epoxy",
    "DGEBD Epoxy Resin",
    "DGEBF Epoxy Resin",
    "Epoxy novolac viny ester resin",
    "ethylene propylene diene monomer rubber",
    "Fluoroelastomer",
    "Kevlar",
    "Neopentyl glycol diglycidyl ether",
    "NH-C4H2S-C4H2S-C4H2S",
    "NH-C4H2S-C4H2S-CH2",
    "NH-C4H2S-C6H4-C4H2S",
    "NH-C4H2S-NH-C4H2S",
    "NH-C6H4-C4H2S-C4H2S",
    "NH-C6H4-C4H2S-C6H4",
    "NH-C6H4-C6H4-C4H2S",
    "NH-C6H4-C6H4-C6H4",
    "NH-C6H4-C6H4-CH2",
    "NH-C6H4-NH-C4H2S",
    "NH-C6H4-NH-C6H4",
    "Nylon 11",
    "Nylon 12",
    "Nylon 6",
    "Nylon 6-10",
    "Nylon 6-12",
    "Nylon 6-6",
    "Nylon 6(3)T",
    "Nylon 8",
    "PA 6T",
    "PEG 100 - HMDI Urethane",
    "PEG 200 - HMDI Urethane",
    "PoIy(1,3-dimethylbutyl acrylate)",
    "PoIy(IH, lH-heptafluorobutyl acrylate)",
    "PoIy(isoprene) cis",
    "Poly-benzothiazole",
    "Poly-trifluoro-methyloxirane",
    "Poly-trifluoroethylene",
    "Poly(1-ethyl-1,4-butadiene)",
    "Poly(1-heptene)",
    "Poly(1-hexene)",
    "Poly(1-octadecene)",
    "Poly(1-pentene)",
    "Poly(1,2-butadiene)",
    "Poly(1,4-butadiene)",
    "Poly(1,4-cyclohexylidene dimethylene terephthalate) (Kodel) (trans)",
    "Poly(1,4-pentadiene)",
    "Poly(1,4-phenylene sulfide)",
    "Poly(10-aminodecanoic acid) (nylon 10)",
    "Poly(2-bromoethyl methacrylate)",
    "Poly(2-butoxycarbonylstyrene)",
    "Poly(2-butoxymethylstyrene)",
    "Poly(2-carboxystyrene)",
    "Poly(2-chloro-p-xylylene)",
    "Poly(2-chloroethyl methacrylate)",
    "Poly(2-chlorophenyl acrylate)",
    "Poly(2-chlorostyrene)",
    "Poly(2-cyanoethyl acrylate)",
    "Poly(2-cyanoethyl methacrylate)",
    "Poly(2-cyanoheptyl acrylate)",
    "Poly(2-cyanohexyl acrylate)",
    "Poly(2-cyanoisobutyl acrylate)",
    "Poly(2-dimethylaminocarbonylstyrene)",
    "Poly(2-ethoxycarbonylphenyl acrylate)",
    "Poly(2-ethoxycarbonylstyrene)",
    "Poly(2-ethoxyethyl methacrylate)",
    "Poly(2-ethoxymethylstyrene)",
    "Poly(2-ethylbutyl acrylate)",
    "Poly(2-ethylbutyl methacrylate)",
    "Poly(2-ethylhexyl acrylate)",
    "Poly(2-ethylhexyl methacrylate)",
    "Poly(2-ethylstyrene)",
    "Poly(2-ethylsulfinylethyl methacrylate)",
    "Poly(2-fluoro-5-methylstyrene)",
    "Poly(2-hexyloxycarbonylstyrene)",
    "Poly(2-hydroxyethyl acrylate)",
    "Poly(2-hydroxyethyl methacrylate)",
    "Poly(2-hydroxymethylstyrene)",
    "Poly(2-hydroxypropyl methacrylate)",
    "Poly(2-isobutoxycarbonylstyrene)",
    "Poly(2-isopentyloxycarbonylstyrene)",
    "Poly(2-methoxycarbonylphenyl acrylate)",
    "Poly(2-methoxyethyl acrylate)",
    "Poly(2-methoxystyrene)",
    "Poly(2-methyl-7-ethyl-4-undecyl acrylate)",
    "Poly(2-methylbutyl acrylate)",
    "Poly(2-methylpentyl acrylate)",
    "Poly(2-methylstyrene)",
    "Poly(2-naphthyl acrylate)",
    "Poly(2-octyl acrylate)",
    "Poly(2-phenylethyl methacrylate)",
    "Poly(2-tert-butylaminoethyl methacrylate)",
    "Poly(2-tert-butylphenyl acrylate)",
    "Poly(2-tert. Butyl-1,3-butadiene)(cis)",
    "Poly(2-vinyl pyridine)",
    "Poly(2,2,2-trifluoroethyl acrylate)",
    "Poly(2,2,2-trimethylhexamethylene terephthalamide)",
    "Poly(2,2,3,3,5,5,5-heptafluoro-4-oxapentyl acrylate)",
    "Poly(2,4-dichlorostyrene)",
    "Poly(2,5-dichlorostyrene)",
    "Poly(2,5-difluorostyrene)",
    "Poly(2,5-diisopropylstyrene)",
    "Poly(2,5-dimethylstyrene)",
    "Poly(2,6-dichlorostyrene)",
    "Poly(2,6-dimethyl-p-phenylene oxide)",
    "Poly(2,6-diphenyl-p-phenylene oxide)",
    "Poly(2,6,3',5'-tetrachloro bisphenol A carbonate)",
    "Poly(3-butoxypropylene oxide)",
    "Poly(3-chlorostyrene)",
    "Poly(3-dimethylaminophenyl acrylate)",
    "Poly(3-ethoxycarbonylphenyl acrylate)",
    "Poly(3-ethoxypropyl acrylate)",
    "Poly(3-ethylstyrene)",
    "Poly(3-hydroxymethylstyrene)",
    "Poly(3-methoxybutyl acrylate)",
    "Poly(3-methoxycarbonylphenyl acrylate)",
    "Poly(3-methoxypropyl acrylate)",
    "Poly(3-methoxypropylene oxide)",
    "Poly(3-methyl hexane)",
    "Poly(3-methyl-1-butene)",
    "Poly(3-methylbutyl acrylate)",
    "Poly(3-methylstyrene)",
    "Poly(3-oxabutyl methacrylate)",
    "Poly(3-pentyl acrylate)",
    "Poly(3-phenoxypropylene oxide)",
    "Poly(3-phenyl-1-propene)",
    "Poly(3-thiabutyl acrylate)",
    "Poly(3-thiapentyl acrylate)",
    "Poly(3,3-dimethyl-2-butyl methacrylate)",
    "Poly(3,3-dimethylbutyl methacrylate)",
    "Poly(3,3-dimethylbutylethylene)",
    "Poly(3,3,5-trimethylcyclohexyl acrylate)",
    "Poly(3,3,5-trimethylcyclohexyl methacrylate)",
    "Poly(3,4-dichlorostyrene)",
    "Poly(3,4-dimethylstyrene)",
    "Poly(3,5-dimethylstyrene)",
    "Poly(3,5,5-trimethylhexyl methacrylate)",
    "Poly(4-acetylstyrene)",
    "Poly(4-aminobutyric acid) (nylon 4)",
    "Poly(4-benzoylstyrene)",
    "Poly(4-biphenylyl acrylate)",
    "Poly(4-bromostyrene)",
    "Poly(4-butoxycarbonylphenyl acrylate)",
    "Poly(4-butoxycarbonylstyrene)",
    "Poly(4-butoxymethylstyrene)",
    "Poly(4-butoxystyrene)",
    "Poly(4-butylstyrene)",
    "Poly(4-butyrylstyrene)",
    "Poly(4-carboxystyrene)",
    "Poly(4-chloro-2-methylstyrene)",
    "Poly(4-chloro-3-fluorostyrene)",
    "Poly(4-chloro-3-methylstyrene)",
    "Poly(4-chlorophenyl acrylate)",
    "Poly(4-chlorostyrene)",
    "Poly(4-cyano-3-thiabutyl acrylate)",
    "Poly(4-cyanobenzyl acrylate)",
    "Poly(4-cyanomethylphenyl methacrylate)",
    "Poly(4-cyanophenyl acrylate)",
    "Poly(4-cyanophenyl methacrylate)",
    "Poly(4-decylstyrene)",
    "Poly(4-diethylcarbamoylstyrene)",
    "Poly(4-dimethylaminocarbonylstyrene)",
    "Poly(4-dodecylstyrene)",
    "Poly(4-ethoxycarbonylphenyl acrylate)",
    "Poly(4-ethoxycarbonylstyrene)",
    "Poly(4-ethoxystyrene)",
    "Poly(4-ethylstyrene)",
    "Poly(4-fluorostyrene)",
    "Poly(4-hexadecylstyrene)",
    "Poly(4-hexanoylstyrene)",
    "Poly(4-hexyloxycarbonylstyrene)",
    "Poly(4-hexyloxymethylstyrene)",
    "Poly(4-hexylstyrene)",
    "Poly(4-hydroxymethylstyrene)",
    "Poly(4-hydroxystyrene)",
    "Poly(4-iodostyrene)",
    "Poly(4-isobutoxycarbonylstyrene)",
    "Poly(4-methoxycarbonylphenyl acrylate)",
    "Poly(4-methoxycarbonylphenyl methacrylate)",
    "Poly(4-methoxyphenyl acrylate)",
    "Poly(4-methoxystyrene)",
    "Poly(4-methyl-1-pentene)",
    "Poly(4-methylstyrene)",
    "Poly(4-phenylstyrene)",
    "Poly(4-sec-butylstyrene)",
    "Poly(4-tert-butylcyclohexyl methacrylate)",
    "Poly(4-tert-butylphenyl acrylate)",
    "Poly(4-tert-butylstyrene)",
    "Poly(4-thiapentyl acrylate)",
    "Poly(4-vinyl pyridine)",
    "Poly(4,4-dimethylpentylethylene)",
    "Poly(4,4'-pentamethylenedibenzoic anhydride)",
    "Poly(4,4'-tetramethylene dioxydibenzoic anhydride)",
    "Poly(4,4'-thiodiphenylene carbonate)",
    "Poly(5-bromo-2-butoxystyrene)",
    "Poly(5-bromo-2-ethoxystyrene)",
    "Poly(5-bromo-2-isopentyloxystyrene)",
    "Poly(5-bromo-2-isopropoxystyrene)",
    "Poly(5-bromo-2-methoxystyrene)",
    "Poly(5-bromo-2-pentyloxystyrene)",
    "Poly(5-bromo-2-propoxystyrene)",
    "Poly(5-cyano-3-oxapentyl acrylate)",
    "Poly(5-cyano-3-thiapentyl acrylate)",
    "poly(5-methyl-1-hexene)",
    "Poly(5-thiahexyl acrylate)",
    "Poly(5,5,5-trifluoro-3-oxapentyl acrylate)",
    "Poly(5,5,6,6,7,7,7-heptafluoro-3-oxaheptyl acrylate)",
    "Poly(6-cyano-3-thiahexyl acrylate)",
    "Poly(6-cyano-4-thiahexyl acrylate)",
    "Poly(8-cyano-7-thiaoctyl acrylate)",
    "Poly(9-aminononanoic acid) (nylon 9)",
    "Poly(acrylamide)",
    "Poly(acrylic acid)",
    "Poly(acrylonitrile)",
    "Poly(alpha-methylstyrene)",
    "Poly(azelaic anhydride)",
    "Poly(benzoxazine)",
    "Poly(benzyl methacrylate)",
    "Poly(bisphenol A carbonate)",
    "Poly(bisphenol A isophthalate)",
    "Poly(bisphenol B carbonate)",
    "Poly(butyl acrylate)",
    "Poly(butyl chloroacrylate)",
    "Poly(butyl cyanoacrylate)",
    "Poly(butyl ethylene)",
    "Poly(butyl methacrylate)",
    "Poly(butyl vinyl ether)",
    "Poly(butyl vinyl thioether)",
    "Poly(butylene adipate-co-terephthalate)",
    "Poly(butylene isophthalate)",
    "Poly(butylene sebacate)",
    "Poly(butylene terephthalate)",
    "Poly(butylene)",
    "Poly(caprolactone)",
    "Poly(chloroprene)",
    "Poly(cyanomethyl acrylate)",
    "Poly(cyclobutyl methacrylate)",
    "Poly(cyclohexyl acrylate)",
    "Poly(cyclohexyl chloroacrylate)",
    "Poly(cyclohexyl methacrylate)",
    "Poly(cyclohexylethylene)",
    "Poly(cyclooctyl methacrylate)",
    "Poly(cyclooctylmethyl methacrylate)",
    "Poly(cyclopentyl methacrylate)",
    "Poly(decamethylene azelamide) (nylon 10,9)",
    "Poly(decamethylene sebacamide) (nylon 10,10)",
    "Poly(decamethylene sebacate)",
    "Poly(decamethylene terephthalate)",
    "Poly(diethyl fumarate)",
    "Poly(dimehyl itaconate)",
    "Poly(dimethyl fumarate)",
    "Poly(diproylfumarate)",
    "Poly(dodecyl acrylate)",
    "Poly(dodecyl methacrylate)",
    "Poly(epichlorohydrin)",
    "Poly(ethene-alt-hexafluoroacetone)",
    "Poly(ether ether ketone)",
    "Poly(ether ketone)",
    "Poly(ethyl acrylate)",
    "Poly(ethyl chloroacrylate)",
    "Poly(ethyl methacrylate)",
    "Poly(ethyl vinyl ether)",
    "Poly(ethyl vinyl thioether)",
    "Poly(ethylene adipate)",
    "Poly(ethylene azelate)",
    "Poly(ethylene carbonate)",
    "Poly(ethylene glycol)",
    "Poly(ethylene isophthalate)",
    "Poly(ethylene naphthalate)",
    "Poly(ethylene oxybenzoate) (A-tell)",
    "Poly(ethylene phthalate)",
    "Poly(ethylene sebacate)",
    "Poly(ethylene succinate)",
    "Poly(ethylene sulfide)",
    "Poly(ethylene terephthalate)",
    "Poly(ethylene-vinyl acetate)",
    "Poly(ethylene)",
    "Poly(ethylene) terephthalate",
    "Poly(ethyleneketone)",
    "Poly(fluoromethyl acrylate)",
    "Poly(furfuryl acrylate)",
    "Poly(glycidyl methacrylate)",
    "Poly(glycidyl methyl ether)",
    "Poly(heptamethylene pimelamide) (nylon 7,7)",
    "Poly(heptyl acrylate)",
    "Poly(hexadecyl acrylate)",
    "Poly(hexafluoro propylene)",
    "Poly(hexamethylene glycol)",
    "Poly(hexamethylene terephthalate)",
    "Poly(hexyl acrylate)",
    "Poly(hexyl methacrylate)",
    "Poly(hexyldecylethylene)",
    "Poly(hexyylene sebacate)",
    "Poly(isobutene)",
    "Poly(isobutyl acrylate)",
    "Poly(isobutyl chloroacrylate)",
    "Poly(isobutyl methacrylate)",
    "Poly(isobutylethylene)",
    "Poly(isohexylethylene)",
    "Poly(isopentylethylene)",
    "Poly(isopropyl acrylate)",
    "Poly(isopropyl chloroacrylate)",
    "Poly(isopropyl methacrylate)",
    "Poly(isopropyl vinyl ether)",
    "Poly(isopropylethylene)",
    "Poly(m-phenylene isophthalamide)",
    "Poly(m-phenylene terephthalamide)",
    "Poly(m-pyridine)",
    "Poly(m-tolyl acrylate)",
    "Poly(methacrylic acid)",
    "Poly(methacrylonitrile)",
    "Poly(methyl acrylate)",
    "Poly(methyl chloroacrylate)",
    "Poly(methyl cyanoacrylate)",
    "Poly(methyl fluoroacrylate)",
    "Poly(methyl isopropenyl ketone)",
    "Poly(methyl methacrylate)",
    "Poly(methyl methacrylate)-block-poly(n-butyl acrylate)",
    "Poly(methyl methacrylate)-block-poly(n-butyl acrylate)-block-poly(methyl methacrylate)",
    "Poly(methyl methacrylate)-block-poly(t-butyl acrylate)",
    "Poly(methyl vinyl ether)",
    "Poly(methyl vinyl thioether)",
    "Poly(morpholinocarbonylethylene)",
    "Poly(N-(1-methylbutyl)acrylamide)",
    "Poly(N-acetylmethacrylamide)",
    "Poly(N-benzyl methacrylamide)",
    "Poly(N-butylacrylamide)",
    "Poly(n-decyl methacrylate)",
    "Poly(N-isopropyl acrylamide)",
    "Poly(N-methyl-N-phenylacrylamide)",
    "Poly(N-octadecylacrylamide)",
    "Poly(n-octyl acrylate)",
    "Poly(N-phenylacrylamide)",
    "Poly(N-sec-butylacrylamide)",
    "Poly(N-tert-butyl acrylamide)",
    "Poly(N-tert-butylmethacrylamide)",
    "Poly(N,N-dibutylacrylamide",
    "Poly(N,N-diisopropylacrylamide)",
    "Poly(N,N-dimethylacrylamide)",
    "Poly(neopentyl acrylate)",
    "Poly(nonamethylene azelamide) (nylon 9,9)",
    "Poly(nonyl acrylate)",
    "Poly(nonylethylene)",
    "Poly(o-tolyl acrylate)",
    "Poly(octadecyl acrylate)",
    "Poly(octadecyl methacrylate)",
    "Poly(octamethylene suberamide) (nylon 8,8)",
    "Poly(octene oxide)",
    "Poly(octene)",
    "Poly(octyl cyanoacrylate)",
    "Poly(octyl methacrylate)",
    "Poly(octyl vinyl ether)",
    "Poly(oxymethylene-oxyethylene)",
    "Poly(oxymethylene-oxytetramethylene)",
    "Poly(p-phenylene benzobisoxazole)",
    "Poly(p-phenylene)",
    "Poly(p-tetramethylenedibenzoic anhydride)",
    "Poly(p-tolyl acrylate)",
    "Poly(p-xylene)",
    "Poly(p-xylylene sebacamide)",
    "Poly(para-hydroxybenzoate) (Ekonol)",
    "Poly(paraphenylene terephthalamide)",
    "Poly(pentabromobenzyl acrylate)",
    "Poly(pentachlorophenyl acrylate)",
    "Poly(pentyl methacrylate)",
    "Poly(phenyl ethyl acrylate)",
    "Poly(phenyl ethyl)acrylate",
    "Poly(phenyl methacrylate)",
    "Poly(phenylene diisothiocyanate Phenylenediamine)",
    "Poly(phenylene sulfide)",
    "Poly(piperazine sebacamide)",
    "Poly(piperidylacrylamide)",
    "Poly(propyl acrylate)",
    "Poly(propyl chloroacrylate)",
    "Poly(propyl methacrylate)",
    "Poly(propyl vinyl ether)",
    "Poly(propyl vinyl thioether)",
    "Poly(propylene carbonate)",
    "Poly(propylene glycol)",
    "Poly(propylene sebacate)",
    "Poly(propylene sulfide)",
    "Poly(propylene)",
    "Poly(propyleneketone)",
    "Poly(sec-butyl acrylate)",
    "Poly(sec-butyl chloroacrylate)",
    "Poly(sec-butyl methacrylate)",
    "Poly(styrene oxide)",
    "Poly(styrene-b-butadiene)",
    "Poly(styrene-co-acrylonitrile)",
    "Poly(tert-butyl acrylate)",
    "Poly(tert-butyl methacrylate)",
    "Poly(tert-butylethylene)",
    "poly(tetrachloroethylene)",
    "Poly(tetradecyl methacrylate)",
    "Poly(tetradecylethylene)",
    "Poly(tetrafluoroethylene-alt-ethylene)",
    "Poly(tetrafluoroethylene)",
    "Poly(tetrafluoropropyl acrylate)",
    "Poly(tetrafluoropropyl methacrylate)",
    "Poly(tetrahydrofuran)",
    "Poly(tetramethyl bisphenol A carbonate)",
    "Poly(tetramethylene adipate)",
    "Poly(trans-2-butene oxide)",
    "Poly(trimethylene glycol)",
    "poly(trimethylene sebacate)",
    "Poly(trimethylene terephthalate)",
    "Poly(vinyl acetate)",
    "Poly(vinyl alcohol)",
    "Poly(vinyl benzoate)",
    "Poly(vinyl butyral)",
    "Poly(vinyl butyrate)",
    "Poly(vinyl caproate)",
    "Poly(vinyl carbazole)",
    "Poly(vinyl chloride)",
    "Poly(vinyl chloroacetate)",
    "Poly(vinyl cyclopentane)",
    "Poly(vinyl ethyl ketone)",
    "Poly(vinyl fluoride)",
    "Poly(vinyl formate)",
    "Poly(vinyl hexyl ether)",
    "Poly(vinyl isobutyl ether)",
    "Poly(vinyl methyl ketone)",
    "Poly(vinyl phenyl ketone)",
    "Poly(vinyl phenyl sulfide)",
    "Poly(vinyl propionate)",
    "Poly(vinyl pyrrolidone)",
    "Poly(vinyl trifluoroacetate)",
    "Poly(vinyl valerate)",
    "Poly(vinylidene chloride)",
    "Poly(vinylidene fluoride-co-chlorotrifluoroethylene)",
    "Poly(vinylidene fluoride-co-hexafluoropropylene)",
    "Poly(vinylidene fluoride-co-trifluoroethylene-co-chlorotrifluoroethylene)",
    "Poly(vinylidene fluoride-co-trifluoroethylene)",
    "Poly(vinylidene fluoride)",
    "Poly[(2-nitrosoethyl) methacrylate]",
    "Poly[(cyclohexylmethyl)ethylene]",
    "Poly[(cyclopentylmethyl)ethylene]",
    "Poly[(tetramethylene terephthalate)",
    "Poly[1,1-(1-phenylethane)bis(4-phenyl)carbonate]",
    "Poly[1,1-(2-methyl propane) bis(4-phenyl)carbonate]",
    "Poly[1,1-bis(chloromethyl)trimethylene oxide]",
    "Poly[1,1-butane bis(4-phenyl)carbonate]",
    "Poly[1,1-cyclohexane bis {(4-(2,6-dichlorophenyl)}carbonate]",
    "Poly[1,1-cyclohexane bis(4-phenyl)carbonate]",
    "Poly[1,1-cyclopentane bis(4-phenyl)carbonate]",
    "Poly[1,1-ethane bis(4-phenyl)carbonate]",
    "Poly[2-(2-dimethylaminoethoxycarbonyl)styrene]",
    "Poly[2,2-pentane bis(4-phenyl)carbonate]",
    "Poly[2,2-propane bis {4-(2-methyl)phenyl}carbonate]",
    "Poly[2,2-propane bis(4-phenyl)carbonate]",
    "Poly[3-(4-biphenylyl)styrene]",
    "Poly[3-chloro-2,2-bis(chloromethyl)propyl acrylate]",
    "Poly[4-(1-hydroxy-1-methylbutyl)styrene]",
    "Poly[4-(1-hydroxy-1-methylethyl)styrene]",
    "Poly[4-(1-hydroxy-1-methylpentyl)styrene]",
    "Poly[4-(1-hydroxy-1-methylpropyl)styrene]",
    "Poly[4-(1-hydroxy-3-morpholinopropyl)styrene]",
    "Poly[4-(1-hydroxy-3-piperidinopropyl)styrene]",
    "Poly[4-(2-dimethylaminoethoxycarbonyl)styrene]",
    "Poly[4-(2-hydroxybutoxymethyl)styrene]",
    "Poly[4-(4-biphenylyl)styrene]",
    "Poly[4-(4-hydroxybutoxymethyl)styrene]",
    "Poly[4,4-heptane bis(4-phenyl)carbonate]",
    "Poly[bis(4-aminocyclohexyl)methane-1,10-decanedicarboxamide](Qiana) (trans)",
    "Poly[di(n-butyl)itaconate]",
    "Poly[di(n-hexyl) itaconate]",
    "Poly[di(n-propyl) itaconate]",
    "Poly[diphenylmethane bis(4-phenyl)carbonate]",
    "Poly[oxy(hexyloxymethyl)ethylene]",
    "Poly[styrene-b-(ethylene-ran-butylene)-b-styrene]",
    "Polyacetal",
    "Polyacetaldehyde",
    "Polyamide",
    "Polyamide-imide",
    "Polybenzimidazole",
    "Polybutadiene",
    "Polybutylene succinate",
    "Polycarbonothioyl",
    "polychloral",
    "Polychloroprene",
    "Polychlorotrifluoroethylene",
    "Polycyanoacrylates",
    "Polydecene",
    "Polyester",
    "Polyether",
    "Polyetherketoneketone",
    "Polyethylene suberate",
    "Polyglycine",
    "Polyglycolide",
    "Polyhydroxybutyrate",
    "Polyimide",
    "Polyisoprene",
    "Polyketone",
    "Polylactic acid",
    "Polynonene",
    "Polypentene",
    "Polyphenylsulfone",
    "Polyrotaxane",
    "Polystyrene",
    "Polythiophene",
    "Polythiourea",
    "Polyurea",
    "Polyurethane",
    "Pullulan",
    "Silicone rubber",
    "Sodium polyacrylate",
    "tetraglycidyl 4,4'-diaminodiphenyl methane",
    "Thermoplastic polyurethane",
    "triglycidyl para-aminophenol",
    "Unsaturated polyester resin",
    "High-density polyethylene (HDPE)",
    "Low-density polyethylene (LDPE)",
    "Polypropylene (PP)",
    "Poly(lactic acid) (PLA)",
    "Poly(butylene succinate) (PBS)",
    "Poly(ether sulfone) (PES)",
    "Poly(phenylene sulfide) (PPS)",
    "Poly(etherimide) (PEI)",
    "Poly(ether ketone ketone) (PEKK)",
    "Poly(butylene adipate-co-terephthalate) (PBAT)",
    "Polyoxymethylene (POM, Acetal)",
    "Poly(vinyl chloride) (PVC)",
    "Poly(ethylene-co-vinyl acetate) (EVA)",
    "Polycaprolactone (PCL)",
    "Polyamide-imide (PAI)",
    "Poly(ethylene oxide) (PEO)",
    "Liquid Crystal Polymer (LCP)",
    "Natural rubber (NR)",
    "Polydicyclopentadiene (pDCPD)",
    "Bio-based epoxy resin",
    "Thermoplastic Polyolefin (TPO)",
]

# MATRIX SYNONYMS & HISTORICAL TERMS
MATRIX_SYNONYMS = [
    "Host polymer",
    "Continuous phase",
    "Binder",
    "Resin",
    "Plastic substrate",
    "Acrylic",
    "Plexiglas",
    "Lucite",
    "Perspex",
    "Teflon",
    "Dacron",
    "Terylene",
    "Lexan",
    "Makrolon",
    "Pyroxylin",
    "Celluloid",
    "India rubber",
    "Hevea rubber",
    "Latex",
    "Carbamates",
]

# Merge synonyms into MATRICES, removing duplicates
MATRICES.extend(MATRIX_SYNONYMS)
MATRICES = list(set(MATRICES))

# ============== PROPERTIES ==============
PROPERTIES = [
    "Tensile Modulus",
    "Tensile Stress At Break",
    "Tensile Stress At Yield",
    "Tensile Toughness",
    "Elongation At Yield",
    "Elongation At Break",
    "Fiber Tensile Modulus",
    "Fiber Tensile Strength",
    "Fiber Tensile Elongation",
    "Poissons Ratio",
    "Flexural Loading Profile",
    "Flexural Modulus",
    "Flexural Stress At Break",
    "Flexural Stress At Yield",
    "Flexural Toughness",
    "Deflection at Break",
    "Compressive Toughness",
    "Compression Modulus",
    "Compression Stress At Break",
    "Compression Stress At Yield",
    "Strain Rate",
    "Strain Amplitude",
    "Shear Modulus",
    "Shear Stress At Break",
    "Shear Stress At Yield",
    "Essential Work Of Fracture",
    "Linear Elastic",
    "Plastic Elastic",
    "Impact Toughness",
    "IZOD Area",
    "IZOD Impact Energy",
    "Charpy Impact Energy",
    "Hardness Test Standard",
    "Hardness Scale",
    "Hardness Value",
    "Compressive Creep",
    "Compressive Creep Rupture Strength",
    "Compressive Creep Rupture Time",
    "Compressive Creep Strain",
    "Tensile Creep Recovery",
    "Tensile Creep Modulus",
    "Tensile Creep Compliance",
    "Tensile Creep Rupture Strength",
    "Tensile Creep Rupture Time",
    "Tensile Creep Strain",
    "Flexural Creep Rupture Strength",
    "Flexural Creep Rupture Time",
    "Flexural Creep Strain",
    "Dielectric Loss Permittivity",
    "Dielectric Real Permittivity",
    "Dielectric Loss Tangent",
    "DC Dielectric Constant",
    "AC Dielectric Dispersion",
    "AC Dielectric Constant",
    "Impedance",
    "Dielectric Breakdown Strength",
    "Volume Resistivity",
    "Surface Resistivity",
    "Electric Conductivity",
    "Arc Resistance",
    "Electric Percolation Threshold",
    "Energy Density",
    "Current Density",
    "Comparative Tracking Index",
    "Flammability",
    "Shielding Effectiveness",
    "Degree of Crystallinity",
    "Growth Rate Of Crystal",
    "Growth Rate Parameter Of Avrami Equation",
    "Nucleation Parameter Of Avrami Equation",
    "Halflife Of Crystallization",
    "Crystallization Temperature",
    "Heat Of Crystallization",
    "Heat Of Fusion",
    "DSC Profile",
    "Thermal Decomposition Temperature",
    "Glass Transition Temperature",
    "LC Phase Transition Temperature",
    "Melting Temperature",
    "Specific Heat Capacity Cp",
    "Specific Heat Capacity Cv",
    "Thermal Conductivity",
    "Thermal Diffusivity",
    "Brittle Temperature",
    "Weight Loss",
    "Interphase Thickness",
    "Density",
    "Linear Expansion Coefficient",
    "Volume Expansion Coefficient",
    "Surface Tension",
    "Interfacial Tension",
    "Water Absorption",
    "Spectroscopy",
    "Stress",
    "Strain",
    "Aspect Ratio",
    "Dynamic Mechanical Analysis",
    "Temperature Sweep",
    "Rheological Complex Modulus",
    "Rheometer Mode",
    "Rheological Storage Modulus",
    "Rheological Loss Modulus",
    "Rheological Loss Tangent",
    "Rheological Viscosity",
    "Dynamic Viscosity",
    "Melt Viscosity",
    "Breakdown Strength",
    "Probability of Failure",
    "Tensile Strength",
    "Stress Relaxation",
    "Strain at Break",
    "Precracking Process",
    "Fracture Energy",
    "K Factor",
    "Weibull Parameter",
    "Work of Adhesion",
    "Work of Spreading",
    "Wetting Angle",
    "Degree of Wetting",
    "Surface Energy",
    "Dispersive Surface Energy",
    "Polar Surface Energy",
    "Alternating Current",
    "Direct Current",
    "Volumetric Shrinkage",
]

PROPERTY_SYNONYMS = [
    "Breaking Stress",
    "Ultimate Tensile Strength (UTS)",
    "Fracture Stress",
    "Maximum Stress",
    "Ultimate Elongation",
    "Elongation to Failure",
    "Strain at Failure",
    "Rupture Elongation",
    "Elastic Modulus",
    "Modulus of Elasticity",
    "Initial Modulus",
    "Lateral Contraction Ratio",
    "Weibull modulus",
    "Weibull exponent",
    "Characteristic Strength",
    "Threshold Stress",
    "Shape Parameter (β)",
    "Scale Parameter (α)",
    "Relative Permittivity",
    "Specific Inductive Capacity",
    "Power Factor",
    "Loss Factor",
    "Dielectric Strength",
    "Puncture Voltage",
    "Second-order transition",
    "Vitrification Temperature",
    "Fusion Point",
    "Melting Range",
    "Pyrolysis Onset",
    "Degradation Temperature",
    "Consistency",
    "Fluidity",
    "Specific Gravity",
    "Indentation Hardness",
    "Penetration Hardness",
    "Crystallinity Index",
    "Crystallinity Fraction",
    "Rate of Stretch",
    "Work of Fracture",
    "Critical Strain Energy Release Rate",
    "LOI",
    "Fire Resistance",
    "Burn Rating",
    "Stress Decay",
    "Tensile Load at Failure",
    "Weibull shape factor",
    "Weibull scale factor",
]

PROPERTIES.extend(PROPERTY_SYNONYMS)
PROPERTIES = list(set(PROPERTIES))

# ============== UNITS ==============
UNITS = [
    "Unit",
    "Pa",
    "V/m",
    "Ohm*m",
    "S/m",
    "J/m^3",
    "C/m^2",
    "V",
    "dB",
    "K",
    "J",
    "kg/m^3",
    "m/K",
    "N",
    "J/m^2",
    "degrees",
    "°C",
    "MPa",
    "GPa",
    "N·m",
    "N·mm",
    "m/s",
    "m/s^2",
    "Pa·s",
    "cP",
    "W/m·K",
    "J/(kg·K)",
    "kV/mm",
    "kJ/m^2",
    "kJ/kg",
    "kPa",
    "bar",
    "torr",
    "m^2/s",
    "s",
    "min",
    "hr",
    "°F",
    "BTU/(hr·ft^2·°F)",
    "mol/L",
    "g",
    "mg",
    "m^3",
    "L",
    "dL",
    "mm^3",
    "μm",
    "nm",
]

UNIT_SYNONYMS = [
    "kg/cm^2",
    "kgf/cm^2",
    "ksi",
    "psi",
    "atm",
    "mmHg",
    "V/mil",
    "deg C",
    "°Ré",
    "°R",
    "Centigrade",
    "cal",
    "erg",
    "kiloJoule",
    "Ohm·cm",
    "specific resistance",
    "Brinell (HB)",
    "Rockwell (HRC)",
    "Vickers (HV)",
    "Monotron hardness",
    "Scleroscope hardness",
    "Poise",
    "Saybolt Universal Seconds (SUS)",
    "mils",
]

UNITS.extend(UNIT_SYNONYMS)
UNITS = list(set(UNITS))

# -----------------------------------------------------------------------------
# PROPERTY_SYNONYMS_MAP to define synonyms -> canonical name
# -----------------------------------------------------------------------------
PROPERTY_SYNONYMS_MAP = {
    "Tensile Modulus": [
        "Elastic Modulus",
        "Modulus of Elasticity",
        "Young’s Modulus",
        "Young's Modulus",
        "Initial Modulus",
        "Tensile modulus",
    ],
    "Tensile Stress At Break": [
        "Breaking Stress",
        "Ultimate Tensile Strength (UTS)",
        "Fracture Stress",
        "Maximum Stress",
        "Tensile load at failure",
    ],
    "Tensile Stress At Yield": [
        "Yield Strength",
        "σy",
        "Stress at Yield",
    ],
    "Melting Temperature": [
        "Fusion Point",
        "Melting Range",
    ],
    "Glass Transition Temperature": [
        "Tg",
        "Second-order transition",
        "Vitrification Temperature",
    ],
    "Thermal Conductivity": [
        "Conductivity (thermal)",
        "Heat Conductivity",
    ],
    "Weibull Parameter": [
        "Weibull exponent",
        "Weibull modulus",
        "Weibull shape factor",
        "Weibull shape parameter",
        "Shape Parameter (β)",
        "Scale Parameter (α)",
    ],
    "Dielectric Breakdown Strength": [
        "Breakdown Strength",
        "Dielectric Strength",
        "Puncture Voltage",
    ],
}

# Invert dictionary to map each synonym to a canonical property
SYNONYM_TO_CANONICAL = {}
for canonical_prop, syn_list in PROPERTY_SYNONYMS_MAP.items():
    SYNONYM_TO_CANONICAL[canonical_prop.lower()] = canonical_prop
    for syn in syn_list:
        SYNONYM_TO_CANONICAL[syn.lower()] = canonical_prop

# -----------------------------------------------------------------------------
# PROPERTY_UNIT_MAP: canonical property -> list of recommended units
# -----------------------------------------------------------------------------
PROPERTY_UNIT_MAP = {
    "Tensile Modulus": ["Pa", "MPa", "GPa"],
    "Tensile Stress At Break": ["Pa", "MPa", "GPa"],
    "Tensile Stress At Yield": ["Pa", "MPa", "GPa"],
    "Melting Temperature": ["°C", "K"],
    "Glass Transition Temperature": ["°C", "K"],
    "Thermal Conductivity": ["W/(m*K)", "mW/(m*K)"],
    "Weibull Parameter": ["dimensionless"],
    "Dielectric Breakdown Strength": ["kV/mm", "MV/m"],
}

# -----------------------------------------------------------------------------
# PROPERTY_RANGES: canonical property -> numeric range
# -----------------------------------------------------------------------------
PROPERTY_RANGES = {
    "Tensile Modulus": (1e6, 3e9),
    "Tensile Stress At Break": (1e6, 1e8),
    "Tensile Stress At Yield": (1e6, 1e8),
    "Melting Temperature": (300, 2500),
    "Glass Transition Temperature": (200, 500),
    "Thermal Conductivity": (0.1, 400),
    "Weibull Parameter": (1.0, 50.0),
    "Dielectric Breakdown Strength": (1e3, 1e6),
}

# Fallback range for any property not listed
DEFAULT_PROPERTY_RANGE = (0, 1e6)

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def get_canonical_property_name(prop_name: str) -> str:
    """
    Convert a property name or synonym to its canonical form.

    :param prop_name: The property name or synonym.
    :return: The canonical property name if recognized, else original.
    """
    return SYNONYM_TO_CANONICAL.get(prop_name.lower(), prop_name)


def get_preferred_units(canonical_prop_name: str):
    """
    Retrieve the recommended list of units for a given canonical property.

    :param canonical_prop_name: The canonical property name.
    :return: A list of valid units, or None if not found.
    """
    return PROPERTY_UNIT_MAP.get(canonical_prop_name)


def get_property_range(canonical_prop_name: str):
    """
    Retrieve the numeric value range for a given canonical property.

    :param canonical_prop_name: The canonical property name.
    :return: A (min_val, max_val) tuple or DEFAULT_PROPERTY_RANGE if not found.
    """
    return PROPERTY_RANGES.get(canonical_prop_name, DEFAULT_PROPERTY_RANGE)


# -----------------------------------------------------------------------------
# Example code (run python data_lists.py)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    test_props = ["Tensile Modulus", "Breaking Stress", "Tg", "Puncture Voltage", "Weibull exponent"]
    for p in test_props:
        canonical = get_canonical_property_name(p)
        units = get_preferred_units(canonical)
        val_range = get_property_range(canonical)

        print(f"Original: {p}")
        print(f" -> Canonical: {canonical}")
        print(f" -> Preferred Units: {units if units else 'Any from UNITS list'}")
        print(f" -> Typical Range: {val_range}")
        print("-" * 50)

