import math
from numpy import mean

scales = {'Fauchere-Pliska': {'A':  0.31, 'R': -1.01, 'N': -0.60,
                              'D': -0.77, 'C':  1.54, 'Q': -0.22,
                              'E': -0.64, 'G':  0.00, 'H':  0.13,
                              'I':  1.80, 'L':  1.70, 'K': -0.99,
                              'M':  1.23, 'F':  1.79, 'P':  0.72,
                              'S': -0.04, 'T':  0.26, 'W':  2.25,
                              'Y':  0.96, 'V':  1.22}}

def calculate_hydrophob_frac_features(sequence, window=4):
    """ Here max_hydrophobic_window_frac, N_hydrophobicity, C_hydrophobicity are calculated"""
    # Max hydrophobic run fraction
    max_hydrophobic_run = 0
    for i in range(len(sequence) - window + 1):
        window_seq = sequence[i:i+window]
        window_hydrophobicity = mean([scales['Fauchere-Pliska'].get(aa, 0) for aa in window_seq])
        max_hydrophobic_run = max(max_hydrophobic_run, window_hydrophobicity)
    
    # N terminal hydrophobicity
    N_hydrophobicity = mean([scales['Fauchere-Pliska'].get(aa, 0) for aa in sequence[:window]])
    # C terminal hydrophobicity
    C_hydrophobicity = mean([scales['Fauchere-Pliska'].get(aa, 0) for aa in sequence[-window:]])

    return {
        'Max_Hydrophobic_Frac': max_hydrophobic_run,
        'N_Hydrophobicity': N_hydrophobicity,
        'C_Hydrophobicity': C_hydrophobicity
    }
        

def assign_hydrophobicity(sequence):  # noqa: E302
    """Assigns a hydrophobicity value to each amino acid in the sequence"""
    hscale = scales.get('Fauchere-Pliska', None)

    hvalues = []
    for aa in sequence:
        sc_hydrophobicity = hscale.get(aa, None)
        if sc_hydrophobicity is None:
            raise KeyError('Amino acid not defined in scale: {}'.format(aa))
        hvalues.append(sc_hydrophobicity)

    return hvalues

def calculate_moment(array, angle=100):
    """Calculates the hydrophobic dipole moment from an array of hydrophobicity
    values. Formula defined by Eisenberg, 1982 (Nature). Returns the average
    moment (normalized by sequence length)

    uH = sqrt(sum(Hi cos(i*d))**2 + sum(Hi sin(i*d))**2),
    where i is the amino acid index and d (delta) is an angular value in
    degrees (100 for alpha-helix).
    """

    sum_cos, sum_sin = 0.0, 0.0
    for i, hv in enumerate(array):
        rad_inc = ((i*angle)*math.pi)/180.0
        sum_cos += hv * math.cos(rad_inc)
        sum_sin += hv * math.sin(rad_inc)
    return math.sqrt(sum_cos**2 + sum_sin**2) / len(array)

def calculate_discrimination(mean_uH, total_charge):
    """Returns a discrimination factor according to Rob Keller (IJMS, 2011)
    A sequence with d>0.68 can be considered a potential lipid-binding region.
    """
    d = 0.944*mean_uH + 0.33*total_charge
    return d

def calculate_hydrophobicity_features(sequence, charge, window:int = 18):
    """ Calculates hydrophobicity features for a given peptide sequence """

    seq_h = assign_hydrophobicity(sequence)
    avg_h = sum(seq_h)/len(seq_h)
    avg_uH = calculate_moment(seq_h)

    d = calculate_discrimination(avg_uH, charge)

    output = {
        'Mean_Hydrophobicity': avg_h,
        'Hydrophobic_Moment': avg_uH,
        'Discrimination_Factor': d
    }

    frac_features = calculate_hydrophob_frac_features(sequence, window=4)

    output.update(frac_features)

    return output
