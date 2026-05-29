import os
import pandas as pd
from colorama import Fore, Style

amino_acids = 'ACDEFGHIKLMNPQRSTVWY'

def is_standard_sequence(seq):
    """Check if a sequence contains only standard amino acids."""
    for aa in seq:
        if aa not in amino_acids:
            return False
    return True

def extractSequencesFromFile(file_path, num_seqs=10):
    """Extracts sequences from a given file and returns them as a list."""
    if not os.path.isfile(file_path):
        raise IOError('File not found/readable: {}'.format(file_path))
    
    df = pd.read_csv(file_path)
    print(Fore.GREEN + f"Loaded {len(df)} sequences" + Style.RESET_ALL)
    # Only keep sequences where the structure is helix
    df = df[df['Structure_3D'] == 'Helix']
    print(Fore.CYAN + f"{len(df)} sequences remaining after filtering for helix structure" + Style.RESET_ALL)

    # Exact num_seqs sequences randomly from the 'Sequence' column but remove sequences with > 17 a.a. first
    df = df[df['Sequence'].str.len() <= 17]
    print(Fore.CYAN + f"{len(df)} sequences remaining after filtering for length <= 17" + Style.RESET_ALL)
    # Also remove sequences with non-standard amino acids
    df = df[df['Sequence'].apply(is_standard_sequence)]
    print(Fore.GREEN + f"{len(df)} sequences remaining after filtering for standard amino acids" + Style.RESET_ALL)
    num_seqs = min(num_seqs, len(df))
    sequences = df['Sequence'].dropna().sample(n=num_seqs, random_state=42).tolist()
    return sequences

if __name__ == '__main__':
    file_path = os.path.join(os.path.dirname(__file__), 'amp_data_1-6338_sorted.csv')
    sequences = extractSequencesFromFile(file_path, num_seqs=1000)

    print(Fore.BLUE + "Extracted Sequences" + Style.RESET_ALL)
    # Save to a CSV file
    output_file = 'data/APD6/amp_helices.csv'
    df_out = pd.DataFrame(sequences, columns=['Sequence'])
    df_out.to_csv(output_file, index=False)
    print(Fore.GREEN + f"Saved extracted sequences to {output_file}" + Style.RESET_ALL)
    