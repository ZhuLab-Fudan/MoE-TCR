import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import tree

import numpy as np
import pandas as pd
from esm.data import Alphabet


blosum50_20aa = {
        'A': np.array((5,-2,-1,-2,-1,-1,-1,0,-2,-1,-2,-1,-1,-3,-1,1,0,-3,-2,0)),
        'R': np.array((-2,7,-1,-2,-4,1,0,-3,0,-4,-3,3,-2,-3,-3,-1,-1,-3,-1,-3)),
        'N': np.array((-1,-1,7,2,-2,0,0,0,1,-3,-4,0,-2,-4,-2,1,0,-4,-2,-3)),
        'D': np.array((-2,-2,2,8,-4,0,2,-1,-1,-4,-4,-1,-4,-5,-1,0,-1,-5,-3,-4)),
        'C': np.array((-1,-4,-2,-4,13,-3,-3,-3,-3,-2,-2,-3,-2,-2,-4,-1,-1,-5,-3,-1)),
        'Q': np.array((-1,1,0,0,-3,7,2,-2,1,-3,-2,2,0,-4,-1,0,-1,-1,-1,-3)),
        'E': np.array((-1,0,0,2,-3,2,6,-3,0,-4,-3,1,-2,-3,-1,-1,-1,-3,-2,-3)),
        'G': np.array((0,-3,0,-1,-3,-2,-3,8,-2,-4,-4,-2,-3,-4,-2,0,-2,-3,-3,-4)),
        'H': np.array((-2,0,1,-1,-3,1,0,-2,10,-4,-3,0,-1,-1,-2,-1,-2,-3,2,-4)),
        'I': np.array((-1,-4,-3,-4,-2,-3,-4,-4,-4,5,2,-3,2,0,-3,-3,-1,-3,-1,4)),
        'L': np.array((-2,-3,-4,-4,-2,-2,-3,-4,-3,2,5,-3,3,1,-4,-3,-1,-2,-1,1)),
        'K': np.array((-1,3,0,-1,-3,2,1,-2,0,-3,-3,6,-2,-4,-1,0,-1,-3,-2,-3)),
        'M': np.array((-1,-2,-2,-4,-2,0,-2,-3,-1,2,3,-2,7,0,-3,-2,-1,-1,0,1)),
        'F': np.array((-3,-3,-4,-5,-2,-4,-3,-4,-1,0,1,-4,0,8,-4,-3,-2,1,4,-1)),
        'P': np.array((-1,-3,-2,-1,-4,-1,-1,-2,-2,-3,-4,-1,-3,-4,10,-1,-1,-4,-3,-3)),
        'S': np.array((1,-1,1,0,-1,0,-1,0,-1,-3,-3,0,-2,-3,-1,5,2,-4,-2,-2)),
        'T': np.array((0,-1,0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1,2,5,-3,-2,0)),
        'W': np.array((-3,-3,-4,-5,-5,-1,-3,-3,-3,-3,-2,-3,-1,1,-4,-4,-3,15,2,-3)),
        'Y': np.array((-2,-1,-2,-3,-3,-1,-2,-3,2,-1,-1,-2,0,4,-3,-2,-2,2,8,-1)),
        'V': np.array((0,-3,-3,-4,-1,-3,-3,-4,-4,4,1,-3,1,-1,-3,-2,0,-3,-1,5))
    }

class AllTCRpepDataset(Dataset):
    """
    Dataset for CDRA-1,2,3 & CDRB-1,2,3 & epitope from CSV format data.
    Applies padding to sequences and merges A and B sequences with eos token as separator.
    """
    def __init__(self, csv_file, config, cv_id=None, return_fold:list=None, return_epi:str=None):
        """
        Args:
            csv_file (str): Path to the csv file with annotations.
            config (dict): Configuration dictionary containing 'peptide', 'a1' to 'a3', 'b1' to 'b3' lengths and the model architecture.
        """
        if isinstance(csv_file, pd.DataFrame):
            data = csv_file
        elif isinstance(csv_file, str):
            data = pd.read_csv(csv_file)
        if cv_id is not None:
            data['partition'] = cv_id
        self.return_fold = return_fold
        data = self.get_fold(data, return_fold)
        
        self.return_epi = return_epi
        data = self.get_epi(data, return_epi)
        self.data = data
        
        # Map weights
        if config.sample_weight_dict == 'equal':
            weight_dict = lambda x: 1.0
        elif config.sample_weight_dict == 'weight':
            weight_dict = np.log2(self.data.shape[0] / self.data['peptide'].value_counts()) / np.log2(len(self.data['peptide'].unique()))
            weight_dict = weight_dict * (self.data.shape[0] / np.sum(weight_dict * self.data['peptide'].value_counts()))            
        else:
            raise ValueError(f"sample_weight_dict value must in [weight, equal]")
        self.data['sample_weight'] = self.data['peptide'].map(weight_dict).fillna(1.0)  # Fill missing weights with 1.0
        
        # blosum
        self.blosum = blosum50_20aa
        
        self.config = config
        self.alphabet = Alphabet.from_architecture(config['esm_alphabet'])
    
    def get_fold(self, data, return_fold):
        if 'partition' not in data.columns:
            return data
        elif return_fold is None:
            return data
        return data[data['partition'].isin(return_fold)]
    
    def get_epi(self, data, return_epi):
        if return_epi is None:
            return data
        return data[data['peptide'] == return_epi]

    def pad_sequence(self, sequence, max_len):
        """
        Pad the sequence to the specified max length.
        """
        padding_length = max_len - len(sequence)
        seq_list = sequence.tolist()[:max_len]
        
        if padding_length > 0:
            seq_list += [0] * padding_length
        return np.asarray(seq_list, dtype=np.int64)

    def enc_single_seq_bl_max_len(self, seq, max_seq_len, pad_value=-5):
            """
            Blosum encoding of a single amino acid sequence with padding to max length.
            """
            n_features = len(self.blosum[seq[0]])  # Get the length of BLOSUM vectors
            e_seq = np.full((max_seq_len, n_features), pad_value)  # Initialize with padding
            
            for j, aa in enumerate(seq):
                if j >= max_seq_len:
                    break  # Stop if the sequence exceeds max length
                try:
                    e_seq[j] = self.blosum[aa]  # Assign BLOSUM encoding for known amino acids
                except KeyError:
                    raise KeyError(f"Unknown amino acid in peptides: {aa}\n")
                        
            return e_seq / 5  # Apply normalization factor and return encoded sequence
    

    def __getitem__(self, index):
        """
        Generates one sample of data.
        """
        row = self.data.iloc[index]
        padded_sequences = {}

        # Pad sequences and prepare merged sequences
        for key in ['peptide', 'A1', 'A2', 'A3', 'B1', 'B2', 'B3']:
            max_len = self.config[key + '_len']
            aa_sequence = row[key]
            sequence = np.asarray(self.alphabet.encode(aa_sequence))
            padded_sequence = self.pad_sequence(sequence, max_len)
            padded_sequences[key] = np.asarray(padded_sequence)
            
            encoded_sequence = self.enc_single_seq_bl_max_len(aa_sequence, max_len)
            padded_sequences[key+'_blosum50'] = np.asarray(encoded_sequence, dtype=np.float32)

        padded_sequences['target'] = np.asarray(row['binder'], dtype=np.float32)
        padded_sequences['sample_weight'] = np.asarray(row['sample_weight'], dtype=np.float32)
        
        padded_sequences = tree.map_structure(
            lambda x: x if torch.is_tensor(x) else torch.tensor(x), padded_sequences)
        

        return padded_sequences

    def __len__(self):
        return len(self.data)
    
