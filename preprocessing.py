import numpy as np
import pandas as pd
import extensions
import argparse
from Bio import SeqIO
from BCBio import GFF


def read_data(dir_fasta, dir_gff, source):
    with open(dir_fasta) as file:
        seq_dict = SeqIO.to_dict(SeqIO.parse(file, "fasta"))

    record_id = list(seq_dict.keys())[0]
    limit_info = dict(gff_id=[record_id], gff_source=source)
    
    with open(dir_gff) as file:
        gen_data = next(GFF.parse(file, limit_info=limit_info, base_dict=seq_dict))

    return gen_data
         
        
def second_model_target_mask(gen_data, target:str):

    def dfs(feature, target_mask):
        for feature in feature.sub_features:
            if feature.type == target:
                start_index = int(feature.location.start)
                end_index = int(feature.location.end)
                if not np.all(target_mask[start_index:end_index]):
                    target_mask[start_index:end_index] = 1
            dfs(feature, target_mask)

    target_mask = np.zeros(len(gen_data.seq))
    target = "CDS"

    for feature in gen_data.features:
        if feature.type == target:
            start_index = int(feature.location.start)
            end_index = int(feature.location.end)
            if not np.all(target_mask[start_index:end_index]):
                target_mask[start_index:end_index] = 1
        dfs(feature, target_mask)
    return target_mask








def make_masked_sequence(sequence: list):
    masked_sequence = np.zeros((len(sequence), 5)) # columns are A, C, G, T, N
    letter_index_dict = {
        'A': 0,
        'C': 1,
        'G': 2,
        'T': 3,
        'N': 4
    }
    for i, letter in enumerate(sequence):
         if letter not in letter_index_dict:    # if letter is unclear, make in N as NaN
              letter = 'N'
         column = letter_index_dict[letter]
         masked_sequence[i][column] = 1
    return masked_sequence



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                        prog="data_preprocessing",
                        description="Script that processing data for the model input",
                        epilog="help")
        
    parser.add_argument("fasta_dir", type=extensions.fasta_type, help='Input dir for fasta file')
    parser.add_argument("gff_dir", type=extensions.gff_type, help='Input dir for gff file')
    parser.add_argument("source", type=extensions.gff_type, help='Source for the limit')

    args = vars(parser.parse_args())


    

    