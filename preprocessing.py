import numpy as np
import pandas as pd
import configparser
from pathlib import Path
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
                end_index = int(feature.location.end)   # End is not included
                if not np.all(target_mask[start_index:end_index]):
                    target_mask[start_index:end_index] = 1
            dfs(feature, target_mask)
    
    target_mask = np.zeros(len(gen_data.seq), dtype=bool)
    for feature in gen_data.features:
        if feature.type == target:
            start_index = int(feature.location.start)
            end_index = int(feature.location.end)
            if not np.all(target_mask[start_index:end_index]):
                target_mask[start_index:end_index] = 1
        dfs(feature, target_mask)
    return target_mask


def make_masked_sequence(sequence):
    sequence = np.array(sequence)
    letters = ['A', 'C', 'G', 'T']
    masked_sequence = np.array([sequence == letter for letter in letters], dtype=bool)
    N_column = np.array(np.isin(sequence, letters, invert=True))
    N_column = np.reshape(N_column, (1, len(N_column)))
    masked_sequence = np.concatenate((masked_sequence, N_column), axis=0)
    masked_sequence = masked_sequence.T

    return masked_sequence


def target_split(target_mask, slice_len=1000, target_ratio=None):
    number_of_slices = len(target_mask) // slice_len
    target_slices = []
    non_target_slices = []

    for i in range(0, len(target_mask) - slice_len, slice_len): 
        # Важно помнить, что тут мы отбурбаем последний слайс, длина, которого меньше 10000
        # когда пишем range(0, len(target_mask) - slice_len, ...)
        left = i
        right = i + slice_len
        if not np.any(target_mask[left: right]):
            non_target_slices.append((left, right))
        else:
            target_slices.append((left, right))

    np.random.shuffle(target_slices)
    np.random.shuffle(non_target_slices)
    if target_ratio == None:
        # Если на задано соотношение, то выводим как есть
        return target_slices, non_target_slices
    else:
        # Тут данные делятся по заданному target_ratio, функция написана так, чтобы она делила данные
        # с сохранением максимально возможного количества данных. 
        # Находим реальное соотношение классов, если заданное соотношение больше чем реальное,
        # (т.е. нам нужно больше слайсов с таргетом, чем есть на самом деле) то мы берем весь массив слайсов с
        # таргетом и уменьшаем массив нон таргетов до нужного соотношения. Если же заданное соотношение меньше/равно 
        # реальному то мы берем весь массив слайсов с нон таргетом и уменьшаем массив таргетов (кароче наобарот)
        real_ratio = len(target_slices) / number_of_slices
        if target_ratio > real_ratio:
            len_of_non_target = int(len(target_slices) * ((1 - target_ratio) / target_ratio))
            non_target_slices = non_target_slices[:len_of_non_target]
        else:
            len_of_target = int(len(non_target_slices) * (target_ratio / (1 - target_ratio)))
            target_slices = target_slices[:len_of_target]

    return target_slices, non_target_slices
    

def make_dataset(masked_sequence, target_mask, target_slices, non_tarter_slices):
    df = pd.DataFrame(columns=["Masked sequence", "Y1", "Y2", "Count target", "Start", "End"])

    for left, right in target_slices:
        data_slice = target_mask[left:right].copy()
        data_slice = np.insert(data_slice, 0, 0) 
        data_slice = np.insert(data_slice, len(data_slice), 0)
        difference = np.diff(data_slice.astype('int'))
        start = np.where(difference == 1)[0]
        end = np.where(difference == -1)[0]
        count_target = len(start)
        df.loc[len(df)] = [masked_sequence[left:right], True, target_mask[left:right], count_target, start, end]

    for left, right in non_tarter_slices:
        df.loc[len(df)] = [masked_sequence[left:right], False, target_mask[left:right], 0, None, None]
    # Может добавить шафл?
    return df


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read("parser_config.ini")

    dir_fasta = Path(config["Paths"]["fasta_file_path"])
    dir_gff = Path(config["Paths"]["gff_file_path"])
    dir_save = Path(config["Paths"]["save_path"])

    source = config.get('Data' , 'source_name').split()
    target_name = config["Data"]["target_name"]
    target_ratio = float(config["Data"]["target_ratio"])
    slice_len = int(config["Data"]["slice_len"])

    gen_data = read_data(dir_fasta, dir_gff, source)
    target_mask = second_model_target_mask(gen_data, target_name)

    target_slices, non_target_slices = target_split(target_mask, slice_len=slice_len, target_ratio=target_ratio)
    masked_sequence = make_masked_sequence(gen_data.seq)
    df = make_dataset(masked_sequence, target_mask, target_slices, non_target_slices)

    df.to_csv(dir_save, index=False)

