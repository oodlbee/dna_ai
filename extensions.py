from pathlib import Path

def fasta_type(directory:str):
    directory = Path(directory)
    if directory.suffix != ".fasta":
        raise NameError("File extension is not fasta")
    return str(directory)

def gff_type(directory:str):
    if directory.suffix != ".gff":
        raise NameError("File extension is not gff")
    return str(directory)