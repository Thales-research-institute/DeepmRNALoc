import numpy as np
import pandas as pd


# basesmap = {"A":0, "C":1, "G":2, "T":2}

def get_tris(k):
    nucle_com = []
    chars = ['A', 'C', 'G', 'T']
    base = len(chars)
    end = len(chars)**k
    for i in range(0, end):
        n = i
        add = ''
        for j in range(k):
            ch = chars[n % base]
            n = int(n/base)
            add += ch
        nucle_com.append(add)
    return nucle_com


def get_kmer(path,k):
    fasta = open(path)
    fasta = fasta.read()
    sequence = "".join(fasta.split("\n")[1:])
    sequence = sequence.replace("N", "")
    print(len(sequence))
    kmerbases = get_tris(k)

    kmermap = {}
    for kmer in  kmerbases:
        kmermap[kmer] = 0

    # print(kmermap)
    for index in range(len(sequence)-k+1):
        kmermap[sequence[index:index+k]] += 1

    # print(kmermap)
    # print(len(kmermap))
    result = []
    for kmer in kmermap:
        result.append(kmermap[kmer])
    return result

# path = '/home/zshen/workplace/mRNA/Data/fasta/Cytoplasm_test/mRNALoc_12.fasta'
# k = 3
# print(get_kmer(path,3))