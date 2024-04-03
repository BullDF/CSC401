import os
import re
import string
from typing import List, Literal
import numpy as np

dataDir = '/u/cs401/A3/data/'


# Class definition for an individual table cell. DO NOT MODIFY
class TableCell(object):
    EditTypes = Literal['del', 'ins', 'match', 'sub', 'first']

    def __init__(self, cost: int = 0, trace: EditTypes = 'first'):
        """
        initialize: defines an individual table cell

        Parameters
        ----------
        cost : minimal number of insertions, deletions, and substitutions to get to this cell
        trace : The last edit that was made to enter this cell.

        Returns
        -------
        N/A

        Examples
        --------
        Creating a cell
        >>> cell = TableCell(1, "ins")

        Checking the cost of a cell
        >>> if cell.cost == 1:
        ...     ...

        Modifying the edit type
        >>> cell.trace = "match"
        """
        super().__init__()
        self.cost, self.trace = cost, trace


# Preprocess text for best result. DO NOT MODIFY!
def preprocess(raw_line):
    # remove label
    line = raw_line.split(" ", 2)[-1]

    # remove tags (official)
    line = re.sub(r'<[A-Z]+>', '', line)

    # remove tags (Kaldi)
    line = re.sub(r'\[[a-z]+\]', '', line)

    # remove punctuations
    line = line.translate(str.maketrans("", "", string.punctuation))

    return line


# Complete the following component for Levenshtein
def initialize(U: int, T: int) -> List[List[TableCell]]:
    """
    initialize: allocate the table and initialize the first row and column

    Parameters
    ----------
    U : int, Number of elements in the reference sequence
    T : int, Number of elements in the hypothesis sequence

    Returns
    -------
    table: list of list of TableCell of size [U + 1, T + 1], the table used for dynamic programming.
        All elements of the table should be `TableCell` instances,
        but only the values of the first row and column need to be computed.
    """

    ############################################
    table = []
    for u in range(U + 1):
        row = []
        for t in range(T + 1):
            if u == 0:
                cell = TableCell(t)
            elif t == 0:
                cell = TableCell(u)
            else:
                cell = TableCell()
            row.append(cell)
        table.append(row)
    ############################################

    # Make sure the type of values are correct
    assert isinstance(table, list) and all(
        isinstance(row, list) and all(isinstance(cell, TableCell) for cell in row) for row in table)
    return table


def step(u: int, t: int, table: List[List[TableCell]], r: List[str], h: List[str]) -> None:
    """
    step: computes the value of the current cell

    **NOTE** :  in case of tie, use the following priority: "match" > "sub" > "ins" > "del"

    Parameters
    ----------
    u : int, row index of the current cell
    t : int, col index of the current cell
    table : list of list of TableCell of size [U + 1][T + 1]
    r : list of strings, representing the reference sentence
    h : list of strings, representing the hypothesis sentence

    Returns
    -------
    N/A
    """
    ############################################
    match = table[u - 1][t - 1].cost
    substitution = table[u - 1][t - 1].cost + 1
    insertion = table[u][t - 1].cost + 1
    deletion = table[u - 1][t].cost + 1

    if r[u - 1] == h[t - 1]:
        table[u][t].cost = min(match, insertion, deletion)
    else:
        table[u][t].cost = min(substitution, insertion, deletion)

    if table[u][t].cost == match:
        table[u][t].trace = 'match'
    elif table[u][t].cost == substitution:
        table[u][t].trace = 'sub'
    elif table[u][t].cost == insertion:
        table[u][t].trace = 'ins'
    else:
        table[u][t].trace = 'del'
    ############################################
    return


def finalize(table: List[List[TableCell]]):
    """
    finalize: computes the final results, including WER, number of all operations

    NOTE: If the reference sequence is of length 0, WER should be `float("inf")`

    Parameters
    ----------
    table : list of list of TableCell of size [U + 1][T + 1]

    Returns
    -------
    (WER, nI, nD, nS): (float, int, int, int) WER, number of insertions, deletions, and substitutions respectively
    """
    # Define results to be returned:

    ############################################
    U, T = len(table) - 1, len(table[0]) - 1
    wer = float('inf') if U == 0 else table[U][T].cost / U

    backward = {'match': [0, (1, 1)], 'sub': [0, (1, 1)], 'ins': [0, (0, 1)], 'del': [0, (1, 0)]}

    def trace(u: int, t: int) -> None:
        if table[u][t].trace == 'first':
            backward['ins'][0] += t
            backward['del'][0] += u
            return

        backward[table[u][t].trace][0] += 1
        du, dt = backward[table[u][t].trace][1]
        trace(u - du, t - dt)

    trace(U, T)
    insertions = backward['ins'][0]
    deletions = backward['del'][0]
    substitutions = backward['sub'][0]
    ############################################

    return (wer, insertions, deletions, substitutions)


def Levenshtein(r, h):
    """
    Calculation of WER with Levenshtein distance.

    You should complete the core component of this function.
    DO NOT MODIFY ANYTHING IN HERE

    Parameters
    ----------
    r : list of strings, representing the reference sentence
    h : list of strings, representing the hypothesis sentence

    Returns
    -------
    (WER, nI, nD, nS): (float, int, int, int) WER, number of insertions, deletions, and substitutions respectively

    Examples
    --------
    >>> Levenshtein("who is there".split(), "is there".split())
    (0.3333333333333333, 0, 1, 0)
    >>> Levenshtein("who is there".split(), "".split())
    (1.0, 0, 3, 0)
    >>> Levenshtein("".split(), "who is there".split())
    (inf, 3, 0, 0)
    """

    # U: length of reference;
    # T: length of hypothesis
    U, T = len(r), len(h)

    ############################################
    ############### Levenshtein: ###############
    ############################################

    # Call initialize() to create table
    table = initialize(U, T)

    # Iterate over the remaining cols and rows
    # Use values of words and the previously computed cells to compute current cell
    for u in range(1, U + 1, 1):
        for t in range(1, T + 1, 1):
            step(u, t, table, r, h)

    # Use table to compute final results
    # A.K.A WER, insertions, deletions and substitutions
    return finalize(table)


if __name__ == "__main__":
    """
    Main Function: Generates a file that has all the result. DO NOT MODIFY! 

    Output Format: ([] contains argument)
    --------------
    [Speaker Name] [System Name] [Line Number i] [WER] I:[# Insertions], D:[# Deletions], S:[# Substitutions]
    [Speaker Name] [System Name] [Line Number i] [WER] I:[# Insertions], D:[# Deletions], S:[# Substitutions]
    ...
    [Speaker Name] [System Name] [Line Number i] [WER] I:[# Insertions], D:[# Deletions], S:[# Substitutions]
    FINAL KALDI AVG = [Some Number] +- [Some Number]
    FINAL GOOGLE WER = [Some Number] +- [Some Number]

    """

    GTF = "transcripts.txt"
    GOOGLE = "transcripts.Google.txt"
    KALDI = "transcripts.Kaldi.txt"


    # Make sure the data is clean and usable
    def check_valid(sdir):
        def exists(name):
            path = os.path.join(sdir, name)
            return os.path.exists(path)

        return exists(GTF) or exists(GOOGLE) or exists(KALDI)


    # Load speakers names
    speakers = os.listdir(dataDir)

    wer_google = []
    wer_kaldi = []

    # Actual Process
    print("a3_levenshtein process is running...")
    with open('a3_levenshtein.out', 'w') as file:
        for s in speakers:
            # Form the full path
            speaker_dir = os.path.join(dataDir, s)
            if not check_valid(speaker_dir):
                continue

            # Read all three transcripts
            google = open(os.path.join(speaker_dir, GOOGLE)).readlines()
            kaldi = open(os.path.join(speaker_dir, KALDI)).readlines()
            groundtruth = open(os.path.join(speaker_dir, GTF)).readlines()

            print(f"Processing Speaker {s}...")

            for i, (g, k, r) in enumerate(zip(google, kaldi, groundtruth)):
                # Preprocess the lines
                google_sample = preprocess(g)
                kaldi_sample = preprocess(k)
                gt_sample = preprocess(r)

                # Caluculate WER for Google and Kaldi
                google_lev = Levenshtein(gt_sample.split(), google_sample.split())
                kaldi_lev = Levenshtein(gt_sample.split(), kaldi_sample.split())

                # Append the result for final output
                wer_google.append(google_lev[0])
                wer_kaldi.append(kaldi_lev[0])

                print(f'{s} Google {i} {google_lev[0]:.6f} I:{google_lev[1]} D:{google_lev[2]} S:{google_lev[3]}',
                      file=file)
                print(f'{s} Kaldi {i} {kaldi_lev[0]:.6f} I:{kaldi_lev[1]} D:{kaldi_lev[2]} S:{kaldi_lev[3]}', file=file)

        wer_google = np.array(wer_google)
        wer_kaldi = np.array(wer_kaldi)

        # Print out the final result
        print(f"FINAL KALDI AVG = {np.mean(wer_kaldi) :.4f} +- {np.std(wer_kaldi):.4f}", file=file)
        print(f"FINAL GOOGLE WER = {np.mean(wer_google):.4f} +- {np.std(wer_google):.4f}", file=file)
    print("a3_levenshtein process has completed!")
