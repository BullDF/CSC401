#  * This code is provided solely for the personal and private use of students
#  * taking the CSC401 course at the University of Toronto. Copying for purposes
#  * other than this use is expressly prohibited. All forms of distribution of
#  * this code, including but not limited to public repositories on GitHub,
#  * GitLab, Bitbucket, or any other online platform, whether as given or with
#  * any changes, are expressly prohibited.
#  
#  * All of the files in this directory and all subdirectories are:
#  * Copyright (c) 2024 Frank Rudzicz, Gerald Penn

import numpy as np
import argparse
import json

import csv

# Provided wordlists.
FIRST_PERSON_PRONOUNS = {
    'i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours'}
SECOND_PERSON_PRONOUNS = {
    'you', 'your', 'yours', 'u', 'ur', 'urs'}
THIRD_PERSON_PRONOUNS = {
    'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them',
    'their', 'theirs'}
SLANG = {
    'smh', 'fwb', 'lmfao', 'lmao', 'lms', 'tbh', 'rofl', 'wtf', 'bff',
    'wyd', 'lylc', 'brb', 'atm', 'imao', 'sml', 'btw', 'bw', 'imho', 'fyi',
    'ppl', 'sob', 'ttyl', 'imo', 'ltr', 'thx', 'kk', 'omg', 'omfg', 'ttys',
    'afn', 'bbs', 'cya', 'ez', 'f2f', 'gtr', 'ic', 'jk', 'k', 'ly', 'ya',
    'nm', 'np', 'plz', 'ru', 'so', 'tc', 'tmi', 'ym', 'ur', 'u', 'sol', 'fml'}

punctuations = {'/$', '/.', '/,', '/:', '/-LRB-', '/-RRB-', '/``', "/''", '/`', "/'", '/HYPH'}

bristol_norms = {}
warringer_norms = {}

liwc_index = {}
liwc_score = {}


def load_norms(args):
    global bristol_norms
    global warringer_norms
    with open(f'{args.a1_dir}../Wordlists/BristolNorms+GilhoolyLogie.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['AoA (100-700)'] and row['IMG'] and row['FAM']:
                bristol_norms[row['WORD']] = (
                    float(row['AoA (100-700)']),
                    float(row['IMG']),
                    float(row['FAM'])
                )

    with open(f'{args.a1_dir}/../Wordlists/Ratings_Warriner_et_al.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            warringer_norms[row['Word']] = (
                float(row['V.Mean.Sum']),
                float(row['A.Mean.Sum']),
                float(row['D.Mean.Sum'])
            )


def load_liwc(args):
    global liwc_index
    global liwc_score

    for cat in ['Alt', 'Center', 'Left', 'Right']:
        i = 0
        d = {}
        with open(f'{args.a1_dir}feats/{cat}_IDs.txt', 'r') as f:
            for line in f:
                d[line.strip()] = i

        liwc_index[cat] = d
        liwc_score[cat] = np.load(f'{args.a1_dir}feats/{cat}_feats.dat.npy')


def extract1(comment: str) -> np.ndarray:
    """ 
    This function extracts features from a single comment.

    Parameters:
    - comment: string, the body of a comment (after preprocessing).

    Returns:
    - feats: NumPy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here).
    """
    # TODO: Extract features that rely on capitalization.
    # TODO: Lowercase the text in comment. Be careful not to lowercase the tags. (e.g. "Dog/NN" -> "dog/NN").
    # TODO: Extract features that do not rely on capitalization.

    global bristol_norms
    global warringer_norms

    feats = np.zeros(174)

    modified_comment = ''
    sentences = comment.split('\n')
    feats[17] = len(sentences)

    num_tokens = 0
    len_tokens = 0
    num_tokens_without_punc = 0

    aoas, imgs, fams = [], [], []
    v_means, a_means, d_means = [], [], []

    for sent in sentences:
        words = sent.split()
        num_tokens += len(words)
        for i in range(len(words)):
            word = words[i]
            slash_index = word.index('/')
            token = word[:slash_index]
            tag = word[slash_index:]

            if token.isupper() and len(token) >= 3:
                feats[1] += 1

            token = token.lower()

            if token in FIRST_PERSON_PRONOUNS:
                feats[2] += 1
            elif token in SECOND_PERSON_PRONOUNS:
                feats[3] += 1
            elif token in THIRD_PERSON_PRONOUNS:
                feats[4] += 1
            if token in SLANG:
                feats[14] += 1

            if "'ll" in token:
                feats[7] += 1
            elif token == 'will' and tag == '/MD':
                feats[7] += 1
            elif token == 'gonna':
                feats[7] += 1
            elif token == 'go':
                try:
                    if words[i + 1][:2] == 'to' and words[i + 2][-3:] == '/VB':
                        feats[7] += 1
                except IndexError:
                    pass

            if tag == '/CC':
                feats[5] += 1
            elif tag == '/VBD':
                feats[6] += 1
            elif tag == '/,':
                feats[8] += 1
            elif tag in {'/NN', '/NNS'}:
                feats[10] += 1
            elif tag in {'/NNP', '/NNPS'}:
                feats[11] += 1
            elif tag in {'/RB', '/RBR', '/RBS'}:
                feats[12] += 1
            elif tag in {'/WDT', '/WP', '/WP$', '/WRB'}:
                feats[13] += 1

            if tag in punctuations:
                if len(token) > 1:
                    feats[9] += 1
            else:
                len_tokens += len(token)
                num_tokens_without_punc += 1

            if token in bristol_norms:
                aoas.append(bristol_norms[token][0])
                imgs.append(bristol_norms[token][1])
                fams.append(bristol_norms[token][2])

            if token in warringer_norms:
                v_means.append(warringer_norms[token][0])
                a_means.append(warringer_norms[token][1])
                d_means.append(warringer_norms[token][2])

    feats[15] = num_tokens / len(sentences) if len(sentences) != 0 else 0
    feats[16] = len_tokens / num_tokens_without_punc if num_tokens_without_punc != 0 else 0

    if aoas:
        feats[18], feats[19], feats[20] = np.mean(aoas), np.mean(imgs), np.mean(fams)
        feats[21], feats[22], feats[23] = np.std(aoas), np.std(imgs), np.std(fams)
    if v_means:
        feats[24], feats[25], feats[26] = np.mean(v_means), np.mean(a_means), np.mean(d_means)
        feats[27], feats[28], feats[29] = np.std(v_means), np.std(a_means), np.std(d_means)

    feats = feats[1:]
    assert len(feats) == 173
    return feats


def extract2(feats, comment_class, comment_id):
    """ This function adds features 30-173 for a single comment.

    Parameters:
    - feats: np.array of length 173.
    - comment_class: str in {"Alt", "Center", "Left", "Right"}.
    - comment_id: int indicating the id of a comment.

    Returns:
    - feats: NumPy Array, a 173-length vector of floating point features (this 
        function adds feature 30-173). This should be a modified version of 
        the parameter feats.
    """
    global liwc_index
    global liwc_score

    feats[29:173] = liwc_score[comment_class][liwc_index[comment_class][comment_id]]

    return feats


def main(args):
    # Declare necessary global variables here. 

    # Load data
    data = json.load(open(args.input))
    feats = np.zeros((len(data), 173 + 1))

    cat_map = {'Left': 0, 'Center': 1, 'Right': 2, 'Alt': 3}
    load_norms(args)
    load_liwc(args)

    # TODO: Call extract1 for each datapoint to find the first 29 features.
    # Add these to feats.
    # TODO: Call extract2 for each feature vector to copy LIWC features (features 30-173)
    # into feats. (Note that these rely on each data point's class,
    # which is why we can't add them in extract1).
    for i, line in enumerate(data):
        comment_class = line['cat']
        body = line['body']
        comment_id = line['id']

        feats_single = extract1(body)
        feats_single = extract2(feats_single, comment_class, comment_id)
        feats[i, :173] = feats_single
        feats[i, 173] = cat_map[comment_class]

    np.savez_compressed(args.output, feats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Specify the output file.", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1.", required=True)
    parser.add_argument("-p", "--a1-dir",
                        help="Path to csc401 A1 directory. By default it is set to the teach.cs directory for the assignment.",
                        default="/u/cs401/A1/")
    args = parser.parse_args()

    main(args)
