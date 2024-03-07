#  * This code is provided solely for the personal and private use of students
#  * taking the CSC401 course at the University of Toronto. Copying for purposes
#  * other than this use is expressly prohibited. All forms of distribution of
#  * this code, including but not limited to public repositories on GitHub,
#  * GitLab, Bitbucket, or any other online platform, whether as given or with
#  * any changes, are expressly prohibited.
#  
#  * All of the files in this directory and all subdirectories are:
#  * Copyright (c) 2024 Frank Rudzicz, Gerald Penn

import sys
import argparse
import os
import json
import re
import spacy
import html

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
nlp.add_pipe('sentencizer')


def preprocess(comment: str) -> str:
    """ 
    This function preprocesses a single comment.

    Parameters:                                                                      
    - comment: string, the body of a comment.

    Returns:
    - modified_comment: string, the modified comment.
    """
    modified_comment = comment

    # STEP 1
    # TODO: Replace newlines with spaces to handle other whitespace chars.
    modified_comment = re.sub(r"(\n+|\r+|\t+)", " ", modified_comment)

    # STEP 2
    # TODO: Remove '[deleted]' or '[removed]' statements.
    modified_comment = re.sub(r'(\[deleted]|\[removed])', '', modified_comment)
    # STEP 3
    # TODO: Unescape HTML.
    modified_comment = html.unescape(modified_comment)
    # Remove URLs.
    modified_comment = re.sub(r"(http|www)\S+", "", modified_comment)

    # STEP 4
    # TODO: Remove duplicate spaces.
    modified_comment = re.sub(r'\s+', ' ', modified_comment)

    # STEP 5
    # TODO: Get Spacy document for modified_comment.
    # TODO: Use Spacy document for modified_comment to create a string.
    # Make sure to:
    #    * Insert "\n" between sentences.
    #    * Split tokens with spaces.
    #    * Write "/POS" (/tag) after each token.
    nlp_doc = nlp(modified_comment)
    modified_comment = ''
    for sent in nlp_doc.sents:
        for token in sent:
            word = token.text
            lemma = token.lemma_
            tag = token.tag_
            if lemma[:1] == '-' and token.text[:1] != '-':
                lemma = word
            if word.isupper():
                lemma = lemma.upper()
            else:
                lemma = lemma.lower()
            modified_comment += f'{lemma}/{tag} '
        modified_comment = modified_comment.strip() + '\n'

    return modified_comment


def main(args):
    all_output = []

    for subdir, dirs, files in os.walk(indir):

        for file in files:
            fullFile = os.path.join(subdir, file)
            print("Processing " + fullFile)

            data = json.load(open(fullFile))

            # TODO: Select appropriate args.max lines.
            # TODO: Read those lines with something like `j = json.loads(line)`.
            # TODO: Choose to retain fields from those lines that are relevant to you.
            # TODO: Add a field to each selected line called 'cat' with the value of 'file' (e.g., 'Alt', 'Right', ...).
            # TODO: Process the body field (j['body']) with preprocess(...) using default for `steps` argument.
            # TODO: Replace the 'body' field with the processed text.
            # TODO: Append the result to 'all_output'.

            start_index = args.ID[0] % len(data)
            if len(data) - start_index < args.max:
                sample = data[start_index:]
                sample.extend(data[:args.max - (len(data) - start_index)])
            else:
                sample = data[start_index:start_index + args.max]
            assert len(sample) == args.max

            for line in sample:
                j = json.loads(line)
                j = {'id': j['id'], 'body': preprocess(j['body']), 'cat': file}
                all_output.append(j)

    fout = open(args.output, 'w')
    fout.write(json.dumps(all_output, indent=2))
    fout.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument('ID', metavar='N', type=int, nargs=1,
                        help='your student ID.')
    parser.add_argument("-o", "--output", help="Specify the output file.", required=True)
    parser.add_argument("--max", type=int, help="The maximum number of comments to read from each file.", default=10000)
    parser.add_argument("--a1-dir", help="The directory for A1. This directory should contain the subdir `data`.",
                        default='/u/cs401/A1')

    args = parser.parse_args()

    if args.max > 200272:
        print("Error: If you want to read more than 200,272 comments per file, you have to read them all.")
        sys.exit(1)

    indir = os.path.join(args.a1_dir, 'data')
    main(args)
