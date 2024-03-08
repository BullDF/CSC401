"""
This code is provided solely for the personal and private use of students
taking the CSC401H/2511H course at the University of Toronto. Copying for
purposes other than this use is expressly prohibited. All forms of
distribution of this code, including but not limited to public repositories on
GitHub, GitLab, Bitbucket, or any other online platform, whether as given or
with any changes, are expressly prohibited.

Authors: Sean Robertson, Jingcheng Niu, Zining Zhu, and Mohamed Abdall
Updated by: Arvie Frydenlund, Raeid Saqur and Jingcheng Niu

All of the files in this directory and all subdirectories are:
Copyright (c) 2024 University of Toronto
"""
import math
import time

"""
Calculate BLEU score for one reference and one hypothesis

You do not need to import anything more than what is here
"""

from math import exp  # exp(x) gives e^x
from collections.abc import Sequence


def grouper(seq: Sequence[str], n: int) -> list:
    """
    Extract all n-grams from a sequence

    An n-gram is a contiguous sub-sequence within `seq` of length `n`. This
    function extracts them (in order) from `seq`.

    Parameters
    ----------
    seq : sequence
        A sequence of words or token ids representing a transcription.
    n : int
        The size of sub-sequence to extract.

    Returns
    -------
    ngrams : list
    """
    ngrams = []
    for i in range(len(seq) - n + 1):
        ngrams.append(seq[i:i + n])

    return ngrams


def n_gram_precision(
        reference: Sequence[str], candidate: Sequence[str], n: int
) -> float:
    """
    Calculate the precision for a given order of n-gram

    Parameters
    ----------
    reference : sequence
        The reference transcription. A sequence of words or token ids.
    candidate : sequence
        The candidate transcription. A sequence of words or token ids
        (whichever is used by `reference`)
    n : int
        The order of n-gram precision to calculate

    Returns
    -------
    p_n : float
        The n-gram precision. In the case that the candidate has length 0,
        `p_n` is 0.
    """
    reference_n_grams = grouper(reference, n)
    candidate_n_grams = grouper(candidate, n)
    count = 0
    total = len(candidate_n_grams)
    for cand_n_gram in candidate_n_grams:
        if cand_n_gram in reference_n_grams:
            count += 1

    return count / total if total != 0 else 0


def brevity_penalty(reference: Sequence[str], candidate: Sequence[str]) -> float:
    """
    Calculate the brevity penalty between a reference and candidate

    Parameters
    ----------
    reference : sequence
        The reference transcription. A sequence of words or token ids.
    candidate : sequence
        The candidate transcription. A sequence of words or token ids
        (whichever is used by `reference`)

    Returns
    -------
    BP : float
        The brevity penalty. In the case that the candidate transcription is
        of 0 length, `BP` is 0.
    """

    r = len(reference)
    c = len(candidate)

    if c == 0:
        return 0
    brevity = r / c

    return 1 if brevity < 1 else exp(1 - brevity)


def BLEU_score(reference: Sequence[str], candidate: Sequence[str], n) -> float:
    """
    Calculate the BLEU score.  Please scale the BLEU score by 100.0

    Parameters
    ----------
    reference : sequence
        The reference transcription. A sequence of words or token ids.
    candidate : sequence
        The candidate transcription. A sequence of words or token ids
        (whichever is used by `reference`)
    n : int
        The maximum order of n-gram precision to use in the calculations,
        inclusive. For example, ``n = 2`` implies both unigram and bigram
        precision will be accounted for, but not trigram.

    Returns
    -------
    bleu : float
        The BLEU score
    """

    p = []
    BP = brevity_penalty(reference, candidate)
    for i in range(1, n + 1):
        p.append(n_gram_precision(reference, candidate, i))

    return BP * (math.prod(p) ** (1 / n)) * 100.0
