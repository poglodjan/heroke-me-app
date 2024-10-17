import autograd.numpy as np
import autograd.numpy.random as npr
from matplotlib import pyplot as plt
import random

def needleman_wunsch(seq1, seq2, match=1, mismatch=-1, gap=-1):
    n = len(seq1)
    m = len(seq2)
    score_matrix = np.zeros((n + 1, m + 1))

    for i in range(1, n + 1):
        score_matrix[i][0] = score_matrix[i - 1][0] + gap
    for j in range(1, m + 1):
        score_matrix[0][j] = score_matrix[0][j - 1] + gap

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match_score = match if seq1[i - 1] == seq2[j - 1] else mismatch
            score_matrix[i][j] = max(
                score_matrix[i - 1][j - 1] + match_score,
                score_matrix[i - 1][j] + gap,
                score_matrix[i][j - 1] + gap
            )

    align1, align2 = '', ''
    i, j = n, m
    while i > 0 or j > 0:
        current_score = score_matrix[i][j]
        if i > 0 and j > 0 and (current_score == score_matrix[i - 1][j - 1] + (match if seq1[i - 1] == seq2[j - 1] else mismatch)):
            align1 += seq1[i - 1]
            align2 += seq2[j - 1]
            i -= 1
            j -= 1
        elif i > 0 and (current_score == score_matrix[i - 1][j] + gap):
            align1 += seq1[i - 1]
            align2 += '-'
            i -= 1
        else:
            align1 += '-'
            align2 += seq2[j - 1]
            j -= 1

    return score_matrix, align1[::-1], align2[::-1]

def generate_random_sequence(length, alphabet="ATGC"):
    return ''.join(random.choices(alphabet, k=length))