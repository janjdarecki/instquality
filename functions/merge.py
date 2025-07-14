import pandas as pd
import numpy as np
import difflib


def check_countries(column, min_length=4):
    similar_pairs = []
    for i in range(len(column)):
        for j in range(i + 1, len(column)):
            if column[i] != column[j]:  # Ensure that identical countries are excluded
                seq = difflib.SequenceMatcher(None, column[i], column[j])
                match = seq.find_longest_match(0, len(column[i]), 0, len(column[j]))
                if match.size >= min_length:
                    similar_pairs.append((column[i], column[j], column[i][match.a: match.a + match.size]))
    return similar_pairs