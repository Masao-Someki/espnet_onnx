
import os
import glob
from pathlib import Path
import argparse

import numpy as np

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('rtf_file', type=str)
    args = parser.parse_args()

    # prepare HP
    with open(args.rtf_file, 'r', encoding='utf-8') as f:
        t = f.readlines()            
    
    a = np.array(t).astype(np.float32)
    print(f'mean: {np.mean(a)} - std: {np.std(a)}')
