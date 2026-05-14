import pandas as pd
import numpy as np

df = pd.read_csv('../data/dynamic_data.csv')

for seq_id in df['sequence_id'].unique():
    seq = df[df['sequence_id'] == seq_id]
    motion = np.sum(np.abs(np.diff(seq['x8'].values)))
    if motion < 0.08:
        print(f'Seq {seq_id}: motion = {motion:.4f} ← too low, consider rerecording')