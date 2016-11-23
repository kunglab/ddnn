import os

import numpy as np
import matplotlib.pyplot as plt

def min_error(traces, save_dir):
    save_dir = os.path.join(save_dir, 'figures/')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    points = [t for t in traces
              if t['action'] == 'add_point']
    plt.figure(figsize=(8, 6.5))
    ys = [1-p['y'] for p in points]
    plt.plot(np.minimum.accumulate(ys), linewidth=1.5)
    plt.xlabel('epoch')
    plt.ylabel('minimum error')
    plt.savefig(save_dir + 'min_error.png')
    plt.clf()
