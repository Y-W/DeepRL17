import matplotlib.pyplot as plt
import numpy as np

from game_replay_parallel import GameEngine


frame_size = (2,)
past_frame = 1
state_shape = (past_frame,) + frame_size
atari_game = 'MountainCar-v0'
state_bounds = [(-1.2, 0.6), (-0.07, 0.07)]

action_n = 3

def null_fn(x):
    return x

class MountainCar(GameEngine):
    def __init__(self, parallel_size, replay_size, max_samp_cnt=2):
        super(MountainCar, self).__init__(
            parallel_size, atari_game, None,
            past_frame, 
            null_fn, frame_size, np.float32, 
            null_fn, frame_size, np.float32,
            null_fn, max_samp_cnt,
            replay_size)

def visualize_states(imgPath, eval_batch_fn, batch_size=32, granularity=64):
    plt.rcParams.update({'font.size': 36})
    x = np.linspace(-1.2, 0.6, granularity)
    y = np.linspace(-0.07, 0.07, granularity)
    xx, yy = np.meshgrid(x, y)
    states = np.stack((xx.reshape(-1), yy.reshape(-1)), axis=1)

    fig, axes = plt.subplots(figsize=(34, 7), nrows=1, ncols=action_n, sharey=True)
    values = np.zeros((states.shape[0], action_n), dtype=np.float32)
    p = 0
    while p < states.shape[0]:
        dp = min(states.shape[0] - p, batch_size)
        values[p:p+dp] = eval_batch_fn(states[p:p+dp])
        p = p + dp

    for a in xrange(action_n):
        axes[a].xaxis.set_tick_params(width=2, length=8, direction='out')
        axes[a].set_xticks([-1.0, -0.5, 0.0, 0.5])
        axes[a].xaxis.set_ticks_position('bottom')
        if a == 0:
            axes[a].yaxis.set_tick_params(width=2, length=8, direction='out')
            axes[a].set_yticks([-0.5, 0.0, 0.5])
            axes[a].yaxis.set_ticks_position('left')
        if a == 0:
            axes[a].set_title("Go to the left", y=1.05)
        if a == 1:
            axes[a].set_title("Do nothing", y=1.05)
        if a == 2:
            axes[a].set_title("Go to the right", y=1.05)
        value_displayed = values[:, a].reshape(granularity, granularity)
        value_displayed = np.ma.array(value_displayed, mask=np.isnan(value_displayed))
        cmap = plt.cm.jet
        cmap.set_bad('grey', 1.)
        mp = axes[a].imshow(value_displayed,
                            extent=(-1.2, 0.6, 0.7, -0.7),
                            interpolation='nearest', vmin=np.min(values)-0.1,
                            vmax=np.max(values)+0.1, cmap=cmap)
        axes[a].set_xlabel('position')
        axes[0].set_ylabel('velosity * 10')

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, -0.01, 0.04, 1.0])
    cbar_ax.xaxis.set_tick_params(width=2, length=8, direction='out')
    cbar_ax.yaxis.set_tick_params(width=2, length=8, direction='out')
    fig.colorbar(mp, cax=cbar_ax)
    fig.savefig(imgPath, bbox_inches='tight')
    plt.close()
