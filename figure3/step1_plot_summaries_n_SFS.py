"""
read pkl file of summary stats (pi & tajima's D) computed for each replicate and plot lineplot
"""
import pickle, sys, time
import numpy as np
import matplotlib.pyplot as plt


def get_medan_n_CI(stat_pool):
    """Return three np.arrays of mean & 95% CI
    :param: stat_pool: list, list of arrays of same length
    :return: ()
    """
    mean_stat = np.nanmean(np.array(stat_pool), axis=0)
    sd_stat = np.nanstd(np.array(stat_pool), axis=0)
    CIupper_stat, CIlower_stat = mean_stat + 1.76 * sd_stat, mean_stat - 1.76 * sd_stat
    return mean_stat, CIupper_stat, CIlower_stat


# let's not do double axes. Just one at a time=['ci95', 'quartile']
def plot_stats(ax, Neut_pool, Sel_pool, stat, xCoords, y_label, condition, skip=1):
    """Sel_stat and Neut_stat are lists of 3 arrays: mean, CI upper, CI lower
    Return plotted ax
    """
    if stat == 'ci95':
        Neut_stat = get_medan_n_CI(Neut_pool)
        Sel_stat = get_medan_n_CI(Sel_pool)
    elif stat == 'quartile':
        Neut_stat = np.nanquantile(Neut_pool, [0.5, 0.75, 0.25], axis=0)
        Sel_stat = np.nanquantile(Sel_pool, [0.5, 0.75, 0.25], axis=0)
    else:
        raise ValueError('`stat` can only be \"ci95\" or \"quartile\"')
    ax.set_xlabel('Position (0.1cM)')
    xCoord_neut, xCoord_sel = xCoords

    # ax.set_ylabel(y_label)
    ax.axhline(y=0, lw=0.75, color='black')
    ax.axvline(x=0, lw=0.75, color='black')

    ax.plot(xCoord_neut[::skip], Neut_stat[0][::skip], '--', color='darkgrey', label='Neutral')
    # ax.fill_between(xCoord_neut[::skip], Neut_stat[2][::skip], Neut_stat[1][::skip], alpha=0.3, color='lightgrey', ls='--')

    ax.plot(xCoord_sel[::skip], Sel_stat[0][::skip], '-', color='black', label=condition)
    # ax.fill_between(xCoord_sel[::skip], Sel_stat[2][::skip], Sel_stat[1][::skip], alpha=0.3, color='lightgrey', ls='-')

    # ax.set_xlim(min(xCoord_sel), max(xCoord_sel))
    print(ax.get_xlim())
    xmin = np.min(xCoords)
    xmax = np.max(xCoords)
    print(xmin, xmax)
    ax.set_xlim(xmin, xmax)
    print(ax.get_ylim())

    return ax


def merge_more_bins(ogX, ogY, step=2):
    assert len(ogX) == len(ogY)
    extra = len(ogY) % step
    if extra != 0:
        newX = ogX[:-extra:step]
        newY = ogY[:-extra:step]
        for i in range(1, step):
            newY += ogY[i::step]
    else:
        newX = ogX[::step]
        newY = ogY[::step]
        for i in range(1,step):
            newY += ogY[i::step]
            newX += ogX[i::step]
        newX /= step
    assert len(newX) == len(newY)
    return newX, newY


# `center_size` should be in bp
def plot_SFS_panel(ax, neut_glb, Sel_segs, center_size, seq_len, num_bins: int, group: str):
    # pick out the center
    center = int((seq_len/1e3)/2)
    num_segs = int(center_size/1e3)
    if num_segs % 2 == 0:
        left = int(center - num_segs/2)
        right = int(center + num_segs/2)
    else:
        left = int(center - num_segs//2)
        right = int(center + num_segs//2)
        num_segs -= 1
    # pick'em out
    sel_center = np.nansum(np.array(Sel_segs[f'{center - 1}-{center}kb']), axis=0)[1:]
    for win_start in range(left, right):
        if win_start == (center - 1): continue
        win = f'{win_start:g}-{win_start+1:g}kb'
        sel_center += np.nansum(np.array(Sel_segs[win]), axis=0)[1:]

    # merge bins if needed
    seq_len_kb = int(seq_len/1e3)
    x_neu, y_neu = merge_more_bins(np.arange(1., len(neut_glb)+1)/len(neut_glb), neut_glb/seq_len_kb, num_bins)
    x_sel, y_sel = merge_more_bins(np.arange(1., len(sel_center)+1)/len(sel_center), sel_center/num_segs, num_bins)
    # print(x_neu, x_sel)
    # plot
    ax.plot(x_neu, y_neu, '--', color='gray', label='Neutral global')
    ax.plot(x_sel, y_sel, '-', color='black', label=f'{group} {num_segs}kb')
    # ax.set_title(f'{group} center {num_segs}kb')
    ax.set_frame_on(False)
    ax.axhline(y=1, lw=0.75, color='black')
    ax.axvline(x=0, lw=0.75, color='black')
    ax.set_yscale('log')
    # ax.set_ylabel('Count / Mb')
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 1e3)
    ax.set_xlabel('Derived Allele Freq')
    # ax.legend(loc='best')

    return ax


def main():
    figname = sys.argv[1]
    window_size = sys.argv[2]
    # skip = int(sys.argv[3])
    skip = 1
    sfs_segSize = float(sys.argv[3])

    PklFiles = {'Neut': f'precomputed/Neut_rep0-999_win{window_size}_step2kb_SFS-n-summaries.pkl',
                'FS-01': f'precomputed/fullSweep_s.01_h.5_rep0-999_win{window_size}_step2kb_SFS-n-summaries.pkl',
                'PS-01': f'precomputed/partialSweep.5_s.01_h.5_rep0-999_win{window_size}_step2kb_SFS-n-summaries.pkl',
                'stVar-01': f'precomputed/fullSweep_stdVar.1_s.01_h.5_rep0-999_win{window_size}_step2kb_SFS-n-summaries.pkl',
                'BS-01': f'precomputed/500kb_8NeAgo_s1e-5_h100_rep0-999_win{window_size}_step2kb_SFS-n-summaries.pkl'}
    # initiate
    Global_SFS, Local_SFS, Pi_pool, D_pool, SFS_windows, Stat_centers = {}, {}, {}, {}, {}, {}
    scenario_list = ['Neut', 'FS-01', 'PS-01', 'stVar-01', 'BS-01'] #

    for scenario in scenario_list:
        with open(PklFiles[scenario], "rb") as pkl_file:
            Global_SFS[scenario], Local_SFS[scenario], Pi_pool[scenario], D_pool[scenario], SFS_windows[scenario], Stat_centers[scenario] = pickle.load(pkl_file)
        pkl_file.close()

    # now start plotting
    xCoord_neut_stat = Stat_centers['Neut']
    # print(xCoord_neut_stat)
    xCoord_neut_stat.sort()
    # print(xCoord_neut_stat/1e4)

    neutSFS_global = np.sum(np.array(Global_SFS['Neut']), axis=0)[1:]

    # 4 panels
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(8, 3.8), sharex="row", sharey='row', frameon=False)
    # print(axes.shape)
    for i, scenario in enumerate(scenario_list[1:]):
        print(i, scenario)
        xCoord_sel = Stat_centers[scenario]
        # print(xCoord_sel)
        xCoord_sel.sort()
        # print(xCoord_sel/1e5)

        # plot pi
        axes[(0, i)] = plot_stats(axes[(0, i)], np.array(Pi_pool['Neut'])*1e4, np.array(Pi_pool[scenario])*1e4, stat='ci95', xCoords=(xCoord_neut_stat/1e5, xCoord_sel/1e5), y_label='$\hat{\pi}/10^{-4}$', condition=scenario, skip=skip)
        # axes[0, i].set_title(scenario + ': 95% C. I., $\pi$')
        axes[(0, i)].set_ylim(0, 10)
        axes[(0, i)].set_xticks([0, 2.5, 5], labels=['0', '2.5', '5'])
        axes[(0, i)].set_yticks([0, 5, 10], labels=['0', '5', '10'])
        axes[(0, i)].set_frame_on(False)
        axes[(0, i)].set_axis_on()

        # now plot SFS
        axes[(1, i)] = plot_SFS_panel(axes[(1, i)], neutSFS_global, Local_SFS[scenario], center_size=sfs_segSize,
                                   seq_len=5e5, num_bins=(2 + skip), group=scenario)
        axes[(1, i)].set_xticks([0, 0.5, 1], labels=['0', '0.5', '1'])
        axes[(1, i)].set_axis_on()

    # add y labels
    axes[(0, 0)].set_ylabel('$\hat{\pi}/10^{-4}$')
    axes[(1, 0)].set_ylabel('Count / Mb')

    plt.tight_layout()
    plt.savefig(figname, bbox_inches='tight', dpi=300, transparent=True)
    print(time.ctime())


if __name__ == "__main__":
    main()
