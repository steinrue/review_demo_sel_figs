""" Modified from `plot_hap_freq_spec_hist.py`
Plot haplotype frequency spectra as histograms
"""
import time, sys, os
import numpy as np
import matplotlib.pyplot as plt
# local import
from plot_hap_freq_spect import read_haps_from_msout, truncate_hap_reps


msoutSkeleton = {'Neut': '500kb_Neut_ConstN/msout/Neut_ConstN_rep%d.msout',
                 'FS-01': '500kb_fullSweep_s.01_h.5_ConstN/msout/fullSweep_s.01_h.5_ConstN_rep%d.msout',
                 'FS-001': '500kb_fullSweep_s.001_h.5_ConstN/msout/fullSweep_s.001_h.5_ConstN_rep%d.msout',
                 'PS-01': '500kb_partialSweep.5_s.01_h.5_ConstN/msout/partialSweep.5_s.01_h.5_ConstN_rep%d.msout',
                 'PS-001': '500kb_partialSweep.5_s.001_h.5_ConstN/msout/partialSweep.5_s.001_h.5_ConstN_rep%d.msout',
                 'stVar-01': '500kb_fullSweep_stdVar.1_s.01_h.5_ConstN/msout/fullSweep_stdVar.1_s.01_h.5_ConstN_rep%d.msout',
                 'stVar-001': '500kb_fullSweep_stdVar.1_s.001_h.5_ConstN/msout/fullSweep_stdVar.1_s.001_h.5_ConstN_rep%d.msout',
                 'BS-100k': '100kb_8NeAgo_s1e-5_h100_ConstN/msout/8NeAgo_s1e-5_h100_ConstN_rep%d.msout',
                 'BS-01': '500kb_8NeAgo_s1e-5_h100_ConstN/msout/8NeAgo_s1e-5_h100_ConstN_rep%d.msout',
                 'BS-001': '500kb_8NeAgo_s1e-5_h10_ConstN/msout/8NeAgo_s1e-5_h10_ConstN_rep%d.msout'}


def tally_haps(samples):
    '''Count haplotypes. Return a np.array of counts (int)'''
    Haps = {}
    for hap in samples:
        if hap not in Haps:
            Haps[hap] = 1
        else:
            Haps[hap] += 1
    counts = list(Haps.values())
    counts = sorted(counts, reverse=True)
    return np.array(counts, dtype=int)


def get_hap_freq_spect(samples, normalize=False, K=None):
    '''Generalized function to produce haplotype frequency spectra
    :param samples: list of 0/1 strings as haplotypes
    :param normalize: bool whether or not to normalize
    :param K: int or None. Required when `normalize=True`.
    :return: array of counts and (when normalized) spectra
    '''
    if normalize:
        assert K is not None
        assert len(samples) > K
    Haps = {}
    for hap in samples:
        if hap not in Haps:
            Haps[hap] = 1
        else:
            Haps[hap] += 1

    # sort descendingly
    full_count = sorted(list(Haps.values()), reverse=True)
    full_spect = np.array(full_count)/sum(full_count)
    if normalize:
        k_spect = np.array(full_spect[:K])
        assert np.all(np.diff(k_spect) <= 0), np.diff(k_spect)
        total = np.sum(k_spect)
        k_spect = k_spect/total
    else:
        k_spect = None
    return full_count, k_spect


def pool_HFS_counts(condition, repIDs, window_size, sample_size, seed=None):
    # initiate
    Hap_counts = np.zeros(sample_size, dtype=int)
    hap_numSites = []
    for i in repIDs:
        filename = msoutSkeleton[condition] % (i)
        samples, positions, numSites, samp_size, sel_loc = read_haps_from_msout(filename)
        assert sample_size <= samp_size, 'Not enough samples in the output'
        # down sample if needed
        if sample_size < samp_size:
            if seed is None:
                seed = np.random.randint(1e8, 1e9, 1)[0]
            # print(f'Down-sampling with seed {seed}')
            rng = np.random.RandomState(seed)
            samples = rng.choice(samples, sample_size, replace=False)
        # now truncate haps
        haps = truncate_hap_reps(samples, positions, sel_loc, window_size)
        # print(f'MLH has {len(haps[0])} snps.')
        hap_numSites.append(len(haps[0]))
        # then get specs (already sorted & trimmed
        batch_counts = tally_haps(haps)
        # sanity check
        # assert batch_counts.shape == Hap_counts.shape, f'batch shape:{batch_counts.shape}, Hap_count shape: {Hap_counts.shape}'
        if batch_counts.shape != Hap_counts.shape:
            assert batch_counts.shape[0] < Hap_counts.shape[0]
            Hap_counts = [Hap_counts[i] + batch_counts[i] if i < batch_counts.shape[0] else Hap_counts[i] for i in range(Hap_counts.shape[0])]
            Hap_counts = np.array(Hap_counts)
        else:
            Hap_counts += batch_counts
    hap_numSites = np.array(hap_numSites)
    print(condition, "number of sites in the center window:\n", min(hap_numSites), np.quantile(hap_numSites, [0.25, 0.5, 0.75]), max(hap_numSites))
    return Hap_counts, hap_numSites


def plot_pool_HFS_hist(ax, condition, repIDs, window_size, sample_size: int, K: int, seed=None):
    """Plot histograms of hap freq./count."""
    # initiate
    Hap_counts, hap_numSites = pool_HFS_counts(condition, repIDs, window_size, sample_size, seed)
    full_spec = Hap_counts / np.sum(Hap_counts)
    k_spec = Hap_counts[:K] / np.sum(Hap_counts[:K])
    # now let's plot
    ax.bar(x=np.arange(1,K+1)*2, height=full_spec[:K], color="#444444", label='Un-normalized')
    ax.bar(x=np.arange(1,K+1)*2 + 1, height=k_spec, color='#AAAAAA', label='Normalized')
    return ax


def contrast_pool_HFSs(ax, spect_sel, K: int, spect_neut=None, ylabel=None):
    if spect_neut is None:
        xCoords = np.arange(1, K+1)
    else:
        assert len(spect_sel) == len(spect_neut)
        xCoords = 2 * np.arange(1, K+1)

    ax.bar(xCoords, spect_sel, color="#444444") #, label=condition
    if spect_neut is not None:
        ax.bar(xCoords + 1, spect_neut, color='#AAAAAA', label="Neutral")
    # ax.set_axis_off()
    ax.set_ylabel(ylabel)
    ax.set_xticks([])
    # ax.set_ylabel('Fraction')
    ax.set_frame_on(False)
    return ax


def test():
    condition = sys.argv[1]
    from_rep, to_rep = int(sys.argv[2]), int(sys.argv[3])
    window_size = float(sys.argv[4])
    sample_size = 100
    K = 5
    sd = int(sys.argv[5])
    figname = sys.argv[6]

    fig, ax = plt.subplots()
    ax = plot_pool_HFS_hist(ax, condition, range(from_rep, to_rep + 1), window_size, sample_size, K, seed=sd)
    ax.set_title(condition)
    ax.set_ylabel('Fraction')
    ax.set_frame_on(False)

    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(figname, bbox_inches='tight', dpi=300, transparent=True)


def main():
    from_rep, to_rep = 0, 999
    window_size = float(sys.argv[1])
    sample_size = int(sys.argv[2])
    K = int(sys.argv[3])
    sd = int(sys.argv[4])
    figPrefix = sys.argv[5]
    scenario_list = ['Neut', 'FS-01', 'FS-001', 'PS-01', 'PS-001', 'stVar-01', 'stVar-001', 'BS-01', 'BS-001']  #

    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(12, 7), sharex=True, sharey=True)
    # ax_idx = [(0,0), (0,1), (1,0), (1,1), (2,0), (2,1), (3,0), (3,1)]
    ax_idx = [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2), (0, 3), (1, 3)]
    for i, scenario in enumerate(scenario_list[1:]):
        idx = ax_idx[i]
        axes[idx] = plot_pool_HFS_hist(axes[idx], scenario, range(from_rep, to_rep + 1), window_size, sample_size, K, seed=sd)
        # axes[idx].invert_yaxis()
        axes[idx].set_title(scenario)
        axes[idx].set_ylabel('Fraction')
        axes[idx].set_frame_on(False)

    plt.tight_layout()
    figname = f'{figPrefix}_rep{from_rep}-{to_rep}_win{window_size/1e3:g}_n{sample_size}_K{K}_seed{sd}.pdf'
    plt.savefig(figname, bbox_inches='tight', dpi=300, transparent=True)


def main_rev():
    from_rep, to_rep = 0, 999
    window_size = float(sys.argv[1])
    sample_size = int(sys.argv[2])
    K = int(sys.argv[3])
    sd = int(sys.argv[4])
    figPrefix = sys.argv[5]
    scenario_list = ['Neut', 'FS-01', 'FS-001', 'PS-01', 'PS-001', 'stVar-01', 'stVar-001', 'BS-01', 'BS-001']  #

    # read through all scenarios and save the counts first
    Hap_counts = {}
    Num_sites = {}
    for i, scenario in enumerate(scenario_list):
        print(time.ctime(), f'Reading {to_rep - from_rep + 1} reps in {scenario}:')
        cond_hap_counts, hap_numSites = pool_HFS_counts(scenario, range(from_rep, to_rep + 1),
                                                        window_size, sample_size, seed=sd)
        Hap_counts[scenario] = cond_hap_counts
        Num_sites[scenario] = hap_numSites

    # finished reading data, start plotting
    ## fig1--4, bars
    Spect_abs, Spect_rel = {}, {}
    Spect_abs['Neut'] = Hap_counts['Neut'][:K]/sum(Hap_counts['Neut'])
    Spect_rel['Neut'] = Hap_counts['Neut'][:K]/sum(Hap_counts['Neut'][:K])
    # ax_idx = [(0,0), (0,1), (1,0), (1,1), (2,0), (2,1), (3,0), (3,1)]
    ax_idx = [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2), (0, 3), (1, 3)]
    ### fig1: only absolute bars
    fig1, axes1 = plt.subplots(nrows=2, ncols=4, figsize=(12, 7), sharex=True, sharey=True)
    for i, scenario in enumerate(scenario_list[1:]):
        idx = ax_idx[i]
        Spect_abs[scenario] = Hap_counts[scenario][:K]/sum(Hap_counts[scenario])
        axes1[idx] = contrast_pool_HFSs(axes1[idx], Spect_abs[scenario], K, ylabel=f'Frac. of all {window_size/1e3:g}kb haps')
        axes1[idx].set_title(scenario)
    plt.tight_layout()
    figname1 = f'{figPrefix}_absFracs_rep{from_rep}-{to_rep}_win{window_size/1e3:g}_n{sample_size}_K{K}_seed{sd}.pdf'
    plt.savefig(figname1, bbox_inches='tight', transparent=True)

    # clean the slate
    plt.clf()
    ### fig2: only relative bars
    fig2, axes2 = plt.subplots(nrows=2, ncols=4, figsize=(12, 7), sharex=True, sharey=True)
    for i, scenario in enumerate(scenario_list[1:]):
        idx = ax_idx[i]
        Spect_rel[scenario] = Hap_counts[scenario][:K]/sum(Hap_counts[scenario][:K])
        axes2[idx] = contrast_pool_HFSs(axes2[idx], Spect_rel[scenario], K, ylabel=f'Frac. of top {K} {window_size/1e3:g}kb haps')
        axes2[idx].set_title(scenario)
    plt.tight_layout()
    figname2 = f'{figPrefix}_relFracs_rep{from_rep}-{to_rep}_win{window_size/1e3:g}_n{sample_size}_K{K}_seed{sd}.pdf'
    plt.savefig(figname2, bbox_inches='tight', transparent=True)

    # clean up
    plt.clf()
    ### fig3: sel vs neut relative bars
    fig3, axes3 = plt.subplots(nrows=2, ncols=4, figsize=(12, 7), sharex=True, sharey=True)
    for i, scenario in enumerate(scenario_list[1:]):
        idx = ax_idx[i]
        axes3[idx] = contrast_pool_HFSs(axes3[idx], Spect_rel[scenario], K, Spect_rel['Neut'], ylabel=f'Frac. of top {K} {window_size/1e3:g}kb haps')
        axes3[idx].set_title(scenario)
    plt.tight_layout()
    figname3 = f'{figPrefix}_relFracs_wNeut_rep{from_rep}-{to_rep}_win{window_size/1e3:g}_n{sample_size}_K{K}_seed{sd}.pdf'
    plt.savefig(figname3, bbox_inches='tight', transparent=True)

    # clean up
    plt.clf()
    ### fig3: sel vs neut relative bars
    fig4, axes4 = plt.subplots(nrows=2, ncols=4, figsize=(12, 7), sharex=True, sharey=True)
    for i, scenario in enumerate(scenario_list[1:]):
        idx = ax_idx[i]
        axes4[idx] = contrast_pool_HFSs(axes4[idx], Spect_abs[scenario], K, Spect_abs['Neut'], ylabel=f'Frac. of top {K} {window_size/1e3:g}kb haps')
        axes4[idx].set_title(scenario)
    plt.tight_layout()
    figname4 = f'{figPrefix}_absFracs_wNeut_rep{from_rep}-{to_rep}_win{window_size/1e3:g}_n{sample_size}_K{K}_seed{sd}.pdf'
    plt.savefig(figname4, bbox_inches='tight', transparent=True)

    # clean up
    plt.clf()
    # the histogram of number of sites doesn't have to do with K
    figname = f'{figPrefix}_numSites-Hist_win{window_size/1e3:g}_n{sample_size}_rep{from_rep}-{to_rep}_seed{sd}.png'
    if not os.path.exists(figname):
        mycolors = plt.cm.Set2(range(1, 1+len(scenario_list)))
        # find x-axis range
        from plot_tmrca_hist import flatten
        all_counts = list(Num_sites.values())
        all_counts = np.sort(flatten(all_counts))
        xmin, xmax = all_counts[0], all_counts[-1]
        binwidth = 5
        numBins = (xmax - xmin)//binwidth + 1
        # plot histogram
        fig, ax = plt.subplots(figsize=(9,6))
        for i, scenario in enumerate(scenario_list):
            ax.hist(Num_sites[scenario], bins=np.linspace(xmin, xmax, numBins + 1), label=scenario,
                    color=mycolors[i], alpha=0.6)
        ax.legend(loc='best', frameon=False, ncol=2, framealpha=0)
        plt.tight_layout()
        plt.savefig(figname, bbox_inches='tight', dpi=300, transparent=True)


if __name__ == "__main__":
    # test()
    main_rev()