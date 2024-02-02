"""
Plot haplotype frequency spectra as histograms
(also plot histograms of number of segregating sites per window, which is not in the paper)
"""
import time, re, sys, os
import numpy as np
import matplotlib.pyplot as plt

msoutSkeleton = {'Neut': '500kb_Neut_ConstN/msout/Neut_ConstN_rep%d.msout',
                 'FS-01': '500kb_fullSweep_s.01_h.5_ConstN/msout/fullSweep_s.01_h.5_ConstN_rep%d.msout',
                 'PS-01': '500kb_partialSweep.5_s.01_h.5_ConstN/msout/partialSweep.5_s.01_h.5_ConstN_rep%d.msout',
                 'stVar-01': '500kb_fullSweep_stdVar.1_s.01_h.5_ConstN/msout/fullSweep_stdVar.1_s.01_h.5_ConstN_rep%d.msout',
                 'BS-01': '500kb_8NeAgo_s1e-5_h100_ConstN/msout/8NeAgo_s1e-5_h100_ConstN_rep%d.msout'}


def read_haps_from_msout(msoutfile):
    '''Read ms-formatted output file, return list of haps (: str), list of segsite position (:int), number of sites (:int), and sample size (: int)'''
    samples = []
    numSites = 0
    samp_size = 0
    positions = []
    seq_length = 0
    sel_loc = None
    regex_01 = re.compile(r'^[0|1]{2,}$')
    with open(msoutfile, 'r') as msout:
        for l in msout:
            l = l.strip()
            if l.startswith('initializeGenomicElement('):
                start, end = map(int, re.findall(r'\b([0-9]+)\b', l))
                seq_length = (end - start + 1)
            elif l.startswith('#OUT') and 'SM' in l:
                samp_size = int(l.split(' ')[-1])
            elif l.startswith('#OUT') and 'T' in l and 'm2' in l:
                sel_loc = int(l.split(' ')[7])
            elif l.startswith('segsites'):
                numSites = int(l.split(' ')[1])
            elif l.startswith('positions'):
                assert seq_length != 0, l
                positions = [int(float(pos) * seq_length) for pos in l.split(' ')[1:]]
                assert len(positions) == numSites, f'len(positions) = {positions}, numSites = {numSites}.'
            elif regex_01.search(l):
                # sanity check
                assert numSites != 0 and samp_size != 0, f'numSites = {numSites}, samp_size = {samp_size}\nl={l}.'
                assert len(l) == numSites, f'len(str)={len(l)}, numSites = {numSites}.'
                samples.append(l)
            else:
                continue
    # last sanity check
    assert len(samples) == samp_size, f'len(samples) = {len(samples)}, samp_size = {samp_size}.'
    # just in case
    msout.close()
    if sel_loc is None:
        sel_loc = int(seq_length / 2)
    return samples, np.array(positions), numSites, samp_size, sel_loc


# hap_pool is a list of strings of 0's & 1's
# positions is a list of segsite positions
def truncate_hap_reps(hap_pool, positions, sel_loc, win_size):
    # sanity check
    assert len(hap_pool[0]) == len(
        positions), f'Length of hap strings ({len(hap_pool[0])}) and positions ({len(positions)}) should match.'

    left = sel_loc - win_size / 2
    right = sel_loc + win_size / 2
    indice_to_keep = np.where((left <= positions) & (positions <= right))[0]
    # positions should be ascending
    ## np.diff works on plain lists too!
    pos_diff = np.diff(positions)
    # remove duplicate
    if not np.all(pos_diff > 0):
        non_ascen = np.where(pos_diff <= 0)[0]
        assert np.all(pos_diff[non_ascen] == 0), print(np.where(pos_diff[non_ascen] < 0, pos_diff))
        dup_idx = np.concatenate((non_ascen, non_ascen + 1))  # , non_ascen-1
        dup_idx = np.array(list(set(dup_idx)))
        dup_idx.sort()
        # print(dup_idx, positions[dup_idx])
        # indice_to_keep = np.delete(indice_to_keep, dup_idx)
        # indice_to_keep = indice_to_keep[indice_to_keep != dup_idx]
        # indice_to_keep = np.where(indice_to_keep not in dup_idx, indice_to_keep, None)
        indice_to_keep = np.array([index for index in indice_to_keep if index not in dup_idx])

    def truncate(hap: str, indice):
        trimmed_hap = [hap[i] for i in range(len(hap)) if i in indice]
        return trimmed_hap

    trimmed_haps = ["".join(truncate(hap, indice_to_keep)) for hap in hap_pool]

    return trimmed_haps


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
    full_spect = np.array(full_count) / sum(full_count)
    if normalize:
        k_spect = np.array(full_spect[:K])
        assert np.all(np.diff(k_spect) <= 0), np.diff(k_spect)
        total = np.sum(k_spect)
        k_spect = k_spect / total
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
        assert sample_size <= samp_size, f'Not enough samples in the output ({samp_size}; requested {sample_size} for {condition} rep {i})'
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
            Hap_counts = [Hap_counts[i] + batch_counts[i] if i < batch_counts.shape[0] else Hap_counts[i] for i in
                          range(Hap_counts.shape[0])]
            Hap_counts = np.array(Hap_counts)
        else:
            Hap_counts += batch_counts
    hap_numSites = np.array(hap_numSites)
    print(condition, "number of sites in the center window:\n", min(hap_numSites),
          np.quantile(hap_numSites, [0.25, 0.5, 0.75]), max(hap_numSites))
    return Hap_counts, hap_numSites


def plot_pool_HFS_hist(ax, condition, repIDs, window_size, sample_size: int, K: int, seed=None):
    """Plot histograms of hap freq./count."""
    # initiate
    Hap_counts, hap_numSites = pool_HFS_counts(condition, repIDs, window_size, sample_size, seed)
    full_spec = Hap_counts / np.sum(Hap_counts)
    k_spec = Hap_counts[:K] / np.sum(Hap_counts[:K])
    # now let's plot
    ax.bar(x=np.arange(1, K + 1) * 2, height=full_spec[:K], color="#444444", label='Un-normalized')
    ax.bar(x=np.arange(1, K + 1) * 2 + 1, height=k_spec, color='#AAAAAA', label='Normalized')
    return ax


def contrast_pool_HFSs(ax, spect_sel, K: int, spect_neut=None, ylabel=None):
    xCoords = np.arange(1, K + 1)

    ax.bar(xCoords, spect_sel, color="lightgray")
    if spect_neut is not None:
        ax.bar(xCoords, spect_neut, color='None', facecolor='None', edgecolor="black", lw=1, ls='--', label="Neutral")
    # ax.set_axis_off()'#444444'
    ax.set_ylabel(ylabel)
    ax.set_xticks([])
    # ax.set_ylabel('Fraction')
    ax.set_frame_on(False)
    return ax


def write_hap_counts(Hap_counts, filename, scenario_list):
    if not os.path.exists(filename):
        with open(filename, 'w') as haps:
            for sce in scenario_list:
                # haps.write(f'{sce}\t{"\t".join(list(map(str, Hap_counts[sce])))}\n')
                haps.write('{}\t{}\n'.format(sce, "\t".join(list(map(str, Hap_counts[sce])))))
        haps.close()


def read_hap_counts(filename):
    Hap_counts = {}
    with open(filename, 'r') as haps:
        for l in haps:
            l = l.strip().split("\t")
            cond, counts = l[0], list(map(int, l[1:]))
            Hap_counts[cond] = np.array(counts)
    haps.close()
    return Hap_counts


def write_num_sites(Num_sites, filename, scenario_list):
    if not os.path.exists(filename):
        # maybe reformat the dict?
        with open(filename, 'w') as sites:
            ## write a header here:
            sites.write('scenario\treps\n')
            for sce in scenario_list:
                sites.write('{}\t{}\n'.format(sce, ",".join(list(map(str, Num_sites[sce])))))
            sites.close()


def read_num_sites(filename):
    Num_sites = {}
    with open(filename, 'r') as sites:
        for l in sites:
            if 'reps' in l:
                continue
            l = l.strip().split("\t")
            cond, counts = l[0], list(map(int, l[1].split(",")))
            Num_sites[cond] = np.array(counts)
    sites.close()
    return Num_sites


def main():
    from_rep, to_rep = 0, 999
    window_size = float(sys.argv[1])
    sample_size = int(sys.argv[2])
    K = int(sys.argv[3])
    sd = int(sys.argv[4])
    figPrefix = sys.argv[5]
    scenario_list = ['Neut', 'FS-01', 'PS-01', 'stVar-01', 'BS-01']  # , 'FS-001', 'PS-001', 'stVar-001', 'BS-001'

    # read through all scenarios and save the counts first
    hapCount_name = f'{figPrefix}_aggregated_hap_counts_win{window_size / 1e3:g}kb_n{sample_size}_rep{from_rep}-{to_rep}_seed{sd}.txt'
    numSites_name = f'{figPrefix}_number-of-SNPs_win{window_size / 1e3:g}kb_n{sample_size}_rep{from_rep}-{to_rep}_seed{sd}.txt'
    if not os.path.exists(hapCount_name) or not os.path.exists(numSites_name):
        Hap_counts = {}
        Num_sites = {}
        for i, scenario in enumerate(scenario_list):
            print(time.ctime(), f'Reading {to_rep - from_rep + 1} reps in {scenario}:')
            cond_hap_counts, hap_numSites = pool_HFS_counts(scenario, range(from_rep, to_rep + 1),
                                                            window_size, sample_size, seed=sd)
            Hap_counts[scenario] = cond_hap_counts
            Num_sites[scenario] = hap_numSites
        # save stuff
        write_hap_counts(Hap_counts, hapCount_name, scenario_list)
        write_num_sites(Num_sites, numSites_name, scenario_list)
    else:
        Hap_counts = read_hap_counts(hapCount_name)
        Num_sites = read_num_sites(numSites_name)

    # finished reading data, start plotting
    Spect_abs = {'Neut': Hap_counts['Neut'][:K] / sum(Hap_counts['Neut'])}

    # ax_idx = [(0, 0), (0, 1), (0, 2), (0, 3)]  # , (1, 0), (1, 1), (1, 2), (1, 3)
    ### only absolute bars
    ### sel vs neut absolute bars
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(8, 2), sharex=True, sharey=True)
    for i, scenario in enumerate(scenario_list[1:]):
        # idx = ax_idx[i]
        axes[i] = contrast_pool_HFSs(axes[i], Spect_abs[scenario], K, Spect_abs['Neut'], ylabel='Prop. of Hap.')
        # axes[idx].set_title(scenario)
        axes[i].set_ylim(0, 1)
        axes[i].set_xlabel(f'Top {K} Haplotypes')
        axes[i].axhline(y=1, lw=0.75, color='black')
        axes[i].axvline(x=0, lw=0.75, color='black')
    plt.tight_layout()
    figname = f'{figPrefix}_absProp_wNeut_rep{from_rep}-{to_rep}_win{window_size / 1e3:g}kb_n{sample_size}_K{K}_seed{sd}.pdf'
    plt.savefig(figname, bbox_inches='tight', transparent=True)

    # clean up
    plt.clf()
    # the histogram of number of sites doesn't have to do with K
    # mycolors = plt.cm.tab20(range(1, 1 + len(scenario_list)))
    mycolors = plt.cm.Set2(range(len(scenario_list)))
    # find x-axis range
    from collections.abc import Iterable

    def flatten(x):
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            return [a for i in x for a in flatten(i)]
        else:
            return [x]

    all_counts = list(Num_sites.values())
    all_counts = np.sort(flatten(all_counts))
    xmin, xmax = all_counts[0], all_counts[-1]
    binwidth = 2
    numBins = (xmax - xmin) // binwidth + 1
    # plot histogram
    fig2, ax = plt.subplots(figsize=(9, 6))
    for i, scenario in enumerate(scenario_list):
        ax.hist(Num_sites[scenario], bins=np.linspace(xmin, xmax, numBins + 1), label=scenario,
                color=mycolors[i], alpha=0.6)
    ax.legend(loc='best', frameon=False, ncol=2, framealpha=0)
    ax.set_frame_on(False)
    plt.tight_layout()
    figname2 = f'{figPrefix}_numSites-Hist_win{window_size / 1e3:g}kb_n{sample_size}_rep{from_rep}-{to_rep}_seed{sd}.png'
    plt.savefig(figname2, bbox_inches='tight', dpi=300, transparent=True)


if __name__ == "__main__":
    # test()
    main()
