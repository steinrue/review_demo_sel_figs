# python script to simulate slim replicates and plot their summaries for a certain condition
import os, re, sys, time, subprocess
import numpy as np
import tskit
# from process_slim_trees import summarize_trees
# wdir = '/gpfs/data/steinruecken-lab/XiaohengStuff/ARreview_sims/'
wdir = os.getcwd()
# slimpath = '/gpfs/data/steinruecken-lab/XiaohengStuff/build/slim'
slimpath = 'slim4'


def get_overlaping_windows(L, window_size):
    '''return borders & centers of the two set of contiguous windows that,
    when merged, would be overlapping with closest neighbors by half of the window.
    :param int,float L: length of the sequence
    :param int,float window_size: length of each window
    :return: (borders, centers), 2xN np.arrays that stores the borders and centers
        of each layer(?) of contiguous windows
    '''
    num_windows = int(L // window_size) + 1
    borders_set1 = np.arange(num_windows) * window_size
    borders_set1 = np.clip(borders_set1, 0, L)
    centers_set2 = borders_set1[1:-1]
    centers_set1 = borders_set1[:-1] + window_size / 2
    borders_set2 = np.array([0] + list(centers_set1) + [L])
    # sanity check
    assert len(borders_set1) - 1 == len(centers_set1)
    assert len(borders_set2) - 3 == len(centers_set2)

    return [borders_set1, borders_set2], [centers_set1, centers_set2]


def get_contiguous_windows(L, window_size):
    '''return borders & centers of the set of contiguous windows that cover the sequence.
    :param int,float L: length of the sequence
    :param int,float window_size: length of each window
    :return: (borders, centers), 1-D np.arrays that stores the borders and centers
    '''
    num_windows = int(L // window_size) + 1
    borders = np.arange(num_windows) * window_size
    borders = np.clip(borders, 0, L)

    centers = (borders[:-1] + borders[1:]) / 2

    return borders, centers


def get_summary_stat(ts, sampleIDs, borders, centers, stat):  # ={'pi', 'tajimas_D'}
    '''
    return an array of summary stats for the sequence, computed from `window_size` windows
    and taking `window_size/2` steps.
    '''
    if stat == 'pi':
        get_stat = ts.diversity
    elif stat == 'tajimas_D':
        get_stat = ts.Tajimas_D
    else:
        return False

    stat_set1 = get_stat(sample_sets=[sampleIDs], windows=borders[0])
    stat_set2 = get_stat(sample_sets=[sampleIDs], windows=borders[1])

    stat_merged = np.empty(len(centers[0]) + len(centers[1]))
    stat_merged[::2] = stat_set1[:, 0]
    stat_merged[1::2] = stat_set2[1:-1, 0]

    return (stat_merged)


def get_summary_stat_sliding(ts, sampleIDs, L, window_size, step_size, stat):
    '''
    return an array of summary stats for the sequence, computed from `window_size` windows
    and taking `step_size` steps. The steps would be evenly spaced only when `window_size % step_size == 0`
    '''
    if stat == 'pi':
        get_stat = ts.diversity
    elif stat == 'tajimas_D':
        get_stat = ts.Tajimas_D

    # go through each layer of windows
    steps_per_win = int(window_size // step_size)
    wins_per_seq = L // window_size
    # initialize containers
    all_centers = np.empty(int(steps_per_win * (wins_per_seq - 1)) + 1)
    all_stats = np.empty(int(steps_per_win * (wins_per_seq - 1)) + 1)
    # print(f'len(all_centers) = {steps_per_win * (wins_per_seq - 1)}')
    for step in range(steps_per_win):
        if step == 0:
            batch_borders = np.arange(wins_per_seq+1) * window_size + step * step_size
        else:
            batch_borders = np.arange(wins_per_seq) * window_size + step * step_size
            batch_borders = np.concatenate(([0], batch_borders, [L]))
        if np.any( np.diff(batch_borders) <= 0 ):
            print("window borders not ascending:", batch_borders)
            print(step, steps_per_win)
        # if batch_borders[-1] < L:
        #     batch_borders = np.concatenate((batch_borders, [L]))

        try:
            batch_stats = get_stat(sample_sets=[sampleIDs], windows=batch_borders)
        except Exception as e:
            print(e)
            print(batch_borders)
            print(step, steps_per_win)
            sys.exit()

        if step == 0:
            all_stats[step::steps_per_win] = batch_stats[:,0]
            batch_centers = window_size/2 + np.arange(wins_per_seq) * window_size
            all_centers[step::steps_per_win] = batch_centers
        else:
            all_stats[step::steps_per_win] = batch_stats[1:-1, 0]
            batch_centers = window_size/2 + np.arange(wins_per_seq - 1) * window_size + step * step_size
            all_centers[step::steps_per_win] = batch_centers

    return all_centers, all_stats


def summarize_trees_sliding(L, Ne, treeOutFile, window_size, step_size, sfs_seg_size, sample_size, sample_seed=None):
    # check out our trees
    ots = tskit.load(treeOutFile)
    # ots = ots.simplify()
    # sanity checks:
    assert ots.sequence_length == L, f'sequence length in eidos: {L}; seq length in tree ouput: {ots.sequence_length}'
    # note both num_individuals and Ne are numbers of diploids
    assert ots.num_individuals == Ne , f'ts.num_individuals={ots.num_individuals}, eidos Ne = {Ne}.'

    # determine seed & get sample set
    if sample_seed is None:
        sd = np.random.randint(1e6,1e7-1,1)
    else:
        sd = sample_seed
    # print(f'sampling with seed {sd}.')
    # np.random.seed(sd)
    rng = np.random.default_rng(sd)
    sampleIDs = rng.choice(ots.samples(), sample_size, replace=False)

    # get summary stats:
    pi_centers, pi_by_win = get_summary_stat_sliding(ots, sampleIDs, L, window_size, step_size, 'pi')
    D_centers, tajimas_Ds = get_summary_stat_sliding(ots, sampleIDs, L, window_size, step_size, 'tajimas_D')

    assert np.all((pi_centers - D_centers) == 0)
    stat_centers = pi_centers

    # get global and per-kb SFS
    # get overall SFS but in counts. 0 & n would be included
    ## note: sfs[0] counts sites polymorphic in the population but not observed in the samples
    sfs_global = ots.allele_frequency_spectrum(sample_sets=[sampleIDs],
                                        span_normalise=False,
                                        polarised=True)
    num_sfs_win = L // sfs_seg_size
    sfs_windows = np.arange(num_sfs_win+1) * sfs_seg_size
    sfss_local = ots.allele_frequency_spectrum(sample_sets=[sampleIDs],
                                 windows=sfs_windows,
                                 polarised=True, span_normalise=False)

    return pi_by_win, tajimas_Ds, stat_centers, sfs_global, sfss_local, sfs_windows


def get_seeds(condition, num_reps):
    seedfile = f'{wdir}500kb_{condition}_ConstN_slimSeeds.txt'
    Seeds = []
    enough = False
    if os.path.exists(seedfile):
        with open(seedfile, 'r') as seeds:
            for l in seeds:
                if l.startswith('rep'):
                    continue
                rep, seed = map(int, l.strip().split("\t"))
                assert rep == len(Seeds), f'rep=={rep}, len(Seeds)=={len(Seeds)}, l: {l}\n{Seeds}'
                Seeds.append(seed)
        seeds.close()
        enough = (len(Seeds) >= num_reps)
    else:
        seed_pool = np.empty((num_reps, 2))
        seed_pool[:,0] = np.arange(num_reps)
        seed_pool[:,1] = np.random.randint(1e12, 1e13 - 1, num_reps)
        np.savetxt(seedfile, seed_pool, fmt="%d", delimiter="\t", header="rep\tseed", comments="")
        enough = True

    if not enough:
        extras = np.random.randint(1e12, 1e13 - 1, num_reps - len(Seeds))
        OG_len = len(Seeds)
        with open(seedfile, 'a') as seeds:
            for i, seed in enumerate(extras):
                seeds.write(f'{OG_len + i}\t{seed}\n')
                Seeds.append(seed)
        seeds.close()
    return Seeds


def check_slim_dirs(condition):
    # root_folder = f'{wdir}500kb_{condition}_ConstN/'
    root_folder = os.path.join(wdir, f'500kb_{condition}_ConstN')
    if not (os.path.exists(root_folder) and os.path.isdir(root_folder)):
        os.makedirs(root_folder)
    else:
        assert os.path.isdir(root_folder)

    for subdir in ('msout', 'trees'):
        subdir_path = os.path.join(root_folder, subdir)
        if not (os.path.exists(subdir_path) and os.path.isdir(subdir_path)):
            os.makedirs(subdir)
        else:
            assert os.path.isdir(subdir_path)


def get_slim_command(condition, rep, seed, sample_size, slim_vars):
    if condition == 'Neut':
        slim_input = wdir + 'slim_input/Slim4_500kb_Neut_ConstN1e4_noRecap_oTrees_lambda20.eidos'
        slim_output = f'{wdir}500kb_Neut_ConstN/msout/Neut_ConstN_rep{rep}.msout'
        slim_command = f'{slimpath} -t -s {seed} -d var_rep={rep} -d sample_size={sample_size} {slim_vars} {slim_input} > {slim_output}'
        treefile = f'{wdir}500kb_{condition}_ConstN/trees/{condition}_ConstN_rep{rep}.trees'
    elif 'stdVar' in condition:
        slim_input = wdir + 'slim_input/Slim4_500kb_track-Sweep_stdVar-f_ConstN1e4_noRecap_oTrees_lambda20.eidos'
        slim_output = f'{wdir}500kb_{condition}_ConstN/msout/{condition}_ConstN_rep{rep}.msout'
        slim_command = f'{slimpath} -t -s {seed} -d var_rep={rep} -d var_cond=\"\'{condition}\'\" -d sample_size={sample_size} {slim_vars} {slim_input} > {slim_output}'
        treefile = f'{wdir}500kb_{condition}_ConstN/trees/{condition}_ConstN_rep{rep}.trees'
    elif 'fullSweep_s' in condition:
        slim_input = wdir + 'slim_input/Slim4_500kb_track-fullSweep_ConstN1e4_noRecap_oTrees_lambda20.eidos'
        slim_output = f'{wdir}500kb_{condition}_ConstN/msout/{condition}_ConstN_rep{rep}.msout'
        slim_command = f'{slimpath} -t -s {seed} -d var_rep={rep} -d var_cond=\"\'{condition}\'\" -d sample_size={sample_size} {slim_vars} {slim_input} > {slim_output}'
        treefile = f'{wdir}500kb_{condition}_ConstN/trees/{condition}_ConstN_rep{rep}.trees'
    elif 'partialSweep' in condition:
        slim_input = wdir + 'slim_input/Slim4_500kb_track-partialSweep_var-f_ConstN1e4_noRecap_bkTrees_lambda20.eidos'
        slim_output = f'{wdir}500kb_{condition}_ConstN/msout/{condition}_ConstN_rep{rep}.msout'
        slim_command = f'{slimpath} -t -s {seed} -d var_rep={rep} -d var_cond=\"\'{condition}\'\" -d sample_size={sample_size} {slim_vars} {slim_input} > {slim_output}'
        treefile = f'{wdir}500kb_{condition}_ConstN/trees/{condition}_ConstN_rep{rep}.trees'
    elif 'NeAgo' in condition:
        slim_input = wdir + 'slim_input/Slim4_500kb_track-bal_ConstN1e4_noRecap_oTrees_lambda20.eidos'
        slim_output = f'{wdir}500kb_{condition}_ConstN/msout/{condition}_ConstN_rep{rep}.msout'
        slim_command = f'{slimpath} -t -s {seed} -d var_rep={rep} -d var_cond=\"\'{condition}\'\" -d sample_size={sample_size} {slim_vars} {slim_input} > {slim_output}'
        treefile = f'{wdir}500kb_{condition}_ConstN/trees/{condition}_ConstN_rep{rep}.trees'
    else:
        raise (ValueError, 'condition not valid')
    return slim_command, slim_input, treefile, slim_output


def extract_param_from_eidos(eidosFile):
    """Read the eidos file to extract chr length, pop size, and mut/rec rates.
    The regex patterns only works for sequences with homogenous mut/rec rates, and none of the quantities to be extracted are variables (all are constants).
    The sequence must be of type "g1", and the population (for which size will be extracted) must be named "p1"
    """
    # initialize regex
    mut_regex = re.compile(r'initializeMutationRate\s*\(\s*([0-9]+\.*[0-9]*e*-*[0-9]+)\s*\)')
    rec_regex = re.compile(
        r'initializeRecombinationRate\s*\(\s*([0-9]+\.*[0-9]*e*-*[0-9]+)\s*,*\s*([0-9]+\.*[0-9]*e*-*[0-9]+)*\s*\)')
    pop_regex = re.compile(r'sim\.addSubpop\s*\(\s*\"p1\"\s*,\s*([0-9]+\.*[0-9]*e*-*[0-9]+)\s*\)')
    len_regex = re.compile(r'initializeGenomicElement\s*\(\s*g1\s*,\s*([0-9]+\.*[0-9]*e*-*[0-9]*)\s*,\s*([0-9]+\.*[0-9]*e*-*[0-9]*)\s*\)')
    # initialize variables
    mut, rho, seq_len, N = 0, 0, 0, 0
    with open(eidosFile, 'r') as eidos:
        for l in eidos:
            # print(mut_regex.findall(l), rec_regex.findall(l), pop_regex.findall(l), len_regex.findall(l))
            if mut_regex.search(l):
                # print(l)
                mut = mut_regex.findall(l)
                assert len(mut) == 1, f'Error in extracting mutation rate; mu={mut};\nOG line: \"{l}\".'
                mut = float(mut[0])
            elif rec_regex.search(l):
                # print(l)
                rec = rec_regex.findall(l)
                assert len(rec) == 1, f'Error in parsing recombination rate; rec={rec}\nOG line: {l}.'
                rho, seq_len_temp = rec[0]
                rho = float(rho)
                if seq_len_temp != '':
                    assert seq_len == 0
                    seq_len = float(seq_len_temp)
                else:
                    assert seq_len_temp == ''
            elif seq_len == 0 and len_regex.search(l):
                # print(l)
                seq = len_regex.findall(l)
                assert len(seq) == 1, seq
                start, end = map(float, seq[0])
                seq_len = end - start + 1
            elif pop_regex.search(l):
                # print(l)
                N = pop_regex.findall(l)
                assert len(N) == 1, f'Error in parsing pop size; N={N}\nOG line: {l}.'
                N = float(N[0])
            # if all vars are extracted, quit reading
            elif mut != 0 and rho != 0 and seq_len != 0 and N != 0:
                # print(f'Quit reading at {l}')
                break
    # close file in case
    eidos.close()
    # print(mut, rho, seq_len, N)
    return mut, rho, seq_len, N


def get_sliding_summaries(condition, start, end, sample_size, window_size, step_size, Seeds, slim_vars=None):
    """variant of `get_pool_summaries()` to record summary stats from windows sliding with smaller steps"""
    # initialize containers
    pi_pool = []
    D_pool = []
    global_SFSs = []
    Local_SFSs = {}
    for i, seed in enumerate(Seeds[start:(end+1)]):
        rep = start + i
        rng = np.random.default_rng(seed)
        slim_seed, stat_seed = rng.integers(1e12, 1e13 - 1, 2)
        slim_input, treefile, slim_msout = get_slim_files(condition, rep)

        if not os.path.exists(treefile) or not os.path.exists(slim_msout):
            assert slim_vars is not None
            print(f'{time.ctime()}. Running slim for {condition} rep {rep} on seed {slim_seed}')
            command, slim_input, treefile, slim_msout = get_slim_command(condition, rep, slim_seed, sample_size, slim_vars)
            print(command)
            subprocess.check_output(command, shell=True)

        # print(f'{time.ctime()}. Compute summaries...')
        # grep param
        mu, rho, L, Ne = extract_param_from_eidos(slim_input)
        # check out summaries
        pi_by_win, tajimas_Ds, stat_centers, sfs_global, sfss_local, sfs_windows = summarize_trees_sliding(L, Ne, treefile, window_size, step_size, sfs_seg_size=1e3, sample_size=sample_size, sample_seed=stat_seed)
        global_SFSs.append(sfs_global)
        # store by window:
        for loc, sfs in enumerate(sfss_local):
            name = f'{sfs_windows[loc]/1e3:g}-{sfs_windows[loc+1]/1e3:g}kb'
            if name not in Local_SFSs:
                Local_SFSs[name] = []
            Local_SFSs[name].append(sfs)
        pi_pool.append(pi_by_win)
        D_pool.append(tajimas_Ds)
    return global_SFSs, Local_SFSs, pi_pool, D_pool, sfs_windows, stat_centers


def main():
    # start = int(sys.argv[1])
    # end = int(sys.argv[2])
    # num_reps = 1000
    num_reps = int(sys.argv[1])
    start = 0; end = num_reps - 1
    condition = sys.argv[2]
    slim_vars = sys.argv[3] # <-- this would be \"-d var_s -d var_h -d var_freq\" etc
    window_size = float(sys.argv[4])
    sample_size = 50 * 2

    # get seeds
    Seeds = get_seeds(condition, num_reps)
    # sanity check
    assert 0 <= start < end <= num_reps, f'start={start}, end={end}, num_reps={num_reps}'

    # make dirs
    check_slim_dirs(condition)

    # do stuff
    # 1kb segments for local SFS is hard-coded
    global_SFSs, Local_SFSs, pi_pool, D_pool, sfs_windows, stat_centers = get_sliding_summaries(condition, start, end, sample_size, window_size, step_size, Seeds)

    # store everything
    import pickle
    # if 'NeAgo' in condition:
    #     pkl_depo = f'{wdir}100kb_{condition}_ConstN/{condition}_rep{start}-{end}_SFS-n-summaries.pkl'
    # else:
    pkl_depo = f'{wdir}500kb_{condition}_ConstN/{condition}_rep{start}-{end}_win{window_size/1e3:g}kb_SFS-n-summaries.pkl'
    print(f'Pickling to {pkl_depo}...')
    with open(pkl_depo, 'wb') as depo:
        pickle.dump((global_SFSs, Local_SFSs, pi_pool, D_pool, sfs_windows, stat_centers), depo)
    depo.close()


if __name__ == "__main__":
    main()