"""Go through the center tree in each rep, and plot the tree with median TMRCA
Based on `Process_slim_trees.ipynb`
"""
import sys, os, re
import numpy as np
import matplotlib.pyplot as plt
import tskit

treeSkeleton = {'Neut': '500kb_Neut_ConstN/trees/Neut_ConstN_rep%d.trees',
                'FS-01': '500kb_fullSweep_s.01_h.5_ConstN/trees/fullSweep_s.01_h.5_ConstN_rep%d.trees',
                'FS-001': '500kb_fullSweep_s.001_h.5_ConstN/trees/fullSweep_s.001_h.5_ConstN_rep%d.trees',
                'PS-01': '500kb_partialSweep.5_s.01_h.5_ConstN/trees/partialSweep.5_s.01_h.5_ConstN_rep%d.trees',
                'PS-001': '500kb_partialSweep.5_s.001_h.5_ConstN/trees/partialSweep.5_s.001_h.5_ConstN_rep%d.trees',
                'stVar-01': '500kb_fullSweep_stdVar.1_s.01_h.5_ConstN/trees/fullSweep_stdVar.1_s.01_h.5_ConstN_rep%d.trees',
                'stVar-001': '500kb_fullSweep_stdVar.1_s.001_h.5_ConstN/trees/fullSweep_stdVar.1_s.001_h.5_ConstN_rep%d.trees',
                'BS-100k': '100kb_8NeAgo_s1e-5_h100_ConstN/trees/8NeAgo_s1e-5_h100_ConstN_rep%d.trees',
                'BS-01': '500kb_8NeAgo_s1e-5_h100_ConstN/trees/8NeAgo_s1e-5_h100_ConstN_rep%d.trees',
                'BS-001': '500kb_8NeAgo_s1e-5_h10_ConstN/trees/8NeAgo_s1e-5_h10_ConstN_rep%d.trees'}

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


def find_m2_position(condition: str, rep):
    filename = msoutSkeleton[condition] % (rep)
    with open(filename, 'r') as msout:
        for l in msout:
            if l.startswith('#OUT') and " m2 " in l:
                l = l.strip().split(" ")
                sel_loc = int(l[7])
                sel_coef = float(l[8])
    msout.close()
    return sel_loc, sel_coef


from IPython.core.display import SVG


def plot_center_tree(samp_trees, sigma: float, sel_loc=None):
    # get center loc
    if sel_loc is None:
        sel_loc = samp_trees.sequence_length / 2
    else:
        assert 0 < sel_loc < samp_trees.sequence_length
    # # simplify
    # samp_trees = ts.simplify(sample_IDs) #, keep_input_roots=True, keep_unary=True
    center_tree = samp_trees.at(sel_loc)

    # remove labels on mutations except for the selected one
    mut_labels = {}  # An array of labels for the mutations
    dup_list = []
    for site in center_tree.sites():
        if len(site.mutations) > 1:
            dup_list.append(site.id)
        else:
            try:
                mut = site.mutations[0]
                assert mut.site == site.id, print(mut, site)
            except Exception as e:
                print(e)
                continue
            if site.position == sel_loc:
                mut_labels[mut.id] = f"4Ns = {sigma}"
            else:
                mut_labels[mut.id] = ""
    # remove dup from ts
    samp_trees = samp_trees.delete_sites(dup_list)

    # remove labels on the internal nodes
    nd_labels = {}  # An array of labels for the nodes
    for n in center_tree.nodes():
        # Set sample node labels from metadata
        nd = samp_trees.node(n)
        if nd.individual == -1:
            nd_labels[n] = ""
        # to use the *individual* name instead, if the individuals in your tree sequence have names
        else:
            nd_labels[n] = nd.individual

    # make y-ticks to label out the root times
    base_ticks = [0, 50, 100, 150, 200, 250, 300, 400, 500, 750, 1000, 1200, 1500]#, 2500, 5000, 7500, 1e4
    yTicks = base_ticks + [center_tree.time(root) for root in center_tree.roots if center_tree.time(root) > 0]
    yTicks = sorted(yTicks)

    svgstr = center_tree.draw_svg(size=(400, 600),
                                  force_root_branch=True,
                                  mutation_labels=mut_labels,  # center_tr.mutations()
                                  node_labels=nd_labels,
                                  max_time=yTicks[-1],
                                  y_axis=True, x_axis=True,
                                  y_ticks=yTicks,
                                  y_gridlines=False,
                                  order="minlex")
                                  # order="tree")

    SVG(svgstr)
    return svgstr


numbers_regex = re.compile(r'^[0-9]+')


def read_median_rep_from_file(heightfile):
    """[obsolete??] Can be replaced by pandas.read_csv
    Read `heightfile` to obtain replicate Tmrca's and return the list of median heights, ordered by rep indice.
    """
    height_list = []
    with open(heightfile, "r") as heights:
        for l in heights:
            if numbers_regex.search(l):
                l = l.strip().split("\t")
                i, H, this_seed, num_roots = int(l[0]), float(l[1]), int(l[2]), int(l[3])
                if l[4] == "-":
                    batch_Hs = [H]
                else:
                    batch_Hs = list(map(float, l[4].split(",")))
                height = (i, H, this_seed, num_roots, batch_Hs)
                height_list.append(height)
    return height_list


def find_median_from_reps(cond, heightfile, treeSkeleton, replist, sample_size, scaling_factor, seed=None):
    # set seed
    if seed is None:
        rng = np.random.RandomState()
        seed = rng.get_state()[1][0]
    else:
        rng = np.random.RandomState(seed)

    # initialize
    # mean_heights = []
    # Heights = {} # change it to list of tupples for convenience of sorting
    height_list = []
    print(f'Initialize sampling with seed {seed}.')
    for i in replist:
        tsfile = treeSkeleton % (i)
        rep_ts = tskit.load(tsfile)
        if cond != "Neut":
            sel_loc, sel_coef = find_m2_position(cond, i)
        else:
            sel_loc, sel_coef = (rep_ts.sequence_length / 2), 0
        # sanity check
        assert sample_size <= rep_ts.num_samples

        this_seed = rng.get_state()[1][0]
        sampleIDs = rng.choice(rep_ts.samples(), sample_size, replace=False)

        # cTree = pick_center_tree(rep_ts, sampleIDs, sel_loc)
        cTree = rep_ts.simplify(sampleIDs).at(sel_loc)
        # remove unary ones
        batch_Hs = [cTree.time(root) * scaling_factor for root in cTree.roots if cTree.time(root) > 0]
        tmrca = np.mean(batch_Hs)
        # print(len(cTree.roots), tmrca)
        height = (i, tmrca, this_seed, cTree.num_roots, batch_Hs)
        height_list.append(height)
    # sort by mean
    height_list = sorted(height_list, key=lambda x: x[1])

    # write output
    recordH = open(heightfile, "w")
    recordH.write(f"# time already scaled back by lambda={scaling_factor}.\n")
    recordH.write("rep\tmean_root_age\tsample_seed\tnum_roots\troot_ages\n")
    for height in height_list:
        (i, tmrca, this_seed, num_roots, batch_Hs) = height
        # write out
        if num_roots > 1:
            recordH.write(f'{i}\t{tmrca}\t{this_seed}\t{num_roots}\t{",".join(list(map(str, batch_Hs)))}\n')
        else:
            recordH.write(f'{i}\t{tmrca}\t{this_seed}\t{num_roots}\t-\n')
    # finished recording, close file
    recordH.close()

    return height_list


def test():
    cond = sys.argv[1]
    from_rep = int(sys.argv[2])
    to_rep = int(sys.argv[3])
    sample_size = 25
    scaling_factor = 20
    seed = int(sys.argv[4])

    numReps = to_rep - from_rep + 1

    # how about we write out the Tmrcas?
    heightfile = f'TreeHeights_{cond}_rep{from_rep}-{to_rep}_n{sample_size}_seed{seed}_lambda20.tsv'

    if os.path.exists(heightfile):
        height_list = read_median_rep_from_file(heightfile)
    else:
        height_list = find_median_from_reps(cond, heightfile, treeSkeleton[cond], range(from_rep, to_rep + 1), sample_size, scaling_factor, seed)
    # note that items are (i, tmrca, this_seed, num_roots, batch_Hs) = height) now and are sorted
    # Tmrca_list = list(zip(*height_list))[1]
    Tmrca_list = [height[1] for height in height_list]

    print(min(Tmrca_list), np.quantile(Tmrca_list, [0.25, 0.5, 0.75]), max(Tmrca_list))
    H_med = np.quantile(Tmrca_list, 0.5)
    # # pick out the median tree
    med_idx, med_height, med_seed, med_num_roots, med_batch_Hs = height_list[numReps//2 + 1]
    print(f"rep {med_idx} has its center tree height closest to median {med_height}.")

    med_trees = tskit.load(treeSkeleton[cond] % (med_idx))
    # now plot tree
    if cond != "Neut":
        med_sel_loc, scaled_sel_coef = find_m2_position(cond, med_idx)
    else:
        med_sel_loc, scaled_sel_coef = (med_trees.sequence_length / 2), 0

    # get pop-scale sel coef
    scaled_Ne = med_trees.num_samples / 2
    sigma = 4 * scaled_Ne * scaled_sel_coef
    print(f'scaled_Ne = {scaled_Ne}, scaled_sel_coef = {scaled_sel_coef}; sigma = {sigma}')

    rng = np.random.RandomState(seed)
    med_seed = rng.get_state()[1][0]
    sampleIDs = rng.choice(med_trees.samples(), sample_size, replace=False)
    # med_tree = pick_center_tree(med_trees, sampleIDs, med_sel_loc)
    med_samp_trees = med_trees.simplify(sampleIDs)
    med_center_tree = med_samp_trees.at(med_sel_loc)
    print(med_center_tree.draw_text())
    batch_Hs = [med_center_tree.time(root) for root in med_center_tree.roots if med_center_tree.time(root) > 0]
    print(f'sampled {sample_size} lineages from rep {med_idx} with seed {med_seed}. Deepest tree height: {max(batch_Hs)}.')
    svgstr = plot_center_tree(med_samp_trees, sigma=sigma, sel_loc=med_sel_loc)
    svgstr = re.sub(r'\>\<', '>\n<', svgstr)
    SVG(svgstr)
    figname = f'{cond}_rep{med_idx}_n{sample_size}_seed{med_seed}_center_tree.svg'
    with open(figname, 'w') as dump:
        dump.write(svgstr)
    dump.close()


if __name__ == "__main__":
    # main()
    test()
