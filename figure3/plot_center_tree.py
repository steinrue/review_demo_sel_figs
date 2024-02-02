'''locate the tree at the center of sequence and plot it as svg and svg string'''
import re, sys
import time, os
import numpy as np
# import matplotlib.pyplot as plt
import tskit


def extract_center_tree(bigTree: tskit.TreeSequence):
    """ [obsolete] (didn't know about TreeSequence.at(position) method when writing this func)
    Load the TreeSequence from SLiM and find the tree that contains the selected locus
    """
    midpoint = bigTree.sequence_length // 2
    # diff = 1e64
    tree_id = 0
    # counter=0tracked_samples=
    for local_tree in bigTree.trees():
        if local_tree.index < 0:
            print('skip tree', local_tree)
            continue
        left, right = local_tree.interval
        if left <= midpoint <= right:
            #         print(left, midpoint, right, local_diff, diff)
            tree_id = local_tree.index
            assert tree_id != -1 and tree_id != 0, print(local_tree)
            break
            #         print(center_tree)
            # diff = local_diff
    print(tree_id, local_tree)
    center_tree = bigTree.at_index(tree_id)
    #     print(local_tree.interval, center_tree)
    return center_tree


def main():
    treefile = sys.argv[1]
    numNodes = int(sys.argv[2])
    outputprefix = sys.argv[3]
    sel_loc = 25e4
    sigma = float(sys.argv[4])

    Tree = tskit.load(treefile)

    # down-sample tree
    if len(sys.argv) > 5:
        seed = int(sys.argv[5])
        rng = np.random.RandomState(seed)
        sampleIDs = rng.choice(Tree.samples(),
                               numNodes, replace=False)
        outputprefix = f'{outputprefix}_seed{seed:g}'
    else:
        sampleIDs = np.random.choice(Tree.samples(),
                                     numNodes, replace=False)

    subTree = Tree.simplify(sampleIDs)

    # center_tree = extract_center_tree(subTree)
    center_tree = subTree.at(sel_loc)
    # print(center_tree)

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
    subTree = subTree.delete_sites(dup_list)
    center_tree = subTree.at(sel_loc)
    for site in center_tree.sites():
        if site.position == sel_loc:
            mut = site.mutations[0]
            mut_labels[mut.id] = f"4Ns = {sigma}"
        else:
            mut = site.mutations[0]
            mut_labels[mut.id] = ""
    # remove labels for mutations and nodes
    nd_labels = {}  # An array of labels for the nodes
    for n in center_tree.nodes():
        nd_labels[n] = ""

    # plot
    from IPython.core.display import SVG

    # make y-ticks to label out the root times
    base_ticks = [0, 50, 100, 150, 200, 250, 300, 400, 500, 750, 1000, 1200, 1500]#, 2500, 5000, 7500, 1e4
    yTicks = base_ticks + [center_tree.time(root) for root in center_tree.roots if center_tree.time(root) > 0]
    yTicks = sorted(yTicks)
    print(center_tree.draw_text())
    topo = "tree"
    # topo = "minlex"
    svgstr = center_tree.draw_svg(size=(400, 600),
                                  # node_labels=nd_labels,
                                  # mutation_labels=mut_labels,
                                  # force_root_branch=True,
                                  x_axis=True, y_axis=True,
                                  # max_time=yTicks[-1],
                                  # y_gridlines=False, #  y_axis=False,
                                  # y_ticks=yTicks, #, 2000, 5000, 1e4
                                  order=topo)

    # print(svgstr)
    # save string to file first
    svgstr = re.sub(r'\>\<', '>\n<', svgstr)
    with open(f'{outputprefix}_n{numNodes}_{topo}.svg', 'w') as pic:
        pic.write(svgstr)
    pic.close()
    # from svglib.svglib import svg2rlg
    # from reportlab.graphics import renderPM, renderPDF
    # drawing = svg2rlg(f'{outputprefix}_n{numNodes}.svg')
    # renderPM.drawToFile(drawing, f'{outputprefix}_n{numNodes}.png', fmt='PNG', dpi=100)
    # renderPDF.drawToFile(drawing, f'{outputprefix}_n{numNodes}.pdf')


if __name__ == "__main__":
    main()
