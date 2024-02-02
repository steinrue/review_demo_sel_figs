import numpy




def getGenoFromSlimOutput (slimLines, sequenceLength, sampleSize):

    # line starting with "positions:" has the positions
    # find it
    posLineIdx = -1
    for (lIdx, thisLine) in enumerate (slimLines):
        if (thisLine.startswith ('positions:')):
            # make sure we have only one
            assert (posLineIdx < 0)
            posLineIdx = lIdx

    # the positions in ms format are fractions,
    positions = numpy.array ([int(float(x) * sequenceLength) for x in slimLines[posLineIdx].split(':')[1].split()])
    # assert that at least not in decreasing order
    # we deal with making them unique if they are not later
    assert (numpy.all (numpy.diff (positions) >= 0))
    assert (positions.min() >=0)
    assert (positions.max() <= sequenceLength)

    # the lines after the position line until the end should be the haplotypes
    sampledHaplotypes = slimLines[(posLineIdx+1):]
    assert (len(sampledHaplotypes) == sampleSize)
    numSegSites = len(sampledHaplotypes[0].strip())
    assert (numSegSites == len(positions))

    # prepare genotype matrix
    geno = -1 * numpy.ones((numSegSites, sampleSize), dtype=int)
    # go through haplotpyes and add them
    for (hIdx, rawHaplo) in enumerate(sampledHaplotypes):
        # convert the raw hapot to numpy
        haplo = numpy.array (list (rawHaplo.strip()), dtype=int)
        assert (len(haplo) == numSegSites)

        # and store it in genotype matrix
        geno[:,hIdx] = haplo

    # everything the light touches is our kingdom
    assert (numpy.all (numpy.isin (geno, [0,1])))

    # if some nucleotide site hit twice, take the first of the two
    uniquePositionMask = numpy.concatenate ([[True], numpy.diff (positions) != 0])
    positions = positions[uniquePositionMask]
    geno = geno[uniquePositionMask,:]

    return (geno, positions)


def singlePopulationBGPiecewise (seed, scalingFactor, mutRate, recoRate, sequenceLength, diploidNes, changeTimes, sampleSize, delMutRate=None, delMutSelCoeff=None):

    # make sure if we want background selection, then we really want it
    assert (((delMutRate is None) and (delMutSelCoeff is None)) or
            ((delMutRate is not None) and (delMutSelCoeff is not None)))
    if (delMutSelCoeff is not None):
        assert ((-1 < delMutSelCoeff) and (delMutSelCoeff <= 0))
    
    # convert rates and sizes with scaling factor
    scaledDiploidNes = [int (x / scalingFactor) for x in diploidNes]
    scaledChangeTimes = [int (x / scalingFactor) for x in changeTimes]
    # one more size than times (the ancestral/equilibrium size)
    assert (len(diploidNes)-1 == len(changeTimes))
    scaledReco = scalingFactor * recoRate

    # if we want deleterious mutations, we need to increase the
    netMutRate = mutRate
    if (delMutRate is not None):
        netMutRate += delMutRate
    scaledMut = scalingFactor * netMutRate

    # initial population size is the one at equilibrium and burn in is ten times that
    burnTime = 10*scaledDiploidNes[0]
    slimSchkript = f"""// simulate genetic variation under neutrality in a population of constant size

// call: slim <scriptname>

initialize() {{
    // set the seed
    setSeed ({seed});

    // slim initialization
    initializeMutationRate ({scaledMut});
    initializeMutationType ("m1", 0.5, "f", 0.0);"""

    # deal with mutations for background selection, if we want them
    if (delMutRate is None):
        slimSchkript += f"""
    initializeGenomicElementType ("g1", m1, 1.0);"""
    else:
        # need the right proportions
        neutMutProp = mutRate / (mutRate + delMutRate)
        delMutProp = delMutRate / (mutRate + delMutRate)
        scaledDelMutSelCoeff = scalingFactor * delMutSelCoeff
        slimSchkript += f"""
    initializeMutationType ("m2", 0.5, "f", {scaledDelMutSelCoeff});
    initializeGenomicElementType ("g1", c(m1,m2), c({neutMutProp},{delMutProp}));"""

    # and this is needed in every case
    slimSchkript += f"""
    initializeGenomicElement (g1, 0, {int(sequenceLength)});
    initializeRecombinationRate ({scaledReco});
}}

1 early() {{
    sim.addSubpop ("p1", {scaledDiploidNes[0]});
}}

// burn in
{burnTime} late() {{"""
    
    # add epochs of changing population size, if we want to
    nextTime = burnTime
    for i in numpy.arange(1, len(scaledDiploidNes)):
        nextTime += scaledChangeTimes[i-1]
        slimSchkript += f"""
    p1.setSubpopulationSize ({scaledDiploidNes[i]});
}}
{nextTime} late() {{"""

    # if we have deleterious mutations (type m2), then we want to remove those before taking the sample
    if (delMutRate is not None):
        slimSchkript += f"""
    // remove mutations of type m2, because shouldn't be used for neutral diversity
    removeMuts = sim.mutationsOfType(m2);
    sim.subpopulations.genomes.removeMutations (removeMuts, T);
"""

    # and finally the sample
    slimSchkript += f"""
    p1.outputMSSample ({int(sampleSize)});
}}

"""
    return slimSchkript