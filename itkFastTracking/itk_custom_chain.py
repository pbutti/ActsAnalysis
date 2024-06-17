#!/usr/bin/env python3
import pathlib, acts, acts.examples, acts.examples.itk

import argparse

from acts.examples.simulation import (
    addParticleGun,
    MomentumConfig,
    EtaConfig,
    ParticleConfig,
    addPythia8,
    addFatras,
    ParticleSelectorConfig,
    addDigitization,
)
from acts.examples.reconstruction import (
    addSeeding,
    SeedingAlgorithm,
    TruthSeedRanges,
    addCKFTracks,
    CkfConfig,
    TrackSelectorConfig,
    addAmbiguityResolution,
    AmbiguityResolutionConfig,
    addVertexFitting,
    VertexFinder,
)

parser = argparse.ArgumentParser(description="ITK ")
parser.add_argument(
    "--output",
    "-o",
    help="Output directory",
    type=pathlib.Path,
    default=pathlib.Path.cwd() / "itk_output",
)
parser.add_argument("--events", "-n", help="Number of events", type=int, default=1000)
parser.add_argument("--skip", "-s", help="Number of events", type=int, default=0)
parser.add_argument(
    "--ttbar",
    help="Use Pythia8 (ttbar, pile-up 200) instead of particle gun",
    action="store_true",
)
parser.add_argument(
    "--ttbar-pu",
    help="Number of pile-up events for ttbar",
    type=int,
    default=200,
)
parser.add_argument(
    "--gun-particles",
    help="Multiplicity (no. of particles) of the particle gun",
    type=int,
    default=4,
)
parser.add_argument(
    "--gun-multiplicity",
    help="Multiplicity (no. of vertices) of the particle gun",
    type=int,
    default=200,
)
parser.add_argument(
    "--gun-eta-range",
    nargs=2,
    help="Eta range of the particle gun",
    type=float,
    default=[-3.0, 3.0],
)

parser.add_argument(
    "--ambi-solver",
    help="Set which ambiguity solver to use, default is the classical one",
    type=str,
    choices=["greedy", "scoring", "ML"],
    default="greedy",
)
parser.add_argument(
    "--ambi-config",
    help="Set the configuration file for the Score Based ambiguity resolution",
    type=pathlib.Path,
    default=pathlib.Path.cwd() / "ambi_config.json",
)

parser.add_argument(
    "--MLSeedFilter",
    help="Use the Ml seed filter to select seed after the seeding step",
    action="store_true",
)
parser.add_argument(
    "--reco",
    help="Switch reco on/off",
    default=True,
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--output-root",
    help="Switch root output on/off",
    default=True,
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--fastTracking",
    help="Swith fastTracking chain on/off",
    default=False,
    action=argparse.BooleanOptionalAction,
)

args = parser.parse_args()


ttbar_pu200 = False
u = acts.UnitConstants
geo_dir = pathlib.Path("/eos/home-p/pibutti/acts-itk")
outputDir = args.output
print("Output directory = ", args.output)
# acts.examples.dump_args_calls(locals())  # show acts.examples python binding calls

detector, trackingGeometry, decorators = acts.examples.itk.buildITkGeometry(geo_dir)
field = acts.examples.MagneticFieldMapXyz(str(geo_dir / "bfield/ATLAS-BField-xyz.root"))
rnd = acts.examples.RandomNumbers(seed=42)

s = acts.examples.Sequencer(events=args.events, numThreads=1, outputDir=str(outputDir))

if not ttbar_pu200:
    addParticleGun(
        s,
        MomentumConfig(1.0 * u.GeV, 10.0 * u.GeV, transverse=True),
        EtaConfig(-4.0, 4.0, uniform=True),
        ParticleConfig(2, acts.PdgParticle.eMuon, randomizeCharge=True),
        rnd=rnd,
    )
else:
    addPythia8(
        s,
        hardProcess=["Top:qqbar2ttbar=on"],
        npileup=200,
        vtxGen=acts.examples.GaussianVertexGenerator(
            stddev=acts.Vector4(0.0125 * u.mm, 0.0125 * u.mm, 55.5 * u.mm, 5.0 * u.ns),
            mean=acts.Vector4(0, 0, 0, 0),
        ),
        rnd=rnd,
        outputDirRoot=outputDir,
    )

addFatras(
    s,
    trackingGeometry,
    field,
    rnd=rnd,
    preSelectParticles=ParticleSelectorConfig(
        rho=(0.0 * u.mm, 28.0 * u.mm),
        absZ=(0.0 * u.mm, 1.0 * u.m),
        eta=(-4.0, 4.0),
        pt=(150 * u.MeV, None),
        removeNeutral=True,
    )
    if ttbar_pu200
    else ParticleSelectorConfig(),
    outputDirRoot=outputDir,
)

addDigitization(
    s,
    trackingGeometry,
    field,
    digiConfigFile=geo_dir / "itk-hgtd/itk-smearing-config.json",
    outputDirRoot=outputDir,
    rnd=rnd,
)



# Add fast seeding configuration
# Fast tracking configuration for seed finder
#def ActsFastPixelSeedingToolCfg(flags,
#                                name: str = "ActsFastPixelSeedingTool",
#                                **kwargs) -> ComponentAccumulator:
#    ## Additional cuts for fast seed configuration
#    kwargs.setdefault("minPt", 1000 * UnitConstants.MeV) --- OK
#    kwargs.setdefault("collisionRegionMin", -150 * UnitConstants.mm) --- OK
#    kwargs.setdefault("collisionRegionMax", 150 * UnitConstants.mm)  --- OK
#    kwargs.setdefault("maxPhiBins", 200) --- OK (default in ACTS)
#    kwargs.setdefault("gridRMax", 250 * UnitConstants.mm) -- OK
#    kwargs.setdefault("deltaRMax", 200 * UnitConstants.mm) -- OK
#    kwargs.setdefault("zBinsCustomLooping" , [2, 10, 3, 9, 6, 4, 8, 5, 7]) -- OK
#    kwargs.setdefault("rRangeMiddleSP", [
#             [40.0, 80.0],
#             [40.0, 200.0],
#             [70.0, 200.0],
#             [70.0, 200.0],
#             [70.0, 250.0],
#             [70.0, 250.0],
#             [70.0, 250.0],
#             [70.0, 200.0],
#             [70.0, 200.0],
#             [40.0, 200.0],
#             [40.0, 80.0]]) --- OK 
#    kwargs.setdefault("useVariableMiddleSPRange", False) --- OK
#    kwargs.setdefault("useExperimentCuts", True) --- Not implemented in ACTS I think

#From original setup
#kwargs.setdefault("numSeedIncrement" , float("inf")) --- 100 in ACTS
#kwargs.setdefault("deltaZMax" , float("inf"))        --- OK
#kwargs.setdefault("maxPtScattering", float("inf"))   --- OK 

addSeeding(
    s,
    trackingGeometry,
    field,
    TruthSeedRanges(pt=(1.0 * u.GeV, None), eta=(-4.0, 4.0), nHits=(9, None))
    if ttbar_pu200
    else TruthSeedRanges(),
    seedingAlgorithm=SeedingAlgorithm.Default,
    *acts.examples.itk.itkSeedingAlgConfig(
        acts.examples.itk.InputSpacePointsType.PixelSpacePoints,
        highOccupancyConfig=args.fastTracking,
    ),
    initialSigmas=[
        1 * u.mm,
        1 * u.mm,
        1 * u.degree,
        1 * u.degree,
        0.1 / u.GeV,
        1 * u.ns,
    ],
    initialVarInflation=[1.0] * 6,
    geoSelectionConfigFile=geo_dir / "itk-hgtd/geoSelection-ITk.json",
    outputDirRoot=outputDir,
)



# Prepare the trackSelector configurations


FastTrackingConfiguration = (
    TrackSelectorConfig(absEta=(None, 2.0), pt=(0.9 * u.GeV, None), loc1=(-150*u.mm,150*u.mm),
                        nMeasurementsMin=9, maxHoles=2, maxOutliers=2, maxSharedHits=2),
    TrackSelectorConfig(absEta=(None, 2.6), pt=(0.4 * u.GeV, None), loc1=(-150*u.mm,150*u.mm),
                        nMeasurementsMin=8, maxHoles=2, maxOutliers=2, maxSharedHits=2),
    TrackSelectorConfig(absEta=(None, 4.0), pt=(0.4 * u.GeV, None), loc1=(-150*u.mm,150*u.mm),
                        nMeasurementsMin=7, maxHoles=2, maxOutliers=2, maxSharedHits=2),
)

z0cut = 200 * u.mm

DefaultConfiguration = (
    TrackSelectorConfig(absEta=(None, 2.0), pt=(1.0 * u.GeV, None),loc1=(-z0cut,z0cut),
                        nMeasurementsMin=7, maxHoles=2, maxOutliers=2, maxSharedHits=100),
    TrackSelectorConfig(absEta=(None, 2.6), pt=(0.4 * u.GeV, None),loc1=(-z0cut,z0cut),
                        nMeasurementsMin=7, maxHoles=2, maxOutliers=2, maxSharedHits=100),
    TrackSelectorConfig(absEta=(None, 4.0), pt=(0.4 * u.GeV, None),loc1=(-z0cut,z0cut),
                        nMeasurementsMin=7, maxHoles=2, maxOutliers=2, maxSharedHits=100),
)



addCKFTracks(
    s,
    trackingGeometry,
    field,
    trackSelectorConfig = FastTrackingConfiguration
    if args.fastTracking
    else DefaultConfiguration,
    ckfConfig=CkfConfig(
        seedDeduplication=True,
        stayOnSeed=True,
    ),
    outputDirRoot=outputDir,
)


if not args.fastTracking :

    addAmbiguityResolution(
        s,
        AmbiguityResolutionConfig(
            maximumSharedHits=3,
            maximumIterations=10000,
            nMeasurementsMin=6,
        ),
        outputDirRoot=outputDir,
    )
else :
    print("Fast Tracking: do not run ambiguity resolution")
    
s.run()
