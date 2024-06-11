#!/usr/bin/env python3

import os
import argparse
import pathlib

import acts
import acts.examples, acts.examples.itk

from acts.examples import (
    CsvTrackingGeometryWriter,
    RootAthInputReader,
    TrackParamsEstimationAlgorithm,
)

from acts.examples.reconstruction import (
    addStandardSeeding,
)

from acts.examples.itk import InputSpacePointsType

geo_dir = pathlib.Path("/eos/home-p/pibutti/acts-itk")
detector, trackingGeometry, decorators = acts.examples.itk.buildITkGeometry(geo_dir)
field = acts.examples.MagneticFieldMapXyz(str(geo_dir / "bfield/ATLAS-BField-xyz.root"))
rnd = acts.examples.RandomNumbers(seed=42)


if "__main__" == __name__:
    
    outputDir = "./"

    s = acts.examples.Sequencer(
        events=10,
        numThreads=1,
        outputDir = str(outputDir)
        )

    # Read Athena input space points and clusters from root file

    athReader = RootAthInputReader(
        level=acts.logging.INFO,
        treename  = "GNN4ITk",
        inputfile = "/eos/home-p/pibutti/sw/run/gnn/Dump_GNN4Itk.root",
    )
    
    s.addReader(athReader)

    spacePoints = athReader.config.outputPixelSpacePoints
    
     # run seeding on Pixel Space points
    inputPixelSeeds = addStandardSeeding(
        s,
        spacePoints,
        *acts.examples.itk.itkSeedingAlgConfig(
            InputSpacePointsType.PixelSpacePoints
            ),
        logLevel = acts.logging.DEBUG,
        )


    # run seeding on Strip Space Points
    

    # run seed to prototracks

    prototracks = "seed-prototracks"
    s.addAlgorithm(
        acts.examples.SeedsToPrototracks(
            level=acts.logging.DEBUG,
            inputSeeds=inputSeeds,
            outputProtoTracks=prototracks,
            )
        )
    
    # estimate seeding performance
    parEstimateAlg = TrackParamsEstimationAlgorithm(
        level=acts.logging.DEBUG,
        inputSeeds=inputSeeds,
        outputTrackParameters="estimatedparameters",
        trackingGeometry=trackingGeometry,
        magneticField=field,
        )
    
    #s.addAlgorithm(parEstimateAlg)


    s.run()

        
    
