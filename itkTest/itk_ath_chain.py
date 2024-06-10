#!/usr/bin/env python3

import os
import argparse
import pathlib

import acts
import acts.examples, acts.examples.itk

from acts.examples import (
    CsvTrackingGeometryWriter,
    RootAthInputReader
)

geo_dir = pathlib.Path("/eos/home-p/pibutti/acts-itk")
detector, trackingGeometry, decorators = acts.examples.itk.buildITkGeometry(geo_dir)
field = acts.examples.MagneticFieldMapXyz(str(geo_dir / "bfield/ATLAS-BField-xyz.root"))
rnd = acts.examples.RandomNumbers(seed=42)


if "__main__" == __name__:
    
    outputDir = "./"


    s = acts.examples.Sequencer(
        events=1,
        numThreads=1,
        outputDir = str(outputDir)
        )

    # Read Athena input space points and clusters from root file

    athReader = RootAthInputReader(
        level=acts.logging.DEBUG,
        treename  = "GNN4ITk",
        inputfile = "/afs/cern.ch/user/p/pibutti/sw/gnn/Dump_GNN4Itk.root",
    )
    
    s.addReader(athReader)


    s.run()

        
    
