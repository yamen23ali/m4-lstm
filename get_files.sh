#!/bin/bash
scp pml_08@cluster.ml.tu-berlin.de:m4-lstm/models/$1.json models/$1.json
scp pml_08@cluster.ml.tu-berlin.de:m4-lstm/models/$1.h5 models/$1.h5