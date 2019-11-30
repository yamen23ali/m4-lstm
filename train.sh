#!/bin/bash

#python src/main.py --EPOCHS $1 --BATCH_SIZE $2 --LOOKBACK 48 --HIDDEN_LAYER_SIZE $3
qsub -l cuda=1 src/main.py --EPOCHS $1 --BATCH_SIZE $2 --LOOKBACK 48 --HIDDEN_LAYER_SIZE $3