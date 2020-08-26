#!/usr/bin/env sh
/home/ice/project/caffe-cpm/build/tools/caffe train \
    --solver=hand_solver.prototxt --weights=pose_iter_102000.caffemodel \
    --gpu=$1 \
     2>&1 | tee ./logs/training_log.txt
