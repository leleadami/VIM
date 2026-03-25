#!/bin/bash
for cat in grid ; do
    for d in nlmeans wavelet rclbp; do
        for e in clahe histeq; do
            echo "=== $cat | $d + $e ==="
            python pipeline.py --dataset dataset --category $cat --denoise $d --enhance $e --img-size 512
        done
    done
done
