#!/bin/bash
for cat in wood; do
    for d in gaussian median bilateral nlmeans wavelet rclbp; do
        for e in clahe histeq; do
            echo "=== $cat | $d + $e ==="
            python pipeline.py --dataset dataset --category $cat --denoise $d --enhance $e 
        done
    done
done

