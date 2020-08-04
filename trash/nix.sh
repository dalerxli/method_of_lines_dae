#!/bin/sh
nix-shell \
-p python37Packages.scikits-odes \
-p python37Packages.numpy \
-p python37Packages.matplotlib \
--run "python3"