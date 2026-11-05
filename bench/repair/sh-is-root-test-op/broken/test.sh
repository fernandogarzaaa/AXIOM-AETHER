#!/bin/sh
. ./lib.sh
is_root 0 || { echo "test.sh:3: is_root 0 should succeed"; exit 1; }
if is_root 1000; then echo "test.sh:4: is_root 1000 should fail"; exit 1; fi
