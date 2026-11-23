#!/usr/bin/env bash
# TurBLiMP verisini (CC BY 4.0) sabit bir commit'te indirir; böylece herkes aynı veriyle çalışır.
set -e
COMMIT=297de13fb7a0ce524fe32e8b175c6b5255d66960
if [ ! -d data/turblimp ]; then
  git clone https://github.com/ezgibasar/turblimp data/turblimp
fi
git -C data/turblimp checkout -q "$COMMIT"
echo "TurBLiMP hazır: data/turblimp (commit $COMMIT)"
