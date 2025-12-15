#!/bin/sh

./scons.py complex_assembly -j12 mode=release bin
./scons.py coupled_moves -j12 mode=release bin
