#!/bin/sh
#
# Select Q
#$ -q albert.q
#
# Your job name
#$ -N hai
#
# Use current working directory
#$ -cwd
#
# Join stdout and stderr
#$ -j y
#$ -o speaker_identification.out
#
# Run job through bash shell
#$ -S /bin/bash
#

./venv/bin/python3 -u speaker_identification.py
