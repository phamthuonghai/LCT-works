#!/usr/bin/env bash
#$ -cwd
#$ -j y
#$ -S /bin/bash

source ../../.env/bin/activate

../../.env/bin/python -uB logreg.py
