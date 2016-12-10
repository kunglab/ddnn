from __future__ import print_function
import os
import sys
sys.path.append('..')
import argparse

from datasets import datasets

#if cam is not specified, will return all cams
train, test = datasets.get_mvmc()
