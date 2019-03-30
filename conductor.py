#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script conducts the entire flow of the multivac system to date. it has the
following flow:
1. collect data
    a. these data come from arxiv, springer, and pubmed in this instance, but
        could be modified to include more.
    b. it saves the downloaded pdf's to a directory and creates a json object
        for further use
"""
from multivac.src.data.make import collect_main

def conduct():
    # step 1: collect data
    collect_main()

    # step 2:



if __name__ == '__main__':
    conduct()
