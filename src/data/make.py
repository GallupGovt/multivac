#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from multivac.src.data.get import collect_get_main
from multivac.src.data.process import collect_process_main


def collect_main():
    # query apis to obtain articles
    collect_get_main()

    # process article data for models
    collect_process_main()


if __name__ == '__main__':
    collect_main()
