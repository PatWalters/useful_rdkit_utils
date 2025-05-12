#!/usr/bin/env python
# This is a simple script to build a ring dictionary from the ChEMBL chemreps file
import sys
import useful_rdkit_utils as uru
import click

@click.command()
@click.option("--mode", prompt="mode [build|search]", help="[build|search]")
@click.option("--infile", prompt="Input chemreps file", help="input file")
@click.option("--outfile", prompt="Output csv file", help="output file")
def main(mode, infile, outfile):
    mode_list = ["build", "search"]
    if mode not in mode_list:
        print(f"mode must be one of {mode_list}")
        sys.exit(0)
    if mode == "build":
        uru.create_ring_dictionary(infile, outfile)
    if mode == "search":
        uru.test_ring_system_lookup(infile, outfile)

if __name__ == "__main__":
    main()
