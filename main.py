#!/usr/bin/env python
# coding: utf-8

import argparse
import train_graphwavenet
import train_dcrnn
import train_autogluon

def main():
    parser = argparse.ArgumentParser(description="NYC Taxi Forecast Models Runner")
    parser.add_argument(
        "--model", 
        type=str, 
        default="autogluon", 
        choices=["graphwavenet", "dcrnn","autogluon"],
        help="Which model to run"
    )
    args = parser.parse_args()

    if args.model == "graphwavenet":
        train_graphwavenet.run()
    elif args.model == "dcrnn":
        train_dcrnn.run()
    elif args.model == "autogluon":
        train_autogluon.run()
    else:
        raise ValueError(f"Unknown model {args.model}")

if __name__ == "__main__":
    main()







