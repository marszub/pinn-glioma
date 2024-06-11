#!/bin/python

from plot.ArgsParser import ArgsParser


if __name__ == "__main__":
    parser = ArgsParser()
    parser.show()
    args = parser.get()
    args.handler(args)
