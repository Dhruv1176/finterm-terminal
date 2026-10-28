"""finterm                      -> interactive shell
finterm quote AAPL MSFT        -> run one command and exit
finterm --demo ...             -> use offline synthetic data"""
import argparse
import shlex
import sys

from . import data
from .shell import FinTerm


def main(argv=None):
    p = argparse.ArgumentParser(prog="finterm", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--demo", action="store_true", help="synthetic offline data")
    p.add_argument("command", nargs=argparse.REMAINDER)
    a = p.parse_args(argv)
    if a.demo:
        data.set_demo(True)
    shell = FinTerm()
    if a.command:
        shell.onecmd(" ".join(shlex.quote(x) for x in a.command))
    else:
        try:
            shell.cmdloop()
        except KeyboardInterrupt:
            print()


if __name__ == "__main__":
    sys.exit(main())
