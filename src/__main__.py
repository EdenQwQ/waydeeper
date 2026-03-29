#!/usr/bin/env python3
"""Main entry point for waydeeper."""

import sys


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "_daemon":
        from src.daemon import main as daemon_main

        sys.argv = [sys.argv[0]] + sys.argv[2:]
        return daemon_main()
    else:
        from src.cli import main as cli_main

        return cli_main()


if __name__ == "__main__":
    sys.exit(main())
