"""Allow ``python -m qc_launcher <subcommand> config.yaml`` where the console script is not installed."""

from .cli import main

if __name__ == "__main__":
    main()
