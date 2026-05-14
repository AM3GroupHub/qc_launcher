import os
import sys
import datetime
from pathlib import Path

from qc_launcher.drivers import QCSisyphusCalc
import pysisyphus.run as pysis_run
# patch the calculator lookup dict
pysis_run.CALC_DICT["qc_launcher"] = QCSisyphusCalc
from pysisyphus.run import parse_args, load_run_dict, print_bibtex, run_from_dict


def run():
    start_time = datetime.datetime.now()
    args = parse_args(sys.argv[1:])

    # Defaults
    run_dict = {}
    yaml_dir = Path(".")

    if args.yaml:
        run_dict = load_run_dict(args.yaml)
        yaml_dir = Path(os.path.abspath(args.yaml)).parent
    elif args.bibtex:
        print_bibtex()
        return

    run_kwargs = {
        "cwd": yaml_dir,
        "set_defaults": True,
        "yaml_fn": args.yaml,
        "cp": args.cp,
        "scheduler": args.scheduler,
        "clean": args.clean,
        "fclean": args.fclean,
        "version": args.version,
        "restart": args.restart,
        "ntimes": args.ntimes,
    }
    run_result = run_from_dict(run_dict, **run_kwargs)

    end_time = datetime.datetime.now()
    duration = end_time - start_time
    # Only keep hh:mm:ss
    duration_hms = str(duration).split(".")[0]
    print(f"pysisyphus run took {duration_hms} h.")

    return 0

if __name__ == "__main__":
    run()