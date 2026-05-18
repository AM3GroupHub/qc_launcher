from __future__ import annotations

from .base_driver import BaseDriver
from .driver_factory import get_driver

__all__ = ["BaseDriver", "get_driver", "QCSisyphusCalc"]


def __getattr__(name: str):
    if name != "QCSisyphusCalc":
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    from .pysisyphus_calc import QCSisyphusCalc

    return QCSisyphusCalc
