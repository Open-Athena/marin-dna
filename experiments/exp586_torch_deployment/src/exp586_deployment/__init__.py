"""Independent exp586 Torch deployment implementation. Optional providers load lazily."""

from typing import Any

__all__ = ["FastMarinDNAMoE", "MarinDNAMoE"]


def __getattr__(name: str) -> Any:
    """Keep benchmark environment selection ahead of Torch/Inductor imports."""
    if name == "MarinDNAMoE":
        from exp586_deployment.model import MarinDNAMoE

        return MarinDNAMoE
    if name == "FastMarinDNAMoE":
        from exp586_deployment.fast import FastMarinDNAMoE

        return FastMarinDNAMoE
    raise AttributeError(name)
