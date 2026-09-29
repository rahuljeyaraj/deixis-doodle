"""Plotter controller: G-code, safety checks, serial link and simulator.

Typical use::

    from pointdraw.controller import Controller, SimulatedPlotter, load_config

    cfg = load_config()
    with Controller(SimulatedPlotter(cfg), cfg) as ctl:
        ctl.set_hands_clear(True)
        ctl.draw_polylines([[(50, 50), (100, 50)]])
        ctl.wait()
"""

from pointdraw.controller.api import ControllerStatus, PlotterController
from pointdraw.controller.config import MachineConfig, load_config, save_config
from pointdraw.controller.controller import Controller
from pointdraw.controller.safety import SafetyError
from pointdraw.controller.serial_link import SerialLink
from pointdraw.controller.simulator import SimulatedPlotter
from pointdraw.controller.transport import LinkError, Transport

__all__ = [
    "Controller",
    "ControllerStatus",
    "LinkError",
    "MachineConfig",
    "PlotterController",
    "SafetyError",
    "SerialLink",
    "SimulatedPlotter",
    "Transport",
    "load_config",
    "save_config",
]
