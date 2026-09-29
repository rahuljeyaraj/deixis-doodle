"""Make sure the right ONNX Runtime DLL is loaded before sherpa-onnx.

Windows 11 ships an old `C:\\Windows\\System32\\onnxruntime.dll` (1.17 on this machine). The
sherpa-onnx wheel doesn't bundle its own, so without help it binds to that one and fails with
"The requested API version [28] is not available". Adding the `onnxruntime` pip package's DLL
folder to the search path first makes sherpa-onnx load the newer DLL instead.
"""

from __future__ import annotations

import os
import sys
from types import ModuleType


def import_sherpa() -> ModuleType:
    """Import and return `sherpa_onnx`, preferring the pip onnxruntime DLL on Windows."""
    if sys.platform == "win32" and "sherpa_onnx" not in sys.modules:
        import onnxruntime

        capi = os.path.join(os.path.dirname(onnxruntime.__file__), "capi")
        if os.path.isdir(capi):
            os.add_dll_directory(capi)
    import sherpa_onnx

    return sherpa_onnx
