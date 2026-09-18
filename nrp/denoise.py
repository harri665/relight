"""Intel Open Image Denoise wrappers.

`GpuDenoiser` binds the OIDN 2.x C API (CUDA device) via ctypes and denoises
torch CUDA tensors in place on the GPU. Set OIDN_DIR to the unpacked release
(`oidn-2.x.x.x64.windows`); by default it is looked up in the work directory.
`Denoiser` is the CPU fallback using the `oidn` wheel (OIDN 1.4).
"""
import ctypes
import os
from pathlib import Path

import numpy as np

from common import WORK_DIR

_FORMAT_FLOAT3 = 3
_DEVICE_CUDA = 3


def _find_oidn2():
    if os.environ.get("OIDN_DIR"):
        return Path(os.environ["OIDN_DIR"]) / "bin"
    hits = sorted(WORK_DIR.glob("oidn-2*/bin"))
    return hits[-1] if hits else None


class GpuDenoiser:
    def __init__(self, albedo, normal):
        """albedo, normal: [H, W, 3] float32 CUDA tensors (guides, fixed per view)."""
        import torch
        bin_dir = _find_oidn2()
        if bin_dir is None:
            raise RuntimeError("OIDN 2.x not found (set OIDN_DIR)")
        os.add_dll_directory(str(bin_dir))
        lib = ctypes.CDLL(str(bin_dir / "OpenImageDenoise.dll"))
        vp, cp, sz = ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t
        sig = {
            "oidnNewDevice": ([ctypes.c_int], vp),
            "oidnCommitDevice": ([vp], None),
            "oidnGetDeviceError": ([vp, ctypes.POINTER(cp)], ctypes.c_int),
            "oidnNewFilter": ([vp, cp], vp),
            "oidnSetSharedFilterImage": ([vp, cp, vp, ctypes.c_int, sz, sz, sz, sz, sz], None),
            "oidnSetFilterBool": ([vp, cp, ctypes.c_bool], None),
            "oidnCommitFilter": ([vp], None),
            "oidnExecuteFilter": ([vp], None),
        }
        for name, (args, res) in sig.items():
            getattr(lib, name).argtypes = args
            getattr(lib, name).restype = res
        self.lib = lib
        self.H, self.W = albedo.shape[:2]
        self.dev = lib.oidnNewDevice(_DEVICE_CUDA)
        lib.oidnCommitDevice(self.dev)
        self._check()
        mk = lambda: torch.zeros(self.H, self.W, 3, dtype=torch.float32, device="cuda")
        self.color, self.output = mk(), mk()
        self.albedo = albedo.float().contiguous().clone()
        self.normal = normal.float().contiguous().clone()
        self.filter = lib.oidnNewFilter(self.dev, b"RT")
        for name, buf in [("color", self.color), ("albedo", self.albedo),
                          ("normal", self.normal), ("output", self.output)]:
            lib.oidnSetSharedFilterImage(self.filter, name.encode(), buf.data_ptr(), _FORMAT_FLOAT3,
                                         self.W, self.H, 0, 0, 0)
        lib.oidnSetFilterBool(self.filter, b"hdr", True)
        lib.oidnCommitFilter(self.filter)
        self._check()

    def _check(self):
        msg = ctypes.c_char_p()
        if self.lib.oidnGetDeviceError(self.dev, ctypes.byref(msg)):
            raise RuntimeError(f"OIDN: {msg.value.decode()}")

    def __call__(self, color):
        """color: [H, W, 3] CUDA tensor -> denoised copy (CUDA)."""
        import torch
        self.color.copy_(color)
        torch.cuda.current_stream().synchronize()
        self.lib.oidnExecuteFilter(self.filter)  # blocking
        return self.output.clone()


class Denoiser:
    """CPU fallback (numpy in, numpy out)."""

    def __init__(self, albedo, normal):
        import oidn
        self.oidn = oidn
        self.albedo = np.ascontiguousarray(albedo, dtype=np.float32)
        self.normal = np.ascontiguousarray(normal, dtype=np.float32)
        self.H, self.W = albedo.shape[:2]
        lib = ctypes.CDLL(os.path.join(os.path.dirname(oidn.__file__), "lib.win.x64", "OpenImageDenoise.dll"))
        self._set1b = lib.oidnSetFilter1b
        self._set1b.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_bool]
        self.device = oidn.NewDevice()
        oidn.CommitDevice(self.device)

    def __call__(self, color):
        oidn = self.oidn
        color = np.ascontiguousarray(color, dtype=np.float32)
        result = np.empty_like(color)
        f = oidn.NewFilter(self.device, "RT")
        for name, buf in [("color", color), ("albedo", self.albedo), ("normal", self.normal), ("output", result)]:
            oidn.SetSharedFilterImage(f, name, buf, oidn.FORMAT_FLOAT3, self.W, self.H)
        self._set1b(f, b"hdr", True)
        oidn.CommitFilter(f)
        oidn.ExecuteFilter(f)
        oidn.ReleaseFilter(f)
        return result


def make_denoiser(albedo, normal):
    """albedo/normal: CUDA tensors. Returns a callable CUDA tensor -> CUDA tensor."""
    import torch
    try:
        d = GpuDenoiser(albedo, normal)
        print("denoiser: OIDN 2 (CUDA)")
        return d
    except Exception as e:  # noqa: BLE001
        print(f"denoiser: GPU OIDN unavailable ({e}); falling back to CPU")
        cpu = Denoiser(albedo.cpu().numpy(), normal.cpu().numpy())
        return lambda img: torch.from_numpy(cpu(img.cpu().numpy())).to(img.device)
