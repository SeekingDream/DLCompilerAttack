"""
Mac-only shim: let ``import src`` succeed on machines without TVM.

WHY THIS EXISTS
---------------
``src/abst_cl_model.py`` and ``src/dlcl.py`` do ``import tvm`` at the very top,
and ``src/__init__.py`` pulls those in (plus detectors that drag in torchvision
/ torch._dynamo). Python runs every import line the *moment* a module is loaded,
long before any compilation code is ever *called*. On Apple Silicon TVM isn't
installed, so that import crashes and nothing under ``src/`` can be imported.

Smoke tests only init a model and run one forward pass; they never compile, so
TVM is never actually *used* -- the code merely needs the *name* ``tvm`` (and its
submodules) to resolve.

HOW
---
Rather than hand-registering a few hollow modules in ``sys.modules`` (which look
malformed to tools that walk modules -- ``inspect``, ``torch._dynamo`` -- and
make them crash), we install a *meta path finder*. When anything imports
``tvm`` or ``tvm.<anything>``, Python's own import machinery asks our finder,
which manufactures a proper stub module (with a real ``__spec__``/``__loader__``)
on demand. Because the stubs are created the official way, every introspection
tool accepts them.

It does NOT modify anything in ``src/`` (those files are frozen). Import this
module BEFORE importing ``src``. On a machine that already has TVM, ``install()``
is a no-op.
"""
import importlib.abc
import importlib.machinery
import sys
import types


class _Stub(types.ModuleType):
    """A 'yes-man' module: any non-dunder attribute is another hollow stub, and
    it is callable, so both ``from tvm import relay`` and an accidental
    ``relay.something()`` at import time resolve without crashing."""

    def __getattr__(self, name: str):
        # Do NOT fabricate dunders (``__file__``, ``__all__``, ...). Real empty
        # modules lack them; faking them confuses module-walking tools.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _Stub(f"{self.__name__}.{name}")

    def __call__(self, *args, **kwargs):
        return _Stub(f"{self.__name__}()")


class _TvmFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Answers imports of ``tvm`` and any ``tvm.*`` submodule with a stub."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "tvm" or fullname.startswith("tvm."):
            # is_package=True so that submodule imports (tvm.contrib.foo) keep
            # routing back through this finder.
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        return _Stub(spec.name)

    def exec_module(self, module):
        module.__path__ = []  # mark as a package


def install() -> None:
    """Install the TVM finder, but only if real TVM is unavailable."""
    # Safety valve: if real TVM is importable (e.g. Simin's Linux/CUDA box),
    # leave everything untouched -- this shim becomes a no-op there.
    try:
        import tvm  # noqa: F401
        return
    except ImportError:
        pass

    if not any(isinstance(f, _TvmFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _TvmFinder())


install()
