"""Compatibility shim for weight_norm removal across PyTorch versions.

The codebase imports :func:`torch.nn.utils.parametrizations.weight_norm`
(the new-style, post-2.0 weight_norm that lives under the
:mod:`torch.nn.utils.parametrize` mechanism).  But several files also call
:func:`torch.nn.utils.remove_weight_norm`, which is the *old-style* removal
function — and depending on the PyTorch version it may raise

    ValueError: weight_norm of 'weight' not found in ...

because the module is parametrized rather than hook-based.

This module provides a single :func:`remove_weight_norm` that handles both
styles so the existing call sites don't have to care which PyTorch version is
installed.  It is a no-op on modules that were never weight-normed, which is
also the original behaviour we want when ``remove_weight_norm`` is called
after a torch.jit export pass that already stripped the hooks.
"""

import torch.nn as nn
import torch.nn.utils.parametrize as P


def remove_weight_norm(module: nn.Module, name: str = "weight") -> nn.Module:
    """Remove weight_norm from ``module``, supporting both old and new styles.

    Parameters
    ----------
    module : nn.Module
        The weight-normed module (typically a ``Conv1d`` / ``ConvTranspose1d``
        / ``Linear`` wrapped with
        :func:`torch.nn.utils.parametrizations.weight_norm`).
    name : str
        The name of the parameter that was weight-normed.  Defaults to
        ``"weight"`` which matches the default of the original weight_norm.

    Returns
    -------
    nn.Module
        The same ``module`` (with the weight-norm parametrization removed so
        the underlying ``weight`` tensor is now a plain parameter).
    """
    # New-style: weight_norm was applied via parametrizations.
    if hasattr(module, "parametrizations") and name in module.parametrizations:
        try:
            P.remove_parametrizations(module, name)
            return module
        except Exception:
            pass

    # Old-style: weight_norm was applied via a forward-pre-hook.
    try:
        nn.utils.remove_weight_norm(module, name=name)
    except (ValueError, AttributeError, KeyError):
        # The module was never weight-normed (or already had it removed);
        # silently no-op so callers can sweep a whole ModuleList without
        # caring which members were normed.
        pass

    return module
