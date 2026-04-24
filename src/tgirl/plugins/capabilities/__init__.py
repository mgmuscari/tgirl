"""tgirl purpose-built proxy modules (PRP Task 6, Y3).

Four proxy modules replace the banned ``os``/``pathlib``/``io``/``shutil``
modules with curated surfaces that expose ONLY the intended semantics of
each capability:

* :mod:`tgirl.plugins.capabilities.env_proxy` — read-only environment vars.
* :mod:`tgirl.plugins.capabilities.subprocess_proxy` — logged subprocess.run.
* :mod:`tgirl.plugins.capabilities.fs_read_proxy` — read-only filesystem.
* :mod:`tgirl.plugins.capabilities.fs_write_proxy` — read + write filesystem.

Escalation vectors (``symlink_to``, ``hardlink_to``, ``chmod``, ``chown``)
are deliberately absent from every proxy.
"""

from __future__ import annotations
