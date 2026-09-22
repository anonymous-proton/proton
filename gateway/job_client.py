"""Backward-compatibility shim — real implementation in gateway/jobs/job_client.py.

The Nextflow gw executor plugin (GwTaskHandler.groovy) hardcodes the path
``/workspace/gateway/job_client.py``, so this file must remain here.
"""
import os
import sys

if __name__ == "__main__":
    _here = os.path.dirname(os.path.abspath(__file__))
    _repo_root = os.path.dirname(_here)
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)

    real_path = os.path.join(_here, "jobs", "job_client.py")
    with open(real_path) as _f:
        _code = compile(_f.read(), real_path, "exec")
    exec(_code, {"__name__": "__main__", "__file__": real_path})
