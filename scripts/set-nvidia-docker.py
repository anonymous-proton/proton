#!/usr/bin/env python3
import json
import os
import shutil

path = "/etc/docker/daemon.json"
backup_path = "/etc/docker/daemon.json.bak"

shutil.copy(path, backup_path)

try:
    with open(path, "r") as f:
        config = json.load(f)
except json.JSONDecodeError:
    config = {}

config["default-runtime"] = "nvidia"
config.setdefault("runtimes", {})
config["runtimes"]["nvidia"] = {
    "path": "/usr/bin/nvidia-container-runtime",
    "runtimeArgs": []
}

with open(path, "w") as f:
    json.dump(config, f, indent=2)

print(f"Updated {path} with NVIDIA runtime.")
print(f"Backup saved to {backup_path}.")
print("Please restart Docker: sudo systemctl restart docker")