"""Test-suite isolation: never pick up a rig's config/local.toml (bench tools write it on the Pi)."""

import os

os.environ["TBOT_LOCAL_CONFIG"] = ""
