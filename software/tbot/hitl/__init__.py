"""Hardware-in-the-loop bench: the rig's camera looks at a monitor that a browser page drives.

The rig (Pi or laptop) runs the real pipeline and serves the target page; the page syncs its clock to the
rig's monotonic clock and draws scheduled targets; the rig measures every delay from the screen photon to
the trigger break (microphone / break-sense). Procedure: docs/HITL.md.
"""
