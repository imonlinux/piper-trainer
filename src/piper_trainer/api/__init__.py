"""HTTP API for piper-trainer (design doc §4).

The API layer calls the same `piper_trainer.*` functions the CLI does; the
UI is a client of this API, never a second implementation of the pipeline.
"""
