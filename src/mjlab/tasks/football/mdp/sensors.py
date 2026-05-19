"""Deprecated module — kept as an empty stub during the
``depth_pipeline_migration_todo.md`` Phase 1 refactor.

The old ``ForwardPinholeCameraPatternCfg`` was replaced by mjlab's
built-in :class:`mjlab.sensor.PinholeCameraPatternCfg` (paired with
:class:`NoisyGroupedRayCasterCameraCfg` whose ``OffsetCfg`` handles the
forward-pitch placement that ``ForwardPinholeCameraPatternCfg`` used to
hard-code). No other code in the repo imports this module anymore — feel
free to ``git rm`` it on the next housekeeping pass.
"""
