"""Motion prior distillation task (work in progress).

Distills one or more frozen TemporalCNN teacher policies (trained via
Teleopit's tracking task) into a VAE / VQ-VAE motion prior. See
``prior.md`` at the repo root for the full design.

This package is being built incrementally. See the ``teacher/`` subpackage
for the frozen teacher utilities; environment / RL plumbing is added later.
"""
