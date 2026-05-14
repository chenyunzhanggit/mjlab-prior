"""AMP-augmented velocity tracking task.

Couples the standard velocity task with an Adversarial Motion Priors
discriminator trained on reference motion clips. Style is shaped by the
discriminator; locomotion goals stay on the existing velocity reward stack.
"""
