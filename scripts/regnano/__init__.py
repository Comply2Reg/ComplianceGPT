"""RegNano: verifiable-reward task synthesis and RL for a compliance SLM.

Modelled on Microsoft FrogNano-4B (RL on Qwen3.5-4B over self-calibrated
synthetic tasks), with software tests replaced by deterministic verifiers
over openly-licensed (GREEN) regulatory text:

    verifiers.py   reward functions, one per task family — the "hidden tests"
    (author.py, validate.py, calibrate.py follow: author model writes a task
    and a gold answer, the gold must pass its own verifier, and the current
    policy's pass rate decides whether the task is kept)

The author model's gold answer is a checker, never a training target — the
point is to train on what can be verified rather than on another model's
labels, whose two labellers agree on only 0.702 of alert classes.
"""
