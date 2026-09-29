"""Developer tools: offline measurement and maintenance scripts.

Not part of the shipped package. Everything here is invoked explicitly:

- ``python -m tools.replay_decisions`` -- replay a recorded fight pack through
  the strategy engine and report what it would have played.
- ``python -m tools.drop_stale_ty_ignores`` -- remove `# ty: ignore` markers that
  ty now reports as unused.
"""
