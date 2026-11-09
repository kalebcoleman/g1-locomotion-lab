"""Entry for Isaac Lab's `--external_callback g1_recovery.register.register` (used by the Isaac MCP jobs): importing the package registers the tasks."""


def register():
    import g1_recovery  # noqa: F401  (registers G1-Recovery-* tasks)
    return None
