"""Pure-data builders for auxiliary model defaults; no configuration I/O."""


def auxiliary_task_defaults(timeout, *, reasoning_effort=True, **extra):
    """Build a standard auxiliary-task model block.

    reasoning_effort=False omits that key (MoA blocks configure depth per slot);
    ``extra`` keys are appended after the standard ones.
    """
    defaults = {"provider": "auto", "model": "", "base_url": "", "api_key": "", "timeout": timeout, "extra_body": {}}
    if reasoning_effort:
        defaults["reasoning_effort"] = ""
    defaults.update(extra)
    return defaults