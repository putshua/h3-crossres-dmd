"""Named experiment and default-group composition, with explicit registration.

Order: experiment -> defaults (left to right) -> local keys -> CLI overrides.
All factories return fresh mappings; composing one run never mutates another.
"""

from copy import deepcopy
from h3dmd.registry import CONFIG_GROUPS, EXPERIMENTS
from . import defaults as defaults, experiments as experiments


def merge_config(base, override):
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def compose_config(raw, stack=()):
    if not isinstance(raw, dict):
        raise TypeError("configuration must be a mapping")
    local = deepcopy(raw)
    experiment = local.pop("experiment", None)
    groups = local.pop("defaults", [])
    if not isinstance(groups, list) or any(
        not isinstance(name, str) for name in groups
    ):
        raise ValueError("defaults must be a list of registered group names")
    references = [] if experiment is None else [(EXPERIMENTS, experiment)]
    references += [(CONFIG_GROUPS, name) for name in groups]
    result = {}
    for registry, name in references:
        token = (registry.kind, name)
        if token in stack:
            raise ValueError(f"cyclic configuration reference: {stack + (token,)}")
        result = merge_config(
            result, compose_config(registry.build(name), stack + (token,))
        )
    return merge_config(result, local)


def experiment_names():
    return EXPERIMENTS.names()
