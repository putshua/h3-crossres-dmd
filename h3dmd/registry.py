from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


class Registry:
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, Callable[..., Any]] = {}

    def register(self, name: str) -> Callable[[T], T]:
        def decorator(factory: T) -> T:
            if name in self._items:
                raise KeyError(f"duplicate {self.kind} registration: {name}")
            self._items[name] = factory  # type: ignore[assignment]
            return factory

        return decorator

    def build(self, name: str, **kwargs: Any) -> Any:
        try:
            factory = self._items[name]
        except KeyError as exc:
            raise KeyError(f"unknown {self.kind} {name!r}; choices={sorted(self._items)}") from exc
        return factory(**kwargs)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._items))


MODELS = Registry("model")
METHODS = Registry("method")
DATASETS = Registry("dataset")
PIPELINES = Registry("pipeline")
CONFIG_GROUPS = Registry("config group")
EXPERIMENTS = Registry("experiment")
TRAINERS = Registry("trainer")
