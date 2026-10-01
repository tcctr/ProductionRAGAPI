"""A/B experiments on /query: each API key is assigned a variant, a set of pipeline settings.

The experiment is set by the AB_EXPERIMENT environment variable, a JSON object such as
    {"name": "compare-pool",
     "variants": {"control": {"weight": 1},
                  "pool10": {"weight": 1, "compare_rerank_pool": 10}}}
A variant's other fields override the settings in TUNABLE; a variant without any (control) runs
the defaults. Without AB_EXPERIMENT there is no experiment and nothing changes.

Assignment is sticky: a key's variant comes from a hash of the experiment name and the key name,
so the same key always gets the same variant (consistent answers for one client) without storing
an assignment table. Including the experiment's name reshuffles the keys for each new experiment,
so the same clients don't always land in B. Weights set the share of keys, not of requests: with
few keys, the split of requests can be far from the weights.

Each /query request notes its experiment and variant (log line and query_log), the variant's
settings go into the cache key (app/cache.py), and ab_report.py compares the variants' latency
and error rates from query_log.
"""
import hashlib
import json
import os
from dataclasses import dataclass, field

# Settings a variant may override, passed to app.versions.retrieve() as keyword arguments.
TUNABLE = {
    "rerank_pool",          # hybrid results reranked in a plain search (app.search.RERANK_POOL)
    "compare_rerank_pool",  # the same per version in compare mode (app.versions.COMPARE_RERANK_POOL)
}


@dataclass(frozen=True)
class Variant:
    name: str
    weight: float
    settings: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class Experiment:
    name: str
    variants: tuple[Variant, ...]


def parse(config: str) -> Experiment:
    """An Experiment from AB_EXPERIMENT's JSON; raises ValueError on anything unexpected, so a
    typo stops the API at startup instead of silently running an experiment that isn't one."""
    data = json.loads(config)
    if not isinstance(data, dict) or not isinstance(data.get("name"), str) or not data["name"]:
        raise ValueError('AB_EXPERIMENT needs a "name"')
    raw = data.get("variants")
    if not isinstance(raw, dict) or len(raw) < 2:
        raise ValueError("AB_EXPERIMENT needs at least two variants")
    variants = []
    for name, spec in raw.items():
        spec = dict(spec)
        weight = spec.pop("weight", 1)
        if not isinstance(weight, (int, float)) or weight <= 0:
            raise ValueError(f"variant {name}: weight must be a positive number")
        if unknown := set(spec) - TUNABLE:
            raise ValueError(f"variant {name}: unknown settings {sorted(unknown)}; allowed: {sorted(TUNABLE)}")
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in spec.values()):
            raise ValueError(f"variant {name}: settings must be non-negative integers")
        variants.append(Variant(name, float(weight), spec))
    return Experiment(data["name"], tuple(variants))


def load() -> Experiment | None:
    config = os.getenv("AB_EXPERIMENT", "").strip()
    return parse(config) if config else None


EXPERIMENT = load()


def bucket(experiment: str, key_name: str) -> float:
    """A number in [0, 1), fixed for an experiment and key and spread evenly over keys."""
    digest = hashlib.sha256(f"{experiment}:{key_name}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2 ** 64


def assign(key_name: str, experiment: Experiment | None = None) -> Variant | None:
    """The key's variant in the experiment (default: the configured one), or None without one.

    The variants split [0, 1) in proportion to their weights, in the order they're listed;
    the key's bucket falls in one of them.
    """
    experiment = experiment or EXPERIMENT
    if experiment is None:
        return None
    point = bucket(experiment.name, key_name) * sum(v.weight for v in experiment.variants)
    for variant in experiment.variants:
        if point < variant.weight:
            return variant
        point -= variant.weight
    return experiment.variants[-1]  # float rounding at the very top
