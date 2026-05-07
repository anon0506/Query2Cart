"""Laptop-specific coherence rules.

These reject constraint sets that no real laptop buyer would hold.
Imported dynamically by DomainConfig via ``coherence_module``.
"""

from typing import Any


def is_coherent(constraints: dict[str, Any]) -> bool:
    price_max = constraints.get("price_max")
    ram_min = constraints.get("ram_min_gb")
    gpu = constraints.get("gpu_type")
    weight_max = constraints.get("weight_max_kg")
    screen_min = constraints.get("screen_min_inches")
    screen_max = constraints.get("screen_max_inches")
    battery_min = constraints.get("battery_min_hours")

    if price_max and ram_min and price_max <= 500 and ram_min >= 32:
        return False

    if price_max and gpu == "dedicated" and price_max <= 600:
        return False

    if gpu == "dedicated" and weight_max and weight_max < 1.6:
        return False

    if screen_min and weight_max and screen_min >= 15.0 and weight_max <= 1.6:
        return False

    if ram_min and ram_min >= 32 and weight_max and weight_max <= 1.4:
        return False

    if battery_min and battery_min >= 10 and gpu == "dedicated":
        return False

    if screen_min and screen_max and screen_min > screen_max:
        return False

    return True
