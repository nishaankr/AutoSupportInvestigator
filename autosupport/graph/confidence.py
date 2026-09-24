"""compute_confidence() — the confidence scale's single definition (docs/design/output-schema.md
§4). Called only from the `verify` node. The formula's constants (K, W_MAX,
SIM_CEILING, factor floors, penalty step/ceiling, caps, band edges) live here as
module constants, not in config.py — changing one changes the meaning of every
stored value, so it goes through code review rather than an environment variable.
Built at CP5."""
