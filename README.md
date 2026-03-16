# pytherapy

> **Early-stage research experiment. Not a medical device, not a therapeutic tool, not validated for clinical use. Do not use with real clients or patients. The authors assume no responsibility for any outcomes resulting from the use of this software. If you or someone you know needs mental health support, please contact a licensed professional.**

**Affordance engine for therapeutic protocols.**

Not an AI therapist — a deterministic runtime that manages therapeutic process state, filters available interventions through guard expressions and phase transitions, and constrains LLM choice space via a weight function. The LLM decides *how* to say something; the engine decides *what* is appropriate to say.

## How it works

```
Client text
    │
    ▼
┌─────────────────────────────┐
│  Parser (LLM)               │
│  Observes, doesn't diagnose │
│  → signals, markers, crisis │
└────────────┬────────────────┘
             ▼
┌─────────────────────────────┐
│  Engine (deterministic)     │
│  Update resources           │
│  Evaluate guards            │
│  Fire triggers              │
│  Advance phase              │
│  Compute weights            │
│  → ranked interventions     │
└────────────┬────────────────┘
             ▼
┌─────────────────────────────┐
│  Formulator (LLM)           │
│  Chosen intervention        │
│  → therapist response       │
└─────────────────────────────┘
```

Alternative: **unified mode** — single LLM call that assesses, picks from engine-built menu, and formulates. Engine validates the pick post-hoc.

## Guards: ethics as code

Interventions have guard expressions — boolean constraints evaluated against session state. LLM cannot bypass them.

```python
confrontation_soft:
    guard = (alliance >= 4) & (experiencing >= 3) & (confrontations_this_session < 3)

radical_acceptance:
    guard = experiencing >= 2

exile_work:    # IFS
    guard = protector_permission == 1
```

Crisis overrides are hard: when `crisis_signal >= 4`, only safety techniques are available. No prompt engineering can circumvent this.

## Weight function

For each available intervention:

```
weight = base_weight
       × marker_bonus        # parser-detected markers this technique responds to
       × evidence_bonus       # accumulated session counters
       × recency_decay        # penalty for recent use
       × phase_affinity       # opening/working/closing fit
       × context_multiplier   # depth range, crisis gradient
```

Marker specificity: rare markers (fewer techniques respond to them) get a stronger bonus. This rewards school-specific signal detection over generic patterns.

## Composable layers (PsychBake)

Therapeutic approaches are defined as **layers** that stack on a crisis safety base. A **stance** combines layers; the **assembler** compiles them into a single executable protocol.

```python
stance = TherapeuticStance(
    id="existential_buddhist",
    layers=[crisis, humanistic, existential, buddhist],
)
bundle = Assembler.build(stance)
```

Each layer contributes:
- **Techniques** with guards, effects, weight metadata, LLM instructions
- **Lens rules** — how to interpret parser markers into resource updates
- **Parser markers** — school-specific signals to detect
- **Phase frames** — school-specific therapeutic framing per phase

Same engine, same session lifecycle — completely different therapeutic voice.

### Available layers

| Layer | Techniques | Tradition |
|-------|-----------|-----------|
| crisis | 3 | Safety baseline (always active) |
| humanistic | 7 | Rogers, person-centered core |
| existential | 4 | Bugental, Yalom |
| focusing | 6 | Gendlin |
| buddhist | 6 | Buddhist psychology |
| gestalt | 6 | Perls, awareness continuum |
| ifs | 5 | Schwartz, parts work |
| somatic | 6 | Body-oriented, Levine |

## Session lifecycle

1. **Opening** — alliance building, resource initialization
2. **Turn loop** — parser → engine → formulator, repeat
3. **Working** — technique availability expands as alliance strengthens and experiencing deepens
4. **Closing** — integration, meaning-making

Process detectors fire automatically: retreat detection, avoidance tracking, winding-down recognition. Evidence accumulates in guards — e.g., `confrontation_soft` only available after `avoidance_count >= 2`.

**Between sessions**: `ClientProfile` stores baseline resource values (EMA), recurring topics, avoidance log, breakthroughs, and affordance history — enabling stronger work over time.

## Why this architecture

| Problem with LLM therapy | Solution |
|--------------------------|----------|
| Sycophancy / stuck on validation | Anti-stagnation weight penalty |
| Superficiality | Guards prevent deepening before readiness |
| No process tracking | Triggers track avoidance, retreat, breakthrough mechanically |
| No memory across sessions | ClientProfile with EMA baselines |
| Ethics rely on prompts | Ethics in guards — code, not text |
| One-size-fits-all | Composable stances from school-specific layers |

## Quick start

```bash
uv sync

# Run tests (no API key needed)
uv run python -m pytest tests/ -x --ignore=tests/calibration.py -k "not llm"

# Run simulation — deterministic stance × scenario comparison
uv run python -m tests.simulation
```

For LLM-integrated sessions, set `ANTHROPIC_API_KEY`:

```bash
uv run python cli.py live
```

## Requirements

- Python >= 3.13
- [attrs](https://www.attrs.org/) >= 24.2.0
- [anthropic](https://github.com/anthropics/anthropic-sdk-python) >= 0.40.0 (LLM integration)

## Status

This is an early research experiment exploring how deterministic state machines can constrain LLM behavior in safety-critical domains. The therapeutic knowledge encoded in layers, guards, and weight functions reflects published literature but has not been validated in clinical settings. The engine's output — ranked intervention suggestions — is not clinical advice.

## License

[AGPL-3.0](LICENSE)
