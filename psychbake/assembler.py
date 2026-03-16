"""Assembler — builds a complete protocol from a TherapeuticStance.

Pipeline:
  1. Validate layers (dependencies, conflicts)
  2. Collect resources from all layers
  3. Collect techniques, filter by stance
  4. Collect lens rules and parser markers
  5. Build phase sequence (macro-phases)
  6. Compile to engine protocol
  7. Return AssembledBundle
"""

from __future__ import annotations

from therapy_engine.state import (
    ResourceDef,
    PhaseDef,
    TransitionDef,
    InterventionDef,
    TriggerDef,
    CompiledProtocol,
)
from therapy_engine.effects import Boost
from therapy_engine.expr import c

from psychbake.recipe import TechniqueRecipe, ProcessRuleSpec
from psychbake.lens import LensRule, MarkerDef
from psychbake.stance import TherapeuticStance
from psychbake.bundle import AssembledBundle, WeightMeta
from psychbake.families import DEFAULT_AFFINITY


# ── Core markers (always included regardless of stance) ───────────

CORE_MARKERS = (
    MarkerDef("body_mention", "Client references bodily sensations",
              "chest tight, stomach knot, hands trembling"),
    MarkerDef("searching_language", "Client searching for words, pausing, uncertain",
              "it's like..., I don't know how to say..., maybe..."),
    MarkerDef("present_tense_self", "Client speaks about self in present tense with feeling",
              "I feel, I notice, right now I..."),
    MarkerDef("felt_shift", "Something shifts — relief, tears, insight, new sensation",
              "oh!, something just..., I feel lighter"),
    MarkerDef("insight", "Client reaches new understanding",
              "I never saw it that way, I realize..."),
    MarkerDef("emotion_expression", "Direct expression of emotion",
              "I'm angry, I feel sad, there's this grief"),
    MarkerDef("metaphor_use", "Client uses metaphor or imagery",
              "it's like a wall, I'm drowning, there's a weight"),
    MarkerDef("engagement", "Active participation, elaboration, energy",
              "and another thing, yes exactly, let me explain"),
    MarkerDef("self_reflection", "Client observes own patterns",
              "I always do this, I notice I..."),
    MarkerDef("vulnerability", "Client shows openness, risk-taking",
              "this is hard to say, I've never told anyone"),
)

# ── Core resources (always included) ──────────────────────────────

CORE_RESOURCES = (
    ResourceDef("experiencing", "core", initial=2.0, min=1.0, max=7.0),
    ResourceDef("alliance", "core", initial=3.0, min=0.0, max=7.0),
    ResourceDef("resistance", "core", initial=2.0, min=0.0, max=7.0),
    ResourceDef("crisis_signal", "core", initial=0.0, min=0.0, max=5.0),
)

TRACKING_RESOURCES = (
    ResourceDef("avoidance_count", "tracking", initial=0.0, min=0.0, max=None),
    ResourceDef("retreat_count", "tracking", initial=0.0, min=0.0, max=None),
    ResourceDef("confrontations_this_session", "tracking", initial=0.0, min=0.0, max=3.0),
    ResourceDef("questions_since_reflection", "tracking", initial=0.0, min=0.0, max=None),
    ResourceDef("contact_depth_peak", "tracking", initial=0.0, min=0.0, max=7.0),
    ResourceDef("winding_down", "tracking", initial=0.0, min=0.0, max=1.0),
)


class AssemblyError(Exception):
    """Raised on structural assembly errors."""


class Assembler:
    """Builds an AssembledBundle from a TherapeuticStance."""

    @staticmethod
    def build(stance: TherapeuticStance) -> AssembledBundle:
        """Assemble a complete protocol bundle from stance configuration."""
        warnings: list[str] = []

        # 1. Validate layers
        _validate_layers(stance.layers, warnings)

        # 2. Collect resources (core + tracking + from layers)
        resources = _collect_resources(stance.layers, warnings)

        # 3. Collect techniques (filter by stance)
        techniques = _collect_techniques(stance, warnings)

        # 4. Collect lens rules and parser markers
        lens_rules = _collect_lens_rules(stance.layers)
        parser_markers = _collect_parser_markers(stance.layers)

        # 5. Build phases (macro-phases with soft transitions)
        phases, phase_frames = _build_phases(stance, warnings)

        # 6. Collect process rules
        process_rules = _collect_process_rules(stance.layers)

        # 7. Build engine protocol
        protocol = _compile_protocol(resources, phases, techniques)

        # 8. Build weight metadata
        weight_meta = _build_weight_meta(techniques)

        # 9. Build instructions
        instructions = {t.id: t.instruction for t in techniques if t.instruction}

        # 10. Family affinity (stance config + defaults)
        family_affinity = dict(DEFAULT_AFFINITY)
        family_affinity.update(stance.weight_config.family_affinity)

        # 11. Phase defaults
        phase_defaults = _build_phase_defaults(techniques)

        return AssembledBundle(
            id=stance.id,
            protocol=protocol,
            weight_meta=weight_meta,
            phase_frames=phase_frames,
            instructions=instructions,
            system_persona=stance.system_persona,
            lens_rules=tuple(lens_rules),
            parser_markers=tuple(CORE_MARKERS + tuple(parser_markers)),
            process_rules=tuple(process_rules),
            weight_config=stance.weight_config,
            session_config=stance.session_config,
            family_affinity=family_affinity,
            phase_defaults=phase_defaults,
            warnings=warnings,
            layer_ids=tuple(layer.id for layer in stance.layers),
        )


# ── Internal helpers ──────────────────────────────────────────────


def _validate_layers(layers: list, warnings: list[str]) -> None:
    """Check dependencies and conflicts between layers."""
    layer_ids = {layer.id for layer in layers}

    for layer in layers:
        for dep in layer.depends_on:
            if dep not in layer_ids:
                raise AssemblyError(
                    f"Layer '{layer.id}' depends on '{dep}' which is not included"
                )
        for conflict in layer.conflicts_with:
            if conflict in layer_ids:
                raise AssemblyError(
                    f"Layer '{layer.id}' conflicts with '{conflict}' "
                    f"— both are included"
                )


def _collect_resources(
    layers: list,
    warnings: list[str],
) -> dict[str, ResourceDef]:
    """Collect all resources: core + tracking + layer-specific."""
    resources: dict[str, ResourceDef] = {}

    # Core always included
    for rdef in CORE_RESOURCES:
        resources[rdef.name] = rdef
    for rdef in TRACKING_RESOURCES:
        resources[rdef.name] = rdef

    # From layers
    for layer in layers:
        for rdef in layer.resources:
            if rdef.name in resources:
                existing = resources[rdef.name]
                if existing.domain != rdef.domain:
                    raise AssemblyError(
                        f"Resource name conflict: '{rdef.name}' defined in "
                        f"domain '{existing.domain}' and '{rdef.domain}'"
                    )
                # Same domain, same name — skip duplicate
                continue
            resources[rdef.name] = rdef

    return resources


def _collect_techniques(
    stance: TherapeuticStance,
    warnings: list[str],
) -> list[TechniqueRecipe]:
    """Collect techniques from all layers, apply stance filters."""
    techniques: list[TechniqueRecipe] = []
    seen_ids: set[str] = set()

    for layer in stance.layers:
        for tech in layer.techniques:
            # Stance exclusions
            if tech.id in stance.exclude_techniques:
                continue
            if tech.family in stance.exclude_families:
                continue

            # Dedup (higher-priority layer wins)
            if tech.id in seen_ids:
                warnings.append(
                    f"Technique '{tech.id}' defined in multiple layers — "
                    f"using version from '{layer.id}'"
                )
                # Replace with higher-priority version
                techniques = [t for t in techniques if t.id != tech.id]

            techniques.append(tech)
            seen_ids.add(tech.id)

    if not techniques:
        raise AssemblyError("No techniques after filtering — empty protocol")

    return techniques


def _collect_lens_rules(layers: list) -> list[LensRule]:
    """Collect all lens rules from layers."""
    rules: list[LensRule] = []
    for layer in layers:
        rules.extend(layer.lens_rules)
    return rules


def _collect_parser_markers(layers: list) -> list[MarkerDef]:
    """Collect unique parser markers from layers."""
    markers: list[MarkerDef] = []
    seen: set[str] = set()
    for layer in layers:
        for m in layer.parser_markers:
            if m.id not in seen:
                markers.append(m)
                seen.add(m.id)
    return markers


def _collect_process_rules(layers: list) -> list[ProcessRuleSpec]:
    """Collect process rules from layers (higher priority wins on ID conflict)."""
    rules: dict[str, ProcessRuleSpec] = {}
    for layer in layers:
        for rule in layer.process_rules:
            rules[rule.id] = rule  # later = higher priority
    return list(rules.values())


def _build_phases(
    stance: TherapeuticStance,
    warnings: list[str],
) -> tuple[tuple[PhaseDef, ...], dict[str, str]]:
    """Build macro-phase definitions with soft transitions."""
    phase_frames: dict[str, str] = {}

    # Collect phase frames from layers
    for layer in stance.layers:
        for phase_name, frame in layer.phase_frames.items():
            if phase_name not in phase_frames:
                phase_frames[phase_name] = frame
            else:
                # Append layer's frame context
                phase_frames[phase_name] += "\n\n" + frame

    sc = stance.session_config
    phases = (
        PhaseDef(
            name="opening",
            transitions=(
                TransitionDef(
                    guard=c.alliance >= 4,
                    target="working",
                ),
            ),
            frame=phase_frames.get("opening", ""),
        ),
        PhaseDef(
            name="working",
            transitions=(
                TransitionDef(
                    guard=c.winding_down >= 1,
                    target="closing",
                ),
            ),
            frame=phase_frames.get("working", ""),
        ),
        PhaseDef(
            name="closing",
            transitions=(),
            frame=phase_frames.get("closing", ""),
            terminal=True,
            terminal_after=sc.closing_turns,
        ),
        PhaseDef(
            name="crisis",
            transitions=(
                TransitionDef(
                    guard=c.crisis_signal <= 2,
                    target="working",
                ),
            ),
            frame=phase_frames.get("crisis",
                "CRISIS MODE. Client safety is the absolute priority. "
                "Use only safety-oriented techniques. Do not explore or deepen."),
        ),
    )

    return phases, phase_frames


def _compile_protocol(
    resources: dict[str, ResourceDef],
    phases: tuple[PhaseDef, ...],
    techniques: list[TechniqueRecipe],
) -> CompiledProtocol:
    """Compile collected data into a CompiledProtocol."""
    interventions: dict[str, InterventionDef] = {}
    triggers: list[TriggerDef] = []

    for tech in techniques:
        interventions[tech.id] = InterventionDef(
            id=tech.id,
            guard=tech.guard,
            effects=tech.effects,
            doc=tech.doc,
        )

        # Generate outcome triggers
        if tech.on_accepted:
            triggers.append(TriggerDef(
                event=tech.on_accepted.event,
                effects=tech.on_accepted.effects or (Boost("experiencing", 0.5),),
                doc=tech.on_accepted.description,
            ))
        if tech.on_rejected:
            triggers.append(TriggerDef(
                event=tech.on_rejected.event,
                effects=tech.on_rejected.effects or (Boost("resistance", 0.5),),
                doc=tech.on_rejected.description,
            ))

    return CompiledProtocol(
        resources=resources,
        phases=phases,
        initial_phase="opening",
        interventions=interventions,
        triggers=triggers,
    )


def _build_weight_meta(techniques: list[TechniqueRecipe]) -> dict[str, WeightMeta]:
    """Extract weight metadata from technique recipes."""
    return {
        tech.id: WeightMeta(
            base_weight=tech.base_weight,
            responds_to=tech.responds_to,
            evidence_keys=tech.evidence_keys,
            min_depth=tech.min_depth,
            max_depth=tech.max_depth,
            preferred_next=tech.preferred_next,
            family=tech.family,
            phase_affinity=tech.phase_affinity,
            depth_range=tech.depth_range,
        )
        for tech in techniques
    }


def _build_phase_defaults(techniques: list[TechniqueRecipe]) -> dict[str, str]:
    """Pick highest-weight technique per macro-phase as fallback."""
    defaults: dict[str, tuple[str, float]] = {}
    for tech in techniques:
        for phase, affinity in tech.phase_affinity.items():
            score = tech.base_weight * affinity
            if phase not in defaults or score > defaults[phase][1]:
                defaults[phase] = (tech.id, score)
    return {phase: tid for phase, (tid, _) in defaults.items()}
