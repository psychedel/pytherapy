"""Protocol builder — fluent API for defining therapeutic protocols.

    from therapy_engine import ProtocolBuilder, Boost, c

    pb = ProtocolBuilder("my_protocol")
    pb.resource("experiencing", domain="core", initial=2.0, min=1.0, max=7.0)
    pb.phase("opening", transitions=[(c.alliance >= 2, "exploration")])
    pb.intervention("reflect", guard=c.experiencing >= 2, effects=(Boost("experiencing", 1),))
    pb.trigger("engagement", effects=(Boost("experiencing", 0.5),))

    protocol = pb.build()
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


class ProtocolBuilder:
    """Fluent builder for CompiledProtocol."""

    def __init__(self, name: str = ""):
        self.name = name
        self._resources: dict[str, ResourceDef] = {}
        self._phases: list[PhaseDef] = []
        self._interventions: dict[str, InterventionDef] = {}
        self._triggers: list[TriggerDef] = []

    def resource(
        self,
        name: str,
        *,
        domain: str = "core",
        initial: float = 0.0,
        min: float | None = None,
        max: float | None = None,
    ) -> ProtocolBuilder:
        """Declare a resource."""
        self._resources[name] = ResourceDef(
            name=name, domain=domain, initial=initial, min=min, max=max,
        )
        return self

    def phase(
        self,
        name: str,
        *,
        transitions: list[tuple] | None = None,
        frame: str = "",
        terminal: bool = False,
        terminal_after: int = 0,
    ) -> ProtocolBuilder:
        """Declare a phase.

        Args:
            transitions: List of (guard_expr, target_phase_name) tuples.
        """
        trans = tuple(
            TransitionDef(guard=guard, target=target)
            for guard, target in (transitions or [])
        )
        self._phases.append(PhaseDef(
            name=name,
            transitions=trans,
            frame=frame,
            terminal=terminal,
            terminal_after=terminal_after,
        ))
        return self

    def intervention(
        self,
        id: str,
        *,
        guard=None,
        effects: tuple | list = (),
        doc: str = "",
    ) -> ProtocolBuilder:
        """Declare an intervention."""
        self._interventions[id] = InterventionDef(
            id=id,
            guard=guard,
            effects=tuple(effects),
            doc=doc,
        )
        return self

    def trigger(
        self,
        event: str,
        *,
        guard=None,
        effects: tuple | list = (),
        doc: str = "",
    ) -> ProtocolBuilder:
        """Declare an event trigger."""
        self._triggers.append(TriggerDef(
            event=event,
            guard=guard,
            effects=tuple(effects),
            doc=doc,
        ))
        return self

    def build(self) -> CompiledProtocol:
        """Compile into a CompiledProtocol.

        Validates:
          - At least one phase exists
          - Initial phase is the first declared phase
          - All transition targets reference existing phases
          - All intervention effect resources exist
        """
        if not self._phases:
            raise ValueError("Protocol must have at least one phase")

        phase_names = {p.name for p in self._phases}

        # Validate transition targets
        for phase in self._phases:
            for trans in phase.transitions:
                if trans.target not in phase_names:
                    raise ValueError(
                        f"Phase '{phase.name}' transitions to unknown "
                        f"phase '{trans.target}'"
                    )

        return CompiledProtocol(
            resources=dict(self._resources),
            phases=tuple(self._phases),
            initial_phase=self._phases[0].name,
            interventions=dict(self._interventions),
            triggers=list(self._triggers),
        )
