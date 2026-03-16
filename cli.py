"""CLI session runner — works without API key.

Instead of making LLM API calls, this runner accepts pre-computed
signals from an external source (e.g. Claude Code acting as both
parser and formulator).

Usage:
    # Start a new session (reads client profile if available)
    uv run python cli.py new-session [--client CLIENT_ID] [--stance STANCE_ID]

    # Process a turn with pre-parsed signals (JSON)
    uv run python cli.py turn '{"experiencing": 3}' --topic work_stress --client-text "My boss is terrible"

    # Show current state
    uv run python cli.py status

    # Show available interventions with weights
    uv run python cli.py deals

    # Execute a specific intervention
    uv run python cli.py execute reflection_simple --therapist-text "It sounds like..."

    # End session — transfer to profile, archive
    uv run python cli.py end-session [--note "Session summary"]

    # Interactive session (requires ANTHROPIC_API_KEY)
    uv run python cli.py live [--client CLIENT_ID] [--stance STANCE_ID]
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

from session import Session
from client_profile import ClientProfile
from weights import compute_weights

# Persistent state file
STATE_FILE = Path(__file__).parent / ".session_state.json"
DATA_DIR = Path(__file__).parent / "data"


def _get_default_bundle():
    """Build the default stance bundle."""
    from psychbake.assembler import Assembler
    from psychbake.stance import TherapeuticStance
    from psychbake.layers.crisis import layer as crisis
    from psychbake.layers.humanistic import layer as humanistic
    from psychbake.layers.existential import layer as existential
    from psychbake.layers.buddhist import layer as buddhist

    stance = TherapeuticStance(
        id="existential_buddhist",
        layers=[crisis, humanistic, existential, buddhist],
        system_persona=(
            "A warm, existentially-grounded therapist drawing on "
            "Buddhist psychology. Present, compassionate, attuned to "
            "the client's experiencing."
        ),
    )
    return Assembler.build(stance)


def _get_bundle(stance_id: str | None = None):
    """Build a stance bundle by ID, or return default."""
    if stance_id is None or stance_id == "existential_buddhist":
        return _get_default_bundle()

    # Import all available layers
    from psychbake.assembler import Assembler
    from psychbake.stance import TherapeuticStance
    from psychbake.layers.crisis import layer as crisis
    from psychbake.layers.humanistic import layer as humanistic

    # Known stances
    stances = {
        "humanistic": ([crisis, humanistic], "A warm, person-centered therapist."),
    }

    # Try loading stance-specific layers dynamically
    try:
        from psychbake.layers.existential import layer as existential
        stances["existential"] = (
            [crisis, humanistic, existential],
            "An existentially-grounded therapist.",
        )
    except ImportError:
        pass

    try:
        from psychbake.layers.focusing import layer as focusing
        stances["focusing"] = (
            [crisis, humanistic, focusing],
            "A Gendlin focusing-oriented therapist.",
        )
    except ImportError:
        pass

    try:
        from psychbake.layers.gestalt import layer as gestalt
        stances["gestalt"] = (
            [crisis, humanistic, gestalt],
            "A Gestalt awareness-oriented therapist.",
        )
    except ImportError:
        pass

    try:
        from psychbake.layers.ifs import layer as ifs
        stances["ifs"] = (
            [crisis, humanistic, ifs],
            "An IFS parts-work therapist.",
        )
    except ImportError:
        pass

    try:
        from psychbake.layers.somatic import layer as somatic
        stances["somatic"] = (
            [crisis, humanistic, somatic],
            "A somatic/body-oriented therapist.",
        )
    except ImportError:
        pass

    try:
        from psychbake.layers.buddhist import layer as buddhist
        stances["buddhist"] = (
            [crisis, humanistic, buddhist],
            "A Buddhist psychology therapist.",
        )
    except ImportError:
        pass

    if stance_id not in stances:
        available = ", ".join(sorted(stances.keys()))
        print(f"Unknown stance: {stance_id}. Available: {available}, existential_buddhist")
        sys.exit(1)

    layers, persona = stances[stance_id]
    stance = TherapeuticStance(id=stance_id, layers=layers, system_persona=persona)
    return Assembler.build(stance)


# -- State persistence --


def _save_session(
    session: Session,
    *,
    session_id: str = "",
    client_id: str = "default",
    stance_id: str = "",
    conversation: list[dict] | None = None,
    turn_log: list[dict] | None = None,
) -> None:
    """Serialize session state to disk."""
    import attrs

    existing = {}
    if STATE_FILE.exists():
        try:
            existing = json.loads(STATE_FILE.read_text())
        except json.JSONDecodeError:
            pass

    data = {
        "session_id": session_id or existing.get("session_id", ""),
        "client_id": client_id or existing.get("client_id", "default"),
        "stance_id": stance_id or existing.get("stance_id", "existential_buddhist"),
        "started_at": existing.get("started_at", datetime.now().isoformat()),
        "turn_number": session.turn_number,
        "deal_history": session.deal_history,
        "arc_values": session.arc.values,
        "topic_history": session.topic_history,
        "conversation": conversation if conversation is not None else existing.get("conversation", []),
        "turn_log": turn_log if turn_log is not None else existing.get("turn_log", []),
        "state": {
            "phase": session.phase,
            "resources": dict(session.state.resources),
            "turn": session.state.turn,
        },
        "presence_ema": session._presence_ema,
        "presence_mode": session.presence_mode,
        "integration_entry_turn": session._integration_entry_turn,
    }
    STATE_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def _load_session() -> tuple[Session, dict]:
    """Restore session from disk. Returns (session, raw_data)."""
    import attrs

    if not STATE_FILE.exists():
        print("No active session. Run: uv run python cli.py new-session")
        sys.exit(1)

    data = json.loads(STATE_FILE.read_text())
    stance_id = data.get("stance_id", "existential_buddhist")
    bundle = _get_bundle(stance_id)

    session = Session(bundle)
    session.start()

    # Restore resources
    for name, value in data["state"]["resources"].items():
        session._set_resource(name, float(value))

    # Restore phase
    target_phase = data["state"]["phase"]
    if session.phase != target_phase:
        phases = list(session.protocol.phases.keys())
        if target_phase in phases:
            session.state = attrs.evolve(session.state, phase=target_phase)

    # Restore metadata
    session.turn_number = data["turn_number"]
    session.deal_history = data["deal_history"]
    session.arc.values = data["arc_values"]
    session.topic_history = data.get("topic_history", [])
    session._presence_ema = data.get("presence_ema", 0.0)
    session._integration_entry_turn = data.get("integration_entry_turn")

    return session, data


def _get_profile_path(client_id: str = "default") -> Path:
    return DATA_DIR / client_id / "client_profile.json"


def _print_state(session: Session) -> None:
    summary = session.state_summary()
    print(json.dumps(summary, indent=2, ensure_ascii=False))


# -- Commands --


def cmd_new_session(
    client_id: str = "default",
    stance_id: str | None = None,
    profile_path: str | None = None,
):
    """Start a new session with client profile context."""
    bundle = _get_bundle(stance_id)

    # Load profile
    path = Path(profile_path) if profile_path else _get_profile_path(client_id)
    profile = ClientProfile.load(path)
    profile.client_id = client_id

    session = Session(bundle)
    session.start()

    # Apply baselines from profile
    for resource_key, baseline_key in [
        ("alliance", "alliance"),
        ("experiencing", "experiencing"),
        ("resistance", "resistance"),
    ]:
        if baseline_key in profile.baselines:
            session._set_resource(resource_key, float(profile.baselines[baseline_key]))

    session_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    _save_session(
        session,
        session_id=session_id,
        client_id=client_id,
        stance_id=stance_id or "existential_buddhist",
        conversation=[],
        turn_log=[],
    )

    print(f"Session started: {session_id}")
    print(f"Client: {client_id} (session #{profile.session_count + 1})")
    print(f"Stance: {stance_id or 'existential_buddhist'}")

    if profile.session_count > 0:
        ctx = profile.context_summary()
        print(f"\nBaselines: {ctx['baselines']}")
        if ctx["recurring_themes"]:
            print(f"Recurring themes: {ctx['recurring_themes']}")

    print(f"\nState:")
    _print_state(session)
    print(f"\nAvailable interventions: {session.available_interventions()}")


def cmd_status():
    session, data = _load_session()
    _print_state(session)
    conv = data.get("conversation", [])
    if conv:
        print(f"\nConversation: {len(conv)} entries")


def cmd_deals(markers_json: str | None = None):
    session, data = _load_session()
    markers = json.loads(markers_json) if markers_json else None

    available = session.available_interventions()
    weighted, trace = compute_weights(
        available, session.state,
        detected_markers=markers,
        weight_meta=session.bundle.weight_meta,
        family_affinity=session.bundle.family_affinity,
        weight_config=session.bundle.weight_config,
        deal_history=session.deal_history,
    )
    session.last_weight_trace = trace

    print(f"Phase: {session.phase}")
    print(f"Available interventions (weighted):")
    for tid, weight in weighted:
        print(f"  {weight:5.2f}  {tid}")

    if trace:
        print(f"\nEngine reasoning: {trace.to_formulator_hint()}")


def cmd_execute(deal_id: str, therapist_text: str | None = None):
    session, data = _load_session()
    ok = session.execute(deal_id)
    if ok:
        print(f"Executed: {deal_id}")

        conversation = data.get("conversation", [])
        if therapist_text:
            conversation.append({
                "role": "therapist",
                "text": therapist_text,
                "turn": session.turn_number,
                "deal_id": deal_id,
            })

        _save_session(
            session,
            client_id=data.get("client_id", "default"),
            conversation=conversation,
        )
        _print_state(session)
    else:
        print(f"BLOCKED: {deal_id} — guard conditions not met")
        print(f"Available: {session.available_interventions()}")


def cmd_turn(
    signals_json: str,
    markers_json: str | None = None,
    triggers_str: str | None = None,
    topic: str | None = None,
    client_text: str | None = None,
):
    session, data = _load_session()

    signals = json.loads(signals_json) if signals_json else {}
    markers = json.loads(markers_json) if markers_json else None
    triggers = triggers_str.split(",") if triggers_str else None

    conversation = data.get("conversation", [])
    if client_text:
        conversation.append({
            "role": "client",
            "text": client_text,
            "turn": session.turn_number + 1,
        })

    old_phase = session.phase
    weighted = session.turn(
        signals=signals,
        markers=markers,
        triggers=triggers,
        topic=topic,
    )

    turn_log = data.get("turn_log", [])
    turn_log.append({
        "turn": session.turn_number,
        "signals": signals,
        "markers": markers or {},
        "triggers": triggers or [],
        "topic": topic or "",
        "phase": session.phase,
        "experiencing": session.get_resource("experiencing"),
        "available_count": len(session.available_interventions()),
    })

    _save_session(
        session,
        client_id=data.get("client_id", "default"),
        conversation=conversation,
        turn_log=turn_log,
    )

    if session.phase != old_phase:
        print(f"Phase: {old_phase} -> {session.phase}")
    else:
        print(f"Phase: {session.phase}")

    print(f"\nWeighted interventions:")
    for tid, weight in weighted:
        print(f"  {weight:5.2f}  {tid}")

    if session.last_weight_trace:
        print(f"\nEngine reasoning: {session.last_weight_trace.to_formulator_hint()}")

    if session.last_process_result and session.last_process_result.events:
        print(f"\nProcess events:")
        for event in session.last_process_result.events:
            print(f"  - {event}")

    print(f"\nState:")
    _print_state(session)


def cmd_end_session(note: str = ""):
    """End the current session: transfer to profile, archive, clear."""
    session, data = _load_session()
    client_id = data.get("client_id", "default")
    session_id = data.get("session_id", "unknown")

    turn_log = data.get("turn_log", [])
    session_data = {
        "session_id": session_id,
        "final_resources": {
            "alliance": session.get_resource("alliance"),
            "experiencing": session.get_resource("experiencing"),
            "resistance": session.get_resource("resistance"),
        },
        "topic_history": session.topic_history,
        "turn_log": turn_log,
        "arc_peak": session.arc.peak,
        "arc_values": session.arc.values,
        "deal_history": session.deal_history,
        "turn_count": session.turn_number,
        "note": note,
    }

    profile_path = _get_profile_path(client_id)
    profile = ClientProfile.load(profile_path)
    profile.client_id = client_id
    profile.update_from_session(session_data)
    profile.save(profile_path)

    archive_dir = DATA_DIR / client_id / "sessions" / session_id
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / "session_state.json"
    shutil.copy2(STATE_FILE, archive_path)

    STATE_FILE.unlink(missing_ok=True)

    print(f"Session ended: {session_id}")
    print(f"Turns: {session.turn_number}")
    print(f"Phase reached: {session.phase}")
    print(f"Experiencing peak: {session.arc.peak}")
    print(f"Interventions used: {session.deal_history}")
    if note:
        print(f"Note: {note}")
    print(f"\nProfile updated: {profile_path}")
    print(f"Session archived: {archive_path}")


def cmd_context(client_id: str = "default"):
    """Show context for Claude Code consumption."""
    if STATE_FILE.exists():
        session, data = _load_session()
        available = session.available_interventions()
        weighted, trace = compute_weights(
            available, session.state,
            weight_meta=session.bundle.weight_meta,
            family_affinity=session.bundle.family_affinity,
            weight_config=session.bundle.weight_config,
            deal_history=session.deal_history,
        )
        trace_hint = trace.to_formulator_hint() if trace else None
        output = {
            "active_session": True,
            "session_id": data.get("session_id", ""),
            "client_id": data.get("client_id", "default"),
            "turn_number": session.turn_number,
            "phase": session.phase,
            "state": session.state_summary(),
            "conversation": data.get("conversation", []),
            "topic_history": session.topic_history,
            "deal_history": session.deal_history,
            "available_interventions": [d for d, _ in weighted],
            "weighted_interventions": [[d, w] for d, w in weighted],
            "engine_reasoning": trace_hint,
        }
    else:
        profile_path = _get_profile_path(client_id)
        profile = ClientProfile.load(profile_path)
        output = {
            "active_session": False,
            "client_id": client_id,
            "profile": profile.context_summary(),
        }

    print(json.dumps(output, indent=2, ensure_ascii=False))


def cmd_live(
    client_id: str = "default",
    mode: str = "unified",
    stance_id: str | None = None,
):
    """Interactive therapy session — uses LLM (requires ANTHROPIC_API_KEY)."""
    from bridge import TherapySession

    bundle = _get_bundle(stance_id)
    profile_path = _get_profile_path(client_id)
    profile = ClientProfile.load(profile_path)
    profile.client_id = client_id

    session = TherapySession(
        bundle=bundle,
        mode=mode,
        profile=profile,
    )
    state = session.start()

    print(f"Live session ({mode} mode) — {stance_id or 'existential_buddhist'}")
    print(f"Client: {client_id} (session #{profile.session_count + 1})")
    print(f"Phase: {state.get('phase', '?')}")
    print("Type 'quit' to end session.\n")

    while not session.session.ended:
        try:
            text = input("Client> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text.lower() in ("quit", "exit", "q"):
            break

        result = session.process(text)
        print(f"\nTherapist: {result.therapist_text}")
        print(f"  [{result.intervention_id} | {result.phase}"
              f" | exp={result.state_summary.get('experiencing', '?')}]")
        print()

    data = session.end_session(note="live session")
    if data:
        profile.save(profile_path)
        print(f"\nSession ended. Turns: {data.get('turn_count', '?')}")
        print(f"Profile saved: {profile_path}")


# -- Argument parsing --


def _parse_kv_args(args: list[str], start: int) -> dict[str, str | None]:
    """Parse --key value pairs from argv."""
    result: dict[str, str | None] = {}
    i = start
    while i < len(args):
        if args[i].startswith("--") and i + 1 < len(args):
            key = args[i][2:]
            result[key] = args[i + 1]
            i += 2
        else:
            i += 1
    return result


def main():
    if len(sys.argv) < 2:
        print("Usage: uv run python cli.py <command> [args]")
        print("Commands: new-session, status, deals, execute, turn, context, end-session, live")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "new-session":
        kv = _parse_kv_args(sys.argv, 2)
        cmd_new_session(
            client_id=kv.get("client", "default"),
            stance_id=kv.get("stance"),
            profile_path=kv.get("profile"),
        )

    elif cmd == "start":
        # Backwards compat alias
        kv = _parse_kv_args(sys.argv, 2)
        cmd_new_session(
            client_id=kv.get("client", "default"),
            stance_id=kv.get("stance"),
        )

    elif cmd == "status":
        cmd_status()

    elif cmd == "deals":
        markers = sys.argv[2] if len(sys.argv) > 2 else None
        cmd_deals(markers)

    elif cmd == "execute":
        if len(sys.argv) < 3:
            print("Usage: execute <intervention_id> [--therapist-text TEXT]")
            sys.exit(1)
        deal_id = sys.argv[2]
        kv = _parse_kv_args(sys.argv, 3)
        cmd_execute(deal_id, therapist_text=kv.get("therapist-text"))

    elif cmd == "turn":
        signals = sys.argv[2] if len(sys.argv) > 2 else "{}"
        kv = _parse_kv_args(sys.argv, 3)
        cmd_turn(
            signals,
            markers_json=kv.get("markers"),
            triggers_str=kv.get("triggers"),
            topic=kv.get("topic"),
            client_text=kv.get("client-text"),
        )

    elif cmd == "context":
        kv = _parse_kv_args(sys.argv, 2)
        cmd_context(client_id=kv.get("client", "default"))

    elif cmd == "end-session":
        kv = _parse_kv_args(sys.argv, 2)
        cmd_end_session(note=kv.get("note", ""))

    elif cmd == "live":
        kv = _parse_kv_args(sys.argv, 2)
        cmd_live(
            client_id=kv.get("client", "default"),
            mode=kv.get("mode", "unified"),
            stance_id=kv.get("stance"),
        )

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
