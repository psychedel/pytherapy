"""Expression AST for therapy-native guard conditions.

Provides ergonomic proxy objects for building expression trees:

    from therapy_engine.expr import c, markers, turn, session

    guard = c.experiencing >= 3 & c.alliance >= 2
    guard = markers.somatic > 0.5
    guard = c.crisis_signal >= 4

Expressions are evaluated against a context dict:

    ctx = {
        "c": {"experiencing": 3.0, "alliance": 4.0, ...},
        "markers": {"somatic": 0.7, ...},
        "turn": {"number": 5, "topic": "grief"},
        "session": {"macro_phase": "working", "last_intervention": "reflect"},
    }
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Any


# ── AST Nodes ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class Ref:
    """Reference to a context value: c.experiencing, markers.somatic."""
    namespace: str
    field: str


@dataclass(frozen=True)
class Lit:
    """Literal value: 3, 0.5, "working"."""
    value: Any


@dataclass(frozen=True)
class Cmp:
    """Comparison: left op right."""
    op: str
    left: Any
    right: Any

    def __and__(self, other):
        return And(self, other)

    def __or__(self, other):
        return Or(self, other)

    def __invert__(self):
        return Not(self)


@dataclass(frozen=True)
class And:
    left: Any
    right: Any

    def __and__(self, other):
        return And(self, other)

    def __or__(self, other):
        return Or(self, other)

    def __invert__(self):
        return Not(self)


@dataclass(frozen=True)
class Or:
    left: Any
    right: Any

    def __and__(self, other):
        return And(self, other)

    def __or__(self, other):
        return Or(self, other)

    def __invert__(self):
        return Not(self)


@dataclass(frozen=True)
class Not:
    expr: Any

    def __and__(self, other):
        return And(self, other)

    def __or__(self, other):
        return Or(self, other)


@dataclass(frozen=True)
class Call:
    """Built-in function call: intervention_count("reflect")."""
    func: str
    args: tuple


# ── Proxy for ergonomic AST building ──────────────────────────────

def _to_node(value: Any) -> Any:
    """Wrap raw values as Lit nodes; pass through existing AST nodes."""
    if isinstance(value, (Ref, Lit, Cmp, And, Or, Not, Call)):
        return value
    return Lit(value)


class _Proxy:
    """Builds Expr AST nodes via Python operators.

    Usage:
        c = _Proxy("c")
        c.experiencing >= 3   # → Cmp(">=", Ref("c", "experiencing"), Lit(3))
    """
    __slots__ = ('_ns', '_field')

    def __init__(self, ns: str, field: str | None = None):
        object.__setattr__(self, '_ns', ns)
        object.__setattr__(self, '_field', field)

    def __getattr__(self, name: str) -> _Proxy:
        if self._field is not None:
            raise AttributeError(
                f"Cannot access '{name}' on '{self._ns}.{self._field}' — "
                f"only one level of attribute access supported"
            )
        return _Proxy(self._ns, name)

    def _ref(self) -> Ref:
        if self._field is None:
            raise ValueError(f"Incomplete reference: '{self._ns}' — need a field")
        return Ref(self._ns, self._field)

    def __ge__(self, other): return Cmp(">=", self._ref(), _to_node(other))
    def __le__(self, other): return Cmp("<=", self._ref(), _to_node(other))
    def __gt__(self, other): return Cmp(">",  self._ref(), _to_node(other))
    def __lt__(self, other): return Cmp("<",  self._ref(), _to_node(other))
    def __eq__(self, other): return Cmp("==", self._ref(), _to_node(other))
    def __ne__(self, other): return Cmp("!=", self._ref(), _to_node(other))

    def __hash__(self):
        return hash((self._ns, self._field))

    def __repr__(self):
        if self._field:
            return f"{self._ns}.{self._field}"
        return self._ns


# ── Public proxies ────────────────────────────────────────────────

c = _Proxy("c")              # client resources
markers = _Proxy("markers")  # parser markers (current turn)
turn = _Proxy("turn")        # turn context
session = _Proxy("session")  # session context


# ── Built-in function constructors ────────────────────────────────

def intervention_count(intervention_id: str) -> Call:
    """Number of times intervention was used this session."""
    return Call("intervention_count", (intervention_id,))


def turns_since(intervention_id: str) -> Call:
    """Turns since intervention was last used. Returns 999 if never used."""
    return Call("turns_since", (intervention_id,))


# ── Evaluator ─────────────────────────────────────────────────────

_OPS = {
    ">=": operator.ge,
    "<=": operator.le,
    ">":  operator.gt,
    "<":  operator.lt,
    "==": operator.eq,
    "!=": operator.ne,
}


def evaluate(expr: Any, ctx: dict) -> Any:
    """Evaluate an expression tree against a context dict.

    Context structure:
        {
            "c": {"experiencing": 3.0, "alliance": 4.0, ...},
            "markers": {"somatic": 0.7, ...},
            "turn": {"number": 5},
            "session": {"macro_phase": "working"},
            "_functions": {"intervention_count": ..., "turns_since": ...},
        }
    """
    if isinstance(expr, Ref):
        ns = ctx.get(expr.namespace, {})
        return ns.get(expr.field, 0.0)

    if isinstance(expr, Lit):
        return expr.value

    if isinstance(expr, Cmp):
        left = evaluate(expr.left, ctx)
        right = evaluate(expr.right, ctx)
        return _OPS[expr.op](left, right)

    if isinstance(expr, And):
        return evaluate(expr.left, ctx) and evaluate(expr.right, ctx)

    if isinstance(expr, Or):
        return evaluate(expr.left, ctx) or evaluate(expr.right, ctx)

    if isinstance(expr, Not):
        return not evaluate(expr.expr, ctx)

    if isinstance(expr, Call):
        funcs = ctx.get("_functions", {})
        fn = funcs.get(expr.func)
        if fn is None:
            raise ValueError(f"Unknown function: {expr.func}")
        return fn(*expr.args)

    if expr is None:
        return True  # no guard = always passes

    raise TypeError(f"Unknown expression type: {type(expr)}")
