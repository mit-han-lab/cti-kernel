# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Layer rules: no import-time cycles, op DAG, private names stay home.

Source-isolation edges (a source importing another source or op-level code) are counted after
the B1 relayout, when every source has a folder; until then they are not in the ledger.
"""

from __future__ import annotations

from ._package import OPS, Module, check_ledger, package, private

# The allowed op -> op imports; every other op is a leaf.
OP_DAG = {
    "attention": {"norm", "rotary", "kvcache"},
    "linear": {"collective"},
    "moe": {"linear", "activation", "collective"},
    "mlp": {"linear"},
    "collective": {"norm"},
}


def _sccs(graph: dict[str, set[str]]) -> list[set[str]]:
    """Strongly connected components with more than one module (iterative Tarjan)."""
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    found = []
    for root in sorted(graph):
        if root in index:
            continue
        work = [(root, iter(sorted(graph[root])))]
        index[root] = low[root] = len(index)
        stack.append(root)
        on_stack.add(root)
        while work:
            node, children = work[-1]
            child = next(children, None)
            if child is not None:
                if child not in index:
                    index[child] = low[child] = len(index)
                    stack.append(child)
                    on_stack.add(child)
                    work.append((child, iter(sorted(graph.get(child, ())))))
                elif child in on_stack:
                    low[node] = min(low[node], index[child])
                continue
            work.pop()
            if work:
                low[work[-1][0]] = min(low[work[-1][0]], low[node])
            if low[node] == index[node]:
                component = set()
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    component.add(member)
                    if member == node:
                        break
                if len(component) > 1:
                    found.append(component)
    return found


def _graph(*, local: bool) -> dict[str, set[str]]:
    """Module graph; without ``local`` only import-time edges plus implicit parent ``__init__``s."""
    pkg = package()
    graph: dict[str, set[str]] = {name: set() for name in pkg.modules}
    for edge in pkg.edges:
        if edge.local and not local:
            continue
        graph[edge.source].add(edge.target)
        if not local:  # importing a.b.c first runs a and a.b, unless the importer is inside them
            parts = edge.target.split(".")
            for size in range(1, len(parts)):
                parent = ".".join(parts[:size])
                if parent in pkg.modules and not edge.source.startswith(parent + "."):
                    graph[edge.source].add(parent)
    return graph


def _in_subfolder(module: Module) -> bool:
    return len(module.parts) > 2 or (module.is_package and len(module.parts) == 2)


def measure() -> dict[str, int]:
    pkg = package()
    pairs = set()
    op_pairs = set()
    cross_private = set()
    lateral_private = set()
    for edge in pkg.edges:
        source, target = pkg.modules[edge.source], pkg.modules[edge.target]
        statement = (edge.source, edge.line)
        touches_private = any(map(private, edge.names)) or any(map(private, target.parts[1:]))
        if source.unit != target.unit:
            pairs.add((source.unit, target.unit))
            if source.unit in OPS and target.unit in OPS:
                op_pairs.add((source.unit, target.unit))
            # Root-private helpers (_storage, _reroute, ...) are shared by design until B3.
            if touches_private and not private(target.unit):
                cross_private.add(statement)
        elif (
            source.unit in OPS
            and _in_subfolder(source)
            and _in_subfolder(target)
            and source.parts[1] != target.parts[1]
            and touches_private
        ):
            lateral_private.add(statement)
    return {
        "import_time_cycles": len(_sccs(_graph(local=False))),
        "local_import_cycles": len(_sccs(_graph(local=True))),
        "function_local_imports": len({(e.source, e.line) for e in pkg.edges if e.local}),
        "cross_unit_pairs": len(pairs),
        "cross_unit_private_imports": len(cross_private),
        "lateral_private_imports": len(lateral_private),
        "op_edges_outside_dag": sum(b not in OP_DAG.get(a, ()) for a, b in op_pairs),
    }


def test_no_import_time_cycles() -> None:
    assert not _sccs(_graph(local=False))


def test_layer_counts_match_ledger() -> None:
    check_ledger("layers", measure())
