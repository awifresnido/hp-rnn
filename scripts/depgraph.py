"""Extract the repository's real dependency graph by parsing it with ``ast``.

Produces ``dependency-graph.json``: modules, imports, classes, functions, and the
call edges between them, classified into stdlib / third-party / internal. Run it
again after changing the code; the diagram is generated from this output rather
than drawn by hand, so it cannot drift from the source.
"""

from __future__ import annotations

import ast
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIRS = ("src", "tests", "scripts")


@dataclass
class Function:
    """A module-level function or a method, with the calls it makes."""

    module: str
    name: str
    qualname: str
    kind: str  # function | method | classmethod | staticmethod | property
    owner: str | None
    lineno: int
    args: list[str] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)
    doc: str = ""


@dataclass
class Class:
    """A class definition with its bases and decorators."""

    module: str
    name: str
    bases: list[str]
    lineno: int
    decorators: list[str] = field(default_factory=list)
    methods: list[str] = field(default_factory=list)
    doc: str = ""


def _call_targets(node: ast.AST) -> list[str]:
    """Every call target inside a function body, as dotted names."""
    targets: list[str] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            func = child.func
            if isinstance(func, ast.Name):
                targets.append(func.id)
            elif isinstance(func, ast.Attribute):
                parts = [func.attr]
                cursor = func.value
                while isinstance(cursor, ast.Attribute):
                    parts.append(cursor.attr)
                    cursor = cursor.value
                if isinstance(cursor, ast.Name):
                    parts.append(cursor.id)
                targets.append(".".join(reversed(parts)))
    return sorted(set(targets))


def _decorator_names(
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
) -> list[str]:
    names = []
    for decorator in node.decorator_list:
        if isinstance(decorator, ast.Name):
            names.append(decorator.id)
        elif isinstance(decorator, ast.Attribute):
            names.append(decorator.attr)
        elif isinstance(decorator, ast.Call):
            inner = decorator.func
            names.append(inner.id if isinstance(inner, ast.Name) else getattr(inner, "attr", "?"))
    return names


def _classify(root: str) -> str:
    if root.startswith("src") or root in {"tests", "scripts"}:
        return "internal"
    return "stdlib" if root in sys.stdlib_module_names else "third_party"


def _imports(tree: ast.Module) -> dict[str, list[str]]:
    buckets: dict[str, list[str]] = {"stdlib": [], "third_party": [], "internal": []}
    seen: set[tuple[str, str]] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                bucket = _classify(alias.name.split(".")[0])
                if (bucket, alias.name) not in seen:
                    seen.add((bucket, alias.name))
                    buckets[bucket].append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            bucket = _classify(module.split(".")[0]) if module else "internal"
            for alias in node.names:
                name = f"{module}.{alias.name}" if module else alias.name
                if (bucket, name) not in seen:
                    seen.add((bucket, name))
                    buckets[bucket].append(name)
    for values in buckets.values():
        values.sort()
    return buckets


def analyse_module(path: Path, module_name: str) -> tuple[dict, list[Class], list[Function]]:
    """Return (module_record, classes, functions) for one source file."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    classes: list[Class] = []
    functions: list[Function] = []

    heading = ast.get_docstring(tree) or ""
    record = {
        "module": module_name,
        "path": str(path.relative_to(REPO_ROOT)),
        "doc": heading.strip().splitlines()[0] if heading else "",
        "lines": len(path.read_text(encoding="utf-8").splitlines()),
        "imports": _imports(tree),
        "classes": [],
        "functions": [],
    }

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            methods = []
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    decorators = _decorator_names(child)
                    kind = "method"
                    for candidate in ("classmethod", "staticmethod", "property"):
                        if candidate in decorators:
                            kind = candidate
                    functions.append(
                        Function(
                            module=module_name,
                            name=child.name,
                            qualname=f"{node.name}.{child.name}",
                            kind=kind,
                            owner=node.name,
                            lineno=child.lineno,
                            args=[a.arg for a in child.args.args],
                            calls=_call_targets(child),
                            doc=(ast.get_docstring(child) or "").strip().splitlines()[:1][0]
                            if ast.get_docstring(child)
                            else "",
                        )
                    )
                    methods.append(child.name)
            bases = []
            for base in node.bases:
                if isinstance(base, ast.Name):
                    bases.append(base.id)
                elif isinstance(base, ast.Attribute):
                    bases.append(f"{getattr(base.value, 'id', '?')}.{base.attr}")
            classes.append(
                Class(
                    module=module_name,
                    name=node.name,
                    bases=bases,
                    lineno=node.lineno,
                    decorators=_decorator_names(node),
                    methods=methods,
                    doc=(ast.get_docstring(node) or "").strip().splitlines()[:1][0]
                    if ast.get_docstring(node)
                    else "",
                )
            )
            record["classes"].append(node.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(
                Function(
                    module=module_name,
                    name=node.name,
                    qualname=node.name,
                    kind="function",
                    owner=None,
                    lineno=node.lineno,
                    args=[a.arg for a in node.args.args],
                    calls=_call_targets(node),
                    doc=(ast.get_docstring(node) or "").strip().splitlines()[:1][0]
                    if ast.get_docstring(node)
                    else "",
                )
            )
            record["functions"].append(node.name)

    return record, classes, functions


def resolve_target(
    call: str,
    caller: Function,
    index: dict[str, list[Function]],
    imported_modules: set[str],
) -> Function | None:
    """Resolve a call to a repo-defined symbol, or ``None`` when it cannot be.

    Deliberately conservative in one specific way: only bare names,
    ``self.method``, and ``Class.method`` resolve. A call on a lower-case
    qualifier (``model.eval``, ``vocab.encode``, ``loss.backward``) is a call on a
    *variable*, and matching its trailing name invents edges — an earlier version
    linked a bare ``model.eval()`` to a test helper that happened to be named
    ``model``. Instance-method calls are therefore not resolved, and the docs say
    so.

    A bare name is looked up in the caller's own module first, then in the modules
    the caller imports, because ``from src.data import build_split_dataloaders``
    is a bare call at the call site even though it crosses modules.
    """
    parts = call.split(".")
    if len(parts) == 1:
        candidates = [f for f in index.get(parts[0], []) if f.owner is None]
        same_module = [f for f in candidates if f.module == caller.module]
        if same_module:
            return same_module[0]
        imported = [f for f in candidates if f.module in imported_modules]
        return imported[0] if imported else None

    if parts[0] == "self" and caller.owner:
        qualified = index.get(f"{caller.owner}.{parts[1]}", [])
        same_module = [f for f in qualified if f.module == caller.module]
        return same_module[0] if same_module else None

    if parts[0][:1].isupper():
        qualified = index.get(f"{parts[0]}.{parts[-1]}", [])
        if qualified:
            same_module = [f for f in qualified if f.module == caller.module]
            return (same_module or qualified)[0]
    return None


def main() -> int:
    modules, classes, functions = [], [], []
    for source_dir in SOURCE_DIRS:
        base = REPO_ROOT / source_dir
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            if path.name == "__init__.py" and path.stat().st_size < 80:
                continue
            if path.name == Path(__file__).name:
                continue
            module_name = str(path.relative_to(REPO_ROOT))[:-3].replace("/", ".")
            record, module_classes, module_functions = analyse_module(path, module_name)
            modules.append(record)
            classes.extend(module_classes)
            functions.extend(module_functions)

    # Resolve call edges conservatively; see resolve_target for what is excluded.
    index: dict[str, list[Function]] = {}
    for function in functions:
        index.setdefault(function.name, []).append(function)
        index.setdefault(function.qualname, []).append(function)

    edges = []
    imported_by_module: dict[str, set[str]] = {}
    for module in modules:
        reachable: set[str] = set()
        for imported in module["imports"]["internal"]:
            reachable.add(imported)
            reachable.add(imported.rsplit(".", 1)[0])
        imported_by_module[module["module"]] = reachable

    for function in functions:
        imported_modules = imported_by_module.get(function.module, set())
        for call in function.calls:
            chosen = resolve_target(call, function, index, imported_modules)
            if chosen is None:
                continue
            source = f"{function.module}:{function.qualname}"
            target = f"{chosen.module}:{chosen.qualname}"
            if source != target:
                edges.append({"from": source, "to": target, "call": call})

    payload = {
        "repo_root": str(REPO_ROOT),
        "modules": modules,
        "classes": [asdict(c) for c in classes],
        "functions": [asdict(f) for f in functions],
        "edges": sorted(edges, key=lambda e: (e["from"], e["to"])),
    }

    out = REPO_ROOT / "docs" / "dependency-graph.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")

    print(f"modules   : {len(modules)}")
    print(f"classes   : {len(classes)}")
    print(f"functions : {len(functions)}")
    print(f"call edges: {len(edges)}")
    for module in modules:
        imports = module["imports"]
        print(
            f"  {module['module']:16} {module['lines']:>4} lines  "
            f"internal={len(imports['internal'])} third_party={len(imports['third_party'])}"
        )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
