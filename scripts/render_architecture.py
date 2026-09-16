"""Render docs/architecture.md and docs/dependency-graph.html from the parsed graph.

Layer membership below is curated (a diagram needs a readable order), but every
symbol named here is checked against the parsed graph and the script fails loudly
if one is missing. The *edges* are read from ``dependency-graph.json``, so they
always reflect the real calls in the source.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GRAPH_PATH = REPO_ROOT / "docs" / "dependency-graph.json"
DOCS = REPO_ROOT / "docs"

#: (layer title, subtitle, [symbol names]) — top-to-bottom reading order.
LAYERS: list[tuple[str, str, list[str]]] = [
    (
        "CLI entry points",
        "argparse mains; each one is a stage you can run",
        ["main"],
    ),
    (
        "Data preparation — Stages 2-4",
        "EPUB -> corpus -> tokens -> ids -> splits",
        [
            "extract_epub_books",
            "extract_epub_chapters",
            "book_start_indices",
            "clean_text",
            "tokenize",
            "Vocabulary",
            "book_token_counts",
            "split_token_ids_by_books",
            "LanguageModelDataset",
            "build_split_dataloaders",
            "prepare_dataset",
        ],
    ),
    (
        "Model — Stage 5",
        "one module, three architectures",
        ["ModelConfig", "RecurrentLanguageModel"],
    ),
    (
        "Train / evaluate / generate — Stages 6-9",
        "the training loop, perplexity, and decoding",
        [
            "overfit_one_batch",
            "clip_gradients_",
            "resolve_device",
            "load_processed",
            "train_model",
            "save_checkpoint",
            "load_checkpoint",
            "sequence_cross_entropy",
            "perplexity_from_loss",
            "split_losses",
            "evaluate_checkpoint",
            "next_word_predictions",
            "sample_next_token",
            "generate_tokens",
            "generate",
            "join_tokens",
        ],
    ),
    (
        "Compare — Stage 14",
        "tables derived from recorded runs",
        ["RunSummary", "discover_runs", "load_run", "format_table"],
    ),
]

#: Third-party libraries the project depends on, in import order.
LIBRARIES = ["torch", "torch.nn", "numpy", "ebooklib", "bs4", "pytest"]

#: The single point where the three architectures diverge.
MODEL_PANEL = [
    ("Vanilla RNN", "rnn", "nn.RNN", "no gate: the hidden state is overwritten each step"),
    ("GRU", "gru", "nn.GRU", "update + reset gates decide what to keep"),
    ("LSTM", "lstm", "nn.LSTM", "input/forget/output gates plus a cell state"),
    ("Improved LSTM", "lstm", "nn.LSTM", "same cell, more capacity + dropout 0.4"),
]


def load_graph() -> dict:
    if not GRAPH_PATH.exists():
        raise SystemExit(
            f"{GRAPH_PATH} is missing - run `python scripts/depgraph.py` first"
        )
    return json.loads(GRAPH_PATH.read_text(encoding="utf-8"))


def symbol_index(graph: dict) -> dict[str, list[str]]:
    """Map every known symbol name to the modules that define it."""
    index: dict[str, list[str]] = {}
    for function in graph["functions"]:
        index.setdefault(function["name"], []).append(function["module"])
        index.setdefault(function["qualname"], []).append(function["module"])
    for klass in graph["classes"]:
        index.setdefault(klass["name"], []).append(klass["module"])
    return index


def verify_layers(graph: dict) -> None:
    """Fail loudly if a layer names a symbol that no longer exists."""
    index = symbol_index(graph)
    missing = [
        name
        for _title, _sub, names in LAYERS
        for name in names
        if name not in index
    ]
    if missing:
        raise SystemExit(
            "these layered symbols are not defined in the code any more: "
            + ", ".join(sorted(missing))
        )


def module_of(graph: dict, symbol: str) -> str | None:
    index = symbol_index(graph)
    owners = index.get(symbol)
    return owners[0] if owners else None


# --- shared layout facts --------------------------------------------------------


def layer_nodes(graph: dict) -> list[list[tuple[str, str]]]:
    """Per layer, the (node_id, label) pairs actually defined in the code."""
    index = symbol_index(graph)
    built: list[list[tuple[str, str]]] = []
    for _title, _sub, names in LAYERS:
        nodes: list[tuple[str, str]] = []
        for name in names:
            for module in dict.fromkeys(index.get(name, [])):
                short = module.split(".")[-1]
                label = name if len(index.get(name, [])) == 1 else f"{short}.{name}"
                nodes.append((f"{module}:{name}", label))
        built.append(nodes)
    return built


def graph_edges(graph: dict, nodes: list[list[tuple[str, str]]]) -> list[tuple[str, str]]:
    """Call edges whose two ends are both drawn, deduplicated."""
    placed = {node_id: depth for depth, layer in enumerate(nodes) for node_id, _ in layer}
    seen: set[tuple[str, str]] = set()
    drawn: list[tuple[str, str]] = []
    for edge in graph["edges"]:
        source, target = edge["from"], edge["to"]
        if source not in placed or target not in placed:
            continue
        if placed[source] == placed[target]:
            continue
        if (source, target) in seen:
            continue
        seen.add((source, target))
        drawn.append((source, target))
    return drawn


def library_users(graph: dict) -> dict[str, list[str]]:
    """Which modules import each third-party library."""
    users: dict[str, list[str]] = {}
    for module in graph["modules"]:
        for imported in module["imports"]["third_party"]:
            root = imported.split(".")[0]
            if root in {lib.split(".")[0] for lib in LIBRARIES}:
                users.setdefault(root, []).append(module["module"])
    return {lib: sorted(set(mods)) for lib, mods in users.items()}


def render_markdown(graph: dict) -> str:
    lines = [
        "# Repository architecture",
        "",
        "Generated from the source by `scripts/depgraph.py` + `scripts/render_architecture.py`.",
        "Regenerate after changing code:",
        "",
        "```bash",
        "python scripts/depgraph.py && python scripts/render_architecture.py",
        "```",
        "",
        "Edges are parsed from the real call sites; if a symbol named here disappears,",
        "the renderer fails instead of drawing a stale diagram.",
        "",
        "What is deliberately **not** resolved: calls made on an instance, such as",
        "`vocab.encode(...)` or `model.eval()`. The qualifier is a runtime variable, so",
        "matching its trailing name would invent edges — an earlier version linked a bare",
        "`model.eval()` to a test helper that happened to be named `model`. Only bare",
        "names, `self.method`, and `Class.method` resolve.",
        "",
        "## Layers",
        "",
        "```",
    ]
    nodes = layer_nodes(graph)
    for depth, (title, subtitle, _names) in enumerate(LAYERS):
        labels = ", ".join(label for _id, label in nodes[depth])
        lines.append(f"L{depth}  {title}")
        lines.append(f"     {subtitle}")
        lines.append(f"     {labels}")
        lines.append("")
    lines.append("```")
    lines.append("")

    lines += ["## Third-party libraries", ""]
    users = library_users(graph)
    for library in sorted(users):
        lines.append(f"- **{library}** — used by {', '.join(users[library])}")
    lines.append("")

    lines += ["## How the three architectures differ", ""]
    lines.append("Everything except one dictionary lookup is shared. `ModelConfig.cell`")
    lines.append("selects the cell in `src/model.py`:")
    lines.append("")
    lines.append("```text")
    lines.append("Embedding -> _CELLS[cell] -> Dropout -> Linear -> [B, T, V] logits")
    lines.append("                 |")
    lines.append('                 +-- "rnn"  -> nn.RNN')
    lines.append('                 +-- "gru"  -> nn.GRU')
    lines.append('                 +-- "lstm" -> nn.LSTM')
    lines.append("```")
    lines.append("")
    lines.append("| Variant | cell | torch module | what changes |")
    lines.append("| --- | --- | --- | --- |")
    for name, cell, module_name, note in MODEL_PANEL:
        lines.append(f"| {name} | `{cell}` | `{module_name}` | {note} |")
    lines.append("")

    lines += ["## Modules", ""]
    for module in graph["modules"]:
        lines.append(f"### `{module['path']}` — {module['lines']} lines")
        if module["doc"]:
            lines.append("")
            lines.append(f"> {module['doc']}")
        imports = module["imports"]
        lines.append("")
        lines.append(f"- internal: {', '.join(imports['internal']) or 'none'}")
        lines.append(f"- third-party: {', '.join(imports['third_party']) or 'none'}")
        lines.append(f"- stdlib: {', '.join(imports['stdlib']) or 'none'}")
        classes = [c for c in graph["classes"] if c["module"] == module["module"]]
        if classes:
            lines.append("")
            lines.append("Classes:")
            for klass in classes:
                bases = f"({', '.join(klass['bases'])})" if klass["bases"] else ""
                lines.append(f"- `{klass['name']}{bases}` — {klass['doc'] or 'no docstring'}")
                if klass["methods"]:
                    lines.append(f"  - methods: {', '.join(klass['methods'])}")
        functions = [f for f in graph["functions"] if f["module"] == module["module"]]
        module_functions = [f for f in functions if f["owner"] is None]
        if module_functions:
            lines.append("")
            lines.append("Functions (with the repo symbols they call):")
            for function in sorted(module_functions, key=lambda f: f["lineno"]):
                lines.append(f"- `{function['name']}({', '.join(function['args'])})` — {function['doc'] or ''}")
                if function["calls"]:
                    calls = ", ".join(f"`{c}`" for c in function["calls"])
                    lines.append(f"  - calls: {calls}")
        lines.append("")

    lines += ["## Cross-module call edges", ""]
    for edge in graph["edges"]:
        source_module = edge["from"].split(":")[0]
        target_module = edge["to"].split(":")[0]
        if source_module != target_module:
            lines.append(f"- `{edge['from']}` → `{edge['to']}`")
    lines.append("")
    return "\n".join(lines)


# --- the rendered diagram ------------------------------------------------------

LAYER_STYLE = [
    ("#22d3ee", "rgba(8, 51, 68, 0.4)"),
    ("#34d399", "rgba(6, 78, 59, 0.4)"),
    ("#a78bfa", "rgba(76, 29, 149, 0.4)"),
    ("#fbbf24", "rgba(120, 53, 15, 0.3)"),
    ("#fb7185", "rgba(136, 19, 55, 0.4)"),
    ("#94a3b8", "rgba(30, 41, 59, 0.5)"),
]
WIDTH = 1280
NODE_AREA_LEFT = 210
MARGIN = 22
NODE_H = 34
NODE_GAP = 12
ROW_GAP = 46
LIB_STYLE = LAYER_STYLE[5]


def _wrap(text: str, width: int) -> list[str]:
    words, lines, current = text.split(), [], ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= width:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _node_width(label: str) -> int:
    return max(112, min(250, int(len(label) * 7.4) + 24))


def render_svg(graph: dict, nodes: list[list[tuple[str, str]]], edges: list[tuple[str, str]]) -> str:
    """Layered top-down graph: boxes per layer, curved edges behind them."""
    geometry: dict[str, tuple[int, int, int, int]] = {}
    parts: list[str] = []
    y = 24
    band_starts: list[int] = []

    for depth, layer in enumerate(nodes):
        colour, fill = LAYER_STYLE[depth]
        title, subtitle, _names = LAYERS[depth]
        band_starts.append(y)
        x = NODE_AREA_LEFT
        row = 0
        for node_id, label in layer:
            width = _node_width(label)
            if x + width > WIDTH - MARGIN:
                x = NODE_AREA_LEFT
                row += 1
            geometry[node_id] = (x, y + 26 + row * ROW_GAP, width, NODE_H)
            x += width + NODE_GAP
        rows = row + 1
        band_height = 26 + rows * ROW_GAP + 14

        parts.append(
            f'<line x1="{MARGIN}" y1="{y + band_height - 8}" x2="{WIDTH - MARGIN}" '
            f'y2="{y + band_height - 8}" stroke="#1e293b" stroke-width="1"/>'
        )
        for index, text in enumerate(_wrap(title, 26)):
            parts.append(
                f'<text x="{MARGIN}" y="{y + 20 + index * 12}" fill="{colour}" '
                f'font-size="11" font-weight="600">{html.escape(text)}</text>'
            )
        sub_y = y + 20 + len(_wrap(title, 26)) * 12
        for index, text in enumerate(_wrap(subtitle, 30)):
            parts.append(
                f'<text x="{MARGIN}" y="{sub_y + index * 10}" fill="#64748b" '
                f'font-size="8.5">{html.escape(text)}</text>'
            )
        y += band_height

    for source, target in edges:
        sx, sy, sw, sh = geometry[source]
        tx, ty, tw, _th = geometry[target]
        sx_c, tx_c = sx + sw / 2, tx + tw / 2
        span = max(18, (ty - (sy + sh)) * 0.45)
        parts.append(
            f'<path d="M {sx_c:.0f} {sy + sh} C {sx_c:.0f} {sy + sh + span:.0f}, '
            f'{tx_c:.0f} {ty - span:.0f}, {tx_c:.0f} {ty}" fill="none" '
            f'stroke="#475569" stroke-width="1" opacity="0.75" marker-end="url(#arrow)"/>'
        )

    for depth, layer in enumerate(nodes):
        colour, fill = LAYER_STYLE[depth]
        for node_id, label in layer:
            x, box_y, width, height = geometry[node_id]
            parts.append(
                f'<rect x="{x}" y="{box_y}" width="{width}" height="{height}" rx="6" fill="#0f172a"/>'
            )
            parts.append(
                f'<rect x="{x}" y="{box_y}" width="{width}" height="{height}" rx="6" '
                f'fill="{fill}" stroke="{colour}" stroke-width="1.5"/>'
            )
            parts.append(
                f'<text x="{x + width / 2:.0f}" y="{box_y + height / 2 + 4:.0f}" '
                f'fill="#e2e8f0" font-size="10.5" text-anchor="middle">{html.escape(label)}</text>'
            )

    users = library_users(graph)
    present = [lib for lib in LIBRARIES if lib.split(".")[0] in users]
    missing = [lib for lib in LIBRARIES if lib.split(".")[0] not in users]
    colour, fill = LIB_STYLE
    band_starts.append(y)
    libraries: list[str] = []
    x = NODE_AREA_LEFT
    for library in present:
        used_by = ", ".join(m.split(".")[-1] for m in users[library.split(".")[0]])
        width = _node_width(library)
        libraries.append(
            f'<rect x="{x}" y="{y + 26}" width="{width}" height="44" rx="6" fill="#0f172a"/>'
            f'<rect x="{x}" y="{y + 26}" width="{width}" height="44" rx="6" fill="{fill}" '
            f'stroke="{colour}" stroke-width="1.5"/>'
            f'<text x="{x + width / 2:.0f}" y="{y + 44}" fill="#e2e8f0" font-size="10.5" '
            f'text-anchor="middle">{html.escape(library)}</text>'
            f'<text x="{x + width / 2:.0f}" y="{y + 58}" fill="#94a3b8" font-size="8" '
            f'text-anchor="middle">used by {html.escape(used_by)}</text>'
        )
        x += width + NODE_GAP
    libraries.append(
        f'<text x="{MARGIN}" y="{y + 20}" fill="{colour}" font-size="11" font-weight="600">'
        f'Libraries</text>'
    )
    if missing:
        libraries.append(
            f'<text x="{NODE_AREA_LEFT}" y="{y + 92}" fill="#64748b" font-size="8.5">'
            f'declared in requirements but not imported by any module: '
            f'{html.escape(", ".join(missing))}</text>'
        )
    height = y + 110

    return (
        f'<svg viewBox="0 0 {WIDTH} {height}" width="100%" '
        f'style="background:#020617;border-radius:10px">'
        "<defs>"
        '<pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">'
        '<path d="M 40 0 L 0 0 0 40" fill="none" stroke="#1e293b" stroke-width="0.5"/>'
        "</pattern>"
        '<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
        'markerHeight="6" orient="auto-start-reverse">'
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#475569"/>'
        "</marker>"
        "</defs>"
        f'<rect width="{WIDTH}" height="{height}" fill="url(#grid)"/>'
        + "".join(parts)
        + "".join(libraries)
        + "</svg>"
    )


CSS = """
*{box-sizing:border-box}
body{margin:0;background:#020617;color:#e2e8f0;
 font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
 padding:28px;line-height:1.5}
h1{font-size:19px;margin:0 0 4px}
h2{font-size:13px;color:#94a3b8;margin:34px 0 12px;letter-spacing:.08em;
 text-transform:uppercase;font-weight:600}
.sub{color:#64748b;font-size:11.5px;margin-bottom:22px}
.card{background:#0f172a;border:1px solid #1e293b;border-radius:10px;padding:0}
.card-inner{padding:16px 18px}
.grid{display:grid;gap:14px}
.models{grid-template-columns:repeat(auto-fit,minmax(268px,1fr))}
.modules{grid-template-columns:repeat(auto-fit,minmax(300px,1fr))}
.card-header{display:flex;align-items:center;gap:8px;margin-bottom:8px}
.card-dot{width:8px;height:8px;border-radius:50%;flex:0 0 8px}
.cyan{background:#22d3ee;box-shadow:0 0 8px #22d3ee}
.violet{background:#a78bfa;box-shadow:0 0 8px #a78bfa}
.amber{background:#fbbf24;box-shadow:0 0 8px #fbbf24}
.emerald{background:#34d399;box-shadow:0 0 8px #34d399}
.rose{background:#fb7185;box-shadow:0 0 8px #fb7185}
.card h3{font-size:12px;margin:0;font-weight:600}
ul{margin:6px 0 0;padding-left:16px;font-size:11px;color:#cbd5e1}
li{margin:3px 0}
code{color:#7dd3fc}
.pulse{display:inline-block;width:7px;height:7px;border-radius:50%;background:#34d399;
 margin-right:7px;animation:pulse 2s ease-in-out infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.25}}
.foot{color:#475569;font-size:10.5px;margin-top:30px;border-top:1px solid #1e293b;padding-top:12px}
"""


def render_html(graph: dict, svg: str) -> str:
    """Assemble the standalone page: diagram, per-model panels, module cards."""
    users = library_users(graph)
    model_panels = []
    for name, cell, torch_module, note in MODEL_PANEL:
        model_panels.append(
            f'<div class="card card-inner"><div class="card-header">'
            f'<div class="card-dot violet"></div><h3>{html.escape(name)}</h3></div>'
            f"<ul>"
            f"<li><code>cell={cell}</code></li>"
            f"<li>torch cell: <code>{html.escape(torch_module)}</code></li>"
            f"<li>{html.escape(note)}</li>"
            f"</ul></div>"
        )

    module_cards = []
    for module in graph["modules"]:
        if not module["module"].startswith("src."):
            continue
        classes = [c["name"] for c in graph["classes"] if c["module"] == module["module"]]
        functions = [
            f["name"]
            for f in graph["functions"]
            if f["module"] == module["module"] and f["owner"] is None
        ]
        third_party = ", ".join(module["imports"]["third_party"]) or "none"
        internal = ", ".join(m.split(".")[-1] for m in module["imports"]["internal"]) or "none"
        module_cards.append(
            f'<div class="card card-inner"><div class="card-header">'
            f'<div class="card-dot cyan"></div><h3>{html.escape(module["path"])}</h3></div>'
            f"<ul>"
            f'<li>{module["lines"]} lines</li>'
            f'<li>imports (internal): {html.escape(internal)}</li>'
            f'<li>imports (third-party): {html.escape(third_party)}</li>'
            f'<li>classes: {html.escape(", ".join(classes)) or "none"}</li>'
            f'<li>functions: {html.escape(", ".join(functions)) or "none"}</li>'
            f"</ul></div>"
        )

    libraries = "".join(
        f"<li><code>{html.escape(lib)}</code> — {html.escape(', '.join(mods))}</li>"
        for lib, mods in sorted(users.items())
    )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>harry-potter-rnn — dependency graph</title><style>{CSS}</style></head>
<body>
<h1><span class="pulse"></span>harry-potter-rnn — dependency graph</h1>
<div class="sub">Parsed from the source with <code>ast</code> ·
 {len(graph["modules"])} modules · {len(graph["classes"])} classes ·
 {len(graph["functions"])} functions · {len(graph["edges"])} call edges</div>

<h2>Layers, entry points and libraries</h2>
<div class="card">{svg}</div>

<h2>How the four models differ</h2>
<div class="grid models">{"".join(model_panels)}</div>

<h2>Source modules</h2>
<div class="grid modules">{"".join(module_cards)}</div>

<h2>Third-party libraries</h2>
<div class="card card-inner"><ul>{libraries}</ul></div>

<div class="foot">Generated by scripts/depgraph.py + scripts/render_architecture.py.
Edges come from real call sites; the renderer exits non-zero if a symbol in the
layer map no longer exists in the code, so this diagram cannot silently go stale.</div>
</body></html>"""


def main() -> int:
    graph = load_graph()
    verify_layers(graph)

    model_source = (REPO_ROOT / "src" / "model.py").read_text(encoding="utf-8")
    for _name, _cell, torch_module, _note in MODEL_PANEL:
        if torch_module not in model_source:
            raise SystemExit(f"src/model.py no longer references {torch_module}")

    nodes = layer_nodes(graph)
    edges = graph_edges(graph, nodes)
    svg = render_svg(graph, nodes, edges)

    DOCS.mkdir(parents=True, exist_ok=True)
    markdown_path = DOCS / "architecture.md"
    html_path = DOCS / "dependency-graph.html"
    markdown_path.write_text(render_markdown(graph), encoding="utf-8")
    html_path.write_text(render_html(graph, svg), encoding="utf-8")
    (DOCS / "dependency-graph.json").write_text(
        json.dumps(graph, indent=1), encoding="utf-8"
    )

    print(f"layers drawn : {len(LAYERS)}")
    print(f"nodes drawn  : {sum(len(layer) for layer in nodes)}")
    print(f"edges drawn  : {len(edges)}")
    print(f"wrote {markdown_path.relative_to(REPO_ROOT)}")
    print(f"wrote {html_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


