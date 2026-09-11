import re
class DocProcessor:
    def __init__(self, annotated_doc):
        self.doc = annotated_doc
        self.texts = {t.get("self_ref"): t for t in annotated_doc.get("texts", []) if "self_ref" in t}
        self.groups = {g.get("self_ref"): g for g in annotated_doc.get("groups", []) if "self_ref" in g}
        self.tables = {tb.get("self_ref"): tb for tb in annotated_doc.get("tables", []) if "self_ref" in tb} if "tables" in annotated_doc else {}

    # ------------------------------------------------
    # Reference resolution — no sorting or reordering
    # ------------------------------------------------
    def resolve_ref(self, ref):
        """Return the actual node from $ref without sorting or reordering."""
        if isinstance(ref, dict) and "$ref" in ref:
            ref_id = ref["$ref"]
            return self.texts.get(ref_id) or self.groups.get(ref_id) or self.tables.get(ref_id)
        return ref

    def resolve_children(self, children):
        """Recursively resolve references (in original JSON order)."""
        resolved = []
        for ch in children or []:
            node = self.resolve_ref(ch)
            if not isinstance(node, dict):
                continue
            if "children" in node:
                node["children"] = self.resolve_children(node["children"])
            resolved.append(node)
        return resolved

    # ------------------------------------------------
    # Markdown generation (pure sequential flow)
    # ------------------------------------------------
    def generate(self):
        print("Generating Markdown (preserving document flow)...")
        body = self.resolve_ref(self.doc.get("body", {}))
        body_children = self.resolve_children(body.get("children", []))
        md_lines = []
        for el in body_children:
            md_lines.extend(self._render_node(el))
        return "\n".join(md_lines).strip()

    def _render_node(self, el):
        """Render a single node and its children in document order."""
        lines = []
        label = el.get("label", "")
        text = (el.get("text") or "").strip()
        level = el.get("level", 1)
        list_level = el.get("list_level", 0)
        children = el.get("children", [])

        # Check if this header was merged into a parent
        if el.get("label") == "section_header" and el.get("merged_into_parent", False):
             # Do not render this node if it was merged
             return []

        # Direct one-to-one Markdown mapping
        if label == "section_header":
            md_level = max(1, min(6, int(level) if isinstance(level, (int, float)) else 1))
            lines.append(f"{'#' * md_level} {text}\n")
        elif label == "list_item":
            indent = "  " * (list_level - 1)
            lines.append(f"{indent}- {text}\n")
        elif label == "table_text":
            lines.append(text + "\n")
        elif label == "table":
            lines.append(self._render_table(el) + "\n")
        else:
            if text:
                lines.append(text + "\n")

        # Render children exactly in sequence
        for child in children:
            lines.extend(self._render_node(child))

        return lines

    # ------------------------------------------------
    # Table rendering
    # ------------------------------------------------
    def _render_table(self, table_obj):
        """Render tables as simple text in original order."""
        data = table_obj.get("data", {})
        table_cells = data.get("table_cells", [])
        if not table_cells:
            return ""

        rows = {}
        for cell in table_cells:
            r = cell.get("start_row_offset_idx", 0)
            c = cell.get("start_col_offset_idx", 0)
            rows.setdefault(r, {})[c] = cell.get("text", "").strip()

        sorted_rows = [rows[k] for k in sorted(rows.keys())]
        num_cols = max(len(r.keys()) for r in sorted_rows)
        lines = []

        for row in sorted_rows:
            cells = [row.get(i, "") for i in range(num_cols)]
            if len(cells) == 2:
                lines.append(f"**{cells[0]}:** {cells[1]}")
            else:
                lines.append(" ".join(c for c in cells if c))

        return "\n".join(lines)
