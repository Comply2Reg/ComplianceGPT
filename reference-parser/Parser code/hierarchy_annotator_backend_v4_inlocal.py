# hierarchy_annotator_main.py
## single column pdf --> like proper document (not a research papaers)


import re
import os
import pandas as pd
from collections import defaultdict

BULLET_RE = re.compile(r'^\s*([-•\u2022\*]|[ivxIVX]+\.)\s+')
NUMBERED_RE = re.compile(r'^\s*(\(\w+\)|\w\)|\d+\.)')

class HierarchyAnnotator:
    """
    Combined annotator:
    - removes furniture (page_header, page_footer, footnote, document_index (TOC), etc.)
    - preserves & computes list hierarchy (list_level on list items)
    - merges CHAPTER / SECTION / ARTICLE with following title lines when visually adjacent
    - normalizes semantic header levels
    - merges text/list items across page breaks (conservative)
    - exports header hierarchy as Excel (RULE_HEADER1..RULE_HEADER8) with improved flow
    """

    def __init__(self, doc, source_name=None):
        self.doc = doc
        self.source_name = source_name
        # Initially build the text mapping, but it will be updated after furniture removal
        self.texts = {t.get("self_ref"): t for t in self.doc.get("texts", [])}


    # Page Number Extraction
    def _get_page_no(self, obj):
        prov = obj.get("prov", [])
        if prov:
            return prov[0].get("page_no", None)
        return None

    # -------------------------
    # FURNITURE / TOC REMOVAL LOGIC
    # -------------------------
    def remove_furniture(self):
        """
        Removes elements like page_header, page_footer, footnote and document_index (TOC)
        from the document structure. This modifies the doc object's body children
        and the texts list directly.
        """
        skip_labels = {
            "page_header", "page_footer", "footnote", "furniture", "document_index", "toc", "table_of_contents",
        }

        refs_to_remove = set()
        for t in self.doc.get("texts", []):
            if t.get("label") in skip_labels and t.get("label") != "section_header":
                refs_to_remove.add(t.get("self_ref"))

        print(f"[remove_furniture] Identified {len(refs_to_remove)} items to remove.")

        # Remove texts
        new_texts_list = [t for t in self.doc.get("texts", []) if t.get("self_ref") not in refs_to_remove]
        self.doc["texts"] = new_texts_list
        # REMOVE document_index TABLES
        new_tables = []
        removed_tables = 0

        for tbl in self.doc.get("tables", []):
            if tbl.get("label") == "document_index":
                removed_tables += 1
                continue
            new_tables.append(tbl)

        self.doc["tables"] = new_tables

        print(f"[remove_furniture] Removed {removed_tables} document_index tables.")

        # Update body children to remove references
        body_children = self.doc.get("body", {}).get("children", [])
        new_body_children = [
            ref_obj for ref_obj in body_children
            if ref_obj.get("$ref", "").split('/')[-1] not in refs_to_remove
        ]
        if "body" in self.doc:
            self.doc["body"]["children"] = new_body_children

        # Update groups' children as well
        for group in self.doc.get("groups", []):
            group_children = group.get("children", [])
            new_group_children = [
                ref_obj for ref_obj in group_children
                if ref_obj.get("$ref", "").split('/')[-1] not in refs_to_remove
            ]
            group["children"] = new_group_children

        # Refresh the local mapping
        self.texts = {t.get("self_ref"): t for t in self.doc.get("texts", [])}
        print(f"[remove_furniture] Removed {len(refs_to_remove)} items. Remaining texts: {len(self.texts)}")


    # -------------------------
    # LIST / BULLET LOGIC
    # -------------------------
    def _is_bullet_like(self, text):
        return bool(text and (BULLET_RE.match(text) or NUMBERED_RE.match(text)))

    # -------------------------
    # LEFT INDENT EXTRACTION
    # -------------------------
    def _get_left_x(self, text_obj):
        prov = text_obj.get("prov", [])
        if prov and isinstance(prov, list) and len(prov) > 0:
            first = prov[0]
            bbox = first.get("bbox")
            if bbox and "l" in bbox:
                try:
                    return float(bbox["l"])
                except (ValueError, TypeError):
                    pass
        t = text_obj.get("text", "") or ""
        leading_spaces = len(t) - len(t.lstrip(" "))
        # coarse fallback mapping of spaces -> pixels
        return float(leading_spaces) * 4.0

    # -------------------------
    # LIST CANDIDATE COLLECTION
    # -------------------------
    def _collect_candidates(self):
        candidates = []
        for ref, t in self.texts.items():  # using local mapping after furniture removal
            text = (t.get("text") or "").strip()
            label = t.get("label", "")
            # skip section headers for list detection
            if label == "section_header":
                continue
            if label == "list_item" or self._is_bullet_like(text):
                parent_ref = t.get("parent", {}).get("$ref")
                left_x = self._get_left_x(t)
                candidates.append((ref, parent_ref, left_x, t))
        return candidates

    # -------------------------
    # INDENTATION LEVEL BUCKETING
    # -------------------------
    def _bucket_levels(self, positions):
        """Cluster left positions into logical indentation levels."""
        if not positions:
            return {}
        sorted_pos = sorted(positions)
        buckets, tol = [], 15.0
        for p in sorted_pos:
            if not buckets or abs(p - (sum(buckets[-1]) / len(buckets[-1]))) > tol:
                buckets.append([p])
            else:
                buckets[-1].append(p)
        pos_to_level = {}
        for idx, bucket in enumerate(buckets, start=1):
            for p in bucket:
                pos_to_level[p] = idx
        return pos_to_level

    # -------------------------
    # HEADER MERGE / NORMALIZE
    # -------------------------
    def rewrite_section_header_levels(self, vertical_tolerance=80):
        """
        Merge consecutive section_header blocks that are visually and semantically
        part of the same heading line (CHAPTER/SECTION/ARTICLE + title).
        Conditions: same page, vertical gap < vertical_tolerance.
        """
        headers = [t for t in self.doc.get("texts", []) if t.get("label") == "section_header"]
        if not headers:
            return

        def get_pos_tuple(h):
            prov = h.get("prov", [])
            if prov and isinstance(prov, list) and len(prov) > 0:
                p = prov[0]
                page_no = p.get("page_no", 0)
                top = p.get("bbox", {}).get("t", 0.0)
                return (page_no, top)
            return (0, 0.0)

        headers.sort(key=lambda h: (get_pos_tuple(h)[0], -get_pos_tuple(h)[1]))

        merged_refs = set()
        for i, h1 in enumerate(headers):
            if h1.get("self_ref") in merged_refs:
                continue
            text1 = (h1.get("text") or "").strip()
            lvl1 = int(h1.get("level", 0) or 0)
            prov1 = h1.get("prov", [{}])[0]
            page1 = prov1.get("page_no", 0)
            t1 = prov1.get("bbox", {}).get("t", 0.0)

            if i + 1 >= len(headers):
                continue
            h2 = headers[i + 1]
            if h2.get("self_ref") in merged_refs:
                continue
            text2 = (h2.get("text") or "").strip()
            lvl2 = int(h2.get("level", 0) or 0)
            prov2 = h2.get("prov", [{}])[0]
            page2 = prov2.get("page_no", 0)
            t2 = prov2.get("bbox", {}).get("t", 0.0)

            same_page = (page1 == page2)
            close_gap = abs(t1 - t2) < vertical_tolerance

            t1_up = text1.upper()
            t2_up = text2.upper()

            is_chapter = bool(re.match(r"^CHAPTER\b", t1_up))
            is_article = bool(re.match(r"^ARTICLE\b", t1_up))
            is_section = bool(re.match(r"^SECTION\b", t1_up))

            if re.match(r"^(CHAPTER|ARTICLE|SECTION)\b", t2_up):
                continue

            should_merge = (same_page and close_gap and (is_chapter or is_article or is_section))

            if should_merge:
                merged_text = f"{text1} {text2}".strip()
                # logger.info(f"[merge] '{text1}' + '{text2}' -> '{merged_text}'")
                h1["text"] = merged_text
                if is_chapter:
                    h1["level"] = 1
                elif is_section:
                    h1["level"] = 2
                elif is_article:
                    h1["level"] = 3
                else:
                    h1["level"] = max(lvl1, lvl2)
                merged_refs.add(h2["self_ref"])

        if merged_refs:
            new_texts = [t for t in self.doc.get("texts", []) if t.get("self_ref") not in merged_refs]
            self.doc["texts"] = new_texts
            self.texts = {t.get("self_ref"): t for t in self.doc.get("texts", [])}
            # logger.info(f"[rewrite_section_header_levels] removed {len(merged_refs)} merged headers")

    def normalize_hierarchy_levels(self):
        """
        Map semantic keywords to consistent levels:
          CHAPTER -> 1
          SECTION -> 2
          ARTICLE -> 3
        Ensure bounds [1..8] for levels.
        """
        headers = [t for t in self.doc.get("texts", []) if t.get("label") == "section_header"]
        for h in headers:
            txt = (h.get("text") or "").strip().upper()
            if re.match(r"^CHAPTER\b", txt):
                h["level"] = 1
            elif re.match(r"^SECTION\b", txt):
                h["level"] = 2
            elif re.match(r"^ARTICLE\b", txt):
                h["level"] = 3
            else:
                try:
                    h["level"] = int(h.get("level", 4))
                except (ValueError, TypeError):
                    h["level"] = 4

            if h["level"] < 1:
                h["level"] = 1
            if h["level"] > 8:
                h["level"] = 8




    # -------------------------
    # PAGE BREAK MERGING LOGIC (CONSERVATIVE + CORRECT)
    # -------------------------
    def merge_page_breaks(self):
        texts_list = self.doc.get("texts", [])
        merged_count = 0

        i = 0
        while i < len(texts_list) - 1:
            t1 = texts_list[i]
            t2 = texts_list[i + 1]

            label1 = t1.get("label", "")
            label2 = t2.get("label", "")

            text1 = (t1.get("text") or "").strip()
            text2 = (t2.get("text") or "").strip()

            if not text1 or not text2:
                i += 1
                continue

            prov1 = t1.get("prov", [{}])[0]
            prov2 = t2.get("prov", [{}])[0]
            page1 = prov1.get("page_no", 0)
            page2 = prov2.get("page_no", 0)

            # ONLY allow merge when actual page break happens
            is_next_page = (page2 == page1 + 1)

            text2_stripped = text2.strip()

            # ------------------------------------------
            # CASE 1: text + text (across page break) --> this is backend code
            # ------------------------------------------
            if label1 == "text" and label2 == "text" and is_next_page:
                no_sentence_end = not re.search(r"[.!?]$", text1)
                starts_like_continuation = bool(
                    re.match(r"(and|or|where|which|that|who|whose|if|as)$\b", text1, re.IGNORECASE) or
                    re.match(r"^[a-z,(]", text2) or
                    re.match(r"^(and|or|where|which|that|who|whose|if|as)\b", text2, re.IGNORECASE)
                )

                if no_sentence_end and starts_like_continuation:
                    t1["text"] = f"{text1} {text2}".strip()
                    t2["text"] = ""
                    merged_count += 1

            # --------------------------------------------------
            # CASE 2: list_item + text (across page break)
            # Example: 6.2 split across pages
            # --------------------------------------------------
            elif label1 == "list_item" and label2 == "text" and is_next_page:
                no_sentence_end = not re.search(r"[.!?]$", text1)

                continuation_start = bool(
                    re.match(r"^[a-z,(]", text2) or
                    re.match(r"^(and|or|where|which|that|who|whose|if|as)\b", text2, re.IGNORECASE)
                )

                not_new_list = not re.match(r"^\(?\d+[\).]|^[A-Z]\.", text2)

                if no_sentence_end and continuation_start and not_new_list:
                    t1["text"] = f"{text1} {text2}".strip()
                    t2["text"] = ""
                    merged_count += 1

            # =========================================================
            # CASE 3: list_item + list_item (across page break)
            # Continuation of SAME bullet split across pages  --new in production code
            # =========================================================
            elif label1 == "list_item" and label2 == "list_item" and is_next_page:

                # 1) HARD STOP → explicit "o " bullet
                if re.match(r"^\s*o\s+", text2_stripped):
                    i += 1
                    continue

                # 2) HARD STOP → symbol bullets
                if re.match(r"^\s*[•\-\*]\s+", text2_stripped):
                    i += 1
                    continue

                # 3) HARD STOP → numbered/alpha bullets
                if re.match(r"^\(?\d+[\).]|^[A-Z]\.", text2_stripped):
                    i += 1
                    continue

                # 4) ONLY merge if previous line did NOT end a sentence
                no_sentence_end = not re.search(r"[.!?;]$", text1) # need to add in next version
                if not no_sentence_end:
                    i += 1
                    continue

                # 5) TRUE CONTINUATION → lowercase / connector words
                continuation_start = bool(
                    re.match(r"^[a-z,(]", text2_stripped) or
                    re.match(
                        r"^(and|or|where|which|that|who|whose|if|as|of|in|to|for|from)\b",
                        text2_stripped,
                        re.IGNORECASE
                    )
                )

                if continuation_start:
                    t1["text"] = f"{text1} {text2}".strip()
                    t2["text"] = ""
                    merged_count += 1

            i += 1

        print(f"[merge_page_breaks] Completed. Merged {merged_count} page-break continuations.")
        self.texts = {t.get("self_ref"): t for t in self.doc.get("texts", [])}
        return self.doc



    def export_hierarchy_to_excel(self, output_path_excel, max_levels=8):

        print("[export_hierarchy_to_excel] Starting export...")

        # ==============================================================
        # STATE
        # ==============================================================
        root_counter = 0
        current_header_signature = None

        last_list_parent_id_by_x = {}
        level_last_id = {}
        level_child_counter = {}

        last_text_parent_id_by_x = {}
        table_child_counter_by_x = {}
        last_table_headers_by_x = {}
        last_table_colcount_by_x = {}

        last_table_row_idx = None
        last_table_row_page = None

        #this is added beccause sunden changebin the table child hierarchy(need to check backend code)
        last_structural_parent_id = None

        # NEW: the last real structural parent in each column (text or list)
        last_structural_parent_id_by_x = {}

        rows = []
        current_headers = [None] * max_levels

        # Track immediate headers local code for header override logic
        prev_was_header = False
        prev_header_level = None

        # ==============================================================
        #  HARD BLOCK: TOC TABLES
        # ==============================================================
        valid_tables = []
        for tbl in self.doc.get("tables", []):
            if tbl.get("label") in {"document_index", "toc", "table_of_contents"}:
                continue
            valid_tables.append(tbl)

        self.doc["tables"] = valid_tables

        # ==============================================================
        #  COLLECT TABLE BOUNDING BOXES
        # ==============================================================
        table_boxes = []
        for tbl in self.doc.get("tables", []):
            for p in tbl.get("prov", []):
                bbox = p.get("bbox", {})
                table_boxes.append(
                    (
                        p.get("page_no"),
                        float(bbox.get("l", 0)),
                        float(bbox.get("t", 0)),
                        float(bbox.get("r", 0)),
                        float(bbox.get("b", 0)),
                    )
                )

        def inside_any_table(text_obj):
            prov = text_obj.get("prov", [])
            if not prov:
                return False

            p = prov[0]
            page = p.get("page_no")
            bbox = p.get("bbox", {}) or {}

            x1 = float(bbox.get("l", 0))
            y1 = float(bbox.get("t", 0))
            x2 = float(bbox.get("r", 0))
            y2 = float(bbox.get("b", 0))

            for t_page, tl, tt, tr, tb in table_boxes:
                if t_page == page:
                    if x1 >= tl and x2 <= tr and y1 >= tt and y2 <= tb:
                        return True
            return False

        # ==============================================================
        #  FINAL CLEAN TEXT TIMELINE (NO TOC, NO TABLE-DUPES)
        # ==============================================================
        all_texts = []

        for t in self.doc.get("texts", []):
            label = t.get("label", "")

            # Kill TOC texts
            if label in {"document_index", "toc", "table_of_contents"}:
                continue

            # Kill table-duplicate texts, but keep section_header
            if inside_any_table(t) and label != "section_header":
                continue

            prov = t.get("prov", [])
            if prov:
                p = prov[0]
                page = p.get("page_no", 0)
                bbox = p.get("bbox", {})
                top = bbox.get("t", 0)
                left = bbox.get("l", 0)
                all_texts.append((page, -top, left, t))
            else:
                all_texts.append((0, 0, 0, t))

        # ==============================================================
        #  SORT TABLES
        # ==================================================
        all_tables = []
        for tbl in self.doc.get("tables", []):
            prov = tbl.get("prov", [])
            if prov:
                p = prov[0]
                page = p.get("page_no", 0)
                bbox = p.get("bbox", {})
                top = bbox.get("t", 0)
                left = bbox.get("l", 0)
                all_tables.append((page, -top, left, tbl))
            else:
                all_tables.append((0, 0, 0, tbl))


        # ==============================================================
        #  FINAL MERGED TIMELINE
        # ==============================================================
        timeline = [(p, y, x, t, "text") for p, y, x, t in all_texts] + [
            (p, y, x, t, "table") for p, y, x, t in all_tables
        ]

        timeline.sort(key=lambda x: (x[0], x[1], x[2]))

        # ==============================================================
        #  MAIN LOOP
        # ==============================================================
        for page_no, y, x, item, item_type in timeline:

            # Reset immediate-header state if current item is NOT a header -local code for header override logic
            if not (item_type == "text" and item.get("label") == "section_header"):
                prev_was_header = False

            col_bucket = int(x // 150)
            last_text_parent_id = last_text_parent_id_by_x.get(col_bucket)
            table_child_counter = table_child_counter_by_x.get(col_bucket, 0)


            # ==========================================================
            #  SECTION HEADERS (NO RULE_TEXT OUTPUT)
            # ==========================================================
            if item_type == "text" and item.get("label") == "section_header":

                lvl = max(1, min(max_levels, int(item.get("level", 1))))

                # ------------------------------------------------------
                # NEW LOGIC: immediate same-level header → push deeper
                # ------------------------------------------------------
                if prev_was_header and prev_header_level == lvl:
                    lvl = min(lvl + 1, max_levels)

                h_idx = lvl - 1
                header_text = (item.get("text") or "").strip()

                current_headers[h_idx] = header_text
                for j in range(h_idx + 1, max_levels):
                    current_headers[j] = None

                header_sig = tuple(current_headers)
                if header_sig != current_header_signature:
                    current_header_signature = header_sig

                    # reset numbering + list state, but keep headers themselves
                    root_counter = 0
                    level_last_id.clear()
                    level_child_counter.clear()
                    last_text_parent_id_by_x.clear()
                    last_list_parent_id_by_x.clear()
                    table_child_counter_by_x.clear()
                    last_table_headers_by_x.clear()
                    last_structural_parent_id_by_x.clear()

                    last_table_row_idx = None
                    last_table_row_page = None

                    # this is added beccause sunden changebin the table child hierarchy and it is my local
                    last_structural_parent_id = None # in my local

                # update header tracking
                prev_was_header = True
                prev_header_level = int(item.get("level", 1))

                continue

            # ==========================================================
            #  TEXT & LIST
            # ==========================================================
            if item_type == "text":
                label = item.get("label", "")
                raw_text = (item.get("text") or "").strip()
                if not raw_text:
                    continue

                raw_text = re.sub(r"\s+", " ", raw_text)
                rule_id = None

                # ---------------- TEXT AS PARENT ----------------
                if label == "text":
                    rule_id = f"pn{root_counter}"
                    root_counter += 1

                    last_text_parent_id = rule_id
                    last_text_parent_id_by_x[col_bucket] = rule_id
                    last_structural_parent_id_by_x[col_bucket] = rule_id

                    # this is added beccause sunden changebin the table child hierarchy and it is my local
                    last_structural_parent_id = rule_id

                    level_last_id.clear()
                    level_child_counter.clear()
                    last_list_parent_id_by_x[col_bucket] = None
                    table_child_counter_by_x[col_bucket] = 0

                # ---------------- LIST ITEM ----------------
                elif label == "list_item":
                    list_level = int(item.get("list_level", 1))

                    # LEVEL 1
                    if list_level == 1:
                        parent_id = last_text_parent_id_by_x.get(col_bucket)

                        if parent_id:
                            idx_child = level_child_counter.get(1, 0)
                            rule_id = f"{parent_id}.pn{idx_child}"
                            level_child_counter[1] = idx_child + 1  # only increment when it's actually a child
                        else:
                            # No text parent → orphan becomes its own root
                            rule_id = f"pn{root_counter}"
                            root_counter += 1
                            level_child_counter[1] = 0  # reset so future children of a real parent start at .pn0

                        level_last_id[1] = rule_id

                        # clear deeper levels
                        for lvl in list(level_last_id.keys()):
                            if lvl > 1:
                                del level_last_id[lvl]
                        for lvl in list(level_child_counter.keys()):
                            if lvl > 1:
                                del level_child_counter[lvl]

                    # LEVEL > 1
                    else:
                        parent_level = list_level - 1
                        while parent_level > 0 and parent_level not in level_last_id:
                            parent_level -= 1

                        parent_id = level_last_id.get(parent_level)

                        if not parent_id:
                            parent_id = last_text_parent_id_by_x.get(col_bucket)

                        if not parent_id:
                            # No parent at any level → orphan becomes its own root
                            rule_id = f"pn{root_counter}"
                            root_counter += 1
                            level_child_counter[list_level] = 0  # reset counter for this level
                            level_last_id[list_level] = rule_id

                            # clear deeper levels
                            for lvl in list(level_child_counter.keys()):
                                if lvl > list_level:
                                    del level_child_counter[lvl]
                            for lvl in list(level_last_id.keys()):
                                if lvl > list_level:
                                    del level_last_id[lvl]

                            last_list_parent_id_by_x[col_bucket] = rule_id
                            last_structural_parent_id_by_x[col_bucket] = rule_id
                            last_structural_parent_id = rule_id
                            # skip normal assignment below
                            # (fall through to row building)
                        else:
                            idx_child = level_child_counter.get(list_level, 0)
                            rule_id = f"{parent_id}.pn{idx_child}"

                            level_child_counter[list_level] = idx_child + 1
                            level_last_id[list_level] = rule_id

                            for lvl in list(level_child_counter.keys()):
                                if lvl > list_level:
                                    del level_child_counter[lvl]
                            for lvl in list(level_last_id.keys()):
                                if lvl > list_level:
                                    del level_last_id[lvl]


                    # list is a structural parent for tables that follow
                    last_list_parent_id_by_x[col_bucket] = rule_id
                    last_structural_parent_id_by_x[col_bucket] = rule_id

                    # this is added beccause sunden changebin the table child hierarchy and it is my local
                    last_structural_parent_id = rule_id

                if rule_id is None:
                    rule_id = f"pn{root_counter}"
                    root_counter += 1

                row = {f"RULE_HEADER{i+1}": (current_headers[i] or "") for i in range(max_levels)}
                row["RULE_ID"] = rule_id
                row["RULE_TEXT_INDICATOR"] = label
                if label == "list_item":
                    last_list_parent_id_by_x[col_bucket] = rule_id
                    marker = (item.get("marker") or "").strip()
                    text_val = raw_text

                    if marker and not text_val.startswith(marker):
                        row["RULE_TEXT"] = f"{marker} {text_val}".strip()
                    else:
                        row["RULE_TEXT"] = text_val
                else:
                    # krishna change 1
                    row["RULE_TEXT"] = raw_text

                row["PAGE_NUMBER"] = self._get_page_no(item)
                rows.append(row)
                continue

            # ==========================================================
            # TABLE PROCESSING — STRUCTURAL COLUMN-AWARE MERGE (PRODUCTION)
            # ==========================================================
            if item_type == "table":

                cells = item.get("data", {}).get("table_cells", [])
                table_rows = {}

                for c in cells:
                    r = c.get("start_row_offset_idx", 0)
                    table_rows.setdefault(r, []).append(c)

                current_num_cols = item.get("data", {}).get("num_cols", 0)

                # ------------------------------------------
                # Extract column headers (if present)
                # ------------------------------------------
                headers_by_col = {}
                for c in table_rows.get(0, []):
                    if c.get("column_header"):
                        col = c.get("start_col_offset_idx", 0)
                        headers_by_col[col] = (c.get("text") or "").strip()

                has_headers = bool(headers_by_col)

                if not has_headers:
                    headers_by_col = last_table_headers_by_x.get(col_bucket, {})

                for r_idx in sorted(table_rows.keys()):
                    row_cells = table_rows[r_idx]

                    if r_idx == 0 and has_headers:
                        continue

                    # ------------------------------------------
                    # Build column values
                    # ------------------------------------------
                    col_values = {}
                    for c in row_cells:
                        col = c.get("start_col_offset_idx", 0)
                        txt = (c.get("text") or "").strip()
                        if txt:
                            col_values[col] = txt

                    # ==========================================================
                    # PAGE BREAK CONTINUATION — STRICT STRUCTURAL CHECK
                    # ==========================================================
                    is_next_page = (
                            last_table_row_page is not None and
                            page_no == last_table_row_page + 1
                    )

                    prev_num_cols = last_table_colcount_by_x.get(col_bucket)

                    filled_cols = len(col_values)
                    has_empty_column = filled_cols < current_num_cols

                    if (
                            last_table_row_idx is not None
                            and is_next_page
                            and prev_num_cols is not None
                            and current_num_cols == prev_num_cols
                            and has_empty_column  # <<< NEW CONDITION
                    ):
                        prev = rows[last_table_row_idx]
                        prev_parts = prev["RULE_TEXT"].split("\n")

                        max_col = max(col_values.keys()) if col_values else -1
                        while len(prev_parts) <= max_col:
                            prev_parts.append("")

                        for col, val in col_values.items():
                            header = headers_by_col.get(col)

                            if header:
                                merged = False
                                for i, part in enumerate(prev_parts):
                                    if part.startswith(f"{header}:"):
                                        prev_parts[i] = prev_parts[i] + " " + val
                                        merged = True
                                        break
                                if not merged:
                                    prev_parts[col] = f"{header}: {val}"
                            else:
                                if prev_parts[col]:
                                    prev_parts[col] = prev_parts[col] + " " + val
                                else:
                                    prev_parts[col] = val

                        prev["RULE_TEXT"] = "\n".join(p.strip() for p in prev_parts if p.strip())
                        last_table_row_page = page_no
                        continue

                    # ==========================================================
                    # NEW TABLE ROW
                    # ==========================================================
                    # Table becomes child ONLY if previous item was text or list - in my local
                    parent_id = last_structural_parent_id if last_structural_parent_id else None

                    if parent_id:
                        rule_id = f"{parent_id}.pn{table_child_counter}"
                    else:
                        rule_id = f"pn{root_counter}"
                        root_counter += 1
                        table_child_counter = 0

                    table_child_counter += 1

                    rule_text_parts = []
                    for col in sorted(col_values.keys()):
                        header = headers_by_col.get(col)
                        if header:
                            rule_text_parts.append(f"{header}: {col_values[col]}")
                        else:
                            rule_text_parts.append(col_values[col])

                    row_out = {
                        f"RULE_HEADER{i + 1}": (current_headers[i] or "")
                        for i in range(max_levels)
                    }
                    row_out["RULE_ID"] = rule_id
                    row_out["RULE_TEXT_INDICATOR"] = "table_row"
                    row_out["RULE_TEXT"] = "\n".join(rule_text_parts).strip()
                    row_out["PAGE_NUMBER"] = page_no

                    rows.append(row_out)
                    last_table_row_idx = len(rows) - 1
                    last_table_row_page = page_no

                last_table_colcount_by_x[col_bucket] = current_num_cols
                table_child_counter_by_x[col_bucket] = table_child_counter

                # keep structural parent as the same text/list for follow-up lists
                last_structural_parent_id_by_x[col_bucket] = parent_id
                last_table_headers_by_x[col_bucket] = headers_by_col
                continue

        # ==============================================================
        #  SAVE EXCEL
        # ==============================================================
        df = pd.DataFrame(rows)
        df["RULE_TEXT"] = df["RULE_TEXT"].fillna("").str.strip()

        # -----------------------------
        # ADD SOURCE + SOURCE_ID
        # -----------------------------
        df.insert(0, "SOURCE", self.source_name if self.source_name else "")
        df.insert(1, "SOURCE_ID", 1)

        os.makedirs(os.path.dirname(output_path_excel), exist_ok=True)
        df.to_excel(output_path_excel, index=False)

        print("Saving Excel to:", output_path_excel)

    # -------------------------
    # TABLE + TOC DUPLICATE REMOVAL
    # -------------------------
    def _remove_table_and_toc_text_duplicates(self):
        """
        Removes:
        1) document_index tables
        2) any text blocks that belong to tables by:
         parent.$ref
         OR spatial overlap (bbox)
        """

        # ----------------------------
        # STEP 1 — Remove document_index tables
        # ----------------------------
        doc_index_table_refs = set()
        all_table_refs = set()

        for tbl in self.doc.get("tables", []):
            if "self_ref" in tbl:
                all_table_refs.add(tbl["self_ref"])
            if tbl.get("label") == "document_index":
                doc_index_table_refs.add(tbl.get("self_ref"))

        self.doc["tables"] = [
            t for t in self.doc.get("tables", [])
            if t.get("self_ref") not in doc_index_table_refs
        ]

        # ----------------------------
        # STEP 2 — Collect ALL table bounding boxes (loose)
        # ----------------------------
        table_boxes = []
        for tbl in self.doc.get("tables", []):
            for p in tbl.get("prov", []):
                bbox = p.get("bbox")
                if bbox:
                    table_boxes.append((
                        p.get("page_no"),
                        float(bbox.get("l", 0)),
                        float(bbox.get("t", 0)),
                        float(bbox.get("r", 0)),
                        float(bbox.get("b", 0)),
                    ))

        def inside_table_or_child(text_obj):

            #  RULE 1: Parent reference check (strongest)

            # Fix for RULE_HEADER issiue
            if text_obj.get("label") == "section_header":
                return False
            parent_ref = text_obj.get("parent", {}).get("$ref")
            if parent_ref in all_table_refs:
                return True

            #  RULE 2: Spatial overlap check (lenient)
            prov = text_obj.get("prov", [])
            if not prov:
                return False

            p = prov[0]
            page = p.get("page_no")
            bbox = p.get("bbox") or {}

            x1 = float(bbox.get("l", 0))
            y1 = float(bbox.get("t", 0))
            x2 = float(bbox.get("r", 0))
            y2 = float(bbox.get("b", 0))

            for t_page, tl, tt, tr, tb in table_boxes:
                if t_page == page:
                    #  allow small margin for PDF drift
                    if not (x2 < tl - 5 or x1 > tr + 5 or y2 < tt - 5 or y1 > tb + 5):
                        return True
            return False

        before = len(self.doc.get("texts", []))

        # ----------------------------
        # STEP 3 — DROP TEXT THAT BELONGS TO TABLES
        # ----------------------------
        self.doc["texts"] = [
            t for t in self.doc.get("texts", [])
            if not inside_table_or_child(t)
        ]

        after = len(self.doc.get("texts", []))

        print(f"[dedupe] Removed {before - after} duplicate table/TOC text blocks")

    def _remove_text_that_duplicates_table_content(self):
        """
        Final safety filter:
        Removes any TEXT whose content already exists inside a table_row RULE_TEXT.
        This fixes Docling's table + flowing-text duplication bug.
        """

        # 1) Collect every string that appears inside tables
        table_text_fragments = set()

        for tbl in self.doc.get("tables", []):
            cells = tbl.get("data", {}).get("table_cells", [])
            for c in cells:
                txt = (c.get("text") or "").strip()
                if txt:
                    table_text_fragments.add(txt)

        def is_duplicate_of_table(text_obj):

            # Fix for RULE_HEADER issiues
            if text_obj.get("label") == "section_header":
                return False

            if text_obj.get("label") == "list_item":
                return False

            txt = (text_obj.get("text") or "").strip()
            if not txt:
                return False

            # Strong exact match
            if txt in table_text_fragments:
                return True

            # Weak containment match
            for t in table_text_fragments:
                if len(txt) > 10 and txt in t:
                    return True

            return False

        before = len(self.doc.get("texts", []))

        self.doc["texts"] = [
            t for t in self.doc.get("texts", [])
            if not is_duplicate_of_table(t)
        ]

        after = len(self.doc.get("texts", []))
        print(f"[dedupe-content] Removed {before - after} flowing text duplicates of table content")

    # -------------------------
    # MAIN: annotate()
    # -------------------------
    def annotate(self, export_excel_path=None, excel_max_levels=8):

        print("[annotate] Starting processing pipeline...")
        print(f"[annotate] Initial text count: {len(self.doc.get('texts', []))}")

        # 0) HARD REMOVE FURNITURE FIRST
        self.remove_furniture()

        #  1) HARD KILL TABLE + TOC DUPLICATES AT SOURCE
        self._remove_table_and_toc_text_duplicates()

        #  2) HARD KILL FLOWING TEXT THAT DUPLICATES TABLE CONTENT
        self._remove_text_that_duplicates_table_content()


        print(f"[annotate] Text count after hard dedupe: {len(self.doc.get('texts', []))}")

        # 2) NOW Merge page breaks (clean input)
        self.merge_page_breaks()

        print(f"[annotate] Text count after merge_page_breaks: {len(self.doc.get('texts', []))}")

        # 3) NOW Detect lists
        candidates = self._collect_candidates()
        print(f"[annotate] List candidates found: {len(candidates)}")

        grouped = defaultdict(list)
        for ref, parent_ref, left_x, text_obj in candidates:
            grouped[parent_ref].append((ref, left_x, text_obj))

        for parent_ref, items in grouped.items():
            positions = sorted({round(x, 2) for (_, x, _) in items})
            pos_to_level = self._bucket_levels(positions)
            for ref, left_x, text_obj in items:
                nearest = min(positions, key=lambda p: abs(p - left_x)) if positions else 0
                level = pos_to_level.get(nearest, 1)
                text_obj["list_level"] = max(1, int(level))
                if text_obj.get("label") != "list_item":
                    text_obj["label"] = "list_item"

        # # 4) Header fix
        # self.rewrite_section_header_levels()
        # self.normalize_hierarchy_levels()

        headers_after_norm = [
            t for t in self.doc.get("texts", [])
            if t.get("label") == "section_header"
        ]
        print(f"[annotate] Section headers after normalization: {len(headers_after_norm)}")

        # 5) EXPORT
        if export_excel_path:
            try:
                self.export_hierarchy_to_excel(export_excel_path, max_levels=excel_max_levels)
            except Exception as e:
                print(f"[annotate] Excel export failed: {e}")

        print("[annotate] Processing pipeline completed.")
        return self.doc
