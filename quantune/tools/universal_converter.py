"""
Universal Data Converter & Normalizer
====================================
Accepts diverse file formats (.jsonl, .json, .csv, .tsv, .txt) and diverse column
schemas (instruction/response, input/output, prompt/completion, question/answer,
messages), and normalizes them into standard OpenAI Chat JSONL format:
{"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}
"""

import os
import csv
import json
import re
from typing import List, Dict, Any, Tuple, Optional


CANDIDATE_SYSTEM_KEYS = {"system", "system_prompt", "system_instruction", "context", "instruction_system"}
CANDIDATE_USER_KEYS = {"instruction", "prompt", "input", "question", "query", "user", "human", "text_input", "problem"}
CANDIDATE_ASSISTANT_KEYS = {"response", "output", "completion", "answer", "assistant", "bot", "solution", "target"}


def _clean_str(val: Any) -> str:
    """Safe string stripping."""
    if val is None:
        return ""
    if isinstance(val, (dict, list)):
        return json.dumps(val, ensure_ascii=False)
    return str(val).strip()


def _is_chat_messages(obj: Any) -> bool:
    """Check if object contains standard messages array with roles."""
    if isinstance(obj, dict) and "messages" in obj and isinstance(obj["messages"], list):
        if len(obj["messages"]) >= 2:
            roles = {m.get("role") for m in obj["messages"] if isinstance(m, dict)}
            if "user" in roles and "assistant" in roles:
                return True
    return False


def _normalize_chat_entry(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Normalizes an existing chat message structure."""
    raw_msgs = obj.get("messages") or obj.get("conversations") or []
    cleaned_msgs = []
    
    for m in raw_msgs:
        if not isinstance(m, dict):
            continue
        role = _clean_str(m.get("role") or m.get("from")).lower()
        if role in ("human", "user", "client", "customer"):
            role = "user"
        elif role in ("assistant", "bot", "gpt", "model"):
            role = "assistant"
        elif role in ("system", "instruction"):
            role = "system"
        else:
            continue
        
        content = _clean_str(m.get("content") or m.get("value") or m.get("text"))
        if content:
            cleaned_msgs.append({"role": role, "content": content})

    # Must have at least 1 user and 1 assistant message
    has_user = any(m["role"] == "user" for m in cleaned_msgs)
    has_assistant = any(m["role"] == "assistant" for m in cleaned_msgs)
    if has_user and has_assistant:
        return {"messages": cleaned_msgs}
    return None


def _detect_column_mapping(columns: List[str]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Finds best matching columns for (system, user, assistant)."""
    col_map = {c.lower().strip(): c for c in columns}
    
    sys_col = next((col_map[k] for k in CANDIDATE_SYSTEM_KEYS if k in col_map), None)
    user_col = next((col_map[k] for k in CANDIDATE_USER_KEYS if k in col_map), None)
    asst_col = next((col_map[k] for k in CANDIDATE_ASSISTANT_KEYS if k in col_map), None)
    
    # Fallback: if user_col is None, search partial match
    if not user_col:
        for k in col_map:
            if any(term in k for term in ("prompt", "instruct", "question", "user")):
                user_col = col_map[k]
                break

    if not asst_col:
        for k in col_map:
            if any(term in k for term in ("response", "output", "answer", "assistant", "completion")):
                asst_col = col_map[k]
                break

    return sys_col, user_col, asst_col


def convert_and_normalize(file_path: str, original_filename: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Ingests file of any supported format and outputs canonical chat rows.
    
    Returns:
        canonical_rows: List[{"messages": [{"role": "...", "content": "..."}]}]
        metadata: Dict with format info, mapped columns, counts, and warnings.
    """
    ext = os.path.splitext(original_filename.lower())[1]
    canonical_rows: List[Dict[str, Any]] = []
    meta: Dict[str, Any] = {
        "original_filename": original_filename,
        "detected_format": ext.replace(".", "").upper() if ext else "UNKNOWN",
        "detected_mapping": "native_messages",
        "total_source_items": 0,
        "valid_converted_items": 0,
        "skipped_items": 0,
        "warnings": []
    }

    # 1. JSONL handling
    if ext == ".jsonl":
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                meta["total_source_items"] += 1
                try:
                    data = json.loads(line)
                    if _is_chat_messages(data):
                        norm = _normalize_chat_entry(data)
                        if norm:
                            canonical_rows.append(norm)
                        else:
                            meta["skipped_items"] += 1
                    elif isinstance(data, dict):
                        # Instruction / Response JSON object on one line
                        sys_c, usr_c, ast_c = _detect_column_mapping(list(data.keys()))
                        if usr_c and ast_c:
                            meta["detected_mapping"] = f"{usr_c} -> user, {ast_c} -> assistant"
                            msgs = []
                            if sys_c and _clean_str(data.get(sys_c)):
                                msgs.append({"role": "system", "content": _clean_str(data[sys_c])})
                            msgs.append({"role": "user", "content": _clean_str(data[usr_c])})
                            msgs.append({"role": "assistant", "content": _clean_str(data[ast_c])})
                            if _clean_str(data[usr_c]) and _clean_str(data[ast_c]):
                                canonical_rows.append({"messages": msgs})
                            else:
                                meta["skipped_items"] += 1
                        else:
                            meta["skipped_items"] += 1
                except Exception:
                    meta["skipped_items"] += 1

    # 2. JSON Array handling
    elif ext == ".json":
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            try:
                data = json.load(f)
            except Exception as e:
                meta["warnings"].append(f"JSON parse error: {str(e)}")
                data = []

        if isinstance(data, dict):
            # Check for container keys: "data", "examples", "rows", "train"
            for k in ("data", "examples", "rows", "train", "dataset", "conversations"):
                if k in data and isinstance(data[k], list):
                    data = data[k]
                    break
            else:
                data = [data]

        if isinstance(data, list):
            meta["total_source_items"] = len(data)
            for item in data:
                if not isinstance(item, dict):
                    meta["skipped_items"] += 1
                    continue
                if _is_chat_messages(item):
                    norm = _normalize_chat_entry(item)
                    if norm:
                        canonical_rows.append(norm)
                    else:
                        meta["skipped_items"] += 1
                else:
                    sys_c, usr_c, ast_c = _detect_column_mapping(list(item.keys()))
                    if usr_c and ast_c:
                        meta["detected_mapping"] = f"{usr_c} -> user, {ast_c} -> assistant"
                        msgs = []
                        if sys_c and _clean_str(item.get(sys_c)):
                            msgs.append({"role": "system", "content": _clean_str(item[sys_c])})
                        u_text = _clean_str(item[usr_c])
                        a_text = _clean_str(item[ast_c])
                        if u_text and a_text:
                            msgs.append({"role": "user", "content": u_text})
                            msgs.append({"role": "assistant", "content": a_text})
                            canonical_rows.append({"messages": msgs})
                        else:
                            meta["skipped_items"] += 1
                    else:
                        meta["skipped_items"] += 1

    # 3. CSV and TSV handling
    elif ext in (".csv", ".tsv"):
        delimiter = "\t" if ext == ".tsv" else ","
        with open(file_path, "r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            if reader.fieldnames:
                columns = reader.fieldnames
                sys_c, usr_c, ast_c = _detect_column_mapping(columns)
                
                # Check if there is a 'messages' column containing serialized JSON
                has_msgs_col = next((c for c in columns if c.lower().strip() == "messages"), None)
                if has_msgs_col:
                    meta["detected_mapping"] = f"{has_msgs_col} (JSON serialized messages)"
                    for row in reader:
                        meta["total_source_items"] += 1
                        try:
                            parsed = json.loads(row[has_msgs_col])
                            if isinstance(parsed, list):
                                norm = _normalize_chat_entry({"messages": parsed})
                                if norm:
                                    canonical_rows.append(norm)
                                    continue
                        except:
                            pass
                        meta["skipped_items"] += 1
                elif usr_c and ast_c:
                    meta["detected_mapping"] = f"{usr_c} -> user, {ast_c} -> assistant"
                    if sys_c:
                        meta["detected_mapping"] += f", {sys_c} -> system"
                    for row in reader:
                        meta["total_source_items"] += 1
                        u_text = _clean_str(row.get(usr_c))
                        a_text = _clean_str(row.get(ast_c))
                        if u_text and a_text:
                            msgs = []
                            if sys_c and _clean_str(row.get(sys_c)):
                                msgs.append({"role": "system", "content": _clean_str(row[sys_c])})
                            msgs.append({"role": "user", "content": u_text})
                            msgs.append({"role": "assistant", "content": a_text})
                            canonical_rows.append({"messages": msgs})
                        else:
                            meta["skipped_items"] += 1
                else:
                    meta["warnings"].append(f"Could not map columns {columns} to user/assistant prompts.")

    # 4. TXT handling (Prompt / Response pairs separated by delimiters or double newlines)
    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            raw_text = f.read()
        
        # Split by double newline or '###' or '---'
        blocks = re.split(r'\n{2,}|\n---+\n|\n===+\n', raw_text)
        meta["total_source_items"] = len(blocks)
        meta["detected_mapping"] = "Delimited text blocks (Q: / A: or User: / Assistant:)"
        
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            
            # Pattern 1: User / Human / Assistant / Bot tags
            m = re.search(r'(?:User|Human|Question|Input):\s*(.+?)\s*(?:Assistant|Bot|Answer|Output):\s*(.+)', block, re.DOTALL | re.IGNORECASE)
            if m:
                u_text = _clean_str(m.group(1))
                a_text = _clean_str(m.group(2))
                if u_text and a_text:
                    canonical_rows.append({
                        "messages": [
                            {"role": "user", "content": u_text},
                            {"role": "assistant", "content": a_text}
                        ]
                    })
                    continue
            meta["skipped_items"] += 1

    else:
        meta["warnings"].append(f"Unsupported extension '{ext}'. Attempting line-by-line JSON fallback.")

    meta["valid_converted_items"] = len(canonical_rows)
    return canonical_rows, meta
