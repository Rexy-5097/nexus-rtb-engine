#!/usr/bin/env python3
import os
import sys
import fnmatch
import zipfile
import io


def match_type(name: str):
    if fnmatch.fnmatch(name, "imp.*.txt"):
        return "IMP"
    if fnmatch.fnmatch(name, "clk.*.txt"):
        return "CLK"
    return None


def discover_items(root: str):
    file_items = {"IMP": [], "CLK": []}
    zip_items = {"IMP": [], "CLK": []}

    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            fpath = os.path.join(dirpath, fname)
            if fname.lower().endswith(".zip"):
                try:
                    with zipfile.ZipFile(fpath) as zf:
                        for info in zf.infolist():
                            if info.is_dir():
                                continue
                            base = os.path.basename(info.filename)
                            typ = match_type(base)
                            if not typ:
                                continue
                            zip_items[typ].append(
                                {
                                    "type": typ,
                                    "kind": "zip",
                                    "zip_path": fpath,
                                    "entry": info.filename,
                                    "display": f"{os.path.abspath(fpath)}::{info.filename}",
                                }
                            )
                except zipfile.BadZipFile:
                    continue
            else:
                typ = match_type(fname)
                if typ:
                    file_items[typ].append(
                        {
                            "type": typ,
                            "kind": "file",
                            "path": fpath,
                            "display": os.path.abspath(fpath),
                        }
                    )

    def pick_one(typ):
        if file_items[typ]:
            return sorted(file_items[typ], key=lambda x: x["display"])[0]
        if zip_items[typ]:
            return sorted(zip_items[typ], key=lambda x: x["display"])[0]
        return None

    return pick_one("IMP"), pick_one("CLK")


def open_text_stream(item):
    if item["kind"] == "file":
        return open(item["path"], "r", encoding="utf-8", errors="replace")
    zf = zipfile.ZipFile(item["zip_path"])
    raw = zf.open(item["entry"])
    return io.TextIOWrapper(raw, encoding="utf-8", errors="replace")


def print_raw_lines(item, label):
    print(f"=== {label} RAW LINES ===")
    with open_text_stream(item) as f:
        for _ in range(5):
            line = f.readline()
            if line == "":
                break
            print(line, end="")
    print("")


def print_column_indexing(item, label):
    print(f"=== {label} COLUMN INDEXING ===")
    with open_text_stream(item) as f:
        for line in f:
            if line.strip() == "":
                continue
            stripped = line.rstrip("\r\n")
            parts = stripped.split("\t")
            for idx, val in enumerate(parts):
                print(f"[{idx}] -> {val}")
            break


def main():
    root = os.getcwd()
    imp_item, clk_item = discover_items(root)

    if not imp_item:
        print("ERROR: No impression file found")
        sys.exit(1)
    if not clk_item:
        print("ERROR: No click file found")
        sys.exit(1)

    print("=== SELECTED IMPRESSION FILE ===")
    print(imp_item["display"])
    print("=== SELECTED CLICK FILE ===")
    print(clk_item["display"])

    print_raw_lines(imp_item, "IMP")
    print_column_indexing(imp_item, "IMP")

    print_raw_lines(clk_item, "CLK")
    print_column_indexing(clk_item, "CLK")


if __name__ == "__main__":
    main()
