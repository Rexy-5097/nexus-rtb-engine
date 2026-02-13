#!/usr/bin/env python3
import os
import sys
import fnmatch
import zipfile
import sqlite3
import zlib
import pickle
import io
import logging

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import SGDClassifier

# Use new config and hashing utils
from src.bidding.config import config
from src.utils.hashing import Hasher
from src.bidding.features import FeatureExtractor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

HASH_SPACE = config.model.hash_space
CHUNK_SIZE = 100_000
SQLITE_DB = "bid_events.db"
SQLITE_BATCH = 10_000
SQLITE_QUERY_BATCH = 900
LOG_EVERY_ROWS = 500_000

N_MAP = config.model.n_map

IMP_USECOLS = [0, 4, 6, 7, 9, 15, 16, 20, 22]
IMP_COLS = [
    "bidid",
    "ua",
    "region",
    "city",
    "domain",
    "vis",
    "fmt",
    "payingprice",
    "advertiser",
]

def match_type(name: str):
    if fnmatch.fnmatch(name, "imp.*.txt"):
        return "IMP"
    if fnmatch.fnmatch(name, "clk.*.txt"):
        return "CLK"
    if fnmatch.fnmatch(name, "conv.*.txt"):
        return "CONV"
    return None

def discover_items(root: str):
    items = []
    # Security: Restrict discovery to specific data directory if possible, currently flexible
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            fpath = os.path.join(dirpath, fname)
            if fname.lower().endswith(".zip"):
                try:
                    with zipfile.ZipFile(fpath) as zf:
                        for info in zf.infolist():
                            if info.is_dir():
                                continue
                            if info.file_size > 2 * 1024 * 1024 * 1024: # Zip bomb check
                                logger.warning(f"Skipping oversized entry: {info.filename}")
                                continue
                                
                            base = os.path.basename(info.filename)
                            typ = match_type(base)
                            if not typ:
                                continue
                            items.append(
                                {
                                    "type": typ,
                                    "kind": "zip",
                                    "zip_path": fpath,
                                    "entry": info.filename,
                                    "size": info.file_size,
                                    "display": f"{os.path.abspath(fpath)}::{info.filename}",
                                    "name": info.filename,
                                }
                            )
                except zipfile.BadZipFile:
                    logger.warning(f"SKIP bad zip: {os.path.abspath(fpath)}")
                continue
            typ = match_type(fname)
            if typ:
                items.append(
                    {
                        "type": typ,
                        "kind": "file",
                        "path": fpath,
                        "size": os.path.getsize(fpath),
                        "display": os.path.abspath(fpath),
                        "name": os.path.basename(fpath),
                    }
                )
    return items

def iter_chunks(item, usecols=None):
    try:
        if item["kind"] == "file":
            reader = pd.read_csv(
                item["path"],
                sep="\t",
                header=None,
                chunksize=CHUNK_SIZE,
                usecols=usecols,
                dtype=str,
                on_bad_lines="skip",
                low_memory=True,
                na_filter=False,
            )
            for chunk in reader:
                yield chunk
        else:
            with zipfile.ZipFile(item["zip_path"]) as zf:
                with zf.open(item["entry"]) as f:
                    text = io.TextIOWrapper(f, encoding="utf-8", errors="replace")
                    reader = pd.read_csv(
                        text,
                        sep="\t",
                        header=None,
                        chunksize=CHUNK_SIZE,
                        usecols=usecols,
                        dtype=str,
                        on_bad_lines="skip",
                        low_memory=True,
                        na_filter=False,
                    )
                    for chunk in reader:
                        yield chunk
    except Exception as e:
        logger.error(f"Error reading {item['display']}: {e}")

def init_db(path: str):
    if os.path.exists(path):
        os.remove(path)
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    # Removed synchronous=OFF for safety as per audit
    cur.execute("CREATE TABLE clicks (bidid TEXT PRIMARY KEY)")
    cur.execute("CREATE TABLE convs (bidid TEXT PRIMARY KEY)")
    conn.commit()
    logger.info("Initialized temporary SQLite database for event indexing")
    return conn

def norm(val):
    if val is None:
        return "unknown"
    s = str(val)
    if s == "" or s.lower() == "nan":
        return "unknown"
    return s

def parse_price(val):
    if val is None:
        return 0.0
    try:
        price = float(val)
    except Exception:
        return 0.0
    if price < 0:
        return 0.0
    return price

# hash_feature now redundant, using Hasher directly in build_feature_matrix
def hash_feature(feature_str: str):
    return Hasher.adler32_hash(feature_str, HASH_SPACE)

def index_events(conn, items, table_name, label):
    cur = conn.cursor()
    total_rows = 0
    for item in items:
        logger.info(f"Processing {label} {item['name']}")
        file_rows = 0
        for chunk in iter_chunks(item, usecols=[0]):
            if chunk.empty:
                continue
            chunk.columns = ["bidid"]
            bidids = chunk["bidid"].tolist()
            cleaned = []
            for bid in bidids:
                b = norm(bid)
                if b != "unknown":
                    cleaned.append(b)
            if not cleaned:
                continue
            total_rows += len(cleaned)
            file_rows += len(cleaned)
            for i in range(0, len(cleaned), SQLITE_BATCH):
                batch = cleaned[i : i + SQLITE_BATCH]
                cur.executemany(
                    f"INSERT OR IGNORE INTO {table_name} (bidid) VALUES (?)",
                    [(b,) for b in batch],
                )
            conn.commit()
            if file_rows % LOG_EVERY_ROWS < len(cleaned):
                logger.info(f"{label} rows so far: {file_rows}")
        logger.info(f"Completed {label} {item['name']} rows={file_rows}")
    logger.info(f"{label} total rows processed: {total_rows}")

def fetch_existing(conn, table_name, bidids):
    found = set()
    cur = conn.cursor()
    if not bidids:
        return found
    
    # Validate table_name to prevent SQLi via table name interpolation
    if table_name not in ("clicks", "convs"):
        raise ValueError(f"Invalid table name: {table_name}")

    for i in range(0, len(bidids), SQLITE_QUERY_BATCH):
        batch = bidids[i : i + SQLITE_QUERY_BATCH]
        placeholders = ",".join("?" for _ in batch)
        # Safe parameterized query with validated table name
        cur.execute(
            f"SELECT bidid FROM {table_name} WHERE bidid IN ({placeholders})",
            batch,
        )
        for row in cur.fetchall():
            found.add(row[0])
    return found

def build_feature_matrix(rows, clicked_mask, conv_mask, stats_map):
    n = len(rows)
    row_indices = []
    col_indices = []
    data = []

    for i, row in enumerate(rows):
        # Unpack usage of FeatureExtractor logic reused here for consistency
        # Construct mock Request object or use helper?
        # For training speed, we inline the logic but use Hasher
        
        bidid, ua, region, city, domain, vis, fmt, payingprice, advertiser = row
        ua_val = norm(ua).lower()
        region_val = norm(region)
        city_val = norm(city)
        domain_val = norm(domain)
        if domain_val != "unknown":
            domain_val = domain_val.lower()
            if domain_val.startswith("www."):
                domain_val = domain_val[4:]
        vis_val = norm(vis)
        fmt_val = norm(fmt)
        adv_val = norm(advertiser)

        os_token, br_token = FeatureExtractor._parse_ua(ua_val)

        feats = [
            ("ua_os", os_token),
            ("ua_browser", br_token),
            ("region", region_val),
            ("city", city_val),
            ("adslot_visibility", vis_val),
            ("adslot_format", fmt_val),
            ("advertiser", adv_val),
            ("domain", domain_val),
        ]
        
        for name, value in feats:
            if value == "unknown":
                continue
            feature = f"{name}:{value}"
            row_indices.append(i)
            # Use strict Hasher
            col_indices.append(Hasher.adler32_hash(feature, HASH_SPACE))
            data.append(1.0)

        stats = stats_map.get(adv_val)
        if stats is None:
            stats = [0, 0.0, 0, 0]
            stats_map[adv_val] = stats
        stats[0] += 1
        price = parse_price(payingprice)
        stats[1] += price
        if clicked_mask[i]:
            stats[2] += 1
        if conv_mask[i]:
            stats[3] += 1

    if not row_indices:
        return csr_matrix((n, HASH_SPACE), dtype=np.float32)
    X = csr_matrix(
        (np.array(data, dtype=np.float32), (np.array(row_indices), np.array(col_indices))),
        shape=(n, HASH_SPACE),
    )
    return X


def process_impressions(items, conn):
    ctr_model = SGDClassifier(loss="log_loss", random_state=42)
    cvr_model = SGDClassifier(loss="log_loss", random_state=42)
    ctr_init = False
    cvr_init = False

    stats_map = {}
    total_imps = 0
    total_clicks = 0
    total_convs = 0

    for item in items:
        logger.info(f"Processing IMP {item['name']}")
        file_rows = 0
        for chunk in iter_chunks(item, usecols=IMP_USECOLS):
            if chunk.empty:
                continue
            if chunk.shape[1] != len(IMP_COLS):
                continue
            chunk.columns = IMP_COLS

            rows = []
            for row in chunk.itertuples(index=False, name=None):
                bidid = norm(row[0])
                if bidid == "unknown":
                    continue
                rows.append(row)

            if not rows:
                continue

            bidids = [r[0] for r in rows]
            clicked_set = fetch_existing(conn, "clicks", bidids)
            clicked_mask = np.fromiter(
                (b in clicked_set for b in bidids), dtype=bool, count=len(bidids)
            )

            clicked_indices = [i for i, flag in enumerate(clicked_mask) if flag]
            clicked_bidids = [bidids[i] for i in clicked_indices]
            conv_set = fetch_existing(conn, "convs", clicked_bidids)

            conv_mask = np.zeros(len(bidids), dtype=bool)
            if clicked_indices:
                for idx in clicked_indices:
                    if bidids[idx] in conv_set:
                        conv_mask[idx] = True

            n_rows = len(rows)
            total_imps += n_rows
            total_clicks += int(clicked_mask.sum())
            total_convs += int(conv_mask.sum())

            X = build_feature_matrix(rows, clicked_mask, conv_mask, stats_map)
            y_ctr = clicked_mask.astype(np.int8)

            if not ctr_init:
                ctr_model.partial_fit(X, y_ctr, classes=np.array([0, 1], dtype=np.int8))
                ctr_init = True
            else:
                ctr_model.partial_fit(X, y_ctr)

            if clicked_indices:
                X_cvr = X[clicked_mask]
                y_cvr = conv_mask[clicked_mask].astype(np.int8)
                if not cvr_init and len(y_cvr) >= 50:
                    cvr_model.partial_fit(
                        X_cvr, np.zeros_like(y_cvr), classes=np.array([0, 1], dtype=np.int8)
                    )
                    cvr_init = True
                if len(np.unique(y_cvr)) == 2:
                    if not cvr_init:
                        cvr_model.partial_fit(
                            X_cvr, y_cvr, classes=np.array([0, 1], dtype=np.int8)
                        )
                        cvr_init = True
                    else:
                        cvr_model.partial_fit(X_cvr, y_cvr)

            file_rows += n_rows
            if file_rows % LOG_EVERY_ROWS < n_rows:
                logger.info(f"IMP rows so far: {file_rows}")

        logger.info(f"Completed IMP {item['name']} rows={file_rows}")

    return (
        ctr_model,
        cvr_model,
        ctr_init,
        cvr_init,
        stats_map,
        total_imps,
        total_clicks,
        total_convs,
    )

def build_stats(stats_map):
    stats_out = {}
    for adv, values in stats_map.items():
        imps, sum_pay, clicks, convs = values
        if imps == 0:
            continue
        n_val = N_MAP.get(str(adv), 0)
        avg_mp = sum_pay / imps
        avg_ev = (clicks + n_val * convs) / imps
        stats_out[str(adv)] = {
            "avg_mp": float(avg_mp),
            "avg_ev": float(avg_ev),
        }
    return stats_out

def model_payload(model, trained):
    if trained and hasattr(model, "coef_"):
        coef = model.coef_
        intercept = model.intercept_
    else:
        coef = np.zeros((1, HASH_SPACE), dtype=np.float32)
        intercept = np.zeros((1,), dtype=np.float32)
    return {
        "coef": coef,
        "intercept": intercept,
        "classes": [0, 1],
        "n_features": HASH_SPACE,
        "hash_space": HASH_SPACE,
    }

def main():
    root = os.getcwd()
    items = discover_items(root)

    items_sorted = sorted(items, key=lambda x: (x["type"], x["display"]))
    logger.info(f"Discovered {len(items_sorted)} files")
    for item in items_sorted:
        logger.info(f"{item['type']}\t{item['name']}\tsize={item['size']}")

    imps = [i for i in items_sorted if i["type"] == "IMP"]
    clks = [i for i in items_sorted if i["type"] == "CLK"]
    convs = [i for i in items_sorted if i["type"] == "CONV"]

    if not imps:
        logger.error("ERROR no impression files found")
        sys.exit(1)

    conn = init_db(SQLITE_DB)
    index_events(conn, clks, "clicks", "CLK")
    index_events(conn, convs, "convs", "CONV")

    (
        ctr_model,
        cvr_model,
        ctr_init,
        cvr_init,
        stats_map,
        total_imps,
        total_clicks,
        total_convs,
    ) = process_impressions(imps, conn)

    stats_out = build_stats(stats_map)

    output = {
        "ctr": model_payload(ctr_model, ctr_init),
        "cvr": model_payload(cvr_model, cvr_init),
        "stats": stats_out,
    }

    with open("src/model_weights.pkl", "wb") as f:
        pickle.dump(output, f)

    logger.info("Final summary")
    logger.info(f"Total impressions: {total_imps}")
    logger.info(f"Total clicks: {total_clicks}")
    logger.info(f"Total conversions: {total_convs}")
    logger.info("Advertiser stats")
    for adv in sorted(stats_out.keys()):
        stat = stats_out[adv]
        logger.info(f"{adv}\tavg_mp={stat['avg_mp']:.6f}\tavg_ev={stat['avg_ev']:.6f}")

    conn.close()

    if os.path.exists(SQLITE_DB):
        os.remove(SQLITE_DB)

if __name__ == "__main__":
    main()
