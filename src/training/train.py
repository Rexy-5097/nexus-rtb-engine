import fnmatch
import io
import logging
import os
import pickle
import random
import sqlite3
import sys
import zipfile

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, vstack
from sklearn.isotonic import IsotonicRegression
# Models
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

from src.bidding.config import config
from src.bidding.features import FeatureExtractor

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
HASH_SPACE = config.model.hash_space
CHUNK_SIZE = 100_000
SQLITE_DB = "bid_events.db"
SQLITE_BATCH = 10_000

IMP_COLS = [
    "bidid",
    "timestamp",
    "userid",
    "ua",
    "region",
    "city",
    "domain",
    "vis",
    "fmt",
    "floorprice",
    "payingprice",
    "advertiser",
]
IMP_USECOLS_NEW = [0, 1, 3, 4, 6, 7, 9, 15, 16, 18, 20, 22]


def match_type(name: str):
    if fnmatch.fnmatch(name, "imp.*.txt"):
        return "IMP"
    if fnmatch.fnmatch(name, "clk.*.txt"):
        return "CLK"
    if fnmatch.fnmatch(name, "conv.*.txt"):
        return "CONV"
    return None


def download_mock_data(root: str):
    """Generate mock data if no data exists."""
    logger.info("Generating Mock Data with LogNormal Market Prices...")
    os.makedirs(root, exist_ok=True)

    n_samples = 50000  # Phase 7: Increased for more robust signal
    with open(os.path.join(root, "imp.20131019.txt"), "w") as f:
        for i in range(n_samples):
            bidid = f"bid_{i}"
            ts = "20131019000000"
            # User ID pool (Simulate repeat users)
            # Pool size 5000 -> Avg 3 imps/user
            userid = f"user_{random.randint(0, 5000)}"
            ua = "Mozilla/5.0"
            reg = random.choice(["CA", "NY", "TX"])
            city = random.choice(["CityA", "CityB"])
            dom = random.choice(["example.com", "news.com", "tech.org"])
            vis = random.choice(["0", "1", "2"])
            fmt = random.choice(["0", "1"])

            # Floor Price (random 0 to 50)
            floor = int(random.choice([0, 10, 20, 50]))

            # Log-Normal Price (Mean ~80, Heavy Tail)
            # mu=4.2, sigma=0.6 -> median=66, mean=80
            pay = int(np.random.lognormal(4.2, 0.6))

            adv = random.choice(["1458", "3358", "3386"])

            row = ["0"] * 25
            row[0] = bidid
            row[1] = ts
            row[3] = userid  # User ID
            row[4] = ua
            row[6] = reg
            row[7] = city
            row[9] = dom
            row[15] = vis
            row[16] = fmt
            row[18] = str(floor)  # Floor
            row[20] = str(pay)
            row[22] = adv

            f.write("\t".join(row) + "\n")

    # Phase 7: Feature-dependent CTR/CVR for meaningful AUC
    # We need to store impression metadata for click generation
    impression_meta = []
    with open(os.path.join(root, "imp.20131019.txt"), "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            impression_meta.append(
                {
                    "bidid": parts[0],
                    "adv": parts[22],
                    "dom": parts[9],
                    "vis": parts[15],
                    "floor": int(parts[18]),
                }
            )

    # CTR depends on features (structured signal)
    adv_ctr = {"1458": 0.08, "3358": 0.03, "3386": 0.05}
    dom_ctr = {"example.com": 1.2, "news.com": 0.8, "tech.org": 1.0}
    vis_ctr = {"0": 0.5, "1": 1.0, "2": 1.5}

    clicked_bids = set()
    with open(os.path.join(root, "clk.20131019.txt"), "w") as f:
        for imp in impression_meta:
            base_ctr = adv_ctr.get(imp["adv"], 0.05)
            ctr = base_ctr * dom_ctr.get(imp["dom"], 1.0) * vis_ctr.get(imp["vis"], 1.0)
            ctr = min(ctr, 0.3)  # Cap
            if random.random() < ctr:
                clicked_bids.add(imp["bidid"])
                f.write(f"{imp['bidid']}\t20131019000001\n")

    # Conv: 10% of clicks, higher for adv 3476 (if present), and floor > 0
    with open(os.path.join(root, "conv.20131019.txt"), "w") as f:
        for imp in impression_meta:
            if imp["bidid"] in clicked_bids:
                conv_rate = 0.10 if imp["adv"] == "1458" else 0.05
                if imp["floor"] > 20:
                    conv_rate *= 1.5  # Higher floor = better quality
                if random.random() < conv_rate:
                    f.write(f"{imp['bidid']}\t20131019000002\n")


def discover_items(root: str):
    items = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            fpath = os.path.join(dirpath, fname)
            if fname.lower().endswith(".zip"):
                try:
                    with zipfile.ZipFile(fpath) as zf:
                        for info in zf.infolist():
                            if info.is_dir() or info.file_size > 2e9:
                                continue
                            typ = match_type(os.path.basename(info.filename))
                            if typ:
                                items.append(
                                    {
                                        "type": typ,
                                        "kind": "zip",
                                        "zip_path": fpath,
                                        "entry": info.filename,
                                        "name": info.filename,
                                        "display": f"{fpath}::{info.filename}",
                                    }
                                )
                except:
                    pass
                continue
            typ = match_type(fname)
            if typ:
                items.append(
                    {
                        "type": typ,
                        "kind": "file",
                        "path": fpath,
                        "name": fname,
                        "display": fpath,
                    }
                )
    return items


def iter_chunks(item, usecols=None):
    try:
        if item["kind"] == "zip":
            with zipfile.ZipFile(item["zip_path"]) as zf:
                with zf.open(item["entry"]) as f:
                    text = io.TextIOWrapper(f, encoding="utf-8", errors="replace")
                    for chunk in pd.read_csv(
                        text,
                        sep="\t",
                        header=None,
                        chunksize=CHUNK_SIZE,
                        usecols=usecols,
                        dtype=str,
                        on_bad_lines="skip",
                        low_memory=True,
                    ):
                        yield chunk
        else:
            for chunk in pd.read_csv(
                item["path"],
                sep="\t",
                header=None,
                chunksize=CHUNK_SIZE,
                usecols=usecols,
                dtype=str,
                on_bad_lines="skip",
                low_memory=True,
            ):
                yield chunk
    except Exception as e:
        logger.error(f"Error reading {item['display']}: {e}")


# SQLite Event Indexing
def init_db(path):
    if os.path.exists(path):
        os.remove(path)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE clicks (bidid TEXT PRIMARY KEY)")
    conn.execute("CREATE TABLE convs (bidid TEXT PRIMARY KEY)")
    return conn


def index_events(conn, items, table):
    cur = conn.cursor()
    for item in items:
        for chunk in iter_chunks(item, usecols=[0]):
            bidids = [b for b in chunk[0] if str(b) != "nan"]
            if bidids:
                cur.executemany(
                    f"INSERT OR IGNORE INTO {table} (bidid) VALUES (?)",
                    [(b,) for b in bidids],
                )
        conn.commit()


def fetch_events(conn, table, bidids):
    found = set()
    cur = conn.cursor()
    for i in range(0, len(bidids), 900):
        batch = bidids[i : i + 900]
        if not batch:
            continue
        q = ",".join("?" * len(batch))
        cur.execute(f"SELECT bidid FROM {table} WHERE bidid IN ({q})", batch)
        found.update(r[0] for r in cur.fetchall())
    return found


# Feature Matrix Builder
def build_matrix(rows, clicked_mask, conv_mask, stats_map, feature_extractor, scaler):
    n = len(rows)
    row_ind, col_ind, data = [], [], []

    class MockRequest:
        __slots__ = (
            "bidId",
            "timestamp",
            "visitorId",
            "userAgent",
            "region",
            "city",
            "domain",
            "adSlotVisibility",
            "adSlotFormat",
            "adSlotFloorPrice",
            "advertiserId",
            "adSlotWidth",
            "adSlotHeight",
            "adv_ctr_1d",
            "adv_ctr_7d",
            "dom_ctr_1d",
            "dom_ctr_7d",
            "slot_ctr",
            "user_ctr",
            "user_count_7d",
        )

        def __init__(self, r):
            self.bidId = r.bidid
            self.timestamp = r.timestamp
            self.visitorId = getattr(r, "userid", "0")
            self.userAgent = r.ua
            self.region = r.region
            self.city = r.city
            self.domain = r.domain
            self.adSlotVisibility = r.vis
            self.adSlotFormat = r.fmt
            self.advertiserId = r.advertiser
            self.adSlotFloorPrice = str(getattr(r, "floorprice", "0"))
            self.adSlotWidth = "0"
            self.adSlotHeight = "0"

            # Rolling Stats (from DataFrame row)
            self.adv_ctr_1d = getattr(r, "adv_ctr_1d", 0.0)
            self.adv_ctr_7d = getattr(r, "adv_ctr_7d", 0.0)
            self.dom_ctr_1d = getattr(r, "dom_ctr_1d", 0.0)
            self.dom_ctr_7d = getattr(r, "dom_ctr_7d", 0.0)
            self.slot_ctr = getattr(r, "slot_ctr", 0.0)

            # Phase 7: User History Signals
            self.user_ctr = getattr(r, "user_ctr", 0.0)
            self.user_count_7d = getattr(r, "user_count_7d", 0.0)

    for i, row in enumerate(rows):
        req = MockRequest(row)
        feats = feature_extractor.extract(req, scaler, stats_map)
        for h, val in feats:
            row_ind.append(i)
            col_ind.append(h)
            data.append(val)

    if not row_ind:
        return None
    return csr_matrix((data, (row_ind, col_ind)), shape=(n, HASH_SPACE))


def calc_ece(y_true, y_prob, n_bins=10):
    """Expected Calibration Error"""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1

    num = len(y_prob)
    ece = 0.0
    for i in range(n_bins):
        idx = binids == i
        if not np.any(idx):
            continue
        avg_pred = np.mean(y_prob[idx])
        avg_true = np.mean(y_true[idx])
        ece += (np.sum(idx) / num) * np.abs(avg_pred - avg_true)
    return ece


def compute_rolling_stats(df, stats):
    """
    Compute 1d and 7d rolling statistics for Advertiser and Domain.
    Updates the 'stats' dictionary with the final snapshot values for inference.
    returns enriched DataFrame.
    """
    logger.info("Computing Rolling Statistics (1d, 7d)...")

    # Sort by timestamp to ensure correct rolling window
    df["timestamp"] = df["timestamp"].astype(float)
    df = df.sort_values("timestamp")

    # Helper for Bayesian Smoothing
    # Alpha/Beta can be tuned. Using heuristics: alpha=10, beta=400 ~ 2.5% CTR prior
    def calc_smoothed_ctr(clicks, count, alpha=10, beta=400):
        return (clicks + alpha) / (count + beta)

    # Config
    windows = {"1d": 86400, "7d": 7 * 86400}

    # Pre-init columns
    for w in windows:
        df[f"adv_ctr_{w}"] = 0.0
        df[f"dom_ctr_{w}"] = 0.0
    df["slot_ctr"] = 0.0

    # State Containers
    # key -> window -> key -> (deque, [clicks, count])
    from collections import defaultdict, deque

    state_adv = {w: defaultdict(lambda: (deque(), [0, 0])) for w in windows}
    state_dom = {w: defaultdict(lambda: (deque(), [0, 0])) for w in windows}
    state_slot = defaultdict(lambda: [0, 0])  # Cumulative for slot

    # We collect results in lists then assign to columns
    res_adv = {w: [] for w in windows}
    res_dom = {w: [] for w in windows}
    res_slot = []

    # Iteration (using dict records for speed)
    # This is O(N) single pass.
    # Use correct column names from IMP_COLS ("advertiser", "fmt", "vis")
    recs = df[["timestamp", "advertiser", "domain", "fmt", "vis", "click"]].to_dict(
        "records"
    )

    for r in recs:
        ts = r["timestamp"]
        adv = str(r["advertiser"])
        dom = str(r["domain"])
        slot_key = f"{r['fmt']}_{r['vis']}"
        clk = r["click"]

        # 1. Update & Read Adv/Dom Windows
        for w, w_sec in windows.items():
            # Adv
            q, s = state_adv[w][adv]
            # Remove old events
            while q and q[0][0] < ts - w_sec:
                old_ts, old_clk = q.popleft()
                s[0] -= old_clk
                s[1] -= 1
            # Compute Feature (Current State)
            res_adv[w].append(calc_smoothed_ctr(s[0], s[1]))
            # Add current to history (for NEXT row)
            q.append((ts, clk))
            s[0] += clk
            s[1] += 1

            # Dom
            q, s = state_dom[w][dom]
            while q and q[0][0] < ts - w_sec:
                old_ts, old_clk = q.popleft()
                s[0] -= old_clk
                s[1] -= 1
            res_dom[w].append(calc_smoothed_ctr(s[0], s[1]))
            q.append((ts, clk))
            s[0] += clk
            s[1] += 1

        # 2. Update & Read Slot (Cumulative)
        s = state_slot[slot_key]
        res_slot.append(calc_smoothed_ctr(s[0], s[1], alpha=1, beta=20))
        s[0] += clk
        s[1] += 1

    # Assign columns
    for w in windows:
        df[f"adv_ctr_{w}"] = res_adv[w]
        df[f"dom_ctr_{w}"] = res_dom[w]
    df["slot_ctr"] = res_slot

    # Export Final States for Inference (Feature Store Snapshot)
    for w in windows:
        for k, (q, s) in state_adv[w].items():
            stats[f"adv_{w}:{k}"] = calc_smoothed_ctr(s[0], s[1])
        for k, (q, s) in state_dom[w].items():
            stats[f"dom_{w}:{k}"] = calc_smoothed_ctr(s[0], s[1])

    for k, s in state_slot.items():
        stats[f"slot:{k}"] = calc_smoothed_ctr(s[0], s[1], alpha=1, beta=20)

    return df


def load_dataset(data_root="data"):
    items = discover_items(data_root)

    # Check if data exists, if not generate
    if not items:
        download_mock_data(data_root)
        items = discover_items(data_root)

    items.sort(key=lambda x: x["name"])

    imps = [i for i in items if i["type"] == "IMP"]
    clks = [i for i in items if i["type"] == "CLK"]
    convs = [i for i in items if i["type"] == "CONV"]

    if not imps:
        sys.exit("No data found even after generation attempt")

    conn = init_db(SQLITE_DB)
    index_events(conn, clks, "clicks")
    index_events(conn, convs, "convs")

    # Collect all impression data into a single DataFrame for rolling stats
    all_imps_df_parts = []

    # We need to iterate chunks but we want to build a full DF for rolling.
    # Warning: If dataset is huge, this might OOM.
    # But user requirement for "rolling 7d" implies global time-awareness.
    # Given the constraints, we will load full dataset for feature engineering.
    # Phase 6 assumes we maximize quality.

    logger.info("loading full dataset for rolling stats...")
    for item in imps:
        for chunk in iter_chunks(item, usecols=IMP_USECOLS_NEW):
            chunk.columns = IMP_COLS
            all_imps_df_parts.append(chunk)

    if not all_imps_df_parts:
        sys.exit("No data found.")

    full_df = pd.concat(all_imps_df_parts, ignore_index=True)

    # Enrich with Labels (Click/Conv)
    bidids = full_df["bidid"].tolist()
    # Optimization: fetch all events in bulk if set is too large?
    # fetch_events returns a set.
    c_set = fetch_events(conn, "clicks", bidids)
    v_set = fetch_events(conn, "convs", bidids)

    full_df["click"] = full_df["bidid"].apply(lambda x: 1 if x in c_set else 0)
    full_df["conversion"] = full_df["bidid"].apply(lambda x: 1 if x in v_set else 0)

    # --- PHASE 7: Signal Quality Filtering ---
    initial_len = len(full_df)
    logger.info(f"Phase 7: Filtering Data (Initial: {initial_len})")

    # 1. Invalid Prices
    # payingprice must be > 0.
    # payingprice < 0 should not exist, but let's be safe.
    # Note: 'payingprice' is string in IMP_COLS (index 20), loaded as object.
    full_df["payingprice"] = pd.to_numeric(
        full_df["payingprice"], errors="coerce"
    ).fillna(0)
    full_df = full_df[full_df["payingprice"] > 0]

    # 1b. Floor Check (New for Phase 7)
    if "floorprice" in full_df.columns:
        full_df["floorprice"] = pd.to_numeric(
            full_df["floorprice"], errors="coerce"
        ).fillna(0)
        # Filter bids where paying price < floor (Invalid market data)
        # Assuming payingprice is the winning price, it must be >= floor.
        full_df = full_df[full_df["payingprice"] >= full_df["floorprice"]]

    # 2. Low-Frequency Advertisers (< 100 imps)
    adv_counts = full_df["advertiser"].value_counts()
    valid_advs = adv_counts[adv_counts >= 100].index
    full_df = full_df[full_df["advertiser"].isin(valid_advs)]

    # 3. Low-Frequency Domains (< 100 imps)
    dom_counts = full_df["domain"].value_counts()
    valid_doms = dom_counts[dom_counts >= 100].index
    full_df = full_df[full_df["domain"].isin(valid_doms)]

    # 4. Cold-Start Noise (Users with < 2 events)
    if "userid" in full_df.columns:
        user_counts = full_df["userid"].value_counts()
        # Keep users with at least 2 events (History available)
        # "Cold start" usually means no history.
        # If we want to use "User Signals", we need history.
        valid_users = user_counts[user_counts >= 2].index
        full_df = full_df[full_df["userid"].isin(valid_users)]

    logger.info(
        f"Phase 7: Data Filtered. New Size: {len(full_df)} (Dropped {initial_len - len(full_df)})"
    )

    # --- Phase 7: Top-K Maps ---
    logger.info("Phase 7: Computing Top-K Encoding Maps...")
    top_k_maps = {}
    for col, k in [("advertiser", 50), ("domain", 200), ("region", 50), ("city", 100)]:
        if col in full_df.columns:
            vc = full_df[col].value_counts()
            top_k_maps[col] = vc.head(k).index.tolist()
    top_k_counts = {k: len(v) for k, v in top_k_maps.items()}
    logger.info(f"Top-K Maps: {top_k_counts}")

    # --- Phase 7: User History Signals ---
    if "userid" in full_df.columns:
        logger.info("Phase 7: Computing User History Signals...")
        # Sort by timestamp for time-aware aggregation
        full_df = full_df.sort_values("timestamp").reset_index(drop=True)

        # Expanding (cumulative) user stats - time-aware, no leakage
        # For each row, compute stats from ALL PRIOR rows for that user
        full_df["cum_user_imps"] = full_df.groupby(
            "userid"
        ).cumcount()  # 0-indexed count before this row
        full_df["cum_user_clicks"] = (
            full_df.groupby("userid")["click"].cumsum() - full_df["click"]
        )  # exclude current

        # user_ctr = prior_clicks / prior_imps (with smoothing)
        full_df["user_ctr"] = (full_df["cum_user_clicks"] + 1) / (
            full_df["cum_user_imps"] + 20
        )
        full_df["user_count_7d"] = full_df[
            "cum_user_imps"
        ]  # Approximation (all history as proxy for 7d)

        # Clean up temp columns
        full_df.drop(columns=["cum_user_imps", "cum_user_clicks"], inplace=True)
    else:
        full_df["user_ctr"] = 0.0
        full_df["user_count_7d"] = 0.0

    # --- Rolling Stats ---
    stats = {}
    full_df = compute_rolling_stats(full_df, stats)

    # --- Pass 1: Global Statistics (Post-Rolling) ---
    logger.info("Pass 1: Computing Global Statistics...")

    num_stats = {
        k: {"sum": 0.0, "sq": 0.0, "n": 0}
        for k in [
            "region",
            "city",
            "adslot_visibility",
            "adslot_format",
            "ad_slot_area",
        ]
    }
    global_stats = {}

    def update_num(n, v):
        try:
            f = float(v)
            num_stats[n]["sum"] += f
            num_stats[n]["sq"] += f * f
            num_stats[n]["n"] += 1
        except:
            pass

    def update_counter(k, price, click, conv):
        if k not in global_stats:
            global_stats[k] = [0, 0.0, 0, 0]
        s = global_stats[k]
        s[0] += 1
        s[1] += price
        if click:
            s[2] += 1
        if conv:
            s[3] += 1

    # Ensure numeric columns for aggregation
    full_df["payingprice"] = pd.to_numeric(
        full_df["payingprice"], errors="coerce"
    ).fillna(0)
    full_df["click"] = pd.to_numeric(full_df["click"], errors="coerce").fillna(0)
    full_df["conversion"] = pd.to_numeric(
        full_df["conversion"], errors="coerce"
    ).fillna(0)

    # 1. Global Aggregates (Adv, Dom, Adv-Dom)
    # Advertiser
    g = full_df.groupby("advertiser").agg(
        {"payingprice": ["count", "sum"], "click": "sum", "conversion": "sum"}
    )
    for adv, row in g.iterrows():
        global_stats[str(adv)] = [
            row[("payingprice", "count")],
            row[("payingprice", "sum")],
            row[("click", "sum")],
            row[("conversion", "sum")],
        ]

    # Domain
    g = full_df.groupby("domain").agg(
        {"payingprice": ["count", "sum"], "click": "sum", "conversion": "sum"}
    )
    for dom, row in g.iterrows():
        global_stats[f"domain:{dom}"] = [
            row[("payingprice", "count")],
            row[("payingprice", "sum")],
            row[("click", "sum")],
            row[("conversion", "sum")],
        ]

    # Adv-Domain
    # NOTE: Groupby multiple columns
    g = full_df.groupby(["advertiser", "domain"]).agg(
        {"payingprice": ["count", "sum"], "click": "sum", "conversion": "sum"}
    )
    for (adv, dom), row in g.iterrows():
        global_stats[f"adv_domain:{adv}_{dom}"] = [
            row[("payingprice", "count")],
            row[("payingprice", "sum")],
            row[("click", "sum")],
            row[("conversion", "sum")],
        ]

    # Numeric Stats
    # Available in IMP_COLS: region, city, vis, fmt
    for col in ["region", "city", "vis", "fmt"]:
        # Ensure numeric
        s = pd.to_numeric(full_df[col], errors="coerce").fillna(0)

        # Manually update num_stats dict
        # s is a series.
        count = len(s)
        total = s.sum()
        sq_sum = (s * s).sum()

        # Re-map keys to match feature names if needed?
        # FeatureScaler keys must match features.py extract() output keys.
        # features.py uses "region", "city", "adslot_visibility", "adslot_format".
        # We need to map 'vis' -> 'adslot_visibility', 'fmt' -> 'adslot_format'.

        target_k = col
        if col == "vis":
            target_k = "adslot_visibility"
        if col == "fmt":
            target_k = "adslot_format"

        if target_k not in num_stats:
            # If target_k was initialized with 0s, we use it.
            # But checks line 356 initialization.
            pass

        num_stats[target_k]["sum"] = total
        num_stats[target_k]["sq"] = sq_sum
        num_stats[target_k]["n"] = count

    # Merge rolling stats snapshot into global_stats for export?
    # features.py expects 'stats' dict.
    # We have 'stats' from compute_rolling_stats.
    # We have 'global_stats' from aggregation.
    # Merge them.
    stats.update(global_stats)

    # Finalize Scaler
    scaler = {}
    for k, v in num_stats.items():
        if v["n"] > 1:
            mean = v["sum"] / v["n"]
            std = np.sqrt((v["sq"] - v["sum"] ** 2 / v["n"]) / (v["n"] - 1))
            scaler[k] = (mean, std)
        else:
            scaler[k] = (0.0, 1.0)

    conn.close()
    if os.path.exists(SQLITE_DB):
        os.remove(SQLITE_DB)
    return full_df, global_stats, scaler, stats, top_k_maps


def main():
    full_df, global_stats, scaler, stats, top_k_maps = load_dataset()

    # --- Pass 2: Data Loading & Splitting ---
    logger.info("Pass 2: Loading Data for Training...")

    X_parts = []
    y_ctr_parts = []
    y_cvr_parts = []

    # Iterate full_df (already loaded and enriched)
    # We process in chunks to match build_matrix structure

    feature_extractor = FeatureExtractor()
    feature_extractor.set_encoding_maps(top_k_maps)

    chunk_size = 100000
    for start in range(0, len(full_df), chunk_size):
        chunk = full_df.iloc[start : start + chunk_size]

        # Get labels directly from DF
        c_mask = chunk["click"].values.astype(np.int8)
        v_mask = chunk["conversion"].values.astype(np.int8)

        rows = [row for row in chunk.itertuples(index=False)]
        mat = build_matrix(
            rows, c_mask, v_mask, global_stats, feature_extractor, scaler
        )

        if mat is not None:
            X_parts.append(mat)
            y_ctr_parts.append(c_mask)
            y_cvr_parts.append(v_mask)

    if not X_parts:
        sys.exit("No training data loaded")

    X_full = vstack(X_parts)
    y_ctr_full = np.concatenate(y_ctr_parts)
    y_cvr_full = np.concatenate(y_cvr_parts)

    logger.info(
        f"Loaded Dataset: {X_full.shape} samples. CTR+: {y_ctr_full.mean():.4f} CVR+: {y_cvr_full.mean():.4f}"
    )

    # Time-based Split
    n_samples = X_full.shape[0]
    n_train = int(n_samples * 0.70)
    n_val = int(n_samples * 0.15)

    X_tr = X_full[:n_train]
    y_ctr_tr = y_ctr_full[:n_train]
    y_cvr_tr = y_cvr_full[:n_train]

    X_va = X_full[n_train : n_train + n_val]
    y_ctr_va = y_ctr_full[n_train : n_train + n_val]
    y_cvr_va = y_cvr_full[n_train : n_train + n_val]

    X_te = X_full[n_train + n_val :]
    y_ctr_te = y_ctr_full[n_train + n_val :]
    y_cvr_te = y_cvr_full[n_train + n_val :]

    def filter_clicks(X, y_click, y_conv):
        mask = y_click == 1
        if mask.sum() == 0:
            return None, None
        return X[mask], y_conv[mask]

    X_tr_cvr, y_tr_cvr = filter_clicks(X_tr, y_ctr_tr, y_cvr_tr)
    X_va_cvr, y_va_cvr = filter_clicks(X_va, y_ctr_va, y_cvr_va)
    X_te_cvr, y_te_cvr = filter_clicks(X_te, y_ctr_te, y_cvr_te)

    logger.info(
        f"CVR Train Samples: {X_tr_cvr.shape[0] if X_tr_cvr is not None else 0}"
    )

    # --- Pass 3: Model Selection ---

    def train_and_select(X_t, y_t, X_v, y_v, X_tt, y_tt, label="CTR"):
        logger.info(f"--- Training {label} Model ---")
        if X_t is None or X_t.shape[0] < 10:
            logger.warning(f"Not enough data for {label}")
            return None, {}, None

        # Handle Class Imbalance
        pos_rate = y_t.mean()
        scale_pos_weight = 1.0
        class_weight = None
        if pos_rate < 0.2:
            scale_pos_weight = (1.0 - pos_rate) / (pos_rate + 1e-6)
            class_weight = "balanced"
            logger.info(
                f"Applying Class Weighting (Pos Rate={pos_rate:.4f}, LGB Scale={scale_pos_weight:.2f})"
            )

        best_mdl = None
        best_score = 0.0
        best_p = {}

        # Grid
        Cs = [0.01, 1.0, 10.0]

        # 1. Logistic Regression
        for C in Cs:
            # logger.info(f"Evaluating LR(C={C}, class_weight={class_weight})...")
            try:
                model = LogisticRegression(
                    solver="saga",
                    penalty="elasticnet",
                    l1_ratio=0.5,
                    C=C,
                    class_weight=class_weight,
                    max_iter=100,
                    random_state=42,
                )
                model.fit(X_t, y_t)
                probs = model.predict_proba(X_v)[:, 1]
                auc = roc_auc_score(y_v, probs)
                # logger.info(f"LR(C={C}) AUC: {auc:.4f}")
                if auc > best_score:
                    best_score = auc
                    best_mdl = model
                    best_p = {"type": "LR", "C": C}
            except Exception:
                pass

        # 2. LightGBM (Phase 7: Tuning Grid w/ Regularization)
        if lgb and X_t.shape[0] > 500:
            lgb_configs = [
                {
                    "n_estimators": 300,
                    "learning_rate": 0.03,
                    "num_leaves": 15,
                    "max_depth": 4,
                    "min_data_in_leaf": 200,
                    "feature_fraction": 0.7,
                    "bagging_fraction": 0.8,
                    "bagging_freq": 5,
                    "lambda_l1": 10.0,
                    "lambda_l2": 10.0,
                },
                {
                    "n_estimators": 500,
                    "learning_rate": 0.02,
                    "num_leaves": 15,
                    "max_depth": 4,
                    "min_data_in_leaf": 300,
                    "feature_fraction": 0.5,
                    "bagging_fraction": 0.6,
                    "bagging_freq": 5,
                    "lambda_l1": 5.0,
                    "lambda_l2": 5.0,
                },
            ]
            for lgb_cfg in lgb_configs:
                try:
                    model = lgb.LGBMClassifier(
                        objective="binary",
                        metric="auc",
                        scale_pos_weight=scale_pos_weight,
                        verbose=-1,
                        **lgb_cfg,
                    )
                    model.fit(
                        X_t,
                        y_t,
                        eval_set=[(X_v, y_v)],
                        eval_metric="auc",
                        callbacks=[lgb.early_stopping(20)],
                    )
                    probs = model.predict_proba(X_v)[:, 1]
                    auc = roc_auc_score(y_v, probs)
                    logger.info(
                        f"LGBM({lgb_cfg['num_leaves']}L, d{lgb_cfg['max_depth']}) Val AUC: {auc:.4f}"
                    )
                    if auc > best_score:
                        best_score = auc
                        best_mdl = model
                        best_p = {"type": "LGBM", **lgb_cfg}
                except Exception as e:
                    logger.warning(f"LGBM config failed: {e}")

        if best_mdl:
            # Train AUC (for overfitting check)
            probs_tr = best_mdl.predict_proba(X_t)[:, 1]
            try:
                auc_tr = roc_auc_score(y_t, probs_tr)
            except:
                auc_tr = 0.5

            probs_te = best_mdl.predict_proba(X_tt)[:, 1]
            try:
                auc_te = roc_auc_score(y_tt, probs_te)
            except:
                auc_te = 0.5
            ll = log_loss(y_tt, probs_te)
            br = brier_score_loss(y_tt, probs_te)
            ece = calc_ece(y_tt, probs_te)

            # Phase 7: Overfitting Check
            auc_gap = auc_tr - auc_te
            overfit_flag = "⚠️ OVERFIT" if auc_gap > 0.03 else "✅ OK"
            logger.info(
                f"{label} TRAIN AUC={auc_tr:.4f}, TEST AUC={auc_te:.4f}, Gap={auc_gap:.4f} {overfit_flag}"
            )
            logger.info(
                f"{label} TEST: LogLoss={ll:.4f}, Brier={br:.4f}, ECE={ece:.4f}"
            )

            # Calibration
            iso = IsotonicRegression(out_of_bounds="clip")
            probs_val = best_mdl.predict_proba(X_v)[:, 1]
            iso.fit(probs_val, y_v)

            return best_mdl, best_p, iso

        return None, {}, None

    final_ctr_model, ctr_params, ctr_calib = train_and_select(
        X_tr, y_ctr_tr, X_va, y_ctr_va, X_te, y_ctr_te, "CTR"
    )
    final_cvr_model, cvr_params, cvr_calib = train_and_select(
        X_tr_cvr, y_tr_cvr, X_va_cvr, y_va_cvr, X_te_cvr, y_te_cvr, "CVR"
    )

    # --- Pass 4: Market Price Modeling ---
    logger.info("--- Training Market Price Model (Regressor) ---")

    # Target: Log1p(payingprice)
    y_price_tr = np.log1p(full_df.iloc[:n_train]["payingprice"].values)
    y_price_va = np.log1p(full_df.iloc[n_train : n_train + n_val]["payingprice"].values)
    y_price_te = np.log1p(full_df.iloc[n_train + n_val :]["payingprice"].values)

    price_model = None
    price_metrics = {}

    if lgb:
        try:
            model = lgb.LGBMRegressor(
                objective="regression",
                metric="rmse",
                n_estimators=100,
                learning_rate=0.05,
                num_leaves=31,
                verbose=-1,
            )
            model.fit(
                X_tr,
                y_price_tr,
                eval_set=[(X_va, y_price_va)],
                eval_metric="rmse",
                callbacks=[lgb.early_stopping(10)],
            )
            price_model = model

            # Test Eval
            preds = model.predict(X_te)
            rmse = np.sqrt(np.mean((y_price_te - preds) ** 2))
            price_metrics = {"rmse": rmse}
            logger.info(f"Price Model Test RMSE: {rmse:.4f}")

        except Exception as e:
            logger.error(f"Price Model Failed: {e}")

    # Save Artifacts
    with open("src/model_weights.pkl", "wb") as f:
        pickle.dump(
            {
                "ctr": (final_ctr_model, ctr_params, ctr_calib),
                "cvr": (final_cvr_model, cvr_params, cvr_calib),
                "price_model": price_model,
                "price_metrics": price_metrics,
                "scaler": scaler,
                "stats": stats,
                "config": config,
                "top_k_maps": top_k_maps,
                "ver": "phase7",
            },
            f,
        )

    # Print stats for report
    total_imps = len(full_df)
    total_spend = full_df["payingprice"].sum()
    total_clicks = full_df["click"].sum()
    total_convs = full_df["conversion"].sum()

    logger.info(
        f"TRAIN STATS: Imps={total_imps} Spend={total_spend:.2f} Clicks={total_clicks} Convs={total_convs}"
    )

    logger.info("Training complete. Weights saved.")


if __name__ == "__main__":
    main()
