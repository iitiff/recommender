"""Are the 7 hand-made path families real? Data-driven clusters of 12-month paths.

Sequence analysis with sequenzo (a Python port of the R package TraMineR):
  - each repeat customer's first 12 months after the first order as a sequence
    of monthly states: "0" no order, "1" one order, "2" two or more;
  - distances by optimal matching (OM) with substitution costs from the
    observed transition rates (TRATE) and automatic indel costs;
  - Ward hierarchical clustering, number of clusters chosen by the average
    silhouette width (ASW);
  - compared with the rule-based families of paths_to_loyalty.py (B-G; one-
    and-done customers have no path to cluster) by the adjusted Rand index.
Concepts: optimal matching, sequence clustering, silhouette, adjusted Rand.

hm: the H&M cohort of paths_to_loyalty.py; cdnow: the Q1 1997 first-ever
buyers. A random sample of repeaters keeps the distance matrix small.

Usage:
  uv run --no-project --with sequenzo --with pyarrow --with scikit-learn python -I \
    path_clusters.py {hm TRANSACTIONS_DIR | cdnow FILE} > out.json
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

SAMPLE = 4000
MONTHS = [f"m{i}" for i in range(1, 13)]


def r(x: float, n: int = 3) -> float:
    return round(float(x), n)


def orders_hm(tx_dir: str) -> pd.DataFrame:
    import pyarrow.compute as pc
    import pyarrow.dataset as ds

    t = ds.dataset(tx_dir).to_table(columns=["t_dat", "customer_id"])
    o = pd.DataFrame(
        {
            "c": pc.dictionary_encode(t["customer_id"]).combine_chunks().indices.to_numpy(),
            "d": (pd.to_datetime(t["t_dat"].to_pandas()) - pd.Timestamp("2018-09-20")).dt.days,
        }
    ).drop_duplicates()
    first = o.groupby("c").d.transform("min")
    return o[first.between(181, 272)].assign(first=first)


def orders_cdnow(path: str) -> pd.DataFrame:
    x = pd.read_csv(path, sep=r"\s+", header=None, names=["c", "date", "n", "v"],
                    dtype={"date": str})  # fmt: skip
    d = (pd.to_datetime(x.date, format="%Y%m%d") - pd.Timestamp("1997-01-01")).dt.days
    o = pd.DataFrame({"c": x.c, "d": d}).drop_duplicates()
    return o.assign(first=o.groupby("c").d.transform("min"))


def sequences(o: pd.DataFrame) -> pd.DataFrame:
    o = o[(o.d > o["first"]) & (o.d <= o["first"] + 365)]
    month = np.minimum((o.d - o["first"] - 1) // 30, 11)
    counts = o.groupby([o.c, month]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=range(12), fill_value=0)
    states = counts.clip(upper=2).astype(str)
    states.columns = MONTHS
    return states


def family(row: pd.Series) -> str:
    active = [i + 1 for i, s in enumerate(row) if s != "0"]
    n, first_rep, last_rep = len(active), active[0], active[-1]
    if last_rep <= 3 and n <= 3:
        return "B quick repeat, then gone"
    if n >= 6:
        return "G steady"
    if last_rep >= 10:
        return "F early repeat, still buying" if first_rep <= 3 else "E slow start, still buying"
    return "D slow return, then gone" if first_rep > 3 else "C early repeat, faded"


def main(mode: str, arg: str) -> dict:
    from sequenzo.clustering import Cluster, ClusterQuality
    from sequenzo.define_sequence_data import SequenceData
    from sequenzo.dissimilarity_measures import get_distance_matrix
    from sklearn.metrics import adjusted_rand_score

    states = sequences(orders_hm(arg) if mode == "hm" else orders_cdnow(arg))
    states = states.sample(n=min(SAMPLE, len(states)), random_state=0)
    df = states.reset_index().rename(columns={"c": "id"})
    seq = SequenceData(df, time=MONTHS, states=["0", "1", "2"],
                       labels=["no order", "1 order", "2+ orders"], id_col="id")  # fmt: skip
    dist = get_distance_matrix(seq, method="OM", sm="TRATE", indel="auto")
    matrix = np.asarray(dist)
    cluster = Cluster(matrix, df.id.to_numpy(), clustering_method="ward")
    quality = ClusterQuality(cluster, max_clusters=10)
    quality.compute_cluster_quality_scores()
    table = quality.get_cqi_table()
    asw = {}
    for k in range(2, 11):
        labels = cluster.get_cluster_labels(num_clusters=k)
        asw[k] = silhouette(matrix, np.asarray(labels))
    best = max(asw, key=asw.get)
    labels = np.asarray(cluster.get_cluster_labels(num_clusters=best))
    fam = states.apply(family, axis=1).to_numpy()
    active = (states != "0").sum(axis=1).to_numpy()
    profiles = {}
    for k in np.unique(labels):
        m = labels == k
        f = pd.Series(fam[m]).value_counts(normalize=True)
        rows = states[m]
        profiles[str(k)] = {
            "share": r(m.mean()),
            "mean_active_months": r(active[m].mean(), 1),
            "active_in_months_10_12": r((rows[MONTHS[9:]] != "0").any(axis=1).mean()),
            "share_with_2plus_in_a_month": r((rows == "2").any(axis=1).mean()),
            "share_active_by_month": [r((rows[c] != "0").mean(), 2) for c in MONTHS],
            "rule_families": {k2: r(v) for k2, v in f.head(3).items()},
        }
    ari = {k: r(adjusted_rand_score(fam, np.asarray(cluster.get_cluster_labels(num_clusters=k))))
           for k in (3, 4, 5, 6, 7, 8)}  # fmt: skip
    return {
        "dataset": mode,
        "repeaters_sampled": int(len(states)),
        "distinct_sequences": int(states.astype(str).agg("".join, axis=1).nunique()),
        "silhouette_by_k": {k: r(v) for k, v in asw.items()},
        "best_k": int(best),
        "cqi_table": json.loads(table.to_json()) if hasattr(table, "to_json") else str(table),
        "clusters": profiles,
        "adjusted_rand_vs_rule_families_by_k": ari,
        "rule_family_shares": {
            k: r(v) for k, v in pd.Series(fam).value_counts(normalize=True).items()
        },  # fmt: skip
    }


def silhouette(matrix: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import silhouette_score

    return float(silhouette_score(matrix, labels, metric="precomputed"))


if __name__ == "__main__":
    import contextlib

    with contextlib.redirect_stdout(sys.stderr):  # sequenzo prints progress to stdout
        result = main(sys.argv[1], sys.argv[2])
    print(json.dumps(result, indent=1, default=str))
