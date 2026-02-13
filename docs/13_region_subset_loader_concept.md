# 13 — Region/Subset Loader Concept (Geo-spatial Ready)

This card defines a **loader concept** that restricts training/evaluation data to a specific **region / subset** (e.g., area-of-interest tiles),
enabling **controlled experiments** and **scenario-based evaluation**.

Even if this repo’s current dataset is ticket text (not imagery), the same pattern applies to incident ops:
- “Region” ⇢ a **scope** (tenant, geography, environment, system, product line)
- “Tiles” ⇢ stable **group IDs** (service_id, cluster_id, customer_id, site_id)

The key is: **subset selection must be reproducible and leakage-safe**.

---

## TL;DR

A **Region/Subset Loader** is two things:

1) A **selector**: “which samples belong to the subset?”
2) A **splitter**: “how do we split train/val/test without leaking across that subset boundary?”

We implement this concept via:
- a **manifest file** (explicit subset membership),
- a **group-aware split** (train/val/test by region or tile groups),
- and **traceability** (manifest hash + dataset version).

---

## 1) Why this exists (what problem it solves)

### Controlled experiments
You want to answer questions like:
- “Does the model generalize to **Region B** if trained on **Region A**?”
- “How does performance change on **high-noise sites** vs **stable sites**?”
- “Does a new augmentation help **dense urban tiles** but harm rural tiles?”

### Scenario evaluation (ops / field realism)
Instead of a single global score, you evaluate **per-scenario**:
- region-by-region performance
- subset-by-subset drift
- boundary cases (rare tile types, unusual incident clusters)

---

## 2) Inputs: how we define “region/subset”

A subset is defined by **one of** the following (ordered by auditability):

### A) Manifest (recommended)
A file that explicitly lists membership:
- tile IDs
- region IDs
- folders / file lists
- optional weights

This is best for auditability: **no hidden logic**.

### B) Bounding box (bbox) / polygon
- BBox: `[min_lon, min_lat, max_lon, max_lat]`
- Polygon: GeoJSON (for complex shapes)

Useful when data is indexed by lat/lon.

### C) Folder-based subset
- “all files under `data/tiles/region_A/`”
Works, but less explicit if the folder contents change silently.

### D) Metadata query (ops analog)
Examples:
- tickets where `system in {"payments", "auth"}`
- environment = `prod`
- source = `pagerduty`

This is still valid, but you must log the query + dataset version.

---

## 3) Manifest format (example)

Store under: `data/subsets/region_eu_west.json`

```json
{
  "schema_version": "1.0",
  "subset_id": "eu-west",
  "subset_type": "tile_ids",
  "tile_ids": ["T_1001", "T_1002", "T_1003"],
  "notes": "EU West AOI tiles for controlled evaluation",
  "created_at": "2026-02-13T00:00:00Z"
}
```

**Alternative: multi-region split manifest**
```json
{
  "schema_version": "1.0",
  "split_id": "regions_holdout_v1",
  "group_field": "region_id",
  "train_groups": ["region_A", "region_B"],
  "val_groups": ["region_C"],
  "test_groups": ["region_D"],
  "notes": "Hold out region_D for generalization test"
}
```

### Manifest invariants (must-haves)
- `schema_version`
- stable identifiers (`subset_id` or `split_id`)
- explicit membership (IDs / groups)
- creation timestamp + notes (why this subset exists)

---

## 4) Leakage-safe splitting rules

The #1 goal is: **no overlap of groups between train and test**.

### Rule 1 — Split by group (region/tile/service)
Use the **group field** (e.g., `region_id`, `tile_id`, `service_id`) and ensure:
- `train_groups ∩ test_groups = ∅`
- `val_groups` also disjoint

This prevents near-duplicate leakage when a “tile/service” produces repeated samples.

### Rule 2 — Time-awareness inside group (optional)
If each group has a timeline, add:
- time-aware split inside the selected group set, or
- global time cutoff after group split

This helps when the same group evolves over time.

### Rule 3 — “Buffer / gap” windows (optional)
Similar to `gap_days`:
- exclude near-boundary data
- reduces leakage via recurring events (deployments, recurring outages)

---

## 5) Metadata fields to carry through the pipeline

For every sample, preserve:

### Identity & traceability
- `sample_id` (stable)
- `region_id` / `tile_id` / `group_id`
- `timestamp` (if exists)
- `source` (file path, system, sensor)
- `label` (category/priority)

### Ops-friendly extras (incident analogy)
- `system`, `service`, `routing_team`, `error_code`, `environment`
- helps explain performance slices later

---

## 6) Traceability: dataset version + manifest hash

To make subsets “defensible”:

### A) Dataset version
Record at runtime:
- dataset file path(s)
- file size + last modified time
- optional dataset checksum / manifest

### B) Subset identity
Compute and persist:
- **SHA256 of the subset manifest** (the JSON itself)
- store it in artifacts + reports

This ensures you can always answer:
> “Exactly which region/subset was used in this run?”

---

## 7) Minimal implementation concept (repo-ready)

### 7.1 DataFrame filter (simple)
Assume the dataset has `region_id` or `tile_id`.

```python
import json
import pandas as pd

manifest = json.load(open("data/subsets/region_eu_west.json", "r", encoding="utf-8"))
tile_set = set(manifest["tile_ids"])

df = pd.read_csv("data/tickets.csv")
df_subset = df[df["tile_id"].isin(tile_set)].copy()
```

### 7.2 Group-aware split (no leakage)
```python
from sklearn.model_selection import GroupShuffleSplit

gss = GroupShuffleSplit(test_size=0.2, random_state=42)
groups = df_subset["tile_id"]

train_idx, test_idx = next(gss.split(df_subset, groups=groups))
df_train = df_subset.iloc[train_idx]
df_test  = df_subset.iloc[test_idx]
```

### 7.3 “Holdout region” split (explicit)
If you explicitly list groups in the manifest:

```python
train_groups = {"region_A", "region_B"}
test_groups = {"region_D"}

df_train = df[df["region_id"].isin(train_groups)]
df_test  = df[df["region_id"].isin(test_groups)]
```

This is the most audit-friendly approach.

---

## 8) Success criteria (how to know this card is “done”)

- ✅ Clear definition of subset inputs (manifest/bbox/folders/query)
- ✅ Split strategy that prevents leakage (group-disjoint)
- ✅ Metadata fields defined (timestamp, group_id, label, source)
- ✅ Traceability note: dataset version + manifest hash

Optional “nice to have”:
- A small demo runner that writes:
  - `reports/subset_demo.md`
  - `artifacts/subset_run/meta.json` (with manifest hash)

---

## 9) GitHub card “Done” checklist (paste)

- [ ] Define region/subset inputs (bbox, tile IDs, folder manifests, metadata query)
- [ ] Specify leakage-safe split strategy (train/val/test disjoint by group)
- [ ] Define metadata fields to keep (source, timestamp, region/tile ID, label)
- [ ] Add traceability note (dataset version + manifest SHA256)
