"""
Bangkok Bless Asset — Real Estate RAG Chatbot
══════════════════════════════════════════════
RAG Pipeline
  [R] Retrieval  : BGE-M3 embed → FAISS similarity search
                   + Nominatim geocode → Haversine distance search
  [A] Augmented  : inject top-K listings as context into LLM prompt
  [G] Generation : Anthropic Claude generates bilingual response

Data source: merged_data.csv (53 k rows, 64 cols)
"""

import os
import re
import json
import math
import hashlib
import urllib.request
import urllib.parse
from pathlib import Path

import numpy as np
import pandas as pd
import faiss
from FlagEmbedding import BGEM3FlagModel
# from groq import Groq                          # Groq
# import google.generativeai as genai            # Gemini
import anthropic                                 # pip install anthropic
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────
# Config
# ──────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
CSV_PATH      = str(BASE_DIR / "merged_data.csv")
INDEX_PATH    = str(BASE_DIR / "real_estate.faiss")
METADATA_PATH = str(BASE_DIR / "real_estate_meta.json")

TOP_K                = 10
SIMILARITY_THRESHOLD = 0.35
LOCATION_RADIUS_KM   = 20.0
LOCATION_RADIUS_WIDE = 35.0   # fallback radius when primary finds 0 results
MAX_HISTORY_TURNS    = 5
HAVERSINE_PREFETCH   = 1.6    # multiply radius before road-distance check to catch all candidates
SEMANTIC_FALLBACK_MAX_KM = 20.0  # max distance for semantic-fallback results when ref_point is known

EMBED_BATCH_SIZE = 16
EMBED_MAX_LENGTH = 512

# LLM_MODEL       = "llama-3.3-70b-versatile"   # Groq
# GEMINI_MODEL    = "gemini-2.0-flash"           # Gemini

CLAUDE_MODEL    = "claude-sonnet-4-6"
LLM_TEMPERATURE = 0.0

AMENITY_COLS = [
    "Elevator", "Parking", "Security", "CCTV", "Pool",
    "Sauna", "Gym", "Garden", "Playground", "Shop", "Restaurant", "Wifi",
]


# ──────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────
def _csv_hash(path: str) -> str:
    h = hashlib.md5()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def _s(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return " ".join(str(x).split()).strip()


def _i(x, default: int = 0) -> int:
    try:
        s = str(x).replace(",", "").strip()
        if not s or s.lower() in {"nan", "none"}:
            return default
        return int(float(s))
    except Exception:
        return default


def _f(x, default=None):
    try:
        s = str(x).replace(",", "").strip()
        if not s or s.lower() in {"nan", "none"}:
            return default
        return float(s)
    except Exception:
        return default


# ──────────────────────────────────────────
# [R] Phase 1 — Load & Build Documents
# ──────────────────────────────────────────
def _coalesce_str(df: pd.DataFrame, *cols: str) -> pd.Series:
    """Return first non-empty string across cols (left-to-right priority)."""
    result = pd.Series("", index=df.index, dtype=str)
    for col in reversed(cols):
        if col not in df.columns:
            continue
        v = df[col].fillna("").astype(str).str.strip()
        v = v.where(~v.str.lower().isin(["", "nan", "none"]), "")
        result = v.where(v != "", result)
    return result


def _coalesce_num(df: pd.DataFrame, *cols: str) -> pd.Series:
    """Return first non-null numeric value across cols (left-to-right priority)."""
    result = pd.Series(np.nan, index=df.index, dtype=float)
    for col in reversed(cols):
        if col not in df.columns:
            continue
        v = pd.to_numeric(
            df[col].astype(str).str.replace(",", "", regex=False),
            errors="coerce",
        )
        result = v.where(v.notna(), result)
    return result


def load_docs_from_csv(path: str) -> list[dict]:
    """
    Read merged_data.csv and convert each row into a doc dict with:
      - structured metadata  (for filtering / display)
      - fused TH+EN text     (for BGE-M3 embedding)
      - lat/lon              (for haversine search)

    Uses vectorized pandas operations for column extraction (fast),
    then a single loop only for per-row text building and amenity lists.
    """
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)

    # ── Vectorized column extraction ──
    name      = _coalesce_str(df, "name_th", "name_x", "name_y")
    ptype_raw = _coalesce_str(df, "propertytype_name_en")
    ptype     = ptype_raw.where(ptype_raw != "", "Condo")
    province  = _coalesce_str(df, "province_name_en")
    district  = _coalesce_str(df, "district_name_th", "district_y")
    nbh       = _coalesce_str(df, "neighborhood_name_th", "subdistrict_name_th")
    developer = _coalesce_str(df, "developer_name_th")
    url       = _coalesce_str(df, "url_project")

    price_min    = _coalesce_num(df, "price_min").fillna(0).astype(int)
    price_sqm    = _coalesce_num(df, "price_sqm_x", "price_sqm_y").fillna(0).astype(int)
    year_built   = _coalesce_num(df, "year_built_x", "year_built_y").fillna(0).astype(int)
    nbr_floors   = _coalesce_num(df, "nbr_floors_x", "nbr_floors_y").fillna(0).astype(int)
    rental_yield = _coalesce_num(df, "rental_yield")

    # ── Coordinates ──
    # latitude_prop / longitude_prop  (prop.csv)   : 20,825 rows, 18,123 unique pairs
    #   → building-level coordinates, accurate for haversine distance search
    # latitude / longitude            (df_cleaned) : 32,745 rows, only 1,018 unique pairs
    #   → district/neighbourhood CENTROIDS — 96.9% duplicates
    #   → must NOT be used for distance search (up to 2 km off from actual building)
    #
    # Rule: coord_accurate=True only when latitude_prop is available.
    lat_prop = _coalesce_num(df, "latitude_prop")
    lon_prop = _coalesce_num(df, "longitude_prop")
    lat_cent = _coalesce_num(df, "latitude")
    lon_cent = _coalesce_num(df, "longitude")

    has_prop     = lat_prop.notna() & lon_prop.notna()
    lat          = lat_prop.where(has_prop, lat_cent)
    lon          = lon_prop.where(has_prop, lon_cent)
    coord_acc    = has_prop

    # ── Transit (BTS takes priority over MRT) ──
    trans_raw = df.get("transportation", pd.Series("", index=df.index))
    trans_raw = trans_raw.fillna("").astype(str).str.lower()
    transit   = pd.Series("", index=df.index, dtype=str)
    transit   = transit.where(~trans_raw.str.contains("mrt", na=False), "MRT")
    transit   = transit.where(~trans_raw.str.contains("bts", na=False), "BTS")

    # ── Amenities matrix (vectorized) ──
    amenity_cols_present = [c for c in AMENITY_COLS if c in df.columns]
    amenity_matrix = (
        df[amenity_cols_present]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .astype(int)
        == 1
    )

    # ── Per-row loop: only for amenity lists + text building ──
    docs: list[dict] = []
    for i in df.index:
        amenities  = [c for c in amenity_cols_present if amenity_matrix.at[i, c]]
        amenity_th = " ".join(amenities) if amenities else "ไม่ระบุ"
        amenity_en = ", ".join(amenities) if amenities else "none listed"

        nm = name.at[i]
        pt = ptype.at[i]
        pr = province.at[i]
        di = district.at[i]
        nb = nbh.at[i]
        dv = developer.at[i]
        pm = int(price_min.at[i])
        ps = int(price_sqm.at[i])
        yb = int(year_built.at[i])
        nf = int(nbr_floors.at[i])
        ry = rental_yield.at[i] if pd.notna(rental_yield.at[i]) else None
        tr = transit.at[i]
        la = lat.at[i] if pd.notna(lat.at[i]) else None
        lo = lon.at[i] if pd.notna(lon.at[i]) else None
        ca = bool(coord_acc.at[i])

        th = (
            f"โครงการ {nm} ประเภท {pt} "
            f"ย่าน {nb} เขต {di} จังหวัด {pr} "
            f"ราคาเริ่มต้น {pm:,} บาท ราคาเฉลี่ย {ps:,} บาท/ตร.ม. "
            f"สร้างปี {yb} จำนวน {nf} ชั้น "
            f"สิ่งอำนวยความสะดวก: {amenity_th}"
            + (f" ใกล้รถไฟฟ้า {tr}" if tr else "")
            + (f" ผลตอบแทนเช่า {ry}%" if ry else "")
            + (f" โดย {dv}" if dv else "")
        )
        en = (
            f"{pt} project {nm} "
            f"in {nb}, {di}, {pr}, "
            f"from {pm:,} THB, avg {ps:,} THB/sqm, "
            f"built {yb}, {nf} floors, "
            f"amenities: {amenity_en}"
            + (f" near {tr}" if tr else "")
            + (f" rental yield {ry}%" if ry else "")
            + (f" by {dv}" if dv else "")
        )

        docs.append({
            "id":            f"MD-{i}",
            "name":          nm,
            "type":          pt,
            "province":      pr,
            "district":      di,
            "neighborhood":  nb,
            "developer":     dv,
            "price_thb":     pm or ps,
            "price_per_sqm": ps,
            "year_built":    yb,
            "nbr_floors":    nf,
            "rental_yield":  ry,
            "near_transit":  tr or None,
            "amenities":     amenities,
            "url":           url.at[i],
            "latitude":      la,
            "longitude":     lo,
            "coord_accurate": ca,
            "text":          f"TH: {th} | EN: {en}",
        })

    print(f"Loaded {len(docs):,} docs from {path}")
    return docs


# ──────────────────────────────────────────
# [R] Phase 2 — Embed & Index (FAISS)
# ──────────────────────────────────────────
def _normalize(vecs: np.ndarray) -> np.ndarray:
    vecs = np.ascontiguousarray(vecs, dtype=np.float32)
    faiss.normalize_L2(vecs)
    return vecs


def build_index(model: BGEM3FlagModel, docs: list[dict], csv_hash: str) -> faiss.Index:
    print(f"[RAG-R] Embedding {len(docs):,} docs with BGE-M3...")
    texts      = [d["text"] for d in docs]
    embeddings = model.encode(
        texts, batch_size=EMBED_BATCH_SIZE, max_length=EMBED_MAX_LENGTH,
    )["dense_vecs"]
    embeddings = _normalize(np.asarray(embeddings))

    dim = embeddings.shape[1]
    idx = faiss.IndexFlatIP(dim)   # cosine similarity on unit vectors
    idx.add(embeddings)

    faiss.write_index(idx, INDEX_PATH)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump({"csv_hash": csv_hash, "docs": docs}, f, ensure_ascii=False)
    print(f"[RAG-R] FAISS index saved → {INDEX_PATH} ({len(docs):,} vectors, dim={dim})")
    return idx


def load_or_build_index(model: BGEM3FlagModel) -> tuple[faiss.Index, list[dict]]:
    csv_hash = _csv_hash(CSV_PATH)

    if Path(INDEX_PATH).exists() and Path(METADATA_PATH).exists():
        with open(METADATA_PATH, encoding="utf-8") as f:
            saved = json.load(f)
        if saved.get("csv_hash") == csv_hash:
            print("[RAG-R] Loading cached FAISS index (CSV unchanged)...")
            return faiss.read_index(INDEX_PATH), saved["docs"]

    print("[RAG-R] CSV changed or index missing — rebuilding...")
    docs = load_docs_from_csv(CSV_PATH)
    idx  = build_index(model, docs, csv_hash)
    return idx, docs


# ──────────────────────────────────────────
# [R] Phase 3a — Semantic Retrieval
# ──────────────────────────────────────────
def retrieve_semantic(
    query: str,
    model: BGEM3FlagModel,
    idx: faiss.Index,
    docs: list[dict],
) -> list[tuple[dict, float]]:
    vec = model.encode([query], max_length=EMBED_MAX_LENGTH)["dense_vecs"]
    vec = _normalize(np.asarray(vec))
    scores, indices = idx.search(vec, k=TOP_K)
    results = [
        (docs[int(i)], float(s))
        for s, i in zip(scores[0], indices[0])
        if i != -1 and float(s) >= SIMILARITY_THRESHOLD
    ]
    results.sort(key=lambda x: x[0].get("price_thb") or float("inf"))
    return results


# ──────────────────────────────────────────
# [R] Phase 3b — Location-based Retrieval
# ──────────────────────────────────────────

_TH_LAT_MIN, _TH_LAT_MAX = 5.5, 20.5
_TH_LON_MIN, _TH_LON_MAX = 97.5, 105.7

_OSM_TYPE_SCORE = {"node": 2.0, "way": 1.0, "relation": 0.0}

# In-memory geocode cache — avoids repeated Nominatim calls for the same place name
_geocode_cache: dict[str, tuple | None] = {}

# Regex for queries that are CLEARLY follow-ups / non-location — skip LLM call entirely.
# Everything else goes to LLM for place extraction (safer: fewer false negatives).
_NO_PLACE_RE = re.compile(
    r"^(?:"
    r"ขอบคุณ|โอเค|โอเค|ได้เลย|ok|okay|thanks|thank you|"
    r"บอกอีก(?:ที)?|แสดงอีก|ดูอีก|show more|"
    r"อธิบาย(?:เพิ่มเติม)?|explain|tell me more|"
    r"ราคาเท่าไหร่|ราคา[อะไร]|price\??|how much\??|"
    r"มีอะไรอีก|anything else|"
    r"ใช่|ไม่ใช่|yes|no|nope|yep"
    r")[\s?!.]*$",
    re.IGNORECASE,
)


def _in_thailand(lat: float, lon: float) -> bool:
    return _TH_LAT_MIN <= lat <= _TH_LAT_MAX and _TH_LON_MIN <= lon <= _TH_LON_MAX


def geocode_place(text: str) -> tuple[float, float, str, str] | None:
    """
    Multi-strategy Nominatim geocoding with OSM geometry-type preference.
    Results are cached in _geocode_cache for the lifetime of the process.

    Why OSM type matters:
      node     = single GPS point  → coordinates are exact
      way      = building polygon  → coordinates are centroid (may be off 50-300 m)
      relation = large area        → centroid can be off 500 m – 2 km

    Returns (lat, lon, display_name, osm_type) or None.
    """
    if text in _geocode_cache:
        return _geocode_cache[text]

    lower = text.lower()
    queries = [text]
    if not any(x in lower for x in ["เชียงใหม่", "ภูเก็ต", "phuket", "chiang mai", "chiangmai"]):
        queries.append(f"กรุงเทพมหานคร {text}")

    all_candidates: list[tuple[float, float, float, str, str]] = []
    for q in queries:
        params = urllib.parse.urlencode({
            "q": q,
            "format": "json",
            "limit": 10,
            "countrycodes": "th",
            "addressdetails": 0,
        })
        req = urllib.request.Request(
            f"https://nominatim.openstreetmap.org/search?{params}",
            headers={
                "User-Agent": "BangkokBlessAsset-Chatbot/1.0",
                "Accept-Language": "th,en;q=0.9",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=6) as resp:
                data = json.loads(resp.read())
            for hit in data:
                lat = float(hit["lat"])
                lon = float(hit["lon"])
                if not _in_thailand(lat, lon):
                    continue
                osm_type   = hit.get("osm_type", "")
                importance = float(hit.get("importance", 0))
                score      = importance + _OSM_TYPE_SCORE.get(osm_type, 0.0)
                all_candidates.append((score, lat, lon, hit.get("display_name", ""), osm_type))
        except Exception:
            pass
        if all_candidates:
            break

    if not all_candidates:
        _geocode_cache[text] = None
        return None

    all_candidates.sort(key=lambda x: -x[0])
    _, lat, lon, display, osm_type = all_candidates[0]
    result = (lat, lon, display, osm_type)
    _geocode_cache[text] = result
    return result


def road_distance_batch(
    origin_lat: float,
    origin_lon: float,
    destinations: list[tuple[float, float]],
) -> list[float | None]:
    """
    Google Maps Distance Matrix API — road distances from one origin to many destinations.
    Returns list[km | None]. Batches 25 destinations per request (API limit).
    Falls back to None for each element on error or missing key.
    """
    api_key = os.getenv("GOOGLE_MAPS_API_KEY", "")
    if not api_key or api_key == "your_key_here":
        return [None] * len(destinations)

    results: list[float | None] = []
    for i in range(0, len(destinations), 25):
        batch = destinations[i : i + 25]
        dest_str = "|".join(f"{lat},{lon}" for lat, lon in batch)
        params = urllib.parse.urlencode({
            "origins":      f"{origin_lat},{origin_lon}",
            "destinations": dest_str,
            "mode":         "driving",
            "key":          api_key,
        })
        req = urllib.request.Request(
            f"https://maps.googleapis.com/maps/api/distancematrix/json?{params}",
            headers={"User-Agent": "BangkokBlessAsset-Chatbot/1.0"},
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
            elements = data.get("rows", [{}])[0].get("elements", [])
            for elem in elements:
                if elem.get("status") == "OK":
                    results.append(elem["distance"]["value"] / 1000.0)
                else:
                    results.append(None)
        except Exception:
            results.extend([None] * len(batch))

    return results


def extract_place_name(query: str, llm: anthropic.Anthropic) -> str:
    """
    Extract the place/location entity from the user's query.

    Fast path: if the query contains no location-like keywords, return ""
    immediately without calling the LLM (saves API cost + ~300 ms latency).
    Only queries that match _HAS_PLACE_RE get sent to Claude for extraction.
    """
    if _NO_PLACE_RE.match(query.strip()):
        return ""

    resp = llm.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=40,
        temperature=0,
        system=(
            "Extract ONLY the place name, landmark, BTS/MRT station, "
            "shopping mall, hospital, district, or area from the user message. "
            "Return ONLY the place name in its original language (Thai or English) "
            "— no explanation, no punctuation, no extra words. "
            "If no specific place is mentioned, return empty string."
        ),
        messages=[{"role": "user", "content": query}],
    )
    return resp.content[0].text.strip()


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km."""
    R    = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a    = (math.sin(dlat / 2) ** 2
            + math.cos(math.radians(lat1))
            * math.cos(math.radians(lat2))
            * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def retrieve_by_location(
    place_name: str,
    docs: list[dict],
    radius_km: float = LOCATION_RADIUS_KM,
) -> tuple[list[tuple[dict, float]], str | None, str, tuple[float, float] | None]:
    """
    Geocode place_name → road distance (Google Maps) or haversine fallback →
    return listings within radius_km sorted by price + context note + dist_label + ref_point.

    Returns: (nearby, context_note, dist_label, ref_point)
      dist_label — "ถนน" when Google Maps is used, "เส้นตรง" otherwise.
      ref_point  — (lat, lon) of the geocoded place, or None if geocoding failed.
                   Returned even when nearby is empty so caller can compute
                   distances for semantic-fallback results.
    """
    if not place_name:
        return [], None, "เส้นตรง", None
    result = geocode_place(place_name)
    if not result:
        return [], None, "เส้นตรง", None
    lat, lon, display_name, osm_type = result

    accuracy_note = {
        "node":     "precise point",
        "way":      "building centroid (±100–300 m)",
        "relation": "area centroid (±500 m–2 km)",
    }.get(osm_type, "unknown accuracy")

    print(
        f"[GEO] '{place_name}' → lat={lat:.6f}, lon={lon:.6f}"
        f"  type={osm_type} ({accuracy_note})"
        f"\n      display: {display_name[:100]}"
    )

    # Step 1: haversine pre-filter at HAVERSINE_PREFETCH × radius to get candidates
    prefetch_r = radius_km * HAVERSINE_PREFETCH
    candidates: list[tuple[dict, float]] = []
    for doc in docs:
        if not doc.get("coord_accurate"):
            continue
        dlat, dlon = doc.get("latitude"), doc.get("longitude")
        if dlat is None or dlon is None:
            continue
        if not _in_thailand(dlat, dlon):
            continue
        if haversine(lat, lon, dlat, dlon) <= prefetch_r:
            candidates.append((doc, dlat, dlon))  # type: ignore[arg-type]

    # Step 2: road distance via Google Maps (batch), fallback to haversine per element
    api_key   = os.getenv("GOOGLE_MAPS_API_KEY", "")
    use_road  = bool(api_key and api_key != "your_key_here")
    dist_label = "ถนน" if use_road else "เส้นตรง"

    if use_road and candidates:
        dests      = [(dlat, dlon) for _, dlat, dlon in candidates]
        road_dists = road_distance_batch(lat, lon, dests)
    else:
        road_dists = [None] * len(candidates)

    nearby: list[tuple[dict, float]] = []
    fallback_count = 0
    for (doc, dlat, dlon), road_d in zip(candidates, road_dists):
        if road_d is not None:
            dist = road_d
        else:
            dist = haversine(lat, lon, dlat, dlon)
            fallback_count += 1
        if dist <= radius_km:
            nearby.append((doc, dist))

    method_str = (
        f"road (Google Maps Driving){f', {fallback_count} haversine fallbacks' if fallback_count else ''}"
        if use_road else "straight-line (haversine)"
    )
    print(f"[DIST] method={method_str}  radius={radius_km} km  candidates={len(candidates)}  matched={len(nearby)}")
    nearby.sort(key=lambda x: (x[0].get("price_thb") or float("inf"), x[1]))

    context_note = (
        f"Geocoded '{place_name}' as: {display_name[:120]}"
        f" | geometry={osm_type} ({accuracy_note})"
        f" | ref_point=({lat:.6f}, {lon:.6f})"
        f" | radius={radius_km} km"
        f" | distance_method={method_str}"
    )
    return nearby, context_note, dist_label, (lat, lon)


# ──────────────────────────────────────────
# [A] Augment — Build LLM Context
# ──────────────────────────────────────────
SYSTEM_PROMPT = """\
You are a bilingual (Thai/English) real estate assistant for Bangkok Bless Asset.

Rules:
1. Detect the user's language — reply in the same language.
2. Use ONLY the provided listing context. Never fabricate data.
3. Present listings sorted cheapest → most expensive (top 10).
4. LOCATION SEARCH: When the user mentions any place name (Thai or English),
   the system automatically geocodes it to obtain the latitude/longitude of that
   place, then finds all listings whose coordinates fall within a 20 km radius
   (or 35 km if no results found at 20 km).
   The RAG context will be labeled "[RAG context — location]" in that case.
   • At the TOP of your reply state the reference point explicitly, e.g.:
     "📍 จุดอ้างอิง: BTS อโศก (lat=13.737593, lon=100.560699)"
   • If the context says radius=20 km, mention it to the user.
   • Distances are shown as "X.XX km (ถนน)" when Google Maps road distance is used,
     or "X.XX km (เส้นตรง)" when straight-line haversine is used.
     Always use the label exactly as it appears in the listing data.
   • Show FULL details only for listings within the primary radius (0–20 km).
   • If the context contains HINT_EXTRA, mention to the user that more listings exist
     beyond 20 km (e.g. "มีอีก X โครงการในรัศมี 35 กม. ถ้าอยากดูบอกได้เลย")
     but do NOT show their names or any details — wait for user to ask.
   • If no coord_accurate listings found, system uses semantic search with distance computed
     from the ref point. Show distances normally.
5. SEMANTIC SEARCH: When no place name is detected, results come from BGE-M3
   vector similarity. Context label will be "[RAG context — semantic]".
6. For every listing you MUST show ALL of the following on separate lines:
   - Name & type
   - Location (neighborhood / district / province)
   - Price (฿ with commas)
   - Transit access (if any)
   - Amenities (if any)
   - Coordinates: 📍 lat=XX.XXXXXX, lon=XXX.XXXXXX  ← MANDATORY, never omit
   - Distance from searched place (location mode only)
7. If nothing matches, politely say so and suggest refining the search
   (e.g. wider radius, different property type, or different area).
8. If the user says the reference point is wrong (e.g. "wrong place", "not that location",
   "reference point is incorrect"), immediately ask:
   "Could you give me the full name of the place? e.g. 'CentralWorld Shopping Mall' or 'Asok BTS Station, Sukhumvit Road'"
   Wait for the full place name before searching again.
9. Keep answers friendly. Use bullet points for multiple listings.

คุณคือผู้ช่วยอสังหาฯ ของ Bangkok Bless Asset พูดได้ทั้งไทย-อังกฤษ
1. ตอบด้วยภาษาเดียวกับลูกค้า
2. ใช้ข้อมูล listing ที่ให้มาเท่านั้น ห้ามสร้างข้อมูลเอง
3. เรียงจากราคาถูกไปแพง 10 อันดับแรก
4. การค้นหาด้วยสถานที่: หากผู้ใช้ระบุชื่อสถานที่ (ภาษาไทยหรืออังกฤษ)
   ระบบจะแปลงชื่อนั้นเป็นละติจูด/ลองจิจูดโดยอัตโนมัติผ่าน Nominatim
   แล้วค้นหาบ้านที่มีพิกัดอยู่ภายในรัศมี 20 กม. จากสถานที่นั้น
   (ถ้าไม่พบในรัศมี 20 กม. จะขยายเป็น 35 กม. โดยอัตโนมัติ)
   • ขึ้นต้นคำตอบด้วยจุดอ้างอิงเสมอ เช่น:
     "📍 จุดอ้างอิง: BTS อโศก (lat=13.737593, lon=100.560699)"
   • ถ้า context ระบุ radius=20 km ให้แจ้งผู้ใช้ด้วย
   • ระยะทางแสดงเป็น "X.XX km (ถนน)" เมื่อใช้ Google Maps หรือ "X.XX km (เส้นตรง)" เมื่อใช้ haversine
     ให้ใช้ label ตามที่ปรากฏในข้อมูล listing ทุกครั้ง
   • แสดงรายละเอียดเต็มเฉพาะ listing ในรัศมี 0–20 กม. เท่านั้น
   • ถ้า context มี HINT_EXTRA ให้แจ้ง user ว่ามีโครงการเพิ่มเติมในรัศมีที่กว้างกว่า
     เช่น "มีอีก X โครงการในรัศมี 35 กม. ถ้าอยากดูบอกได้เลย"
     แต่ห้ามแสดงชื่อหรือรายละเอียดของโครงการเหล่านั้น รอ user ถามก่อน
   • หากไม่มีพิกัดอาคารในรัศมี ระบบใช้ semantic search พร้อมคำนวณระยะทางจากจุดอ้างอิง แสดงตามปกติ
5. การค้นหาแบบ semantic: หากไม่มีชื่อสถานที่ ระบบจะใช้ BGE-M3 เวกเตอร์
6. ทุก listing ต้องแสดงครบทุกข้อต่อไปนี้:
   - ชื่อโครงการ และประเภท
   - ทำเล (ย่าน/เขต/จังหวัด)
   - ราคา (฿ มีจุลภาค)
   - รถไฟฟ้า (ถ้ามี)
   - สิ่งอำนวยความสะดวก (ถ้ามี)
   - พิกัด: 📍 lat=XX.XXXXXX, lon=XXX.XXXXXX  ← บังคับแสดงทุกครั้ง ห้ามละเว้น
   - ระยะห่างจากสถานที่ที่ค้นหา (เฉพาะ location mode)
7. หากไม่พบรายการ ให้แจ้งสุภาพและแนะนำให้ปรับเกณฑ์ (ขยายรัศมี/เปลี่ยนทำเล/ประเภท)
8. หากผู้ใช้บอกว่าจุดอ้างอิงผิด (เช่น "สถานที่ผิด", "ไม่ใช่ที่นั่น", "จุดอ้างอิงไม่ตรง")
   ให้ถามกลับทันทีว่า: "ช่วยบอกชื่อเต็มของสถานที่ที่ต้องการได้มั้ย เช่น 'ห้างสรรพสินค้าเซ็นทรัล ลาดพร้าว' หรือ 'สถานี BTS อโศก'"
   แล้วรอรับชื่อเต็มจากผู้ใช้ก่อนค้นหาใหม่
"""


def _format_listing(rank: int, doc: dict, value: float, mode: str, dist_label: str = "เส้นตรง") -> str:
    tag   = f"{value:.2f} km ({dist_label})" if mode == "location" else f"score={value:.3f}"
    price = f"฿{doc['price_thb']:,}" if doc.get("price_thb") else "-"
    sqm   = f"฿{doc['price_per_sqm']:,}/sqm" if doc.get("price_per_sqm") else ""
    loc   = ", ".join(x for x in [doc.get("neighborhood"), doc.get("district"), doc.get("province")] if x)
    parts = [
        f"[{rank}] {doc.get('name') or '-'} ({tag})",
        f"type={doc.get('type', '-')}",
        f"loc={loc}" if loc else "",
        f"price={price}",
        sqm,
        f"built={doc['year_built']}" if doc.get("year_built") else "",
        f"near {doc['near_transit']}" if doc.get("near_transit") else "",
        f"amenities={','.join(doc['amenities'])}" if doc.get("amenities") else "",
        f"yield={doc['rental_yield']}%" if doc.get("rental_yield") else "",
        f"by {doc['developer']}" if doc.get("developer") else "",
        f"url={doc['url']}" if doc.get("url") else "",
        (f"coords=({doc['latitude']:.6f}, {doc['longitude']:.6f})"
         if doc.get("latitude") is not None and doc.get("longitude") is not None else ""),
    ]
    return " | ".join(p for p in parts if p)


def build_rag_context(
    results: list[tuple],
    mode: str,
    geocoded_place: str | None = None,
    dist_label: str = "เส้นตรง",
) -> str:
    if not results:
        return "No matching listings found. / ไม่พบรายการที่ตรงกับเงื่อนไข"
    if mode == "location":
        sort_note = "sorted by price ฿ low→high then distance"
        header = geocoded_place or ""
        if geocoded_place and "centroid" in geocoded_place:
            header += (
                "\nNOTE: reference point is a polygon centroid — distances shown are"
                " approximate (may differ from map by 100 m – 2 km depending on building size)."
                " แจ้งผู้ใช้ว่าระยะทางเป็นค่าประมาณเนื่องจากพิกัดสถานที่ไม่ใช่จุดแน่นอน"
            )
    else:
        sort_note = "sorted by price ฿ low→high"
        header = ""
    rows  = "\n".join(_format_listing(i, d, v, mode, dist_label) for i, (d, v) in enumerate(results, 1))
    parts = [p for p in [header, f"[{sort_note}]", rows] if p]
    return "\n".join(parts)


# ──────────────────────────────────────────
# [G] Generate — LLM Response
# ──────────────────────────────────────────
def rag_chat(
    user_query: str,
    history: list[dict],
    embed_model: BGEM3FlagModel,
    idx: faiss.Index,
    docs: list[dict],
    llm: anthropic.Anthropic,
) -> tuple[str, list[tuple], str]:
    # [R] Retrieve — primary radius (full details)
    place_name = extract_place_name(user_query, llm)
    location_results, geocoded_place, dist_label, ref_point = retrieve_by_location(place_name, docs)

    # Count extras in wider radius (show count only — no details)
    extra_count = 0
    if place_name and location_results:
        wide_all, _, _, _ = retrieve_by_location(place_name, docs, radius_km=LOCATION_RADIUS_WIDE)
        extra_count = max(0, len(wide_all) - len(location_results))
        if extra_count > 0 and geocoded_place:
            geocoded_place += (
                f" | HINT_EXTRA: นอกจากนี้ยังมีอีก {extra_count} โครงการในรัศมี {LOCATION_RADIUS_WIDE} กม."
                f" — แจ้ง user ว่า 'มีอีก {extra_count} โครงการในรัศมี {LOCATION_RADIUS_WIDE} กม."
                f" ถ้าอยากดูรายละเอียดบอกได้เลย' แต่ห้ามแสดงรายชื่อหรือข้อมูลเพิ่มเติม รอ user ถาม"
            )

    if location_results:
        results = location_results[:TOP_K]
        mode    = "location"
    else:
        # Semantic fallback with distance computation when ref_point is known
        semantic_results = retrieve_semantic(user_query, embed_model, idx, docs)

        api_key  = os.getenv("GOOGLE_MAPS_API_KEY", "")
        use_road = bool(api_key and api_key != "your_key_here")

        if ref_point and semantic_results:
            ref_lat, ref_lon = ref_point
            dist_label  = "ถนน" if use_road else "เส้นตรง"
            with_coords = [(doc, s) for doc, s in semantic_results
                           if doc.get("latitude") and doc.get("longitude")]

            if use_road and with_coords:
                dests     = [(doc["latitude"], doc["longitude"]) for doc, _ in with_coords]
                road_dsts = road_distance_batch(ref_lat, ref_lon, dests)
                results   = []
                for (doc, _), rd in zip(with_coords, road_dsts):
                    dist = rd if rd is not None else haversine(ref_lat, ref_lon, doc["latitude"], doc["longitude"])
                    if dist <= SEMANTIC_FALLBACK_MAX_KM:
                        results.append((doc, dist))
            else:
                results = [
                    (doc, haversine(ref_lat, ref_lon, doc["latitude"], doc["longitude"]))
                    for doc, _ in with_coords
                    if haversine(ref_lat, ref_lon, doc["latitude"], doc["longitude"]) <= SEMANTIC_FALLBACK_MAX_KM
                ]

            results = results[:TOP_K]
            mode    = "location"
            if not geocoded_place:
                geocoded_place = (
                    f"ref_point=({ref_lat:.6f}, {ref_lon:.6f})"
                    f" | distance_method={'road (Google Maps Driving)' if use_road else 'straight-line (haversine)'}"
                )
        else:
            results        = semantic_results
            mode           = "semantic"
            geocoded_place = None
            dist_label     = "เส้นตรง"

    # [A] Augment
    context = build_rag_context(results, mode, geocoded_place, dist_label)
    claude_history = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in history[-(MAX_HISTORY_TURNS * 2):]
    ]

    # [G] Generate
    resp = llm.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=4096,
        temperature=LLM_TEMPERATURE,
        system=SYSTEM_PROMPT + f"\n\n[RAG context — {mode}]\n{context}",
        messages=[*claude_history, {"role": "user", "content": user_query}],
    )
    return resp.content[0].text, results, mode


# ──────────────────────────────────────────
# Pipeline initialiser (for FastAPI / web use)
# ──────────────────────────────────────────
def init_pipeline() -> dict:
    """
    Load all heavy resources once at startup and return a pipeline dict.
    Call this from FastAPI lifespan or any web server boot routine.
    """
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise EnvironmentError("Set ANTHROPIC_API_KEY in .env  (console.anthropic.com)")
    if not Path(CSV_PATH).exists():
        raise FileNotFoundError(f"{CSV_PATH} not found — expected at {CSV_PATH}")

    print("Loading BGE-M3... (first run downloads ~2.3 GB)")
    embed_model  = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    faiss_index, docs = load_or_build_index(embed_model)
    llm_client   = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    return {
        "embed_model": embed_model,
        "idx":         faiss_index,
        "docs":        docs,
        "llm":         llm_client,
    }


# ──────────────────────────────────────────
# Main (CLI)
# ──────────────────────────────────────────
def main():
    pipeline = init_pipeline()
    embed_model  = pipeline["embed_model"]
    faiss_index  = pipeline["idx"]
    metadata     = pipeline["docs"]
    llm_client   = pipeline["llm"]

    history: list[dict] = []

    geo_ok = geocode_place("Bangkok") is not None
    print(f"\n{'=' * 60}")
    print(f"Bangkok Bless Asset RAG Chatbot — {len(metadata):,} listings")
    print(f"  Semantic search : BGE-M3 + FAISS  |  threshold={SIMILARITY_THRESHOLD}")
    print(f"  Location search : Nominatim  |  radius={LOCATION_RADIUS_KM}/{LOCATION_RADIUS_WIDE} km  |  {'online' if geo_ok else 'OFFLINE'}")
    print(f"  LLM             : {CLAUDE_MODEL}")
    print("Type in Thai or English  |  'exit' to quit")
    print(f"{'=' * 60}")

    while True:
        try:
            q = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not q:
            continue
        if q.lower() in {"exit", "quit", "ออก", "ลาก่อน"}:
            print("Goodbye!")
            break

        try:
            answer, sources, mode = rag_chat(
                q, history, embed_model, faiss_index, metadata, llm_client
            )
        except Exception as e:
            print(f"[Error] {e}")
            continue

        history.append({"role": "user",      "content": q})
        history.append({"role": "assistant",  "content": answer})

        print(f"\nBot: {answer}")
        if sources:
            print(f"\n[{mode.upper()} — top {len(sources)} results]")
            for doc, val in sources:
                label = f"{val:.2f} km" if mode == "location" else f"score={val:.3f}"
                price = f"฿{doc['price_thb']:,}" if doc.get("price_thb") else "N/A"
                print(f"  • {doc.get('name', '-'):30s} | {doc.get('district', '-'):15s} | {price:>15s} | {label}")


if __name__ == "__main__":
    main()
