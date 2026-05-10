"""
Bangkok Bless Asset — Real Estate RAG Chatbot
══════════════════════════════════════════════
RAG Pipeline
  [R] Retrieval  : BGE-M3 embed → FAISS similarity search
                   + Nominatim geocode → Haversine distance search
  [A] Augmented  : inject top-K listings as context into LLM prompt
  [G] Generation : Groq LLM generates bilingual response

Data source: merged_data.csv (53 k rows, 64 cols)
"""

import os
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
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────
# Config
# ──────────────────────────────────────────
CSV_PATH      = "merged_data.csv"
INDEX_PATH    = "real_estate.faiss"
METADATA_PATH = "real_estate_meta.json"

TOP_K                = 10
SIMILARITY_THRESHOLD = 0.35
LOCATION_RADIUS_KM   = 5.0
MAX_HISTORY_TURNS    = 5

EMBED_BATCH_SIZE = 16
EMBED_MAX_LENGTH = 512

LLM_MODEL       = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.3

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


def _pick_f(row: pd.Series, *cols) -> float | None:
    """Return first non-null float from a list of column names.
    Safe alternative to chaining `or` — NaN is truthy in Python so
    `NaN or next_val` returns NaN instead of falling through."""
    for col in cols:
        v = _f(row.get(col))
        if v is not None:
            return v
    return None


def _pick_s(row: pd.Series, *cols) -> str:
    """Return first non-empty string from a list of column names."""
    for col in cols:
        v = _s(row.get(col))
        if v:
            return v
    return ""


def _pick_i(row: pd.Series, *cols, default: int = 0) -> int:
    """Return first non-zero int from a list of column names."""
    for col in cols:
        v = _i(row.get(col))
        if v:
            return v
    return default


# ──────────────────────────────────────────
# [R] Phase 1 — Load & Build Documents
# ──────────────────────────────────────────
def load_docs_from_csv(path: str) -> list[dict]:
    """
    Read merged_data.csv and convert each row into a doc dict with:
      - structured metadata  (for filtering / display)
      - fused TH+EN text     (for BGE-M3 embedding)
      - lat/lon              (for haversine search)
    """
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    docs = []

    for i, r in df.iterrows():
        # ── Core fields ──
        name      = _pick_s(r, "name_th", "name_x", "name_y")
        ptype     = _pick_s(r, "propertytype_name_en") or "Condo"
        province  = _pick_s(r, "province_name_en")
        # district_x is 0 non-null → skip; district_y (32,745) and district_name_th (20,820)
        district  = _pick_s(r, "district_name_th", "district_y")
        nbh       = _pick_s(r, "neighborhood_name_th", "subdistrict_name_th")
        developer = _pick_s(r, "developer_name_th")
        url       = _pick_s(r, "url_project")

        price_min    = _pick_i(r, "price_min")
        price_sqm    = _pick_i(r, "price_sqm_x", "price_sqm_y")
        year_built   = _pick_i(r, "year_built_x", "year_built_y")
        nbr_floors   = _pick_i(r, "nbr_floors_x", "nbr_floors_y")
        rental_yield = _f(r.get("rental_yield"))

        # ── Coordinates ──
        # Use only real source coordinates — never lat_round (join key, not a property coordinate)
        # latitude_prop: 20,825 rows (from prop.csv)
        # latitude     : 32,745 rows (from df_cleaned/scraped)
        # Together they cover all 53,466 rows with no gaps
        lat = _pick_f(r, "latitude_prop", "latitude")
        lon = _pick_f(r, "longitude_prop", "longitude")

        # ── Transit ──
        transit_raw = _s(r.get("transportation", "")).lower()
        if "bts" in transit_raw:
            transit = "BTS"
        elif "mrt" in transit_raw:
            transit = "MRT"
        else:
            transit = ""

        # ── Amenities ──
        amenities  = [c for c in AMENITY_COLS if _i(r.get(c)) == 1]
        amenity_th = " ".join(amenities) if amenities else "ไม่ระบุ"
        amenity_en = ", ".join(amenities) if amenities else "none listed"

        # ── Fused TH + EN text for embedding ──
        th = (
            f"โครงการ {name} ประเภท {ptype} "
            f"ย่าน {nbh} เขต {district} จังหวัด {province} "
            f"ราคาเริ่มต้น {price_min:,} บาท ราคาเฉลี่ย {price_sqm:,} บาท/ตร.ม. "
            f"สร้างปี {year_built} จำนวน {nbr_floors} ชั้น "
            f"สิ่งอำนวยความสะดวก: {amenity_th}"
            + (f" ใกล้รถไฟฟ้า {transit}" if transit else "")
            + (f" ผลตอบแทนเช่า {rental_yield}%" if rental_yield else "")
            + (f" โดย {developer}" if developer else "")
        )
        en = (
            f"{ptype} project {name} "
            f"in {nbh}, {district}, {province}, "
            f"from {price_min:,} THB, avg {price_sqm:,} THB/sqm, "
            f"built {year_built}, {nbr_floors} floors, "
            f"amenities: {amenity_en}"
            + (f" near {transit}" if transit else "")
            + (f" rental yield {rental_yield}%" if rental_yield else "")
            + (f" by {developer}" if developer else "")
        )

        docs.append({
            "id":           f"MD-{i}",
            "name":         name,
            "type":         ptype,
            "province":     province,
            "district":     district,
            "neighborhood": nbh,
            "developer":    developer,
            "price_thb":    price_min or price_sqm,
            "price_per_sqm": price_sqm,
            "year_built":   year_built,
            "nbr_floors":   nbr_floors,
            "rental_yield": rental_yield,
            "near_transit": transit or None,
            "amenities":    amenities,
            "url":          url,
            "latitude":     lat,
            "longitude":    lon,
            "text":         f"TH: {th} | EN: {en}",
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
    # Sort by price ascending
    results.sort(key=lambda x: x[0].get("price_thb") or float("inf"))
    return results


# ──────────────────────────────────────────
# [R] Phase 3b — Location-based Retrieval
# ──────────────────────────────────────────

# Bounding box for Thailand — used to reject obviously wrong coordinates
_TH_LAT_MIN, _TH_LAT_MAX = 5.5, 20.5
_TH_LON_MIN, _TH_LON_MAX = 97.5, 105.7

# OSM geometry type scoring: node = exact point (most accurate),
# way = polygon centroid (moderate), relation = large area centroid (least accurate)
_OSM_TYPE_SCORE = {"node": 2.0, "way": 1.0, "relation": 0.0}


def _in_thailand(lat: float, lon: float) -> bool:
    return _TH_LAT_MIN <= lat <= _TH_LAT_MAX and _TH_LON_MIN <= lon <= _TH_LON_MAX


def geocode_place(text: str) -> tuple[float, float, str, str] | None:
    """
    Multi-strategy Nominatim geocoding with OSM geometry-type preference.

    Why OSM type matters:
      node     = single GPS point  → coordinates are exact
      way      = building polygon  → coordinates are centroid (may be off 50-300 m)
      relation = large area        → centroid can be off 500 m – 2 km

    Strategy:
      1. Try plain query (e.g. "เซ็นทรัลเวิลด์")
      2. Retry prefixed with กรุงเทพมหานคร if no candidates yet
         (skip for provinces like Chiang Mai / Phuket)
      3. Score all candidates: importance + OSM type bonus
      4. Return best-scoring result inside Thailand's bounding box.

    Returns (lat, lon, display_name, osm_type) or None.
    """
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
                osm_type  = hit.get("osm_type", "")
                importance = float(hit.get("importance", 0))
                score = importance + _OSM_TYPE_SCORE.get(osm_type, 0.0)
                all_candidates.append((score, lat, lon, hit.get("display_name", ""), osm_type))
        except Exception:
            pass
        if all_candidates:
            break   # good enough — don't try the next query variant

    if not all_candidates:
        return None

    all_candidates.sort(key=lambda x: -x[0])
    _, lat, lon, display, osm_type = all_candidates[0]
    return lat, lon, display, osm_type


def extract_place_name(query: str, llm: Groq) -> str:
    """
    Use a fast LLM call to pull out just the place / location entity from
    the user's free-text query.  Returns the bare place name, or empty string
    if no specific place is mentioned.
    """
    resp = llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": (
                "Extract ONLY the place name, landmark, BTS/MRT station, "
                "shopping mall, hospital, district, or area from the user message. "
                "Return ONLY the place name in its original language (Thai or English) "
                "— no explanation, no punctuation, no extra words. "
                "If no specific place is mentioned, return empty string."
            )},
            {"role": "user", "content": query},
        ],
        temperature=0,
        max_tokens=40,
    )
    return resp.choices[0].message.content.strip()


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
) -> tuple[list[tuple[dict, float]], str | None]:
    """
    Geocode place_name → find all docs within radius_km →
    return (results, context_note).

    context_note is passed to the LLM so it can tell the user:
      - which place was geocoded
      - how accurate the reference point is (node vs polygon centroid)
    """
    if not place_name:
        return [], None
    result = geocode_place(place_name)
    if not result:
        return [], None
    lat, lon, display_name, osm_type = result

    # Accuracy note based on OSM geometry type
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

    nearby = []
    for doc in docs:
        dlat = doc.get("latitude")
        dlon = doc.get("longitude")
        if dlat is None or dlon is None:
            continue
        if not _in_thailand(dlat, dlon):
            continue
        dist = haversine(lat, lon, dlat, dlon)
        if dist <= radius_km:
            nearby.append((doc, dist))

    nearby.sort(key=lambda x: (x[0].get("price_thb") or float("inf"), x[1]))

    # Context note tells LLM what was geocoded and how precise it is
    context_note = (
        f"Geocoded '{place_name}' as: {display_name[:120]}"
        f" | geometry={osm_type} ({accuracy_note})"
        f" | ref_point=({lat:.6f}, {lon:.6f})"
    )
    return nearby, context_note


# ──────────────────────────────────────────
# [A] Augment — Build LLM Context
# ──────────────────────────────────────────
SYSTEM_PROMPT = """\
You are a bilingual (Thai/English) real estate assistant for Bangkok Bless Asset.

Rules:
1. Detect the user's language — reply in the same language.
2. Use ONLY the provided listing context. Never fabricate data.
3. Present listings sorted cheapest → most expensive (top 5–10).
4. LOCATION SEARCH: When the user mentions any place name (Thai or English),
   the system automatically geocodes it to obtain the latitude/longitude of that
   place, then finds all listings whose coordinates fall within a 5 km radius.
   The RAG context will be labeled "[RAG context — location]" in that case.
   • Always acknowledge the place name and confirm search was done by proximity.
   • Show each listing's distance (km) from that place.
   • If geocoding fails or no listings are nearby, say so and fall back to keyword search.
5. SEMANTIC SEARCH: When no place name is detected, results come from BGE-M3
   vector similarity. Context label will be "[RAG context — semantic]".
6. For every listing include: name, type, location (neighborhood/district/province),
   price (฿ with commas), transit access, and available amenities.
7. If nothing matches, politely say so and suggest refining the search
   (e.g. wider radius, different property type, or different area).
8. Keep answers concise and friendly. Use bullet points for multiple listings.

คุณคือผู้ช่วยอสังหาฯ ของ Bangkok Bless Asset พูดได้ทั้งไทย-อังกฤษ
1. ตอบด้วยภาษาเดียวกับลูกค้า
2. ใช้ข้อมูล listing ที่ให้มาเท่านั้น ห้ามสร้างข้อมูลเอง
3. เรียงจากราคาถูกไปแพง 5–10 อันดับแรก
4. การค้นหาด้วยสถานที่: หากผู้ใช้ระบุชื่อสถานที่ (ภาษาไทยหรืออังกฤษ)
   ระบบจะแปลงชื่อนั้นเป็นละติจูด/ลองจิจูดโดยอัตโนมัติผ่าน Nominatim
   แล้วค้นหาบ้านที่มีพิกัดอยู่ภายในรัศมี 5 กม. จากสถานที่นั้น
   • บอกชื่อสถานที่ที่ค้นหา และยืนยันว่าผลลัพธ์มาจากการค้นหาตามระยะทาง
   • แสดงระยะห่าง (กม.) ของแต่ละรายการจากสถานที่นั้นด้วย
   • หากหาพิกัดสถานที่ไม่ได้ หรือไม่มี listing อยู่ในรัศมี ให้แจ้งและใช้การค้นหาแบบ semantic แทน
5. การค้นหาแบบ semantic: หากไม่มีชื่อสถานที่ ระบบจะใช้ BGE-M3 เวกเตอร์
6. ระบุ ชื่อ ประเภท ทำเล (ย่าน/เขต/จังหวัด) ราคา รถไฟฟ้า สิ่งอำนวยความสะดวก
7. หากไม่พบรายการ ให้แจ้งสุภาพและแนะนำให้ปรับเกณฑ์ (ขยายรัศมี/เปลี่ยนทำเล/ประเภท)
"""


def _format_listing(rank: int, doc: dict, value: float, mode: str) -> str:
    tag   = f"{value:.2f} km" if mode == "location" else f"score={value:.3f}"
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
    ]
    return " | ".join(p for p in parts if p)


def build_rag_context(results: list[tuple], mode: str, geocoded_place: str | None = None) -> str:
    if not results:
        return "No matching listings found. / ไม่พบรายการที่ตรงกับเงื่อนไข"
    if mode == "location":
        sort_note = "sorted by distance then price ฿ low→high"
        header = geocoded_place or ""
        # Surface accuracy warning to LLM if geometry is imprecise
        if geocoded_place and "centroid" in geocoded_place:
            header += (
                "\nNOTE: reference point is a polygon centroid — distances shown are"
                " approximate (may differ from map by 100 m – 2 km depending on building size)."
                " แจ้งผู้ใช้ว่าระยะทางเป็นค่าประมาณเนื่องจากพิกัดสถานที่ไม่ใช่จุดแน่นอน"
            )
    else:
        sort_note = "sorted by price ฿ low→high"
        header = ""
    rows = "\n".join(_format_listing(i, d, v, mode) for i, (d, v) in enumerate(results, 1))
    parts = [p for p in [header, f"[{sort_note}]", rows] if p]
    return "\n".join(parts)


# ──────────────────────────────────────────
# [G] Generate — LLM Response
# ──────────────────────────────────────────
def rag_chat(
    user_query: str,
    history: list[dict],
    model: BGEM3FlagModel,
    idx: faiss.Index,
    docs: list[dict],
    llm: Groq,
) -> tuple[str, list[tuple], str]:
    # [R] Retrieve
    # Step 1: extract just the place name so Nominatim gets an accurate query
    place_name = extract_place_name(user_query, llm)
    location_results, geocoded_place = retrieve_by_location(place_name, docs)
    if location_results:
        results = location_results[:TOP_K]
        mode    = "location"
    else:
        results = retrieve_semantic(user_query, model, idx, docs)
        mode    = "semantic"
        geocoded_place = None

    # [A] Augment
    context  = build_rag_context(results, mode, geocoded_place)
    trimmed  = history[-(MAX_HISTORY_TURNS * 2):]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": f"[RAG context — {mode}]\n{context}"},
        *trimmed,
        {"role": "user", "content": user_query},
    ]

    # [G] Generate
    resp = llm.chat.completions.create(
        model=LLM_MODEL, messages=messages, temperature=LLM_TEMPERATURE,
    )
    return resp.choices[0].message.content, results, mode


# ──────────────────────────────────────────
# Main
# ──────────────────────────────────────────
def main():
    if not os.getenv("GROQ_API_KEY"):
        raise EnvironmentError("Set GROQ_API_KEY in .env  (free key at console.groq.com)")
    if not Path(CSV_PATH).exists():
        raise FileNotFoundError(f"{CSV_PATH} not found in project root")

    print("Loading BGE-M3... (first run downloads ~2.3 GB to E drive)")
    embed_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

    faiss_index, metadata = load_or_build_index(embed_model)
    llm_client = Groq()
    history: list[dict] = []

    geo_ok = geocode_place("Bangkok") is not None  # connectivity check
    print(f"\n{'=' * 60}")
    print(f"Bangkok Bless Asset RAG Chatbot — {len(metadata):,} listings")
    print(f"  Semantic search : BGE-M3 + FAISS  |  threshold={SIMILARITY_THRESHOLD}")
    print(f"  Location search : Nominatim  |  radius={LOCATION_RADIUS_KM} km  |  {'online' if geo_ok else 'OFFLINE'}")
    print(f"  LLM             : {LLM_MODEL}")
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
            print(f"\n[{mode.upper()} — top {len(sources)} results, sorted by price]")
            for doc, val in sources:
                label = f"{val:.2f} km" if mode == "location" else f"score={val:.3f}"
                price = f"฿{doc['price_thb']:,}" if doc.get("price_thb") else "N/A"
                print(f"  • {doc.get('name', '-'):30s} | {doc.get('district', '-'):15s} | {price:>15s} | {label}")


if __name__ == "__main__":
    main()
