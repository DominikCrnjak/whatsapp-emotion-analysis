import argparse
import csv
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# -----------------------------
# WhatsApp parsing regexi
# -----------------------------
LINE_START_RE = re.compile(
    r"""^
    (?P<date>(?:\d{1,2}[\/\.]\d{1,2}[\/\.]\d{2,4}))
    [,\.\s]+
    (?P<time>\d{1,2}:\d{2}(?:\s?[AP]M)?)
    \s*-\s*
    (?P<rest>.*)
    $""",
    re.VERBOSE,
)
SENDER_MSG_RE = re.compile(r"^(?P<sender>[^:]+?):\s(?P<msg>.*)$")

SYSTEM_HINTS = [
    "Messages to this chat",
    "Messages and calls are end-to-end encrypted",
    "<Media omitted>",
    "changed",
    "added",
    "left",
    "joined",
    "created group",
    "You created group",
    "pinned a message",
]

EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_RE = re.compile(r"(?<!\w)(\+?\d[\d\s().-]{6,}\d)(?!\w)")
URL_RE = re.compile(r"\bhttps?://\S+\b", re.IGNORECASE)

EXPECTED_HEADER = ["turn_id", "conversation_id", "datetime", "speaker", "text"]

# -----------------------------
# Pomoćne strukture
# -----------------------------
@dataclass
class ParsedMsg:
    dt: str
    sender_raw: str
    text_raw: str


# -----------------------------
# Utility
# -----------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def looks_like_system(text: str) -> bool:
    t = text.strip().lower()
    return any(h.lower() in t for h in SYSTEM_HINTS)


def parse_datetime(date_str: str, time_str: str) -> str:
    d = date_str.replace(".", "/")
    raw = f"{d} {time_str}".strip()

    fmts = [
        "%d/%m/%Y %H:%M",
        "%d/%m/%y %H:%M",
        "%m/%d/%Y %H:%M",
        "%m/%d/%y %H:%M",
        "%d/%m/%Y %I:%M %p",
        "%d/%m/%y %I:%M %p",
        "%m/%d/%Y %I:%M %p",
        "%m/%d/%y %I:%M %p",
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(raw, fmt).isoformat(sep=" ")
        except ValueError:
            pass

    return raw


def scrub_text(text: str) -> str:
    t = text
    t = EMAIL_RE.sub("[EMAIL]", t)
    t = PHONE_RE.sub("[PHONE]", t)
    t = URL_RE.sub("[URL]", t)
    return t


# -----------------------------
# Parsiranje WhatsApp txt-a
# -----------------------------
def whatsapp_txt_to_msgs(txt_path: Path, keep_system: bool = False) -> List[ParsedMsg]:
    msgs: List[ParsedMsg] = []
    current: Optional[ParsedMsg] = None

    with txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            m = LINE_START_RE.match(line)
            if m:
                if current is not None:
                    msgs.append(current)

                dt_iso = parse_datetime(m.group("date"), m.group("time"))
                rest = m.group("rest")

                sm = SENDER_MSG_RE.match(rest)
                if sm:
                    sender = sm.group("sender").strip()
                    msg = sm.group("msg").strip()
                    if msg and not looks_like_system(msg):
                        current = ParsedMsg(dt=dt_iso, sender_raw=sender, text_raw=msg)
                    else:
                        current = None
                else:
                    if keep_system and rest.strip() and not looks_like_system(rest):
                        current = ParsedMsg(dt=dt_iso, sender_raw="SYSTEM", text_raw=rest.strip())
                    else:
                        current = None
            else:
                if current is not None:
                    extra = line.strip()
                    if extra:
                        current.text_raw += "\n" + extra

    if current is not None:
        msgs.append(current)

    return msgs


# -----------------------------
# State (speaker map + import log)
# -----------------------------
def load_json(path: Path, default):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def get_or_create_speaker(sender_raw: str, speaker_map: Dict[str, str]) -> str:
    key = sender_raw.strip()
    if key in speaker_map:
        return speaker_map[key]

    used = set(speaker_map.values())
    i = 1
    while True:
        candidate = f"Speaker_{i:03d}"
        if candidate not in used:
            speaker_map[key] = candidate
            return candidate
        i += 1


# -----------------------------
# Input expansion
# -----------------------------
def expand_inputs(inputs: List[str]) -> List[Path]:
    paths: List[Path] = []
    for s in inputs:
        p = Path(s)
        if any(ch in s for ch in ["*", "?", "["]):  # glob pattern
            paths.extend(Path().glob(s))
        elif p.is_dir():
            paths.extend(sorted(p.glob("*.txt")))
        else:
            paths.append(p)

    uniq: List[Path] = []
    seen = set()
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            uniq.append(rp)
            seen.add(rp)
    return uniq


# -----------------------------
# One-file -> one-CSV
# -----------------------------
def write_one_csv(
    txt_path: Path,
    out_csv_path: Path,
    speaker_map: Dict[str, str],
    keep_system: bool = False,
) -> Tuple[bool, str]:
    msgs = whatsapp_txt_to_msgs(txt_path, keep_system=keep_system)
    if not msgs:
        return (False, f"SKIP: nema poruka u {txt_path.name}")

    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    conversation_id = f"conv_{sha256_file(txt_path)[:8]}"  # stabilan id po fileu

    with out_csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(EXPECTED_HEADER)

        turn_id = 1
        for m in msgs:
            speaker = get_or_create_speaker(m.sender_raw, speaker_map)
            text = scrub_text(m.text_raw)

            w.writerow([turn_id, conversation_id, m.dt, speaker, text])
            turn_id += 1

    return (True, f"OK: {txt_path.name} -> {out_csv_path.name} ({len(msgs)} poruka)")


def main():
    ap = argparse.ArgumentParser(
        description="WhatsApp TXT -> (za svaki input) zasebni CSV u isti folder"
    )
    ap.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Putanja do .txt filea, foldera (uzima sve *.txt), ili glob (npr chats/*.txt).",
    )
    ap.add_argument(
        "--state-dir",
        default=".wa_state",
        help="Folder gdje se čuva speaker_map (konzistentna anonimizacija).",
    )
    ap.add_argument(
        "--keep-system",
        action="store_true",
        help="Ako uključiš, zadržava i SYSTEM poruke (inače se ignoriraju).",
    )

    args = ap.parse_args()

    state_dir = Path(args.state_dir).resolve()
    speaker_map_path = state_dir / "speaker_map.json"
    speaker_map = load_json(speaker_map_path, default={})

    inputs = expand_inputs(args.input)
    if not inputs:
        print("Nema ulaznih fileova za obradu.")
        return

    ok_any = False
    for txt in inputs:
        txt = txt.resolve()
        if not txt.exists():
            print(f"SKIP: ne postoji {txt}")
            continue

        out_csv = txt.with_suffix(".csv")  # isti folder, inputname.csv
        ok, msg = write_one_csv(
            txt_path=txt,
            out_csv_path=out_csv,
            speaker_map=speaker_map,
            keep_system=args.keep_system,
        )
        print(msg)
        ok_any = ok_any or ok

    # spremi speaker map (da anonimizacija bude konzistentna kroz sve runove)
    save_json(speaker_map_path, speaker_map)

    if ok_any:
        print(f"\nState (speaker map): {speaker_map_path}")


if __name__ == "__main__":
    main()
