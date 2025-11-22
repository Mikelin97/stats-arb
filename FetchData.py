import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import databento as db
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DATA_BENTO_KEY = os.getenv("DATA_BENTO_KEY")
DEFAULT_SCHEMA = "ohlcv-1m"
DATA_DIR = Path("data")


@dataclass(frozen=True)
class AssetRequest:
    dataset: str
    symbol: str


@dataclass(frozen=True)
class PairRequest:
    identifier: str
    description: str
    assets: Dict[str, AssetRequest]


PAIR_CONFIGS: Dict[str, PairRequest] = {
    "pair1": PairRequest(
        identifier="pair1",
        description="WTI Future vs. Brent Future",
        assets={
            "wti_oil_future": AssetRequest(dataset="GLBX.MDP3", symbol="CLZ5"),
            "brent_oil_future": AssetRequest(dataset="IFEU.IMPACT", symbol="BRN FMF0026!"),
        },
    ),
    "pair2": PairRequest(
        identifier="pair2",
        description="Gold Future vs. Silver Future",
        assets={
            "gold_future": AssetRequest(dataset="GLBX.MDP3", symbol="GCZ5"),
            "silver_future": AssetRequest(dataset="GLBX.MDP3", symbol="SIZ5"),
        },
    ),
    "pair3": PairRequest(
        identifier="pair3",
        description="SOFR 3M Future vs. DUK Spot",
        assets={
            "sofr_3m_future": AssetRequest(dataset="GLBX.MDP3", symbol="SR3Z5"),
            "duk_spot": AssetRequest(dataset="XNAS.ITCH", symbol="DUK"),
        },
    ),
    "pair4": PairRequest(
        identifier="pair4",
        description="Corn Future vs. Soybean Oil Future",
        assets={
            "corn_future": AssetRequest(dataset="GLBX.MDP3", symbol="ZCZ5"),
            "soybean_oil_future": AssetRequest(dataset="GLBX.MDP3", symbol="ZLZ5"),
        },
    ),
    "pair5": PairRequest(
        identifier="pair5",
        description="Bitcoin ETF vs. Ethereum ETF",
        assets={
            "ibit_etf": AssetRequest(dataset="XNAS.ITCH", symbol="IBIT"),
            "etha_etf": AssetRequest(dataset="XNAS.ITCH", symbol="ETHA"),
        },
    ),
    "pair6": PairRequest(
        identifier="pair6",
        description="NatGas HH Future vs. NatGas LS Future",
        assets={
            "natgas_hh_future": AssetRequest(dataset="GLBX.MDP3", symbol="NGZ25"),
            "natgas_ls_future": AssetRequest(dataset="IFEU.IMPACT", symbol="G   FMZ0025!"),
        },
    ),
    "pair7": PairRequest(
        identifier="pair7",
        description="MSTR Spot vs. IBIT ETF",
        assets={
            "mstr_spot": AssetRequest(dataset="XNAS.ITCH", symbol="MSTR"),
            "ibit_etf": AssetRequest(dataset="XNAS.ITCH", symbol="IBIT"),
        },
    ),
    "pair8": PairRequest(
        identifier="pair8",
        description="TXN Spot vs. ADI Spot",
        assets={
            "txn_spot": AssetRequest(dataset="XNAS.ITCH", symbol="TXN"),
            "adi_spot": AssetRequest(dataset="XNAS.ITCH", symbol="ADI"),
        },
    ),
    "pair9": PairRequest(
        identifier="pair9",
        description="RBOB Gas Future vs. ULSD Gas Future",
        assets={
            "rbob_gas_future": AssetRequest(dataset="GLBX.MDP3", symbol="RBZ5"),
            "ulsd_gas_future": AssetRequest(dataset="GLBX.MDP3", symbol="HOZ5"),
        },
    ),
    "pair10": PairRequest(
        identifier="pair10",
        description="Gold Future vs. Micro Gold Future",
        assets={
            "gold_future": AssetRequest(dataset="GLBX.MDP3", symbol="GCZ5"),
            "micro_gold_future": AssetRequest(dataset="GLBX.MDP3", symbol="MGCZ5"),
        },
    ),
    "pair11": PairRequest(
        identifier="pair11",
        description="Silver Future vs. Micro Silver Future",
        assets={
            "silver_future": AssetRequest(dataset="GLBX.MDP3", symbol="SIZ5"),
            "micro_silver_future": AssetRequest(dataset="GLBX.MDP3", symbol="SILZ5"),
        },
    ),
}


def ensure_api_key() -> str:
    if not DATA_BENTO_KEY:
        raise EnvironmentError("DATA_BENTO_KEY is not set. Please configure your Databento API key.")
    return DATA_BENTO_KEY


def ensure_iso_datetime(value: str) -> str:
    ts = pd.to_datetime(value)
    if ts.tzinfo is None:
        return ts.isoformat()
    return ts.tz_convert("UTC").isoformat().replace("+00:00", "Z")


def list_pairs() -> None:
    print("Available pairs:")
    for pair_id, cfg in PAIR_CONFIGS.items():
        print(f"  {pair_id:<6} {cfg.description}")


def prompt_for_pairs() -> List[str]:
    list_pairs()
    selection = input("Enter comma-separated pair identifiers (or 'all'): ").strip()
    if selection.lower() in {"all", "a", "*"}:
        return list(PAIR_CONFIGS.keys())
    picks: List[str] = []
    for token in selection.split(","):
        token = token.strip()
        if not token:
            continue
        if token not in PAIR_CONFIGS:
            raise ValueError(f"Unknown pair identifier: {token}")
        picks.append(token)
    return picks


def download_pair(
    client: db.Historical,
    pair_cfg: PairRequest,
    start: str,
    end: str,
    output_dir: Path,
    schema: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for asset_suffix, req in pair_cfg.assets.items():
        print(
            f"Downloading {pair_cfg.identifier} ({pair_cfg.description}) "
            f"- {asset_suffix} [{req.symbol}] from {start} to {end}"
        )
        data = client.timeseries.get_range(
            dataset=req.dataset,
            schema=schema,
            stype_in="raw_symbol",
            symbols=[req.symbol],
            start=start,
            end=end,
        )
        df = data.to_df()
        note_message = None
        if "IFEU" in req.dataset.upper():
            if "publisher_id" in df.columns:
                before = len(df)
                df = df[df["publisher_id"] == 57]
                removed = before - len(df)
                note_message = (
                    f"Filtered to publisher_id == 57 for {req.symbol} "
                    f"in dataset {req.dataset}; dropped {removed} rows."
                )
                print(" ", note_message)
            else:
                note_message = (
                    f"Expected publisher_id column missing for {req.symbol}; "
                    "no publisher-based filtering applied."
                )
                print(" ", note_message)
        file_path = output_dir / f"{pair_cfg.identifier}_{asset_suffix}_{schema}.csv"
        df.to_csv(file_path)
        print(f"Saved {file_path}")
        if note_message:
            note_path = file_path.with_suffix(file_path.suffix + ".note.txt")
            note_path.write_text(note_message + "\n")


def download_pairs(
    pair_ids: Iterable[str],
    start: str,
    end: str,
    output_dir: Path,
    schema: str,
) -> None:
    ensure_api_key()
    client = db.Historical(key=DATA_BENTO_KEY)
    normalized_start = ensure_iso_datetime(start)
    normalized_end = ensure_iso_datetime(end)

    for pair_id in pair_ids:
        pair_cfg = PAIR_CONFIGS.get(pair_id)
        if not pair_cfg:
            raise ValueError(f"Unknown pair identifier: {pair_id}")
        download_pair(client, pair_cfg, normalized_start, normalized_end, output_dir, schema)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Databento data for configured pairs.")
    parser.add_argument("--pairs", nargs="+", help="Pair identifiers to download (e.g. pair1 pair7).")
    parser.add_argument("--all", action="store_true", help="Download all configured pairs.")
    parser.add_argument("--start", type=str, help="ISO start (e.g. 2024-01-15T09:30:00Z).")
    parser.add_argument("--end", type=str, help="ISO end time.")
    parser.add_argument("--schema", type=str, default=DEFAULT_SCHEMA, help="Databento schema (default: ohlcv-1m).")
    parser.add_argument("--output", type=str, default=str(DATA_DIR), help="Output directory for CSV files.")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Prompt for missing information instead of relying on defaults.",
    )
    parser.add_argument("--list", action="store_true", help="List all available pairs and exit.")
    return parser.parse_args()


def resolve_pairs(args: argparse.Namespace) -> List[str]:
    if args.all:
        return list(PAIR_CONFIGS.keys())
    if args.pairs:
        unknown = [p for p in args.pairs if p not in PAIR_CONFIGS]
        if unknown:
            raise ValueError(f"Unknown pair identifiers: {', '.join(unknown)}")
        return args.pairs
    if args.interactive:
        return prompt_for_pairs()
    raise ValueError("No pairs specified. Use --pairs, --all, or --interactive.")


def resolve_date(args_value: Optional[str], prompt_text: str, default: Optional[str] = None) -> str:
    if args_value:
        return args_value
    if default and not args_value and not prompt_text:
        return default
    if default:
        prompt = f"{prompt_text} [{default}]: "
    else:
        prompt = f"{prompt_text}: "
    value = input(prompt).strip() if prompt_text else ""
    if not value:
        if default:
            value = default
        else:
            raise ValueError(f"{prompt_text} is required.")
    return value


def main() -> None:
    args = parse_args()
    if args.list:
        list_pairs()
        if not (args.pairs or args.all or args.interactive):
            return
    try:
        pair_ids = resolve_pairs(args)
    except ValueError as exc:
        if args.interactive:
            print(exc)
            pair_ids = prompt_for_pairs()
        else:
            raise

    default_start = pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=5)
    default_end = pd.Timestamp.utcnow()
    start = resolve_date(args.start, "Enter start datetime", default_start.isoformat())
    end = resolve_date(args.end, "Enter end datetime", default_end.isoformat())

    output_dir = Path(args.output)
    download_pairs(pair_ids, start, end, output_dir, args.schema)


if __name__ == "__main__":
    main()
