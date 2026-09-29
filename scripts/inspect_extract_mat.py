from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocess_functions.manifest import load_session_records
from preprocess_functions.neural import find_extract_outputs, load_extract_labels
from preprocess_functions.pipeline import matlab_output_dir


def main() -> None:
    args = parse_args()
    records = load_session_records(
        args.manifest,
        sheet_name=args.sheet,
        lab_drive=args.lab_drive,
    )
    selected_records = [record for record in records if selected(record, args.only)]
    if not selected_records:
        raise SystemExit(f"No manifest records matched --only {args.only!r}")

    for record in selected_records[: args.max_records]:
        print("=" * 88)
        print(f"recording_id: {record.recording_id}")
        print(f"session_id:   {record.session_id}")
        print(f"mouse_id:     {record.mouse_id}")
        out_dir = record.matlab_output_dir or matlab_output_dir(args.output_root, record)
        print(f"matlab dir:   {out_dir}")

        traces_mat, labels_mat = explicit_or_discovered_outputs(record, out_dir)
        labels = load_extract_labels(labels_mat)
        print(f"traces mat:   {traces_mat}")
        print(f"labels mat:   {labels_mat}")
        print(f"labels shape: {labels.shape}; accepted label 1: {int(np.sum(labels == 1))}")
        inspect_h5_mat(traces_mat, args.max_items)


def explicit_or_discovered_outputs(record, output_dir: Path) -> tuple[Path, Path]:
    if record.extract_output_mat and record.extract_labels_mat:
        return record.extract_output_mat, record.extract_labels_mat
    return find_extract_outputs(
        output_dir,
        recording_id=record.recording_id,
        session_id=record.session_id,
    )


def inspect_h5_mat(path: Path, max_items: int) -> None:
    import h5py

    with h5py.File(path, "r") as h5f:
        print("\nTop-level keys:")
        for key in h5f.keys():
            print(f"  {key}: {describe_node(h5f[key])}")

        print("\nCandidate spatial_weights nodes:")
        candidate_names: list[str] = []

        def collect(name: str, obj: Any) -> None:
            if name.endswith("spatial_weights") or "/spatial_weights/" in name:
                candidate_names.append(name)

        h5f.visititems(collect)
        if not candidate_names:
            print("  none found")

        for name in candidate_names[:max_items]:
            node = h5f[name]
            print_node(h5f, name, node, indent="  ", max_items=max_items)

        print("\nSparse-looking groups with data/ir/jc:")
        sparse_names: list[str] = []

        def collect_sparse(name: str, obj: Any) -> None:
            if hasattr(obj, "keys") and {"data", "ir", "jc"}.issubset(set(obj.keys())):
                sparse_names.append(name)

        h5f.visititems(collect_sparse)
        if not sparse_names:
            print("  none found")
        for name in sparse_names[:max_items]:
            print_node(h5f, name, h5f[name], indent="  ", max_items=max_items)


def print_node(h5f: Any, name: str, node: Any, *, indent: str, max_items: int) -> None:
    import h5py

    print(f"{indent}{name}: {describe_node(node)}")
    attrs = dict(node.attrs.items())
    if attrs:
        print(f"{indent}  attrs: {format_attrs(attrs)}")

    if isinstance(node, h5py.Dataset):
        ref_type = h5py.check_dtype(ref=node.dtype)
        if ref_type is not None:
            refs = np.asarray(node).ravel(order="F")
            print(f"{indent}  reference dtype: {ref_type}; refs: {len(refs)}")
            shown = 0
            for ref in refs:
                if not ref:
                    continue
                target = h5f[ref]
                print(f"{indent}  -> {target.name}: {describe_node(target)}")
                shown += 1
                if shown >= max_items:
                    break
        else:
            values = np.asarray(node)
            flat = values.ravel(order="F")
            sample = flat[: min(len(flat), max_items)]
            print(f"{indent}  sample: {sample!r}")

    if hasattr(node, "keys"):
        for key in list(node.keys())[:max_items]:
            child = node[key]
            print(f"{indent}  {key}: {describe_node(child)}")
            child_attrs = dict(child.attrs.items())
            if child_attrs:
                print(f"{indent}    attrs: {format_attrs(child_attrs)}")


def describe_node(node: Any) -> str:
    if hasattr(node, "shape") and hasattr(node, "dtype"):
        return f"Dataset shape={node.shape} dtype={node.dtype}"
    if hasattr(node, "keys"):
        keys = list(node.keys())
        return f"Group keys={keys[:8]}{'...' if len(keys) > 8 else ''}"
    return type(node).__name__


def format_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in attrs.items():
        if isinstance(value, bytes):
            out[key] = value.decode(errors="replace")
        elif isinstance(value, np.ndarray):
            out[key] = value.tolist()
        else:
            out[key] = value
    return out


def selected(record, only: str | None) -> bool:
    if only is None:
        return True
    return only in {
        record.recording_id,
        record.session_id,
        record.trial_type,
        record.mouse_id,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect EXTRACT MAT files used for cell registration."
    )
    parser.add_argument("manifest", help="CSV/XLSX manifest, preferably manifest_with_cells.csv")
    parser.add_argument("--sheet", help="Excel sheet to process; defaults to all sheets")
    parser.add_argument("--lab-drive", help="Local mount for lab paths such as \\Data\\May")
    parser.add_argument("--output-root", default="preprocess_out")
    parser.add_argument("--only", help="Mouse ID, recording ID, session ID, or trial type to inspect")
    parser.add_argument("--max-records", type=int, default=1)
    parser.add_argument("--max-items", type=int, default=12)
    return parser.parse_args()


if __name__ == "__main__":
    main()
