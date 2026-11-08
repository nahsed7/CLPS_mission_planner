#!/usr/bin/env python3

import sys
import math
from pathlib import Path

ROOT = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else Path.home() / "Desktop/clpsdatasets"

RASTER_EXTENSIONS = {".tif", ".tiff", ".img", ".vrt"}
TEXT_EXTENSIONS = {".xyz", ".csv", ".txt"}

def inspect_raster(path):
    print(f"\n{'='*75}\nRASTER: {path.name}\n{'='*75}")

    try:
        import rasterio
        import numpy as np

        with rasterio.open(path) as ds:
            print("Dimensions:", ds.width, "x", ds.height)
            print("Bands:", ds.count)
            print("Data types:", ds.dtypes)
            print("CRS:", ds.crs)
            print("Transform:", ds.transform)
            print("Bounds:", ds.bounds)
            print("Nodata:", ds.nodata)
            print("Pixel size:", ds.res)
            print("Units:", ds.units)
            print("Descriptions:", ds.descriptions)
            print("Tags:", ds.tags())

            for band in range(1, ds.count + 1):
                scale = max(1, math.ceil(max(ds.width, ds.height) / 1500))
                sample = ds.read(
                    band,
                    out_shape=(
                        max(1, ds.height // scale),
                        max(1, ds.width // scale)
                    ),
                    masked=True
                )
                values = sample.compressed()
                values = values[np.isfinite(values)]

                print(f"\nBand {band}:")
                if values.size:
                    print("  Sample count:", values.size)
                    print("  Min:", float(values.min()))
                    print("  Max:", float(values.max()))
                    print("  Mean:", float(values.mean()))
                    print("  Median:", float(np.median(values)))
                    print("  Std:", float(values.std()))
                else:
                    print("  No valid sample values")

    except Exception as exc:
        print("RASTER INSPECTION ERROR:", repr(exc))


def inspect_text(path):
    print(f"\n{'='*75}\nTEXT DATA: {path.name}\n{'='*75}")

    rows = []
    total = 0

    try:
        with path.open("r", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(("#", ";")):
                    continue

                total += 1
                if len(rows) < 10000:
                    try:
                        vals = [float(x) for x in line.replace(",", " ").split()]
                        if vals:
                            rows.append(vals)
                    except ValueError:
                        if total <= 5:
                            print("Header/sample:", line[:250])

        print("Total non-comment lines:", total)
        print("Numeric rows sampled:", len(rows))

        if rows:
            width = min(map(len, rows))
            print("Numeric columns:", width)

            for col in range(width):
                values = [
                    r[col] for r in rows
                    if len(r) > col and math.isfinite(r[col])
                ]
                if values:
                    print(
                        f"Column {col + 1}: "
                        f"min={min(values):.6g}, "
                        f"max={max(values):.6g}, "
                        f"mean={sum(values)/len(values):.6g}"
                    )

            print("First numeric row:", rows[0])

    except Exception as exc:
        print("TEXT INSPECTION ERROR:", repr(exc))


def main():
    if not ROOT.exists():
        raise SystemExit(f"Dataset directory does not exist: {ROOT}")

    print("LUNAR DATASET INVENTORY")
    print("Directory:", ROOT)

    files = sorted(p for p in ROOT.rglob("*") if p.is_file())
    print("Files found:", len(files))

    for path in files:
        size = path.stat().st_size
        print(f"{path.name:55} {size / (1024**2):10.2f} MiB")

    for path in files:
        if path.suffix.lower() in RASTER_EXTENSIONS:
            inspect_raster(path)
        elif path.suffix.lower() in TEXT_EXTENSIONS:
            inspect_text(path)

    print("\nINSPECTION COMPLETE")
    print("No source datasets were modified.")


if __name__ == "__main__":
    main()
