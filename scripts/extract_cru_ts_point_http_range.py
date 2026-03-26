#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import re
import struct
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


NC_DIMENSION = 10
NC_VARIABLE = 11
NC_ATTRIBUTE = 12

TYPE_SIZES = {
    1: 1,  # byte
    2: 1,  # char
    3: 2,  # short
    4: 4,  # int
    5: 4,  # float
    6: 8,  # double
}

TYPE_CODES = {
    1: ">i1",
    2: "S1",
    3: ">i2",
    4: ">i4",
    5: ">f4",
    6: ">f8",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a single CRU TS point series from an HTTP Range-enabled NetCDF classic file."
    )
    parser.add_argument("--url", required=True, help="HTTP URL to a CRU TS NetCDF file.")
    parser.add_argument("--variable", default="cld", help="Variable to extract. Defaults to cld.")
    parser.add_argument("--latitude", type=float, required=True, help="Target latitude.")
    parser.add_argument("--longitude", type=float, required=True, help="Target longitude.")
    parser.add_argument("--output-csv", type=Path, required=True, help="Output CSV path.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=120,
        help="How many monthly point offsets to request per HTTP multi-range batch.",
    )
    parser.add_argument(
        "--header-bytes",
        type=int,
        default=262144,
        help="How many leading bytes to fetch for header parsing.",
    )
    return parser.parse_args()


def fetch_range(url: str, range_spec: str) -> tuple[bytes, dict[str, str]]:
    req = Request(url, headers={"Range": f"bytes={range_spec}"})
    with urlopen(req, timeout=120) as resp:
        data = resp.read()
        headers = {k.lower(): v for k, v in resp.headers.items()}
    return data, headers


def parse_multipart_ranges(body: bytes, content_type: str) -> dict[int, bytes]:
    match = re.search(r"boundary=([^\s;]+)", content_type)
    if not match:
        raise RuntimeError(f"Missing multipart boundary in content-type: {content_type}")
    boundary = b"--" + match.group(1).encode("ascii")
    out: dict[int, bytes] = {}
    for part in body.split(boundary):
        if b"Content-range:" not in part and b"Content-Range:" not in part:
            continue
        try:
            head, payload = part.split(b"\r\n\r\n", 1)
        except ValueError:
            continue
        payload = payload.rstrip(b"\r\n")
        head_text = head.decode("latin1", errors="ignore")
        m = re.search(r"Content-[Rr]ange:\s*bytes\s+(\d+)-(\d+)/", head_text)
        if not m:
            continue
        start = int(m.group(1))
        out[start] = payload
    return out


class HeaderReader:
    def __init__(self, payload: bytes):
        self.buf = io.BytesIO(payload)

    def read(self, n: int) -> bytes:
        out = self.buf.read(n)
        if len(out) != n:
            raise EOFError("Unexpected end of header payload")
        return out

    def read_int(self) -> int:
        return struct.unpack(">i", self.read(4))[0]

    def read_string(self) -> str:
        n = self.read_int()
        raw = self.read(n)
        pad = (-n) % 4
        if pad:
            self.read(pad)
        return raw.decode("latin1")

    def skip_values(self, nc_type: int, count: int) -> None:
        item_size = TYPE_SIZES[nc_type]
        nbytes = item_size * count
        self.read(nbytes)
        pad = (-nbytes) % 4
        if pad:
            self.read(pad)

    def read_att_array(self) -> None:
        tag = self.read_int()
        if tag == 0:
            self.read_int()
            return
        if tag != NC_ATTRIBUTE:
            raise RuntimeError(f"Unexpected attribute tag {tag}")
        count = self.read_int()
        for _ in range(count):
            _ = self.read_string()
            nc_type = self.read_int()
            nelems = self.read_int()
            self.skip_values(nc_type, nelems)


def parse_netcdf_header(header: bytes) -> dict[str, Any]:
    rd = HeaderReader(header)
    magic = rd.read(4)
    if magic != b"CDF\x01":
        raise RuntimeError(f"Unsupported NetCDF magic {magic!r}; expected classic CDF\\x01")
    num_records = rd.read_int()

    dims_tag = rd.read_int()
    if dims_tag == 0:
        _ = rd.read_int()
        dims: list[dict[str, Any]] = []
    elif dims_tag == NC_DIMENSION:
        dims_count = rd.read_int()
        dims = []
        for _ in range(dims_count):
            name = rd.read_string()
            length = rd.read_int()
            dims.append({"name": name, "length": None if length == 0 else length})
    else:
        raise RuntimeError(f"Unexpected dimension tag {dims_tag}")

    rd.read_att_array()

    vars_tag = rd.read_int()
    if vars_tag != NC_VARIABLE:
        raise RuntimeError(f"Unexpected variable tag {vars_tag}")
    vars_count = rd.read_int()
    variables: dict[str, dict[str, Any]] = {}
    recsize = 0
    for _ in range(vars_count):
        name = rd.read_string()
        dim_count = rd.read_int()
        dim_ids = [rd.read_int() for _ in range(dim_count)]
        rd.read_att_array()
        nc_type = rd.read_int()
        vsize = rd.read_int()
        begin = rd.read_int()
        dim_names = [dims[i]["name"] for i in dim_ids]
        dim_sizes = [dims[i]["length"] for i in dim_ids]
        is_record = bool(dim_sizes) and dim_sizes[0] is None
        if is_record:
            recsize += vsize
        variables[name] = {
            "name": name,
            "dim_names": dim_names,
            "dim_sizes": dim_sizes,
            "nc_type": nc_type,
            "dtype": np.dtype(TYPE_CODES[nc_type]),
            "itemsize": TYPE_SIZES[nc_type],
            "vsize": vsize,
            "begin": begin,
            "is_record": is_record,
        }
    return {
        "num_records": num_records,
        "dimensions": dims,
        "variables": variables,
        "record_size": recsize,
    }


def read_contiguous_array(url: str, begin: int, count: int, dtype: np.dtype) -> np.ndarray:
    end = begin + count * dtype.itemsize - 1
    payload, _ = fetch_range(url, f"{begin}-{end}")
    return np.frombuffer(payload, dtype=dtype).astype(dtype.newbyteorder("="), copy=False)


def fetch_strided_point_series(
    url: str,
    variable: dict[str, Any],
    record_count: int,
    record_size: int,
    flat_index: int,
    batch_size: int,
) -> np.ndarray:
    itemsize = int(variable["itemsize"])
    base = int(variable["begin"]) + flat_index * itemsize
    offsets = [base + i * record_size for i in range(record_count)]
    values = np.empty(record_count, dtype=np.float32)

    for batch_start in range(0, record_count, batch_size):
        batch_offsets = offsets[batch_start : batch_start + batch_size]
        range_spec = ",".join(f"{off}-{off + itemsize - 1}" for off in batch_offsets)
        payload, headers = fetch_range(url, range_spec)
        content_type = headers.get("content-type", "")
        if "multipart/byteranges" not in content_type.lower():
            raise RuntimeError(f"Server did not return multipart/byteranges for batch starting {batch_start}")
        parts = parse_multipart_ranges(payload, content_type)
        for i, off in enumerate(batch_offsets):
            chunk = parts.get(off)
            if chunk is None or len(chunk) < itemsize:
                raise RuntimeError(f"Missing range payload for offset {off}")
            values[batch_start + i] = np.frombuffer(chunk[:itemsize], dtype=variable["dtype"])[0]
    return values


def infer_monthly_index(url: str, record_count: int) -> pd.DatetimeIndex:
    match = re.search(r"\.(\d{4})\.(\d{4})\.", url)
    if not match:
        raise RuntimeError(f"Could not infer year span from URL: {url}")
    start_year = int(match.group(1))
    end_year = int(match.group(2))
    months = (end_year - start_year + 1) * 12
    if months != record_count:
        raise RuntimeError(f"Expected {months} monthly records from URL span, got {record_count}")
    return pd.date_range(f"{start_year}-01-01", periods=record_count, freq="MS")


def main() -> None:
    args = parse_args()
    header_payload, _ = fetch_range(args.url, f"0-{args.header_bytes - 1}")
    header = parse_netcdf_header(header_payload)
    variables = header["variables"]

    if args.variable not in variables:
        raise RuntimeError(f"Variable {args.variable!r} not found. Available: {sorted(variables)}")
    if "lat" not in variables or "lon" not in variables:
        raise RuntimeError("NetCDF missing lat/lon coordinate variables")

    lon_var = variables["lon"]
    lat_var = variables["lat"]
    lon_vals = read_contiguous_array(args.url, lon_var["begin"], lon_var["dim_sizes"][0], lon_var["dtype"])
    lat_vals = read_contiguous_array(args.url, lat_var["begin"], lat_var["dim_sizes"][0], lat_var["dtype"])

    lon_idx = int(np.argmin(np.abs(lon_vals - args.longitude)))
    lat_idx = int(np.argmin(np.abs(lat_vals - args.latitude)))
    lon_sel = float(lon_vals[lon_idx])
    lat_sel = float(lat_vals[lat_idx])

    target_var = variables[args.variable]
    if not target_var["is_record"]:
        raise RuntimeError(f"Variable {args.variable} is not a record variable")
    if target_var["dim_sizes"][1:] != [len(lat_vals), len(lon_vals)]:
        raise RuntimeError(f"Unexpected variable shape for {args.variable}: {target_var['dim_sizes']}")

    flat_index = lat_idx * len(lon_vals) + lon_idx
    values = fetch_strided_point_series(
        url=args.url,
        variable=target_var,
        record_count=header["num_records"],
        record_size=header["record_size"],
        flat_index=flat_index,
        batch_size=max(1, int(args.batch_size)),
    )

    values = values.astype(np.float64)
    values = np.where(np.isclose(values, 9.96921e36), np.nan, values)
    index = infer_monthly_index(args.url, header["num_records"])
    frame = pd.DataFrame(
        {
            "timestamp": index,
            f"{args.variable}_pct": values,
            "grid_lat": lat_sel,
            "grid_lon": lon_sel,
        }
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_csv, index=False)

    print(f"Wrote {args.output_csv}")
    print(
        {
            "rows": int(len(frame)),
            "variable": args.variable,
            "grid_lat": lat_sel,
            "grid_lon": lon_sel,
            "min": float(np.nanmin(values)),
            "max": float(np.nanmax(values)),
        }
    )


if __name__ == "__main__":
    main()
