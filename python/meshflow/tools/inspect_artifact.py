"""CLI tool to dump a compiled artifact (.mpk) as pretty-printed JSON."""

import sys
from pathlib import Path

import msgpack
import orjson


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python -m meshflow.tools.inspect_artifact <path.mpk>", file=sys.stderr)
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    raw = msgpack.unpackb(path.read_bytes(), raw=False, strict_map_key=False)
    json_bytes = orjson.dumps(raw, option=orjson.OPT_INDENT_2)
    sys.stdout.buffer.write(json_bytes + b"\n")


if __name__ == "__main__":
    main()
