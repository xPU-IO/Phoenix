#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import re
import subprocess
import tempfile

parser = argparse.ArgumentParser(description="CPU-only tests of the actual Phoenix and snvme registration code")
parser.add_argument("--tutti-source", type=Path, help="Optional extra snvme integration regression")
parser.add_argument("--tmpdir", type=Path, required=True, help="Existing scratch directory; never defaults to /tmp")
checker = parser.add_mutually_exclusive_group()
checker.add_argument("--sanitize", action="store_true", help="Requires compiler ASan/UBSan runtimes")
checker.add_argument("--valgrind", action="store_true", help="Run binaries under Valgrind memcheck")
args = parser.parse_args()
here = Path(__file__).resolve().parent
phoenix = here.parent.parent


def function(source, name):
    match = re.search(r"^(?:static )?(?:void|int) " + name + r"\([^;]*?\)\s*\{", source, re.M)
    if not match:
        raise RuntimeError("Function not found: " + name)
    depth = 1
    end = match.end()
    while depth:
        if source[end] == "{":
            depth += 1
        elif source[end] == "}":
            depth -= 1
        end += 1
    return source[match.start():end]


with tempfile.TemporaryDirectory(prefix="p2p-unit-", dir=args.tmpdir) as tmp:
    out = Path(tmp)
    service = (phoenix / "module/phxfs-p2p-service.c").read_text()
    service = re.sub(r"^#include[^\n]*\n", "", service, flags=re.M)
    (out / "service-under-test.c").write_text(service)
    tests = ["test_service"]
    if args.tutti_source:
        mapping = (args.tutti_source / "csrc/device_manager/nvme/kernel_modules/snvme/map.c").read_text()
        (out / "map-under-test.c").write_text(function(mapping, "release_gpu_memory") + "\n" + function(mapping, "map_gpu_memory"))
        tests.append("test_map")
    for name in tests:
        binary = out / name
        flags = ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"] if args.sanitize else []
        subprocess.run([os.environ.get("CC", "cc"), "-std=gnu11", "-Wall", "-Wextra", "-Werror",
                        "-g", "-pthread"] + flags +
                       ["-I", str(out), str(here / (name + ".c")), "-o", str(binary)], check=True)
        command = [str(binary)]
        if args.valgrind:
            command = ["valgrind", "--leak-check=full", "--errors-for-leak-kinds=all",
                       "--error-exitcode=1"] + command
        subprocess.run(command, check=True)
