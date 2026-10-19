import re


def extract_tables(file_path, out_file):
    with open(file_path, "r") as f:
        content = f.read()

    m = re.search(
        r"static const int triTable\[128\]\[16\] = \{(.*?)\};", content, re.DOTALL
    )
    if m is None:
        raise ValueError("triTable pattern not found in source file")
    table_str = m.group(1)

    rows = []
    for line in table_str.split("\n"):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        line = line.replace("{", "[").replace("}", "]")
        rows.append("    " + line)

    with open(out_file, "w") as f:
        f.write("import torch\n\n")
        f.write("gridAngles = [\n")
        f.write("    [0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0],\n")
        f.write("    [1, 0, 0], [1, 0, 1], [1, 1, 1], [1, 1, 0]\n")
        f.write("]\n\n")
        f.write("vertList = [\n")
        f.write("    [0, 0, 0.5], [0, 0.5, 1], [0, 1, 0.5], [0, 0.5, 0],\n")
        f.write("    [1, 0, 0.5], [1, 0.5, 1], [1, 1, 0.5], [1, 0.5, 0],\n")
        f.write("    [0.5, 0, 0], [0.5, 0, 1], [0.5, 1, 1], [0.5, 1, 0]\n")
        f.write("]\n\n")
        f.write("triTable = [\n")
        f.write("\n".join(rows) + "\n")
        f.write("]\n")


extract_tables("tests/cShape.c", "fastrad/features/shape_tables.py")
