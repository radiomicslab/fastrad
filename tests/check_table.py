import re


def parse_c_table(file_path):
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
        # remove { and }
        line = line.replace("{", "").replace("}", "").replace(",", " ")
        nums = [int(x) for x in line.split() if x]
        if nums:
            rows.append(nums)
    return rows


from fastrad.features.shape_utils import triTable as py_table

c_table = parse_c_table("tests/cShape.c")

for i in range(128):
    if c_table[i] != py_table[i]:
        print(f"Mismatch at {i}:")
        print(f"C  : {c_table[i]}")
        print(f"Py : {py_table[i]}")

print("Table check complete.")
