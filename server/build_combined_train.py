import os
import shutil
from glob import glob

BASE_TRAIN = os.path.join("chest_xray", "train")
LABELED = os.path.join("server", "storage", "labeled")
OUT = os.path.join("server", "storage", "combined_train")

CLASSES = ["NORMAL", "PNEUMONIA"]
EXTS = (".jpeg", ".jpg", ".png", ".bmp", ".webp")

def copy_all(src_dir, dst_dir, prefix):
    if not os.path.isdir(src_dir):
        return 0
    os.makedirs(dst_dir, exist_ok=True)
    count = 0
    for root, _, files in os.walk(src_dir):
        for fn in files:
            if fn.lower().endswith(EXTS):
                src = os.path.join(root, fn)
                # Avoid name collisions
                dst = os.path.join(dst_dir, f"{prefix}_{count:07d}_{fn}")
                shutil.copy2(src, dst)
                count += 1
    return count

def main():
    os.makedirs(OUT, exist_ok=True)

    totals = {}
    for c in CLASSES:
        out_c = os.path.join(OUT, c)

        # Start fresh each time (simple approach)
        if os.path.isdir(out_c):
            for f in glob(os.path.join(out_c, "*")):
                if os.path.isfile(f):
                    os.remove(f)
        else:
            os.makedirs(out_c, exist_ok=True)

        base_src = os.path.join(BASE_TRAIN, c)
        labeled_src = os.path.join(LABELED, c)

        n1 = copy_all(base_src, out_c, "base")
        n2 = copy_all(labeled_src, out_c, "new")

        totals[c] = (n1, n2, n1 + n2)

    print("Combined train built:")
    for c, (base_n, new_n, total) in totals.items():
        print(f"  {c}: base={base_n}, new={new_n}, total={total}")

if __name__ == "__main__":
    main()
