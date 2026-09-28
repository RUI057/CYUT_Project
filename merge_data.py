#!/usr/bin/env python3
"""把組員傳來的手語資料合併進 dynamic_dataset（組長用）。

來源可以是收集包產生的 zip，也可以是組員直接給的資料夾。

用法：
  python merge_data.py 手語資料_小明_20260920_1530.zip [更多.zip ...]
  python merge_data.py ~/Downloads/dynamic_dataset              # 直接給資料夾
  python merge_data.py 組員的資料/ 手語資料_小美.zip              # 混著給也可以
  python merge_data.py <來源> --dry-run                          # 只看會合併什麼，不寫入

不要用 Finder／檔案總管直接把資料夾拖過去覆蓋：選「取代」會把整個詞資料夾換掉，
原本的資料就沒了。這支程式的做法：
  - 只取 dynamic_dataset/<詞>/<檔名>.npy，其他檔案一律忽略（也擋掉 ../ 這類路徑）
  - 已存在的同名檔案一律跳過、不覆蓋 → 同一個 zip 合併兩次也沒關係
  - 詞不在詞彙表、檔案讀不了或形狀不對 → 跳過並列出
  - 別人自己手動壓縮、檔名沒有標 UTF-8 的 zip，會嘗試修回正確的中文
"""
import argparse
import io
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.vocab import get_all_labels
from src.features import VALID_DIMS

SEQ_LEN = 30
ENTRY = re.compile(r"(?:^|/)dynamic_dataset/([^/]+)/([^/]+\.npy)$")


def entry_name(zi):
    """zip 內的檔名。沒標 UTF-8 的（Windows 手動壓縮／舊版 Mac）Python 會用 cp437 解，
    中文就變亂碼 → 還原成原始位元組，依序試 UTF-8、Big5(cp950)。"""
    name = zi.filename
    if not zi.flag_bits & 0x800:
        raw = name.encode("cp437", errors="replace")
        for enc in ("utf-8", "cp950"):
            try:
                name = raw.decode(enc)
                break
            except UnicodeDecodeError:
                continue
    return name.replace("\\", "/")


def kit_version(zpath):
    """讀 zip 裡的收集包版本字串（沒有就回傳空字串）。"""
    with zipfile.ZipFile(zpath) as z:
        for zi in z.infolist():
            if not zi.is_dir() and entry_name(zi).endswith("KIT_VERSION.txt"):
                return z.read(zi).decode("utf-8", errors="replace").strip()
    return ""


def iter_zip(zpath):
    """走訪 zip 裡的資料檔 → (詞, 檔名, 內容位元組)。"""
    with zipfile.ZipFile(zpath) as z:
        for zi in z.infolist():
            if zi.is_dir():
                continue
            m = ENTRY.search(entry_name(zi))
            if m:
                word, fname = m.groups()
                yield word, fname, z.read(zi)


def iter_folder(fpath):
    """走訪資料夾裡的 .npy → (詞, 檔名, 內容位元組)。

    給 dynamic_dataset 本身或它的上層資料夾都可以；
    詞取自 .npy 的上層資料夾名，不在詞彙表的會在 merge() 被擋下。
    """
    for f in sorted(Path(fpath).rglob("*.npy")):
        if f.is_file():
            yield f.parent.name, f.name, f.read_bytes()


def merge(src, into, labels, dry_run=False):
    """合併一個來源（zip 或資料夾）。

    回傳 (新增 Counter, 已存在略過數, 不在詞彙表 Counter, 壞檔 list, 收集包版本)。
    """
    added, unknown, bad, exist = Counter(), Counter(), [], 0
    src, kit_info = Path(src), ""

    if src.is_dir():
        into_r, src_r = into.resolve(), src.resolve()
        if src_r == into_r:
            raise ValueError("來源資料夾就是目的地本身，不用合併")
        if into_r in src_r.parents or src_r in into_r.parents:
            raise ValueError(f"來源與目的地互相包含（{src_r} vs {into_r}），"
                             "請把組員的資料放到專案外面再合併")
        items = iter_folder(src)
    else:
        kit_info = kit_version(src)
        items = iter_zip(src)

    for word, fname, data in items:
        if word not in labels:
            unknown[word] += 1
            continue
        dest = into / word / fname
        if dest.exists():          # 已存在一律跳過，絕不覆蓋既有資料
            exist += 1
            continue
        try:
            arr = np.load(io.BytesIO(data), allow_pickle=False)
        except Exception:
            bad.append(f"{word}/{fname}（讀不了）")
            continue
        if arr.ndim != 2 or arr.shape[0] != SEQ_LEN or arr.shape[1] not in VALID_DIMS:
            bad.append(f"{word}/{fname}（形狀 {arr.shape}）")
            continue
        if not dry_run:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
        added[word] += 1
    return added, exist, unknown, bad, kit_info


def main():
    ap = argparse.ArgumentParser(description="合併組員傳來的手語資料（zip 或資料夾）")
    ap.add_argument("sources", nargs="+", help="組員傳來的 zip 檔或資料夾")
    ap.add_argument("--into", default=str(ROOT / "dynamic_dataset"), help="合併到哪個資料夾")
    ap.add_argument("--dry-run", action="store_true", help="只列出會合併什麼，不寫入")
    args = ap.parse_args()

    labels, into = set(get_all_labels()), Path(args.into)
    total = 0
    for zp in args.sources:
        src = Path(zp)
        print(f"\n══ {src.name}{'/（資料夾）' if src.is_dir() else ''} ══")
        if not src.exists():
            print(f"  ❌ 找不到：{zp}")
            continue
        try:
            added, exist, unknown, bad, kit_info = merge(src, into, labels, args.dry_run)
        except (zipfile.BadZipFile, FileNotFoundError, ValueError) as e:
            print(f"  ❌ 無法處理：{e}")
            continue
        if kit_info:
            print("  收集包版本：" + kit_info.splitlines()[0])
        verb = "會新增" if args.dry_run else "新增"
        if added:
            print(f"  {verb} {sum(added.values())} 筆：")
            for w, n in sorted(added.items(), key=lambda kv: -kv[1]):
                print(f"    {w:<8}{n:>5}")
        else:
            print(f"  沒有{verb}任何資料")
        if exist:
            print(f"  已存在、略過 {exist} 筆（之前合併過）")
        if unknown:
            print(f"  ⚠️ 詞不在詞彙表、略過 {sum(unknown.values())} 筆："
                  + "、".join(f"{w}({n})" for w, n in unknown.most_common(8)))
            print("     詞彙表若有改過，請組員更新收集包；若是亂碼，請組員改用選單的「打包資料」重新打包。")
        if bad:
            print(f"  ⚠️ 壞檔、略過 {len(bad)} 筆：" + "、".join(bad[:5]))
        total += sum(added.values())

    if args.dry_run:
        print("\n（試跑模式，沒有寫入任何檔案）")
    else:
        print(f"\n✅ 合併完成，共新增 {total} 筆。建議接著跑 python check_dataset.py 檢查品質。")


if __name__ == "__main__":
    main()
