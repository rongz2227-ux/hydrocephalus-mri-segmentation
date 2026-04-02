import os
import glob

project_root = os.path.expanduser("~/projects/Hydro_Seg_Project")
data_dir = os.path.join(project_root, "data")
split_dir = os.path.join(project_root, "splits")

train_file = os.path.join(split_dir, "train_subjects.txt")
val_file = os.path.join(split_dir, "val_subjects.txt")
test_file = os.path.join(split_dir, "test_subjects.txt")


def read_subjects(txt_path):
    with open(txt_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


train_subjects = read_subjects(train_file)
val_subjects = read_subjects(val_file)
test_subjects = read_subjects(test_file)

train_set = set(train_subjects)
val_set = set(val_subjects)
test_set = set(test_subjects)

print("=" * 60)
print("1. SPLIT SIZE")
print(f"Train: {len(train_subjects)}")
print(f"Val:   {len(val_subjects)}")
print(f"Test:  {len(test_subjects)}")

print("=" * 60)
print("2. DUPLICATES INSIDE EACH SPLIT")


def find_duplicates(items):
    seen = set()
    dup = set()
    for x in items:
        if x in seen:
            dup.add(x)
        seen.add(x)
    return sorted(list(dup))


train_dup = find_duplicates(train_subjects)
val_dup = find_duplicates(val_subjects)
test_dup = find_duplicates(test_subjects)

print("Train duplicates:", train_dup if train_dup else "None")
print("Val duplicates:  ", val_dup if val_dup else "None")
print("Test duplicates: ", test_dup if test_dup else "None")

print("=" * 60)
print("3. OVERLAP BETWEEN SPLITS")

train_val_overlap = sorted(list(train_set & val_set))
train_test_overlap = sorted(list(train_set & test_set))
val_test_overlap = sorted(list(val_set & test_set))

print("Train ∩ Val :", train_val_overlap if train_val_overlap else "None")
print("Train ∩ Test:", train_test_overlap if train_test_overlap else "None")
print("Val ∩ Test  :", val_test_overlap if val_test_overlap else "None")

print("=" * 60)
print("4. ALL SUBJECTS UNION CHECK")

all_split_subjects = train_set | val_set | test_set
print(f"Unique subjects in all splits: {len(all_split_subjects)}")

all_data_subjects = sorted([
    os.path.basename(x)
    for x in glob.glob(os.path.join(data_dir, "sub_*"))
    if os.path.isdir(x)
])

all_data_set = set(all_data_subjects)

missing_in_splits = sorted(list(all_data_set - all_split_subjects))
missing_in_data = sorted(list(all_split_subjects - all_data_set))

print("Subjects in data but NOT in any split:")
print(missing_in_splits if missing_in_splits else "None")

print("Subjects in split files but NOT found in data:")
print(missing_in_data if missing_in_data else "None")

print("=" * 60)
print("5. SCAN COUNT PER SUBJECT")

multi_scan_subjects = []
missing_scan_subjects = []
bad_image_subjects = []
bad_label_subjects = []

for sid in sorted(all_split_subjects):
    sub_dir = os.path.join(data_dir, sid)
    scan_dirs = sorted(glob.glob(os.path.join(sub_dir, "scan_*")))

    if len(scan_dirs) == 0:
        missing_scan_subjects.append(sid)
        continue

    if len(scan_dirs) > 1:
        multi_scan_subjects.append((sid, len(scan_dirs), [os.path.basename(s) for s in scan_dirs]))

    for scan_dir in scan_dirs:
        image_path = os.path.join(scan_dir, "eT1W_FFE_SVR.nii.gz")
        label_path = os.path.join(scan_dir, "segmentation", "gt.nii.gz")

        if not os.path.exists(image_path):
            bad_image_subjects.append((sid, scan_dir))
        if not os.path.exists(label_path):
            bad_label_subjects.append((sid, scan_dir))

print("Subjects with NO scan_* folder:")
print(missing_scan_subjects if missing_scan_subjects else "None")

print("\nSubjects with MULTIPLE scan_* folders:")
if multi_scan_subjects:
    for item in multi_scan_subjects:
        sid, nscan, scan_list = item
        print(f"{sid}: {nscan} scans -> {scan_list}")
else:
    print("None")

print("\nSubjects with missing image file:")
if bad_image_subjects:
    for sid, scan_dir in bad_image_subjects:
        print(f"{sid}: {scan_dir}")
else:
    print("None")

print("\nSubjects with missing label file:")
if bad_label_subjects:
    for sid, scan_dir in bad_label_subjects:
        print(f"{sid}: {scan_dir}")
else:
    print("None")

print("=" * 60)
print("6. SIMPLE LEAKAGE RISK SUMMARY")

leakage_flag = False

if train_val_overlap or train_test_overlap or val_test_overlap:
    leakage_flag = True
    print("[WARNING] Overlap exists between splits.")
else:
    print("[OK] No overlap between train/val/test.")

if multi_scan_subjects:
    print("[CAUTION] Some subjects have multiple scans. You should verify they are not causing leakage.")
else:
    print("[OK] No subject with multiple scans found in split subjects.")

if missing_in_data:
    print("[WARNING] Some split subjects do not exist in data.")
if missing_in_splits:
    print("[INFO] Some data subjects are unused by the split files.")

if not leakage_flag and not multi_scan_subjects:
    print("[SUMMARY] No obvious data leakage found at the subject/split level.")
else:
    print("[SUMMARY] Further manual inspection is recommended.")