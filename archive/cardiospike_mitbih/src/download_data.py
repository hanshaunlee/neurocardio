"""Download MIT-BIH Arrhythmia Database from PhysioNet."""
import os
import wfdb

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "mitdb")
os.makedirs(DATA_DIR, exist_ok=True)

RECORDS = [
    "100", "101", "103", "105", "106", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119",
    "121", "122", "123", "124",
    "200", "201", "202", "203", "205", "207", "208", "209", "210",
    "212", "213", "214", "215", "217", "219",
    "220", "221", "222", "223", "228",
    "230", "231", "232", "233", "234",
]


def main():
    print(f"Downloading {len(RECORDS)} MIT-BIH records to {DATA_DIR}")
    wfdb.dl_database("mitdb", dl_dir=DATA_DIR, records=RECORDS)
    print("Done.")
    print("Files:", len(os.listdir(DATA_DIR)))


if __name__ == "__main__":
    main()
