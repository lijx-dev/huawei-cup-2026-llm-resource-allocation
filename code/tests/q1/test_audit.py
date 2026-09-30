"""人工构造微型样例，仅验证 P0 规则，不参与正式结果。"""
import csv
import json
import lzma
from pathlib import Path
import sqlite3
import tempfile
import unittest

from q1.audit.duplicates import classify, init_db
from q1.audit.inventory import sha256_file
from q1.audit.quality import scan_quality
from q1.audit.tables import audit_mapping, audit_pair, read_table


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = {"rule_version": "test", "quality_scalar_fields": ["score"],
                       "quality_list_fields": ["signal"], "fraction_fields": [], "count_fields": [],
                       "provisional_percentage_upper_bound": 100,
                       "mixture_fields": ["train_the_pile_a", "train_the_pile_b"],
                       "loss_fields": ["metric/the_pile_a_val_loss"],
                       "mixture_sum_tolerance": 0.008500001}
        self.db = init_db(self.root / "index.sqlite")
        self.addCleanup(self.db.close)

    def quality_file(self, lines):
        path = self.root / "input.jsonl.xz"
        with lzma.open(path, "wb") as stream:
            stream.write(b"".join(lines))
        return path

    def test_jsonl_good_bad_and_missing(self):
        path = self.quality_file([b'{"id":"a","score":1,"signal":[0.1]}\n',
                                  b'{bad json}\n', b'{"id":"b","score":2}\n'])
        issues = []
        result = scan_quality(path, "A1", self.root, self.config, self.db, issues)
        self.assertTrue(result["complete"])
        self.assertEqual(result["lines_read"], 3)
        self.assertEqual(result["counts"], {"valid": 1, "rejected_corrupt": 2})
        self.assertEqual(result["stats"]["signal"]["missing"], 1)
        self.assertEqual(len(issues), 2)

    def test_truncated_xz_quarantines_read_rows(self):
        path = self.quality_file([b'{"id":"a","score":1,"signal":[0.1]}\n'])
        path.write_bytes(path.read_bytes()[:-10])
        issues = []
        result = scan_quality(path, "A1", self.root, self.config, self.db, issues)
        self.assertFalse(result["complete"])
        self.assertTrue(any(x["reason_code"] == "compressed_stream_incomplete" for x in issues))
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM records WHERE status='valid'").fetchone()[0], 0)

    def test_duplicate_exact_and_conflict(self):
        statuses = [classify(self.db, "a", "A1", 1, "x", "raw1", "same", "test"),
                    classify(self.db, "b", "A2", 1, "x", "raw2", "same", "test")]
        self.assertEqual(statuses, ["valid", "duplicate_exact"])
        classify(self.db, "c", "A3", 1, "x", "raw3", "different", "test")
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM records WHERE status='duplicate_conflict'").fetchone()[0], 3)

    def test_structurally_absent_auxiliary_field(self):
        path = self.quality_file([b'{"id":"a","score":1,"signal":[0.1]}\n'])
        result = scan_quality(path, "A2", self.root, self.config, self.db, [])
        self.assertEqual(result["counts"]["valid"], 1)
        self.assertNotIn("content", result["keys"])

    def test_provisional_fraction_is_suspected_not_corrupt(self):
        self.config["quality_scalar_fields"] = ["frac"]
        self.config["fraction_fields"] = ["frac"]
        path = self.quality_file([b'{"id":"a","frac":120,"signal":[0.1]}\n'])
        result = scan_quality(path, "A1", self.root, self.config, self.db, [])
        self.assertEqual(result["counts"].get("suspected_unreliable"), 1)

    def test_nonfinite_retains_id(self):
        path = self.quality_file([b'{"id":"a","score":NaN,"signal":[0.1]}\n'])
        issues = []
        result = scan_quality(path, "A1", self.root, self.config, self.db, issues)
        self.assertEqual(result["counts"].get("rejected_corrupt"), 1)
        self.assertEqual(issues[0]["record_id"], "a")

    def test_compressed_and_plain_copy_same_source(self):
        raw = b'{"id":"a","score":1,"signal":[0.1]}\n'
        path = self.quality_file([raw])
        copy = self.root / "plain.jsonl"
        copy.write_bytes(raw)
        scan = scan_quality(path, "A1", self.root, self.config, self.db, [])
        self.assertEqual(scan["decoded_sha256"], sha256_file(copy))

    def write_table(self, name, header, rows):
        path = self.root / name
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(rows)
        return path

    def test_mixture_rounding_and_invalid_sum(self):
        path = self.write_table("mix.csv", ["index", *self.config["mixture_fields"]],
                                [["1", "0.503", "0.503"], ["2", "0.8", "0.1"], ["3", "-0.1", "1.1"]])
        issues = []
        result = read_table(path, "mixture", self.config, self.root, issues)
        self.assertEqual((result["valid"], result["invalid"]), (1, 2))

    def test_pair_index_mismatch(self):
        mix = self.write_table("mix.csv", ["index", *self.config["mixture_fields"]], [["1", "0.5", "0.5"]])
        loss = self.write_table("loss.csv", ["index", *self.config["loss_fields"]], [["2", "1.2"]])
        left = read_table(mix, "mixture", self.config, self.root, [])
        right = read_table(loss, "loss", self.config, self.root, [])
        self.assertEqual((audit_pair(left, right)["mixture_only"], audit_pair(left, right)["loss_only"]), (1, 1))

    def test_duplicate_csv_index_is_rejected(self):
        path = self.write_table("mix.csv", ["index", *self.config["mixture_fields"]],
                                [["1", "0.5", "0.5"], ["1", "0.5", "0.5"]])
        result = read_table(path, "mixture", self.config, self.root, [])
        self.assertEqual((result["valid"], result["invalid"]), (1, 1))

    def test_mapping_missing_and_duplicate(self):
        path = self.write_table("map.csv", ["mixture_domain", "quality_domain", "mapping_type"],
                                [["a", "arxiv", "direct"], ["a", "book", "inferred"]])
        rows, summary = audit_mapping(path, self.config["mixture_fields"], self.root, [], "test")
        self.assertEqual(summary["missing_domains"], ["b"])
        self.assertEqual(rows[1]["reason_code"], "duplicate_mapping")


if __name__ == "__main__":
    unittest.main()
