from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from record_product_uat import record_owner_uat  # noqa: E402


class RecordProductUATTest(unittest.TestCase):
    def _write_results(self, root: Path, *, automated: str = "PASS") -> Path:
        path = root / "results.json"
        path.write_text(
            json.dumps(
                {
                    "case_results": [
                        {
                            "eval_case_id": "CASE-A",
                            "automated_result": automated,
                            "owner_uat_result": "WAITING_FOR_UAT",
                            "evidence": [{"kind": "test", "reference": "existing"}],
                        }
                    ]
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return path

    def test_record_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_results(Path(tmp))
            record_owner_uat(path, case_id="CASE-A", result="PASS", notes="通过")
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["case_results"][0]["owner_uat_result"], "PASS")

    def test_record_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_results(Path(tmp))
            record_owner_uat(path, case_id="CASE-A", result="FAIL", notes="失败")
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["case_results"][0]["owner_uat_result"], "FAIL")

    def test_unknown_case_fails_closed_without_rewriting_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_results(Path(tmp))
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "unknown eval_case_id"):
                record_owner_uat(path, case_id="MISSING", result="PASS")
            self.assertEqual(path.read_bytes(), before)

    def test_invalid_result_is_rejected_without_rewriting_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_results(Path(tmp))
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "invalid Owner UAT result"):
                record_owner_uat(path, case_id="CASE-A", result="MAYBE")
            self.assertEqual(path.read_bytes(), before)

    def test_automated_result_is_preserved_and_owner_evidence_appended(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_results(Path(tmp), automated="PARTIAL")
            record_owner_uat(
                path,
                case_id="CASE-A",
                result="PASS",
                reference="Owner UAT screenshot",
                notes="播放第二首 → 好的 → exact Preview",
            )
            record = json.loads(path.read_text(encoding="utf-8"))["case_results"][0]
            self.assertEqual(record["automated_result"], "PARTIAL")
            self.assertEqual(record["evidence"][-1]["kind"], "owner_uat")
            self.assertEqual(record["evidence"][-1]["reference"], "Owner UAT screenshot")
            self.assertEqual(record["evidence"][-1]["notes"], "播放第二首 → 好的 → exact Preview")


if __name__ == "__main__":
    unittest.main()
