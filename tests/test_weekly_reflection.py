import os
import tempfile
import unittest

from memory.db import MemoryDB
from skills.reflection import handler


class TestWeeklyReflection(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        db_path = os.path.join(self.tmpdir.name, "test.db")
        self.db = MemoryDB(db_path)
        schema_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "memory", "schema.sql")
        )
        self.db.init_schema(schema_path)

    def tearDown(self) -> None:
        self.db.close()
        self.tmpdir.cleanup()

    def test_reflection_rollup(self) -> None:
        day1 = "2026-02-01"
        day2 = "2026-02-07"

        # Storage snapshots
        for mount, first_used, last_used in [("/", 1000, 1500), ("/data", 2000, 2500), ("/data2", 3000, 3300)]:
            self.db.insert_disk_snapshot(
                taken_at=f"{day1}T09:00:00-06:00",
                snapshot_local_date=day1,
                hostname="host-a",
                mountpoint=mount,
                filesystem="/dev/root",
                total_bytes=10000,
                used_bytes=first_used,
                free_bytes=9000,
            )
            self.db.insert_disk_snapshot(
                taken_at=f"{day2}T09:00:00-06:00",
                snapshot_local_date=day2,
                hostname="host-a",
                mountpoint=mount,
                filesystem="/dev/root",
                total_bytes=10000,
                used_bytes=last_used,
                free_bytes=9000,
            )

        self.db.insert_dir_size_samples(
            f"{day1}T09:00:00-06:00",
            "root_top",
            [("/var", 100)],
        )
        self.db.insert_dir_size_samples(
            f"{day2}T09:00:00-06:00",
            "root_top",
            [("/var", 200)],
        )
        self.db.insert_dir_size_samples(
            f"{day1}T09:00:00-06:00",
            "home_top",
            [("/home/user/Downloads", 50)],
        )
        self.db.insert_dir_size_samples(
            f"{day2}T09:00:00-06:00",
            "home_top",
            [("/home/user/Downloads", 80)],
        )
        self.db.insert_storage_scan_stats(f"{day1}T09:00:00-06:00", "root_top", 10, 2)
        self.db.insert_storage_scan_stats(f"{day2}T09:00:00-06:00", "home_top", 10, 3)

        # Resource snapshots
        self.db.insert_resource_snapshot(
            taken_at=f"{day1}T09:00:00-06:00",
            snapshot_local_date=day1,
            hostname="host-a",
            load_1m=1.0,
            load_5m=1.2,
            load_15m=1.4,
            mem_total=8000,
            mem_used=1000,
            mem_free=7000,
            swap_total=1000,
            swap_used=100,
        )
        self.db.insert_resource_snapshot(
            taken_at=f"{day2}T09:00:00-06:00",
            snapshot_local_date=day2,
            hostname="host-a",
            load_1m=3.0,
            load_5m=2.2,
            load_15m=2.4,
            mem_total=8000,
            mem_used=3000,
            mem_free=5000,
            swap_total=1000,
            swap_used=200,
        )
        self.db.replace_resource_process_samples(
            f"{day2}T09:00:00-06:00",
            "rss",
            [(123, "procA", 10, 4096)],
        )

        # Network snapshots
        self.db.insert_network_snapshot(
            taken_at=f"{day1}T09:00:00-06:00",
            snapshot_local_date=day1,
            hostname="host-a",
            default_iface="eth0",
            default_gateway="10.0.0.1",
        )
        self.db.insert_network_snapshot(
            taken_at=f"{day2}T09:00:00-06:00",
            snapshot_local_date=day2,
            hostname="host-a",
            default_iface="eth0",
            default_gateway="10.0.0.2",
        )
        self.db.replace_network_interfaces(
            f"{day1}T09:00:00-06:00",
            [("eth0", "up", 10, 20, 0, 0)],
        )
        self.db.replace_network_interfaces(
            f"{day2}T09:00:00-06:00",
            [("eth0", "up", 60, 90, 0, 0)],
        )
        self.db.replace_network_nameservers(f"{day1}T09:00:00-06:00", ["1.1.1.1"])
        self.db.replace_network_nameservers(f"{day2}T09:00:00-06:00", ["8.8.8.8"])

        result = handler.weekly_reflection({"db": self.db, "timezone": "America/Regina"})
        text = result.get("text", "")

        self.assertIn("Weekly reflection (2026-02-01 to 2026-02-07", text)
        self.assertIn("/ used change", text)
        self.assertIn("/ largest dir growth: /var (+100B)", text)
        self.assertIn("home largest dir growth: /home/user/Downloads (+30B)", text)
        self.assertIn("storage scan errors_skipped (7d): 5", text)
        self.assertIn("load_1m min/avg/max: 1.00/2.00/3.00", text)
        self.assertIn("mem_used min/avg/max:", text)
        self.assertIn("swap_used min/avg/max", text)
        self.assertIn("top processes by RSS", text)
        self.assertIn("procA", text)
        self.assertIn("default gateway changes", text)
        self.assertIn("2026-02-07: 10.0.0.1 -> 10.0.0.2", text)
        self.assertIn("eth0 rx=+50B tx=+70B", text)
        self.assertIn("nameserver changes: 2026-02-07", text)

    def test_missing_data_domains(self) -> None:
        day1 = "2026-02-01"
        day2 = "2026-02-07"
        self.db.insert_disk_snapshot(
            taken_at=f"{day1}T09:00:00-06:00",
            snapshot_local_date=day1,
            hostname="host-a",
            mountpoint="/",
            filesystem="/dev/root",
            total_bytes=10000,
            used_bytes=1000,
            free_bytes=9000,
        )
        self.db.insert_disk_snapshot(
            taken_at=f"{day2}T09:00:00-06:00",
            snapshot_local_date=day2,
            hostname="host-a",
            mountpoint="/",
            filesystem="/dev/root",
            total_bytes=10000,
            used_bytes=1200,
            free_bytes=8800,
        )

        result = handler.weekly_reflection({"db": self.db, "timezone": "America/Regina"})
        text = result.get("text", "")
        self.assertIn("Resources:\n- insufficient data", text)
        self.assertIn("Network:\n- insufficient data", text)


if __name__ == "__main__":
    unittest.main()
