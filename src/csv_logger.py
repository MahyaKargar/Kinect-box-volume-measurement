import csv
import os
from datetime import datetime


class CSVLogger:

    def __init__(self, filename="light_test_results.csv"):
        self.filename = filename

    def log_light_result(
        self,
        brightness,
        invalid_pct,
        noise_min_height_mm,
        mask_pixels,
        width_cm,
        length_cm,
        height_cm,
        volume_cm3,
    ):
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "brightness_0_255": (
                round(brightness, 1) if brightness is not None else ""
            ),
            "invalid_pct_current": round(invalid_pct, 2),
            "noise_min_height_mm": round(noise_min_height_mm, 2),
            "mask_pixels": mask_pixels,
            "width_cm": round(width_cm, 2),
            "length_cm": round(length_cm, 2),
            "height_cm": round(height_cm, 2),
            "volume_cm3": round(volume_cm3, 2),
        }

        file_exists = os.path.exists(self.filename)

        with open(
            self.filename, "a", newline="", encoding="utf-8-sig"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

        brightness_str = (
            f"{brightness:.1f}" if brightness is not None else "N/A"
        )
        print(
            f"[Light Log] brightness={brightness_str}  invalid%={invalid_pct:.1f}  "
            f"-> appended to {self.filename}"
        )