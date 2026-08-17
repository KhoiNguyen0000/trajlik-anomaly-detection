import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image
from torchvision import transforms

import _bootstrap  # noqa: F401

from src.datasets.mpdd import MPDD, MPDD_CLASSES
from src.datasets.mvtec_ad import AD_CLASSES, MVTecAD
from src.datasets.visa import VISA_CLASSES, VisA


class TrainingNormalOnlyDatasetsTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.transform = transforms.ToTensor()

    def tearDown(self):
        self.temp_dir.cleanup()

    @staticmethod
    def write_image(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), color="white").save(path)

    def assert_training_normal_sample(self, dataset, category: str) -> None:
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset.normal_indices, [0])
        self.assertEqual(dataset.anom_indices, [])

        sample = dataset[0]
        self.assertEqual(sample["clsnames"], category)
        self.assertEqual(int(sample["clslabels"]), 0)
        self.assertEqual(tuple(sample["samples"].shape), (3, 8, 8))

    def test_mvtec_training_normal_only(self):
        category = AD_CLASSES[0]
        dataset_root = self.root / "mvtec"
        self.write_image(dataset_root / category / "train" / "good" / "000.png")

        dataset = MVTecAD(
            data_root=str(dataset_root),
            category=category,
            input_res=8,
            split="train",
            transform=self.transform,
            normal_only=True,
            cls_label=True,
        )

        self.assert_training_normal_sample(dataset, category)

    def test_mpdd_training_normal_only(self):
        category = MPDD_CLASSES[0]
        dataset_root = self.root / "mpdd"
        self.write_image(dataset_root / category / "train" / "good" / "000.png")

        dataset = MPDD(
            data_root=str(dataset_root),
            category=category,
            input_res=8,
            split="train",
            transform=self.transform,
            normal_only=True,
            cls_label=True,
        )

        self.assert_training_normal_sample(dataset, category)

    def test_visa_training_normal_only(self):
        category = VISA_CLASSES[0]
        dataset_root = self.root / "visa"
        relative_image = Path(category) / "Data" / "Images" / "Normal" / "000.JPG"
        self.write_image(dataset_root / relative_image)

        split_dir = dataset_root / "split_csv"
        split_dir.mkdir(parents=True)
        with (split_dir / "1cls.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=["object", "split", "label", "image"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "object": category,
                    "split": "train",
                    "label": "normal",
                    "image": relative_image.as_posix(),
                }
            )

        dataset = VisA(
            data_root=str(dataset_root),
            category=category,
            input_res=8,
            split="train",
            transform=self.transform,
            normal_only=True,
            cls_label=True,
        )

        self.assert_training_normal_sample(dataset, category)

    def test_mutually_exclusive_filters(self):
        category = AD_CLASSES[0]
        dataset_root = self.root / "mvtec"
        self.write_image(dataset_root / category / "train" / "good" / "000.png")

        with self.assertRaisesRegex(ValueError, "cannot both be True"):
            MVTecAD(
                data_root=str(dataset_root),
                category=category,
                input_res=8,
                split="train",
                transform=self.transform,
                normal_only=True,
                anom_only=True,
            )

    def test_test_split_anomaly_type_is_cross_platform(self):
        root = self.root / "mvtec"
        category = "bottle"
        self.write_image(root / category / "test" / "good" / "000.png")
        self.write_image(root / category / "test" / "broken" / "001.png")
        self.write_image(
            root / category / "ground_truth" / "broken" / "001_mask.png"
        )
        dataset = MVTecAD(
            data_root=str(root),
            category=category,
            input_res=8,
            split="test",
            transform=self.transform,
            is_mask=True,
            cls_label=True,
        )

        self.assertEqual(dataset[0]["anom_type"], "good")
        self.assertEqual(dataset[1]["anom_type"], "broken")


if __name__ == "__main__":
    unittest.main()
