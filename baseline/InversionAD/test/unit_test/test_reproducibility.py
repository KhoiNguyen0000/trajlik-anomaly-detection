import unittest

from src.reproducibility import validate_main_protocol


def valid_config():
    return {
        "data": {
            "img_size": 256,
            "transform_type": "imagenet",
            "dataset_name": "visa_all",
            "data_root": "data/visa",
        },
        "backbone": {
            "model_type": "efficientnet-b4",
            "outblocks": [1, 5, 9, 21],
            "stride": 16,
        },
        "diffusion": {"model_type": "dit"},
        "evaluation": {"eval_step": 3},
    }


class ReproducibilityTest(unittest.TestCase):
    def test_locked_main_protocol_accepts_matching_config(self):
        self.assertEqual(validate_main_protocol(valid_config()), [])

    def test_audit_reports_every_protocol_mismatch(self):
        config = valid_config()
        config["data"]["transform_type"] = "default"
        config["data"]["data_root"] = "data/mpdd"
        config["evaluation"]["eval_step"] = 4

        errors = validate_main_protocol(config)

        self.assertTrue(any("transform_type" in error for error in errors))
        self.assertTrue(any("data_root" in error for error in errors))
        self.assertTrue(any("eval_step" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
