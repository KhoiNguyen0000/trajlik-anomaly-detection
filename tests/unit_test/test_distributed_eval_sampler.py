import unittest

import _bootstrap  # noqa: F401

from src.train_distributed import DistributedEvalSampler


class DistributedEvalSamplerTest(unittest.TestCase):
    def test_shards_have_no_duplicates_or_missing_samples(self):
        dataset = list(range(10))
        shards = [
            list(DistributedEvalSampler(dataset, num_replicas=3, rank=rank))
            for rank in range(3)
        ]
        flattened = [index for shard in shards for index in shard]

        self.assertEqual(sorted(flattened), list(range(10)))
        self.assertEqual(len(flattened), len(set(flattened)))


if __name__ == "__main__":
    unittest.main()
