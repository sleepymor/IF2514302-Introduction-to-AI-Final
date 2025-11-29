import sys

sys.path.append("src")

import unittest
from environment.environment import TacticalEnvironment


class TestClone(unittest.TestCase):

    def test_clone_creates_different_object(self):
        """
        Test clone creates a different object
        """
        original = TacticalEnvironment(width=15, height=10, seed=32)
        clone = original.clone()

        self.assertIsNot(clone, original, "Clone should be a different object")

    def test_clone_traps_independent(self):
        """
        Test modifying clone is traps doesn't affect original
        """
        original = TacticalEnvironment(width=15, height=10, seed=32)
        clone = original.clone()

        clone.traps.add((5, 5))

        self.assertIn((5, 5), clone.traps, "Trap should be in clone")
        self.assertNotIn((5, 5), original.traps, "Trap should NOT be in original")
        self.assertNotEqual(clone.traps, original.traps, "Traps should be different")

    def test_clone_dimensions_copied(self):
        """
        Test dimensions are copied correctly
        """
        original = TacticalEnvironment(width=15, height=10, seed=32)
        clone = original.clone()

        self.assertEqual(clone.width, original.width)
        self.assertEqual(clone.height, original.height)

    def test_clone_player_pos_independent(self):
        """
        Test agent position is independent
        """
        original = TacticalEnvironment(width=15, height=10, seed=32)
        clone = original.clone()

        original_pos = original.player_pos
        clone.player_pos = (1, 1)

        self.assertNotEqual(clone.player_pos, original.player_pos)
        self.assertEqual(original.player_pos, original_pos)


if __name__ == "__main__":
    unittest.main()
