import unittest

import numpy as np

from gaussian_bandit import simulate_gaussian_bandit, update_posterior


class GaussianBanditTests(unittest.TestCase):
    def test_hand_calculation(self):
        mean, variance = update_posterior(0, 25, 2, 1)
        self.assertAlmostEqual(mean, 25 / 13)
        self.assertAlmostEqual(variance, 25 / 26)
        mean, variance = update_posterior(0, 25, 2, 3)
        self.assertAlmostEqual(mean, 25 / 17)
        self.assertAlmostEqual(variance, 225 / 34)

    def test_required_experiments(self):
        for sigma, rounds in [(1, 100), (1, 1000), (3, 1000)]:
            with self.subTest(sigma=sigma, rounds=rounds):
                result = simulate_gaussian_bandit([1, 2, 3], sigma, rounds, 42)
                repeated = simulate_gaussian_bandit([1, 2, 3], sigma, rounds, 42)
                for key in result:
                    np.testing.assert_array_equal(result[key], repeated[key])
                self.assertEqual(result["counts"].sum(), rounds)
                np.testing.assert_array_equal(
                    result["counts"], np.bincount(result["selected_arms"], minlength=3)
                )
                for arm in range(3):
                    rewards = result["rewards"][result["selected_arms"] == arm]
                    variance = 1 / (1 / 25 + len(rewards) / sigma**2)
                    self.assertAlmostEqual(result["posterior_variances"][arm], variance)
                    self.assertAlmostEqual(
                        result["posterior_means"][arm], variance * rewards.sum() / sigma**2
                    )
                self.assertEqual(result["total_reward"], result["rewards"].sum())
                self.assertEqual(result["recommendation"], np.argmax(result["posterior_means"]) + 1)

    def test_one_round_uses_posterior_and_updates_only_selected_arm(self):
        rng = np.random.default_rng(42)
        arm = int(np.argmax(rng.normal(np.zeros(3), np.full(3, 5.0))))
        reward = rng.normal([-100, -200, -300][arm], 1)
        result = simulate_gaussian_bandit([-100, -200, -300], 1, 1, 42)
        self.assertEqual(result["selected_arms"][0], arm)
        self.assertEqual(result["rewards"][0], reward)
        self.assertLess(reward, 0)
        for other in range(3):
            if other != arm:
                self.assertEqual(result["posterior_means"][other], 0)
                self.assertEqual(result["posterior_variances"][other], 25)
        self.assertAlmostEqual(result["posterior_means"][arm], reward * 25 / 26)

    def test_invalid_inputs(self):
        for sigma in [0, -1, np.nan, np.inf]:
            with self.subTest(sigma=sigma), self.assertRaises(ValueError):
                simulate_gaussian_bandit([1, 2, 3], sigma, 10, 42)
        for rounds in [0, -1, 1.5, True]:
            with self.subTest(rounds=rounds), self.assertRaises(ValueError):
                simulate_gaussian_bandit([1, 2, 3], 1, rounds, 42)
        for means in [[1, 2], [1, 2, np.inf]]:
            with self.assertRaises(ValueError):
                simulate_gaussian_bandit(means, 1, 10, 42)
        for seed in [-1, 1.5, True]:
            with self.assertRaises(ValueError):
                simulate_gaussian_bandit([1, 2, 3], 1, 10, seed)


if __name__ == "__main__":
    unittest.main()
