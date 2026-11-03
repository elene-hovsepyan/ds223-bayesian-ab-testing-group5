import numpy as np


def update_posterior(mean, variance, reward, sigma):
    reward_precision = 1.0 / sigma**2
    new_variance = 1.0 / (1.0 / variance + reward_precision)
    new_mean = new_variance * (mean / variance + reward * reward_precision)
    return new_mean, new_variance


def simulate_gaussian_bandit(true_means, sigma, rounds, seed):
    true_means = np.asarray(true_means, dtype=float)
    if true_means.shape != (3,) or not np.all(np.isfinite(true_means)):
        raise ValueError("Enter three finite true means.")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Sigma must be positive and finite.")
    if isinstance(rounds, bool) or not isinstance(rounds, (int, np.integer)) or rounds <= 0:
        raise ValueError("Rounds must be a positive integer.")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("Seed must be a nonnegative integer.")

    rng = np.random.default_rng(seed)
    posterior_means = np.zeros(3)
    posterior_variances = np.full(3, 25.0)
    counts = np.zeros(3, dtype=int)
    selected_arms = np.empty(rounds, dtype=int)
    rewards = np.empty(rounds)

    for round_index in range(rounds):
        sampled_means = rng.normal(posterior_means, np.sqrt(posterior_variances))
        arm = int(np.argmax(sampled_means))
        reward = rng.normal(true_means[arm], sigma)
        posterior_means[arm], posterior_variances[arm] = update_posterior(
            posterior_means[arm], posterior_variances[arm], reward, sigma
        )
        counts[arm] += 1
        selected_arms[round_index] = arm
        rewards[round_index] = reward

    return {
        "posterior_means": posterior_means,
        "posterior_variances": posterior_variances,
        "counts": counts,
        "selected_arms": selected_arms,
        "rewards": rewards,
        "total_reward": float(rewards.sum()),
        "recommendation": int(np.argmax(posterior_means)) + 1,
    }
