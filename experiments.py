from gaussian_bandit import simulate_gaussian_bandit


SEED = 42
TRUE_MEANS = [1, 2, 3]


def main():
    print(f"True means: {TRUE_MEANS}; prior: Normal(0, 25); seed: {SEED}")
    print("| Experiment | σ | Rounds | Posterior means (arms 1, 2, 3) | Selection counts | Total reward | Recommendation |")
    print("|---|---:|---:|---|---|---:|---|")
    for name, sigma, rounds in [
        ("Short run", 1, 100),
        ("Longer run", 1, 1000),
        ("Higher noise", 3, 1000),
    ]:
        result = simulate_gaussian_bandit(TRUE_MEANS, sigma, rounds, SEED)
        means = ", ".join(f"{mean:.6f}" for mean in result["posterior_means"])
        counts = ", ".join(str(count) for count in result["counts"])
        print(f"| {name} | {sigma} | {rounds} | {means} | {counts} | "
              f"{result['total_reward']:.6f} | Arm {result['recommendation']} |")


if __name__ == "__main__":
    main()
