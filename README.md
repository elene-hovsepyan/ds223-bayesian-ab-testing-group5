# DS223 Gaussian Bandit

This standalone Streamlit frontend extends the [DS223 starter project](https://github.com/hovhannisyan91/DS223_Group_Project). The starter URL redirects to [DS223_Bayesian_AB_Testing](https://github.com/hovhannisyan91/DS223_Bayesian_AB_Testing). Its `app/front/` files were copied from commit `3573a04fc6329a3f08a268b72093097d917c4156`. The original Create Carousels, Interaction, and Analytics pages are preserved.

- My GitHub repository: **TODO: ADD GITHUB URL**
- My deployed app: **TODO: ADD STREAMLIT URL**

## Run locally

Install Python 3.12 and [uv](https://docs.astral.sh/uv/getting-started/installation/). From this folder:

```bash
uv sync --locked
uv run streamlit run main.py
```

Open the local URL printed by Streamlit. No backend, database, or API key is needed. The Python version is recorded in `.python-version`; dependencies are pinned in `uv.lock`.

```bash
uv run python -m unittest -v
uv run python experiments.py
```

The nested `DS223_Bayesian_AB_Testing/` folder is the original reference checkout. It is ignored by Git and is not needed to run or publish this frontend. Environments and generated caches are also ignored.

## Original carousel model

Each alternative starts with a Beta(1, 1) prior for its click probability. For a new display, the app samples one probability from each alternative's Beta posterior and shows the alternatives with the largest samples, up to the visible count. The current display stays the same until feedback or an explicit resample.

A feedback round adds one impression to every displayed alternative. Clicking an alternative adds one click and one to its alpha; every other displayed alternative adds one to its beta. With no click, every displayed alternative adds one to beta. Alternatives outside the display are unchanged. History records a round reward of 1 for a click or 0 for no click. Resampling without feedback changes the display without learning from it.

The posterior mean is `alpha / (alpha + beta)`. Analytics uses it to choose a champion and shows counts, posterior distributions, and cumulative round rewards. Its "Average Reward" is total clicks divided by total impressions, so it is a click rate across displayed alternatives.

`main.py` registers the pages and initializes session state. `st.session_state.carousels` holds each carousel's alternatives, posteriors, display, round number, and history. `active_carousel_id` connects the selected carousel across pages. The three pages read and update those same objects. State belongs to one browser session and is not saved to a database; a new session starts fresh. Some starter management actions remain placeholders, including Delete selected carousel.

## Gaussian model

The new page has exactly three arms. The simulator knows their true means to generate rewards, but the learner selects using its posterior beliefs. Each unknown mean starts with an independent Normal prior with mean 0 and variance 25 (standard deviation 5).

Every round, sample one possible mean from each posterior, select the largest, then draw one reward from `Normal(true_mean, sigma²)` for that arm. Rewards stay continuous, including negative values. Only the selected arm learns from that reward.

For old posterior mean `m`, variance `v`, and reward `r`:

```text
reward_precision = 1 / sigma²
new_variance = 1 / (1 / v + reward_precision)
new_mean = new_variance * (m / v + r * reward_precision)
```

Sigma is the known shared reward standard deviation, not its variance. A larger sigma makes rewards noisier and gives each reward less weight in the update. For example, from the initial prior, reward 2 with sigma 1 gives variance `25/26` and mean `25/13`. With sigma 3, that same reward gives variance `225/34` and mean `25/17`: more uncertainty remains.

`gaussian_bandit.py` provides the simulation shared by the page and `experiments.py`. Each run starts fresh with a local NumPy generator, so it does not change the original carousel's random state. The final recommendation uses the largest final posterior mean, not the true means, selection counts, or last random draw. Ties choose the first arm. Returned history uses zero-based arm indices; the UI numbers arms 1–3.

The page keeps its last completed result and input configuration in `st.session_state.gaussian_result`, separately from the carousels. Changing form inputs does not run the simulation until Run is pressed. The stored results remain visible when navigating between pages and are labeled with the configuration that produced them.

## Experiments

All three runs use true means `[1, 2, 3]`, independent Normal(0, 25) priors, and seed **42**. Each run resets the generator and priors. Run `uv run python experiments.py` to reproduce the table.

| Experiment | σ | Rounds | Posterior means (arms 1, 2, 3) | Selection counts | Total reward | Recommendation |
|---|---:|---:|---|---|---:|---|
| Short run | 1 | 100 | 1.570217, 1.728703, 2.994983 | 4, 2, 94 | 291.518390 | Arm 3 |
| Longer run | 1 | 1000 | 1.306247, 2.142116, 2.979191 | 5, 13, 982 | 2960.201477 | Arm 3 |
| Higher noise | 3 | 1000 | 1.153692, 1.483757, 2.954223 | 21, 23, 956 | 2884.604431 | Arm 3 |

With more rounds at sigma 1, arm 3's share grows from 94% to 98.2%. Arms 1 and 2 move closer to their true means, but they receive few observations. Surprisingly, arm 3's estimate moves slightly farther from 3: from 2.994983 to 2.979191. More data reduces posterior uncertainty; it does not guarantee that a particular random estimate gets closer on every run. The 100-round run is the prefix of the 1000-round run because both start with the same seed and sigma.

At sigma 3, the learner selects the weaker arms 44 times instead of 18. Each reward has precision 1/9 rather than 1, so learning is slower. Arm 2 is estimated at only 1.483757 despite receiving more selections than in the lower-noise run. More noisy observations need not beat fewer precise observations in a single experiment. All three runs still recommend the truly best arm, arm 3.

The longer run has more total reward mainly because it has ten times as many rounds; its average observed reward is 2.960201 versus 2.915184 in the short run. Higher noise gives average reward 2.884604 here. Those totals mix random reward noise with arm choices. These are measured outcomes for one seed, not proof of average performance or a guarantee that higher noise always lowers observed reward. Rarely selected arms can keep inaccurate estimates while the final recommendation is correct.

## Checks

The unit tests check exact seeded reproducibility, selection totals, recommendations, negative rewards, input validation, and a hand-calculated posterior update. They also compare final posteriors against the closed-form calculation from each arm's actual rewards. A one-round check verifies posterior-based selection and that unselected arms keep their priors.

Streamlit AppTest checks the Gaussian controls, outputs, invalid inputs, repeated runs, and results across navigation. It also creates a carousel, submits click/no-click feedback, checks resampling without updates, opens Analytics, and verifies the Gaussian page leaves carousel state intact.

Verified locally with Python 3.12.0, NumPy 2.4.4, and Streamlit 1.56.0: `uv sync --locked`, all six tests, and all three experiments passed. The local Streamlit server started successfully and its health endpoint returned `ok`. In Chrome, the Gaussian page ran the default configuration and displayed total reward 2960.201477 and recommendation Arm 3. The initial UI test timed out during the first SciPy import; after import completed, the full suite passed in 1.757 seconds. The original pages and utilities and the lockfile match the starter byte for byte. The deployed app has not been verified yet.

## Publish and verify

1. Create an empty GitHub repository without adding a README. In this project root, run:

   ```bash
   git init -b main
   git add .
   git status --short
   git commit -m "Add Gaussian Thompson sampling frontend"
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
   git push -u origin main
   ```

2. Before committing, confirm the staged list excludes `.venv/`, caches, and the nested starter checkout. Keep the root `main.py`, `pages/`, Python modules, `pyproject.toml`, `uv.lock`, and `.python-version`.
3. In Streamlit Community Cloud, create an app from your repository. Select branch **main**, entrypoint **main.py**, and **Python 3.12** in Advanced settings, then deploy. See the [official deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).
4. Open Gaussian Bandit. Run means `[1, 2, 3]`, sigma `1`, rounds `1000`, seed `42`. Compare the posterior means, counts, total reward, and recommendation with the Longer run row above. Repeat to check reproducibility, and verify the counts sum to 1000.
5. Check sigma `0` and rounds `0` display validation errors. Run sigma `3`, rounds `1000`, seed `42`, and compare with Higher noise. Navigate away and back to confirm the last result remains visible. Also create a carousel, submit a click and a no-click, and open Analytics.
6. Replace both URL placeholders above, commit and push the README, and submit the GitHub and deployed Streamlit links. Deployment is a separate handoff step; local checks do not verify a deployed app.

## Four-minute demo outline

- **0:00–0:40:** Introduce the starter. Create a carousel, click once, then submit no click. Show how Analytics reads the same session data and how alpha/beta change.
- **0:40–1:30:** Open Gaussian Bandit. Explain continuous rewards, three unknown means, and Normal(0, 25) priors. Walk through sample → choose → reward → update only that arm.
- **1:30–2:15:** Explain the update above using reward 2. Sigma controls reward noise; its squared reciprocal is the precision. Larger sigma means less information per observation. Negative rewards are valid.
- **2:15–3:25:** Run the recorded configurations on the deployed page. Compare posterior means, counts, recommendations, and total reward using the experiment table. Point out the interpretation below the table rather than claiming every estimate improves.
- **3:25–4:00:** Repeat one configuration to show the seed works. Explain why the recommendation uses posterior means, why neglected arms may remain uncertain, and why one seed cannot establish average performance. Show the repository and app links.
