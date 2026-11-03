import unittest
from pathlib import Path

import numpy as np
from streamlit.testing.v1 import AppTest

from gaussian_bandit import simulate_gaussian_bandit


class FrontendTests(unittest.TestCase):
    def setUp(self):
        self.app = AppTest.from_file(str(Path(__file__).with_name("main.py")), default_timeout=120).run()
        self.assertEqual(len(self.app.exception), 0)

    def test_gaussian_page(self):
        app = self.app.switch_page("pages/Gaussian_Bandit.py").run()
        self.assertEqual(app.title[0].value, "Gaussian Bandit")
        self.assertEqual(len(app.number_input), 6)
        app.button[0].click().run()
        self.assertEqual(len(app.exception), 0)
        expected = simulate_gaussian_bandit([1, 2, 3], 1, 1000, 42)
        table = app.dataframe[0].value
        np.testing.assert_array_equal(table["Final posterior mean"], expected["posterior_means"])
        np.testing.assert_array_equal(table["Selection count"], expected["counts"])
        self.assertEqual(app.metric[0].value, f"{expected['total_reward']:.6f}")
        self.assertIn(f"Arm {expected['recommendation']}", app.markdown[0].value)
        app.button[0].click().run()
        np.testing.assert_array_equal(
            app.session_state["gaussian_result"]["result"]["rewards"], expected["rewards"]
        )
        for sigma in [0.0, -1.0]:
            app.number_input[3].set_value(sigma)
            app.button[0].click().run()
            self.assertIn("Sigma must be positive", app.error[0].value)
        app.number_input[3].set_value(1.0)
        app.number_input[4].set_value(0)
        app.button[0].click().run()
        self.assertIn("positive integer", app.error[0].value)
        app.switch_page("pages/Analytics.py").run()
        app.switch_page("pages/Gaussian_Bandit.py").run()
        np.testing.assert_array_equal(app.dataframe[0].value["Selection count"], expected["counts"])
        app.number_input[0].set_value(-100.0)
        app.number_input[1].set_value(-200.0)
        app.number_input[2].set_value(-300.0)
        app.number_input[4].set_value(1)
        app.button[0].click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertLess(app.session_state["gaussian_result"]["result"]["total_reward"], 0)

    def test_original_flow(self):
        app = self.app
        app.button[0].click().run()
        self.assertEqual(len(app.exception), 0)
        carousel_id = app.session_state["active_carousel_id"]
        carousel = app.session_state["carousels"][carousel_id]
        self.assertEqual(len(carousel["alternatives"]), 10)
        self.assertEqual(len(carousel["current_display_ids"]), 4)
        app.switch_page("pages/Interaction.py").run()
        displayed = carousel["current_display_ids"].copy()
        clicked = displayed[0]
        app.button(key=f"click_{carousel_id}_{clicked}").click().run()
        carousel = app.session_state["carousels"][carousel_id]
        stats = carousel["alternatives"].set_index("alt_id")
        for arm in stats.index:
            self.assertEqual(stats.at[arm, "impressions"], int(arm in displayed))
            self.assertEqual(stats.at[arm, "alpha"], 1 + int(arm == clicked))
            self.assertEqual(stats.at[arm, "beta"], 1 + int(arm in displayed and arm != clicked))
        self.assertEqual(carousel["history"][0]["reward"], 1)
        before = carousel["alternatives"].copy()
        displayed = carousel["current_display_ids"].copy()
        next(button for button in app.button if button.label == "No click / Next round").click().run()
        carousel = app.session_state["carousels"][carousel_id]
        after = carousel["alternatives"]
        np.testing.assert_array_equal(after["alpha"], before["alpha"])
        np.testing.assert_array_equal(after["clicks"], before["clicks"])
        np.testing.assert_array_equal(
            after["beta"] - before["beta"], before["alt_id"].isin(displayed).astype(int)
        )
        self.assertEqual(carousel["history"][-1]["reward"], 0)
        self.assertEqual(carousel["round_number"], 3)
        before = after.copy()
        next(button for button in app.button if button.label == "Resample without update").click().run()
        self.assertTrue(carousel["alternatives"].equals(before))
        self.assertEqual(len(carousel["history"]), 2)
        app.switch_page("pages/Analytics.py").run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual([metric.value for metric in app.metric[:3]], ["2", "8", "1"])
        self.assertEqual(len(app.dataframe), 3)
        app.switch_page("pages/Gaussian_Bandit.py").run()
        app.button[0].click().run()
        self.assertTrue(app.session_state["carousels"][carousel_id]["alternatives"].equals(before))
        app.switch_page("pages/Create_Carousels.py").run()
        self.assertEqual(app.session_state["active_carousel_id"], carousel_id)
        self.assertEqual(len(app.exception), 0)


if __name__ == "__main__":
    unittest.main()
