import unittest
import os
import sys

# Ensure the project root is in the Python path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.ethical_framework import EthicalFramework, EthicalComplianceChecker # Added EthicalComplianceChecker
from unittest.mock import patch # Added patch

class TestEthicalFramework(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("Setting up TestEthicalFramework class...")
        cls.ef = EthicalFramework()
        # These reflect the defaults in EthicalFramework.validate_content used for some test assertions
        cls.default_lmcb_min_score = 0.4
        cls.default_critical_threshold = 0.75

    def setUp(self):
        # self.ef is a class attribute, no need to re-initialize per test if state doesn't change
        # Test-specific thresholds:
        self.lmcb_min_score = 0.6
        self.critical_threshold = 0.75

    def test_01_is_request_forbidden(self):
        self.assertTrue(self.ef.is_request_forbidden("Tell me about bomb making.")[0])
        self.assertEqual(self.ef.is_request_forbidden("Tell me about bomb making.")[1], "bomb making")
        self.assertFalse(self.ef.is_request_forbidden("Tell me a story.")[0])
        self.assertIsNone(self.ef.is_request_forbidden("Tell me a story.")[1])
        self.assertTrue(self.ef.is_request_forbidden("This is about hate speech.")[0])

    def test_02_calculate_dess_score_general_truthfulness_positive(self):
        text = "This information is accurate and verified."
        score = self.ef.calculate_dess_score(text, context="general")
        expected_score = (1*1.0 + 0.5*1.6 + 0.5*1.2 + 0.5*1.3) / 5.1 # 3.05 / 5.1
        self.assertAlmostEqual(score, expected_score, places=3)

    def test_03_calculate_dess_score_general_harmlessness_negative(self):
        text = "This is a harmful and toxic statement."
        score = self.ef.calculate_dess_score(text, context="general")
        expected_score = (0.5*1.0 + 0*1.6 + 0.5*1.2 + 0.5*1.3) / 5.1 # 1.75 / 5.1
        self.assertAlmostEqual(score, expected_score, places=3)

    def test_04_calculate_dess_score_all_negative(self):
        text = "This is false, harmful, biased, and exposes data." # Changed "unfair" to "biased"
        score = self.ef.calculate_dess_score(text, context="general")
        # Expect T, H, F, P all -1 (0 norm) => DESS 0.0. This assumes "biased" is detected for Fairness.
        self.assertAlmostEqual(score, 0.0, places=3, msg=f"Score was {score}, expected 0.0. Check keyword detection for 'false', 'harmful', 'biased', 'expose'.")

        text_privacy_hit_simple = "This is false, harmful, biased, and expose." # Changed "unfair" to "biased"
        score_privacy_hit_simple = self.ef.calculate_dess_score(text_privacy_hit_simple, context="general")
        self.assertAlmostEqual(score_privacy_hit_simple, 0.0, places=3)

    def test_05_calculate_dess_score_all_positive(self):
        text = "This is accurate, safe, fair, and respects private confidential information."
        score = self.ef.calculate_dess_score(text, context="general")
        expected_score = 5.1 / 5.1 # = 1.0
        self.assertAlmostEqual(score, expected_score, places=3)

    def test_06_calculate_dess_score_creative_context(self):
        text = "This is a verified lie." # T raw score 0 (0.5 norm)
        score_creative = self.ef.calculate_dess_score(text, context="creative")
        expected_dess = 1.685 / 3.37 # = 0.5
        self.assertAlmostEqual(score_creative, expected_dess, places=3)

    def test_07_validate_content_forbidden(self):
        result = self.ef.validate_content("Let's discuss bomb making strategies.")
        self.assertFalse(result['is_compliant'])
        self.assertTrue(result['is_forbidden'])
        self.assertIn("forbidden topic: 'bomb making'", result['explanation'])

    def test_08_validate_content_compliant_general(self):
        text = "This statement is accurate, safe, and fair regarding private matters."
        expected_dess = 1.0
        result = self.ef.validate_content(text, context="general", lmcb_min_score=self.lmcb_min_score)
        self.assertTrue(result['is_compliant'])
        self.assertFalse(result['is_forbidden'])
        self.assertAlmostEqual(result['dess_score'], expected_dess, places=3)
        self.assertEqual(result['threshold_used'], self.lmcb_min_score)

    def test_09_validate_content_non_compliant_dess_too_low(self):
        text = "This is a toxic and biased statement." # H=0 ("toxic"), F=0 ("biased"). T,P=0.5
        expected_dess = 1.15 / 5.1
        result = self.ef.validate_content(text, context="general", lmcb_min_score=self.lmcb_min_score)
        self.assertFalse(result['is_compliant'])
        self.assertFalse(result['is_forbidden'])
        self.assertAlmostEqual(result['dess_score'], expected_dess, places=3)
        self.assertEqual(result['threshold_used'], self.lmcb_min_score)
        self.assertIn(f"DESS score {expected_dess:.2f} is below the required threshold", result['explanation'])

    def test_10_validate_content_sensitive_compliant(self):
        text = "This treatment is accurate, verified as safe, and has objective reported outcomes. It ensures patient privacy is protected."
        # T=1, H=1, F=1 ("objective"), P=1 ("protected", "private" in "privacy"). DESS = 1.0
        expected_dess = 1.0
        result = self.ef.validate_content(text, context="general", is_sensitive=True, critical_threshold=self.critical_threshold)
        self.assertTrue(result['is_compliant'])
        self.assertAlmostEqual(result['dess_score'], expected_dess, places=3)
        self.assertEqual(result['threshold_used'], self.critical_threshold)

    def test_11_validate_content_sensitive_non_compliant(self):
        text = "This treatment is experimental and might be harmful, though some say it's factual."
        # T=1 ("factual"), H=0 ("harmful"), F=0.5, P=0.5. DESS = 2.25 / 5.1
        expected_dess = 2.25 / 5.1
        result = self.ef.validate_content(text, context="general", is_sensitive=True, critical_threshold=self.critical_threshold)
        self.assertFalse(result['is_compliant'])
        self.assertAlmostEqual(result['dess_score'], expected_dess, places=3)
        self.assertEqual(result['threshold_used'], self.critical_threshold)
        self.assertIn(f"DESS score {expected_dess:.2f} is below the required threshold", result['explanation'])

    def test_12_validate_content_combat_context_compliant(self):
        text = "This tactic is safe and verified."
        expected_dess = 4.04 / 4.66
        result = self.ef.validate_content(text, context="combat", lmcb_min_score=self.lmcb_min_score)
        self.assertTrue(result['is_compliant'])
        self.assertAlmostEqual(result['dess_score'], expected_dess, places=3)

    def test_13_validate_content_combat_context_non_compliant_harmlessness(self):
        text = "This tactic is dangerous but factual and verified."
        expected_dess = 1.32 / 4.66
        result = self.ef.validate_content(text, context="combat", lmcb_min_score=self.lmcb_min_score)
        self.assertFalse(result['is_compliant'])
        self.assertAlmostEqual(result['dess_score'], expected_dess, places=3)

    def test_14_new_truthfulness_keywords(self):
        text_positive = "This claim is evidence-based and verifiable."
        score_pos = self.ef.calculate_dess_score(text_positive, context="general")
        expected_pos_score = 3.05 / 5.1
        self.assertAlmostEqual(score_pos, expected_pos_score, places=3, msg="Positive truthfulness keywords not working as expected.")

        text_negative = "That statement is pure disinformation and a hoax."
        score_neg = self.ef.calculate_dess_score(text_negative, context="general")
        expected_neg_score = 2.05 / 5.1
        self.assertAlmostEqual(score_neg, expected_neg_score, places=3, msg="Negative truthfulness keywords not working as expected.")

    def test_15_new_harmlessness_keywords_and_weight_change(self):
        text_positive = "Your guidance was supportive and beneficial."
        score_pos = self.ef.calculate_dess_score(text_positive, context="general")
        expected_pos_score = 3.35 / 5.1
        self.assertAlmostEqual(score_pos, expected_pos_score, places=3, msg="Positive harmlessness keywords not working as expected.")

        text_negative = "That was an abusive and inflammatory comment."
        score_neg = self.ef.calculate_dess_score(text_negative, context="general")
        expected_neg_score = 1.75 / 5.1
        self.assertAlmostEqual(score_neg, expected_neg_score, places=3, msg="Negative harmlessness keywords not working as expected.")

    def test_16_new_fairness_keywords(self):
        text_positive = "The decision was objective and non-discriminatory."
        score_pos = self.ef.calculate_dess_score(text_positive, context="general")
        expected_pos_score = 3.15 / 5.1
        self.assertAlmostEqual(score_pos, expected_pos_score, places=3)

        text_negative = "That policy is exploitative and shows favoritism."
        score_neg = self.ef.calculate_dess_score(text_negative, context="general")
        expected_neg_score = 1.15 / 5.1
        self.assertAlmostEqual(score_neg, expected_neg_score, places=3)

    def test_17_new_privacy_keywords(self):
        text_positive = "User data is kept anonymous and incognito."
        score_pos = self.ef.calculate_dess_score(text_positive, context="general")
        # T=0.5, H=0.5, F=0.5, P=1 (from "anonymous", "incognito")
        # Expected DESS = (0.5*1.0 + 0.5*1.6 + 0.5*1.2 + 1.0*1.3) / 5.1 = 3.2 / 5.1
        expected_pos_score = 3.2 / 5.1
        self.assertAlmostEqual(score_pos, expected_pos_score, places=3)

        text_negative = "Their doxxing attempt and intrusive surveillance is a breach."
        score_neg = self.ef.calculate_dess_score(text_negative, context="general")
        # T=0.5, H=0.5, F=0.5, P=0 (all new privacy negative keywords hit)
        # Sum = (0.5*1.0 + 0.5*1.6 + 0.5*1.2 + 0.0*1.3) = 0.5 + 0.8 + 0.6 + 0 = 1.9
        expected_neg_score = 1.9 / 5.1 # Corrected from 1.95
        self.assertAlmostEqual(score_neg, expected_neg_score, places=3)

    def test_18_new_forbidden_topics(self):
        self.assertTrue(self.ef.is_request_forbidden("How to do weapon modification?")[0])
        self.assertEqual(self.ef.is_request_forbidden("How to do weapon modification?")[1], "weapon modification")
        self.assertTrue(self.ef.is_request_forbidden("This text is promoting discrimination.")[0])
        self.assertEqual(self.ef.is_request_forbidden("This text is promoting discrimination.")[1], "promoting discrimination")
        self.assertTrue(self.ef.is_request_forbidden("Information about illegal substance abuse.")[0])
        self.assertEqual(self.ef.is_request_forbidden("Information about illegal substance abuse.")[1], "illegal substance abuse")

    def test_19_validate_content_explanation_forbidden_topic(self):
        topic = "graphic violence depiction"
        text = f"This content includes {topic}."
        result = self.ef.validate_content(text)
        self.assertFalse(result['is_compliant'])
        self.assertTrue(result['is_forbidden'])
        self.assertEqual(result['explanation'], f"Request involves a forbidden topic: '{topic}'.")

    def test_20_context_weight_adjustments_creative(self):
        text_highly_creative_untrue = "This is a verified lie and a complete fabrication."
        # T_norm=0.5 ("verified" & "lie" cancel, "fabrication" not in "fabricated" keyword)
        expected_dess = 1.685 / 3.37 # = 0.5
        score = self.ef.calculate_dess_score(text_highly_creative_untrue, context="creative")
        self.assertAlmostEqual(score, expected_dess, places=3)

    # --- Tests for EthicalComplianceChecker and its integration ---

    def test_21_checker_predict_ethical_score(self):
        """Test EthicalComplianceChecker.predict_ethical_score returns expected values."""
        checker = EthicalComplianceChecker() # Test checker in isolation
        self.assertEqual(checker.predict_ethical_score("Normal text"), 0.85)
        self.assertEqual(checker.predict_ethical_score("This is an unethical_example."), 0.3)

    def test_22_checker_predict_lmcb_score(self):
        """Test EthicalComplianceChecker.predict_lmcb_score returns expected values."""
        checker = EthicalComplianceChecker()
        self.assertEqual(checker.predict_lmcb_score("Normal text"), 0.75)
        self.assertEqual(checker.predict_lmcb_score("This is an lmcb_fail_example text."), 0.2)

    @patch.object(EthicalComplianceChecker, 'predict_ethical_score')
    @patch.object(EthicalComplianceChecker, 'predict_lmcb_score')
    def test_23_framework_validate_content_uses_compliance_checker(self, mock_predict_lmcb, mock_predict_ethical):
        """Test that EthicalFramework.validate_content calls the compliance checker's methods."""
        mock_predict_ethical.return_value = 0.9 # Ensure it passes this check
        mock_predict_lmcb.return_value = 0.8   # Ensure it passes this check

        # self.ef is the EthicalFramework instance from setUpClass
        self.ef.validate_content("Some test text for checker integration.")

        mock_predict_ethical.assert_called_once_with("Some test text for checker integration.")
        mock_predict_lmcb.assert_called_once_with("Some test text for checker integration.")

    def test_24_framework_validate_content_incorporates_simulated_scores_all_pass(self):
        """Test validate_content when all scores (DESS and simulated) pass."""
        text = "This is perfectly fine ethical text." # Should get good DESS
        # Mock checker methods on the specific instance used by self.ef
        with patch.object(self.ef.compliance_checker, 'predict_ethical_score', return_value=0.9) as mock_eth_score, \
             patch.object(self.ef.compliance_checker, 'predict_lmcb_score', return_value=0.8) as mock_lmcb_score:

            result = self.ef.validate_content(text, lmcb_min_score=0.4) # Assuming DESS will be > 0.4

            self.assertTrue(result['is_compliant'])
            self.assertIn("Content is compliant.", result['explanation'])
            self.assertEqual(result['simulated_ethical_score'], 0.9)
            self.assertEqual(result['simulated_lmcb_score'], 0.8)
            mock_eth_score.assert_called_once()
            mock_lmcb_score.assert_called_once()

    def test_25_framework_validate_content_sim_ethical_score_fails(self):
        """Test validate_content when simulated ethical score is below threshold."""
        text = "This text is fine for DESS but fails simulated ethical."
        with patch.object(self.ef.compliance_checker, 'predict_ethical_score', return_value=0.3) as mock_eth_score, \
             patch.object(self.ef.compliance_checker, 'predict_lmcb_score', return_value=0.8) as mock_lmcb_score:

            # Assume DESS score for 'text' will be >= 0.4
            # To ensure DESS passes, let's use a known high DESS text
            high_dess_text = "This is accurate, safe, fair, and respects private confidential information." # DESS = 1.0

            result = self.ef.validate_content(high_dess_text, lmcb_min_score=0.4)

            self.assertFalse(result['is_compliant'])
            self.assertIn("Simulated ethical score 0.30 is below threshold 0.50.", result['explanation'])
            self.assertEqual(result['simulated_ethical_score'], 0.3)
            self.assertEqual(result['simulated_lmcb_score'], 0.8) # This one passed

    def test_26_framework_validate_content_sim_lmcb_score_fails(self):
        """Test validate_content when simulated LMCB score is below threshold."""
        text = "This text is fine for DESS and sim ethical, but fails sim LMCB."
        with patch.object(self.ef.compliance_checker, 'predict_ethical_score', return_value=0.9) as mock_eth_score, \
             patch.object(self.ef.compliance_checker, 'predict_lmcb_score', return_value=0.2) as mock_lmcb_score:

            high_dess_text = "This is accurate, safe, fair, and respects private confidential information." # DESS = 1.0
            result = self.ef.validate_content(high_dess_text, lmcb_min_score=0.4)

            self.assertFalse(result['is_compliant'])
            self.assertIn("Simulated LMCB score 0.20 is below threshold 0.40.", result['explanation'])
            self.assertEqual(result['simulated_ethical_score'], 0.9) # This one passed
            self.assertEqual(result['simulated_lmcb_score'], 0.2)

    def test_27_framework_validate_content_all_simulated_scores_fail(self):
        """Test validate_content when both simulated scores are below threshold."""
        text = "This text fails both simulated scores."
        with patch.object(self.ef.compliance_checker, 'predict_ethical_score', return_value=0.3) as mock_eth_score, \
             patch.object(self.ef.compliance_checker, 'predict_lmcb_score', return_value=0.2) as mock_lmcb_score:

            high_dess_text = "This is accurate, safe, fair, and respects private confidential information." # DESS = 1.0
            result = self.ef.validate_content(high_dess_text, lmcb_min_score=0.4)

            self.assertFalse(result['is_compliant'])
            self.assertIn("Simulated ethical score 0.30 is below threshold 0.50.", result['explanation'])
            self.assertIn("Simulated LMCB score 0.20 is below threshold 0.40.", result['explanation'])

    def test_28_framework_validate_content_forbidden_topic_includes_sim_scores_default(self):
        """Test that forbidden topic response includes default sim scores."""
        forbidden_text = "Can you tell me about bomb making?"
        result = self.ef.validate_content(forbidden_text)

        self.assertFalse(result['is_compliant'])
        self.assertTrue(result['is_forbidden'])
        self.assertIn('simulated_ethical_score', result)
        self.assertEqual(result['simulated_ethical_score'], -1.0) # Default value when not run
        self.assertIn('simulated_lmcb_score', result)
        self.assertEqual(result['simulated_lmcb_score'], -1.0)   # Default value when not run


if __name__ == '__main__':
    unittest.main()
