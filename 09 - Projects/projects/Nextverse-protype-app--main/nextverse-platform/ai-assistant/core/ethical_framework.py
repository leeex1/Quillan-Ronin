"""
Ethical Framework module for the AI Assistant.

This module provides functionalities to assess content against a defined set of
ethical guidelines. It includes checks for forbidden topics and calculates a
Dynamic Ethical Safety Score (DESS) based on moral factors, keywords, and context.
"""

import re
import logging

logger = logging.getLogger(__name__)


class EthicalComplianceChecker:
    """
    Simulates an advanced ethical compliance checker.
    In a real scenario, this might involve a separate model or more complex heuristics.
    """

    def __init__(self):
        """Initializes the EthicalComplianceChecker."""
        logger.info("EthicalComplianceChecker initialized (simulated).")

    def predict_ethical_score(self, text: str) -> float:
        """
        Simulates predicting a general ethical desirability score.
        Higher is better.
        """
        logger.debug(f"Simulating ethical score prediction for: {text[:50]}...")
        # Example: a simple heuristic based on text length or keywords
        # For now, returns a fixed value for predictability in tests.
        if "unethical_example" in text.lower():
            return 0.3  # Low score for specific test cases
        return 0.85

    def predict_lmcb_score(self, text: str) -> float:
        """
        Simulates predicting a score related to specific LMCB (Llama Guard / Moral Compass Benchmark) type checks.
        Higher means more aligned with specific safety benchmarks.
        """
        logger.debug(f"Simulating LMCB score prediction for: {text[:50]}...")
        if "lmcb_fail_example" in text.lower():
            return 0.2  # Low score for specific test cases
        return 0.75


class EthicalFramework:
    """
    Implements the ethical guidelines and content validation for the AI Assistant.

    This framework checks for forbidden topics and calculates a DESS score
    to determine content compliance. The configuration for moral factors,
    keywords, context weights, and forbidden topics is hardcoded for now
    but can be externalized in future versions.
    """

    def __init__(self):
        """
        Initializes the EthicalFramework with predefined configurations.

        Loads moral factors, their associated positive/negative keywords,
        base weights, context-specific weights, and a list of forbidden topics.
        """
        # Configuration for moral factors, keywords, and their base weights
        # Base weights signify the intrinsic importance of each factor.
        self.moral_factors_config = {
            "truthfulness": {
                "positive_keywords": [
                    "accurate",
                    "factual",
                    "verified",
                    "true",
                    "correct",
                    "authentic",
                    "evidence-based",
                    "verifiable",
                    "substantiated",
                    "genuine",
                    "sound",
                    "valid",
                ],
                "negative_keywords": [
                    "false",
                    "misleading",
                    "unverified",
                    "lie",
                    "incorrect",
                    "fabricated",
                    "fake",
                    "deceptive",
                    "unsubstantiated",
                    "hoax",
                    "disinformation",
                    "counterfeit",
                    "fraudulent",
                ],
                "base_score_weight": 1.0,
            },
            "harmlessness": {
                "positive_keywords": [
                    "safe",
                    "harmless",
                    "respectful",
                    "constructive",
                    "positive",
                    "supportive",
                    "safe-guard",
                    "helpful",
                    "beneficial",
                    "protective",
                    "considerate",
                    "empathetic",
                ],
                "negative_keywords": [
                    "harmful",
                    "dangerous",
                    "toxic",
                    "offensive",
                    "threat",
                    "attack",
                    "hate",
                    "abusive",
                    "inflammatory",
                    "malicious",
                    "violent",
                    "harass",
                    "bully",
                    "exploit",
                ],  # 'exploit' added
                "base_score_weight": 1.6,  # Slightly increased as per plan
            },
            "fairness": {
                "positive_keywords": [
                    "fair",
                    "unbiased",
                    "impartial",
                    "equitable",
                    "just",
                    "objective",
                    "balanced",
                    "even-handed",
                    "neutral",
                    "equal",
                    "non-discriminatory",
                    "representative",
                ],
                "negative_keywords": [
                    "unfair",
                    "biased",
                    "partial",
                    "discriminatory",
                    "unjust",
                    "stereotypical",
                    "prejudiced",
                    "subjective",
                    "one-sided",
                    "nepotism",
                    "favoritism",
                    "unequal",
                ],
                "base_score_weight": 1.2,
            },
            "privacy": {
                "positive_keywords": [
                    "private",
                    "confidential",
                    "secure",
                    "anonymized",
                    "protected",
                    "encrypted",
                    "discreet",
                    "non-tracking",
                    "anonymous",
                    "incognito",
                ],
                "negative_keywords": [
                    "expose",
                    "reveal",
                    "track",
                    "monitor",
                    "share personal data",
                    "doxxing",
                    "surveillance",
                    "identifiable",
                    "breach",
                    "leak",
                    "intrusive",
                    "eavesdrop",
                ],
                "base_score_weight": 1.3,
            },
        }

        # Context weights reviewed:
        # - Social: Increased harmlessness & privacy slightly more.
        # - Creative: Reduced truthfulness further, but slightly increased fairness to avoid harmful creative outputs.
        # - Combat: Harmlessness is paramount.
        self.context_weights = {
            "general": {
                "truthfulness": 1.0,
                "harmlessness": 1.0,
                "fairness": 1.0,
                "privacy": 1.0,
            },
            "combat": {
                "truthfulness": 0.7,
                "harmlessness": 1.7,
                "fairness": 0.6,
                "privacy": 0.4,
            },
            "social": {
                "truthfulness": 0.8,
                "harmlessness": 1.5,
                "fairness": 1.3,
                "privacy": 1.4,
            },
            "creative": {
                "truthfulness": 0.2,
                "harmlessness": 0.9,
                "fairness": 0.9,
                "privacy": 0.5,
            },
        }
        # Comments from before are still relevant.

        self.forbidden_topics = [
            "illegal activities",
            "drug manufacturing",
            "bomb making",
            "weapon modification",
            "hate speech",
            "inciting violence",
            "promoting discrimination",
            "child exploitation",
            "self-harm encouragement",
            "non-consensual sexual content",
            "graphic violence depiction",
            "terrorism promotion",
            "malware creation or distribution",
            "human trafficking",
            "illegal substance abuse",
        ]

        temp_forbidden_regexes = []
        for topic in self.forbidden_topics:
            # For multi-word topics, ensure they are treated as a phrase
            escaped_topic_parts = [re.escape(part) for part in topic.split()]
            # Regex to match whole words/phrase: \bword1\s+word2\b
            # Or for single word: \bword\b
            if len(escaped_topic_parts) > 1:
                regex_str = r"\b" + r"\s+".join(escaped_topic_parts) + r"\b"
            else:
                regex_str = r"\b" + escaped_topic_parts[0] + r"\b"
            temp_forbidden_regexes.append(re.compile(regex_str, re.IGNORECASE))
        self.forbidden_topics_regex = temp_forbidden_regexes

        logger.info("EthicalFramework initialized with predefined configurations.")
        self.compliance_checker = EthicalComplianceChecker()

    def _calculate_single_factor_score(
        self, text_lower: str, factor_config: dict
    ) -> float:
        """
        Calculates a raw score for a single moral factor based on keyword presence.

        The method tokenizes the input text and checks for exact matches of
        positive and negative keywords associated with the moral factor.

        Args:
            text_lower (str): The lowercase input text.
            factor_config (dict): The configuration for the moral factor,
                                  containing 'positive_keywords' and 'negative_keywords'.

        Returns:
            float: A score of 1.0 if positive keywords dominate, -1.0 if negative
                   keywords dominate, or 0.0 if neutral or balanced.
        """
        score = 0
        # Using substring check `kw in text_lower` to support multi-word keywords.
        # This is a simpler approach than complex tokenization for this specific need.
        # Ensure keywords are specific enough to avoid unintended partial matches if using this.
        for kw in factor_config["positive_keywords"]:
            if kw in text_lower:
                score += 1
        for kw in factor_config["negative_keywords"]:
            if kw in text_lower:
                score -= 1

        # Normalize score:
        # This simple approach scales the raw keyword count score.
        # More sophisticated methods could consider keyword frequency, TF-IDF, proximity, etc.
        # Currently: +1 for net positive, -1 for net negative, 0 for neutral/balanced.
        # Count total possible keyword hits for normalization (max positive + max negative occurrences)
        # For simplicity, let's assume a bounded score, e.g., if any keyword hits, it contributes.
        # A more robust normalization would consider frequency, proximity, etc.
        # Current: If positive and negative keywords for the same factor appear, they cancel out.
        # Let's scale it to be between -1 and 1. If score > 0, it's 1. If < 0, it's -1. If 0, it's 0.
        if score > 0:
            return 1.0
        if score < 0:
            return -1.0
        return 0.0

    def calculate_dess_score(self, text: str, context: str = "general") -> float:
        """
        Calculates the Dynamic Ethical Safety Score (DESS) for a given text and context.

        The DESS score is a normalized value between 0.0 and 1.0, derived from:
        - Raw scores for each moral factor (based on keyword presence).
        - Normalization of raw factor scores to a [0, 1] scale.
        - Application of base weights for each moral factor.
        - Application of context-specific weights for each moral factor.
        - Final normalization by the sum of all applicable weights.

        Args:
            text (str): The input text to evaluate.
            context (str, optional): The operational context (e.g., 'general', 'combat').
                                     Defaults to "general".

        Returns:
            float: The calculated DESS score, ranging from 0.0 (least compliant)
                   to 1.0 (most compliant).
        """
        text_lower = text.lower()  # Ensure text is lowercase for consistent processing

        if context not in self.context_weights:
            logger.warning(
                f"Context '{context}' not found in predefined context_weights. Defaulting to 'general' context."
            )
            context = "general"  # Fallback to a default context

        current_context_weights = self.context_weights[context]

        weighted_factor_scores = []
        total_applicable_weights = 0

        for factor_name, factor_config in self.moral_factors_config.items():
            raw_factor_score = self._calculate_single_factor_score(
                text_lower, factor_config
            )  # -1, 0, or 1

            # Convert to 0-1 scale for DESS: 0 for -1, 0.5 for 0, 1 for 1
            normalized_factor_score = (raw_factor_score + 1) / 2.0

            contextual_weight = current_context_weights.get(
                factor_name, 1.0
            )  # Default to 1 if factor somehow missing in context
            base_score_weight = factor_config.get("base_score_weight", 1.0)

            final_weight = contextual_weight * base_score_weight

            weighted_factor_scores.append(normalized_factor_score * final_weight)
            total_applicable_weights += (
                final_weight  # Sum of weights for factors considered in this context
            )

        if total_applicable_weights == 0:
            return 0.0  # Avoid division by zero if no factors or weights apply

        dess_score_raw_sum = sum(weighted_factor_scores)

        # Normalize DESS score to be between 0 and 1
        # Max possible score is sum of all final_weights (when normalized_factor_score is 1 for all)
        # Min possible score is 0 (when normalized_factor_score is 0 for all for every factor)
        # Max possible score is total_applicable_weights (when normalized_factor_score is 1 for all for every factor)
        # Thus, dividing by total_applicable_weights scales DESS score to be between 0 and 1.
        dess_score = dess_score_raw_sum / total_applicable_weights

        # Clamp the DESS score to ensure it's strictly within the [0.0, 1.0] range.
        return max(0.0, min(1.0, dess_score))

    def is_request_forbidden(self, text: str) -> tuple[bool, str | None]:
        """
        Checks if the input text contains any predefined forbidden topics.

        Args:
            text (str): The input text to check.

        Returns:
            tuple[bool, str | None]: A tuple where the first element is True if a
                                     forbidden topic is found (False otherwise), and
                                     the second element is the name of the forbidden
                                     topic if found, otherwise None.
        """
        text_lower = text.lower()  # Perform case-insensitive matching
        for i, regex_pattern in enumerate(self.forbidden_topics_regex):
            if regex_pattern.search(text_lower):
                forbidden_topic_name = self.forbidden_topics[i]
                logger.warning(
                    f"Forbidden topic detected: '{forbidden_topic_name}' in text: '{text[:100]}...'"
                )  # Log first 100 chars
                return True, forbidden_topic_name
        return False, None

    def validate_content(
        self,
        text: str,
        context: str = "general",
        is_sensitive: bool = False,
        lmcb_min_score: float = 0.4,
        critical_threshold: float = 0.75,
    ) -> dict:
        """
        Validates content based on forbidden topic checks and DESS score.

        Args:
            text (str): The text content to validate.
            context (str, optional): The context of the content (e.g., 'general', 'combat').
                                     Defaults to "general".
            is_sensitive (bool, optional): Flag indicating if the content is sensitive,
                                           which may apply a stricter threshold. Defaults to False.
            lmcb_min_score (float, optional): The minimum DESS score for compliance for
                                              non-sensitive content. Defaults to 0.4.
            critical_threshold (float, optional): The minimum DESS score for compliance
                                                  for sensitive content. Defaults to 0.75.

        Returns:
            dict: A dictionary containing validation results:
                - 'is_compliant' (bool): Whether the content is compliant.
                - 'dess_score' (float): The calculated DESS score.
                - 'threshold_used' (float): The DESS score threshold applied.
                - 'is_forbidden' (bool): Whether a forbidden topic was detected.
                - 'explanation' (str): An explanation for the validation outcome.
        """
        is_forbidden, forbidden_topic_name = self.is_request_forbidden(text)
        sim_ethical_score = -1.0  # Default to invalid if not run
        sim_lmcb_score = -1.0  # Default to invalid if not run

        if is_forbidden:
            logger.warning(
                f"Content validation failed due to forbidden topic: '{forbidden_topic_name}'. Input text: '{text[:200]}...'"
            )
            return {
                "is_compliant": False,
                "dess_score": 0.0,
                "threshold_used": -1.0,
                "is_forbidden": True,
                "explanation": f"Request involves a forbidden topic: '{forbidden_topic_name}'.",
                "simulated_ethical_score": sim_ethical_score,  # Still include keys even if not run
                "simulated_lmcb_score": sim_lmcb_score,
            }

        # Retrieve Simulated Scores
        sim_ethical_score = self.compliance_checker.predict_ethical_score(text)
        sim_lmcb_score = self.compliance_checker.predict_lmcb_score(text)
        logger.info(
            f"Simulated Ethical Score: {sim_ethical_score:.2f}, Simulated LMCB Score: {sim_lmcb_score:.2f} for text: '{text[:50]}...'"
        )

        dess_score = self.calculate_dess_score(text, context)
        threshold_used = critical_threshold if is_sensitive else lmcb_min_score

        # Define thresholds for simulated scores
        sim_ethical_threshold = 0.5
        sim_lmcb_threshold = 0.4

        # Combined compliance logic
        is_compliant = True
        explanation_parts = []

        if not (dess_score >= threshold_used):
            is_compliant = False
            explanation_parts.append(
                f"DESS score {dess_score:.2f} is below the required threshold {threshold_used:.2f}."
            )

        if not (sim_ethical_score >= sim_ethical_threshold):
            is_compliant = False
            explanation_parts.append(
                f"Simulated ethical score {sim_ethical_score:.2f} is below threshold {sim_ethical_threshold:.2f}."
            )

        if not (sim_lmcb_score >= sim_lmcb_threshold):
            is_compliant = False
            explanation_parts.append(
                f"Simulated LMCB score {sim_lmcb_score:.2f} is below threshold {sim_lmcb_threshold:.2f}."
            )

        if is_compliant:
            explanation = f"Content is compliant. DESS score {dess_score:.2f} meets threshold {threshold_used:.2f}. Simulated scores also meet their thresholds."
        else:
            explanation = "Content failed compliance checks: " + " ".join(
                explanation_parts
            )
            if not explanation_parts:  # Should not happen if is_compliant is False
                explanation = (
                    "Content failed compliance checks for unspecified reasons."
                )

        return {
            "is_compliant": is_compliant,
            "dess_score": dess_score,
            "threshold_used": threshold_used,
            "is_forbidden": False,
            "explanation": explanation,
            "simulated_ethical_score": sim_ethical_score,
            "simulated_lmcb_score": sim_lmcb_score,
        }


if __name__ == "__main__":
    # Basic Test
    ef = EthicalFramework()

    print("\n--- Forbidden Topic Test ---")
    forbidden_text = "Can you tell me about bomb making?"
    print(f"Input: '{forbidden_text}'")
    validation_result = ef.validate_content(forbidden_text)
    print(validation_result)
    assert not validation_result["is_compliant"]
    assert validation_result["is_forbidden"]

    print("\n--- Truthfulness Test (Positive) ---")
    truthful_text = "This information is accurate and verified."
    print(f"Input: '{truthful_text}'")
    validation_result = ef.validate_content(truthful_text, context="general")
    print(validation_result)
    # assert validation_result['is_compliant'] # Depends on threshold and other factors

    print("\n--- Harmlessness Test (Negative) ---")
    harmful_text = "This is a harmful and toxic statement."
    print(f"Input: '{harmful_text}'")
    validation_result = ef.validate_content(harmful_text, context="social")
    print(validation_result)
    # assert not validation_result['is_compliant'] # Depends on threshold

    print("\n--- Creative Context Test (Looser truthfulness) ---")
    creative_text = "Once upon a time, in a land of pure fiction..."  # Should have lower truthfulness score but might pass in creative.
    print(
        f"Input: '{creative_text}' (truthfulness negative keywords: ['false', 'misleading', 'unverified', 'lie', 'incorrect', 'fabricated', 'fake'])"
    )
    # To make it clearly "untrue" for testing:
    creative_text_with_lie = (
        "The moon is made of cheese, which is a verified fact, or is it a lie?"
    )
    print(f"Input: '{creative_text_with_lie}'")
    validation_result = ef.validate_content(
        creative_text_with_lie, context="creative", lmcb_min_score=0.4
    )  # Lower threshold for creative
    print(validation_result)

    print("\n--- Sensitive Content Test ---")
    sensitive_query = "Is this treatment safe and verified for a critical condition?"
    print(f"Input: '{sensitive_query}'")
    validation_result = ef.validate_content(
        sensitive_query, context="general", is_sensitive=True, critical_threshold=0.85
    )
    print(validation_result)

    print("\n--- General Compliance (High Score) ---")
    positive_general_text = "This is accurate, safe, fair, and respects privacy."
    print(f"Input: '{positive_general_text}'")
    validation_result = ef.validate_content(positive_general_text)
    print(validation_result)
    assert validation_result["is_compliant"]

    print("\n--- General Non-Compliance (Low Score) ---")
    negative_general_text = "This is false, harmful, unfair, and exposes personal data."
    print(f"Input: '{negative_general_text}'")
    validation_result = ef.validate_content(negative_general_text)
    print(validation_result)
    assert not validation_result["is_compliant"]

    print("\n--- Combat Context (High Harmlessness Weight) ---")
    combat_text_bad = "Ignore safety, this is a dangerous approach but it is factual."
    print(f"Input: '{combat_text_bad}'")
    validation_result = ef.validate_content(combat_text_bad, context="combat")
    print(validation_result)
    # Expect non-compliant due to high harmlessness weight and negative harmlessness keyword

    combat_text_good = "This is a safe and verified tactic."
    print(f"Input: '{combat_text_good}'")
    validation_result = ef.validate_content(combat_text_good, context="combat")
    print(validation_result)
    # Expect compliant

    print("Basic demonstration finished.")
