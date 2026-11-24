#include "gtest/gtest.h"
#include "DVVE.hpp"
#include "Game.hpp" // Added for GameDVVETest
#include <string>    // For std::string used in Game class and tests

// #include <vector>   // Not strictly needed for these tests
// #include <iomanip>  // Not strictly needed for these tests

// Test fixture for DVVECalculator tests
class DVVECalculatorTest : public ::testing::Test {
protected:
    DVVECalculator calculator;
    DVVEParameters params; // Will be initialized per test or use defaults from DVVE.hpp
};

TEST_F(DVVECalculatorTest, BasicCalculation) {
    params.P_core = 2.0e9;
    params.F_v = 1.0;
    params.omega_v = 0.1;
    params.nu_v = 0.05;
    // R_p = 2.0e9 * 1.0 * (1 + 0.1) / (1 + 0.05)
    // R_p = 2.0e9 * 1.1 / 1.05
    // R_p = 2.2e9 / 1.05 = 2095238095.238095...
    double expected_Rp = (2.0e9 * 1.0 * (1.0 + 0.1)) / (1.0 + 0.05);
    ASSERT_NEAR(calculator.calculatePerformance(params), expected_Rp, 1e-3); // Using 1e-3 for floating point comparison
}

TEST_F(DVVECalculatorTest, ZeroPCore) {
    params.P_core = 0.0;
    // nu_v is default 0.05, F_v default 1.0, omega_v default 0.0 from DVVEParameters struct
    ASSERT_DOUBLE_EQ(calculator.calculatePerformance(params), 0.0);
}

TEST_F(DVVECalculatorTest, DivisionByZeroNuvShouldReturnZero) {
    params.P_core = 1.0e9; // Ensure P_core is non-zero to isolate nu_v effect
    params.nu_v = -1.0; // This makes (1 + nu_v) zero
    // The implementation DVVECalculator::calculatePerformance has a check for (1.0 + params.nu_v) == 0
    ASSERT_DOUBLE_EQ(calculator.calculatePerformance(params), 0.0);
}

TEST_F(DVVECalculatorTest, DetermineAdaptiveBoostLogic) {
    // Assuming target_p_core_for_max_quality is implicitly 2.0e9 for this test,
    // matching the Game class's usage context for which determineAdaptiveBoost was designed.
    // A more robust DVVECalculator might take target_P_core as an argument to determineAdaptiveBoost.
    // For now, test against the fixed thresholds in the current implementation.
    double target_max_q_for_game_context = 2.0e9;

    // Test case 1: P_core_current (0.8e9) < target_max_q_for_game_context * 0.5 (1.0e9) -> should be 0.2
    ASSERT_DOUBLE_EQ(calculator.determineAdaptiveBoost(0.8e9, target_max_q_for_game_context), 0.2);

    // Test case 2: P_core_current (1.2e9) is >= (0.5 * target) and < (0.75 * target which is 1.5e9) -> should be 0.1
    ASSERT_DOUBLE_EQ(calculator.determineAdaptiveBoost(1.2e9, target_max_q_for_game_context), 0.1);

    // Test case 3: P_core_current (1.6e9) is >= (0.75 * target) -> should be 0.0
    ASSERT_DOUBLE_EQ(calculator.determineAdaptiveBoost(1.6e9, target_max_q_for_game_context), 0.0);

    // Test case 4: P_core_current (2.0e9) is at target -> should be 0.0
    ASSERT_DOUBLE_EQ(calculator.determineAdaptiveBoost(2.0e9, target_max_q_for_game_context), 0.0);

    // Test case 5: P_core_current (0.99e9) < 0.5 * target (1.0e9) -> should be 0.2
     ASSERT_DOUBLE_EQ(calculator.determineAdaptiveBoost(0.99e9, target_max_q_for_game_context), 0.2);

    // Test case 6: P_core_current (1.49e9) < 0.75 * target (1.5e9) -> should be 0.1
    ASSERT_DOUBLE_EQ(calculator.determineAdaptiveBoost(1.49e9, target_max_q_for_game_context), 0.1);

    // Test case 7: P_core_current (1.5e9) == 0.75 * target -> should be 0.0 (falls into the else)
    ASSERT_DOUBLE_EQ(calculator.determineAdaptiveBoost(1.5e9, target_max_q_for_game_context), 0.0);
}


// Test fixture for Game class DVVE integration tests
class GameDVVETest : public ::testing::Test {
protected:
    Game game; // Game object. Its constructor sets default DVVE params.
               // It also has target_p_core_for_max_quality_ = 2.0e9;
};

TEST_F(GameDVVETest, HighPCoreHighComplexity) {
    game.setMockPCore(2.0e9);
    game.setVisualComplexity(1.8); // High complexity
    game.update(0.016); // Call update to trigger DVVE logic

    // Game's target_p_core_for_max_quality_ is 2.0e9.
    // P_core = 2.0e9. adaptive omega_v = determineAdaptiveBoost(2.0e9, 2.0e9) = 0.0.
    // R_p = 2.0e9 * 1.8 * (1 + 0.0) / (1 + 0.05) = 3.6e9 / 1.05 approx 3.42857e9
    // R_p > 1.8e9, so num_objects = 100, detail = "Very High".
    // omega_v is 0.0, so no "(Simplified...)" suffix.
    ASSERT_EQ(game.getNumObjectsToRender(), 100);
    ASSERT_EQ(game.getDetailLevel(), "Very High");
    ASSERT_NEAR(game.getDVVEParams().omega_v, 0.0, 1e-6);
    ASSERT_NEAR(game.getCurrentRp(), (2.0e9 * 1.8 * 1.0) / 1.05, 1e3);
}

TEST_F(GameDVVETest, MediumPCoreLowComplexity) {
    game.setMockPCore(1.0e9);
    game.setVisualComplexity(0.8); // Low complexity
    game.update(0.016);

    // P_core = 1.0e9. target_p_core_for_max_quality_ = 2.0e9.
    // adaptive omega_v = determineAdaptiveBoost(1.0e9, 2.0e9) results in 0.1.
    // (Since 1.0e9 is not < 2.0e9 * 0.5, but 1.0e9 < 2.0e9 * 0.75 is true)
    // R_p = 1.0e9 * 0.8 * (1 + 0.1) / (1 + 0.05) = 0.88e9 / 1.05 approx 0.838095e9
    // R_p is > 0.8e9 and <= 1.2e9, so num_objects = 50, detail = "Medium".
    // omega_v is 0.1, so no "(Simplified...)" suffix (suffix is if omega_v > 0.15).
    // The suffix logic in Game.cpp (Step 3 refinement) is:
    // if (dvve_params_.omega_v > 0.15) { detail_level_ += " (Simplified due to low P_core)"; }
    // else if (dvve_params_.omega_v > 0.05) { detail_level_ += " (Slightly Simplified)"; }
    ASSERT_EQ(game.getNumObjectsToRender(), 50);
    ASSERT_EQ(game.getDetailLevel(), "Medium (Slightly Simplified)");
    ASSERT_NEAR(game.getDVVEParams().omega_v, 0.1, 1e-6);
    ASSERT_NEAR(game.getCurrentRp(), (1.0e9 * 0.8 * 1.1) / 1.05, 1e3);
}

TEST_F(GameDVVETest, LowPCoreMediumComplexity) {
    game.setMockPCore(0.5e9);
    game.setVisualComplexity(1.0); // Medium complexity
    game.update(0.016);

    // P_core = 0.5e9. target_p_core_for_max_quality_ = 2.0e9.
    // adaptive omega_v = determineAdaptiveBoost(0.5e9, 2.0e9) results in 0.2.
    // (Since 0.5e9 < 2.0e9 * 0.5 is true)
    // R_p = 0.5e9 * 1.0 * (1 + 0.2) / (1 + 0.05) = 0.6e9 / 1.05 approx 0.571428e9
    // R_p < 0.8e9, so num_objects = 30, detail = "Low".
    // omega_v is 0.2 (> 0.15), so suffix " (Simplified due to low P_core)" is added.
    ASSERT_EQ(game.getNumObjectsToRender(), 30);
    ASSERT_EQ(game.getDetailLevel(), "Low (Simplified due to low P_core)");
    ASSERT_NEAR(game.getDVVEParams().omega_v, 0.2, 1e-6);
    ASSERT_NEAR(game.getCurrentRp(), (0.5e9 * 1.0 * 1.2) / 1.05, 1e3);
}

TEST_F(GameDVVETest, VeryLowPCoreCausesMaxSimplification) {
    game.setMockPCore(0.1e9);
    game.setVisualComplexity(1.0);
    game.update(0.016);

    // P_core = 0.1e9. target_p_core_for_max_quality_ = 2.0e9.
    // adaptive omega_v = determineAdaptiveBoost(0.1e9, 2.0e9) = 0.2.
    // R_p = 0.1e9 * 1.0 * (1 + 0.2) / (1 + 0.05) = 0.12e9 / 1.05 approx 0.114285e9
    // R_p < 0.8e9, so num_objects = 30, detail = "Low".
    // omega_v is 0.2 (> 0.15), so suffix " (Simplified due to low P_core)" is added.
    ASSERT_EQ(game.getNumObjectsToRender(), 30);
    ASSERT_EQ(game.getDetailLevel(), "Low (Simplified due to low P_core)");
    ASSERT_NEAR(game.getDVVEParams().omega_v, 0.2, 1e-6);
    ASSERT_NEAR(game.getCurrentRp(), (0.1e9 * 1.0 * 1.2) / 1.05, 1e3);
}
