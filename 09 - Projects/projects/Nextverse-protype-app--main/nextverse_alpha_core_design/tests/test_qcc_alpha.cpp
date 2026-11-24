#include "gtest/gtest.h"
#include "qcc_alpha.hpp"
#include "quantum_common_alpha.hpp"
#include <cmath> // For std::pow and std::log2
#include <optional> // For std::optional

// Define a test suite for QuantumComputationalCore (QCC)
class QCCAlphaTests : public ::testing::Test {
protected:
    NextVerse::QuantumComputationalCore::JQLDParams params; // Correctly scope JQLDParams
    // You can add setup/teardown logic here if needed for multiple tests
};

// Test Case 1: DetailedJQLD_DefaultParams
// Tests calculateJQLD_detailed() with default JQLDParams.
// Expects fallback to simplifiedExponent because algo_optimizations is empty.
TEST_F(QCCAlphaTests, DetailedJQLD_DefaultParams) {
    NextVerse::QuantumComputationalCore qcc(params); // Default params

    // Expected calculation:
    // Default algo_optimizations is empty, so it falls back to calculateJQLD_simplifiedExponent().
    // C = 1.1e9 (default)
    // effective_Nq_exponent_factor = 10.0 (default)
    // Expected = C * 2^(effective_Nq_exponent_factor)
    //          = 1.1e9 * std::pow(2.0, 10.0)
    //          = 1.1e9 * 1024.0
    double expected_value = 1.1e9 * 1024.0; // 1.1264e12
    std::optional<double> result = qcc.calculateJQLD_detailed();
    ASSERT_TRUE(result.has_value());
    ASSERT_NEAR(result.value(), expected_value, 1e5); // Allow some tolerance for floating point
}

// Test Case 2: DetailedJQLD_CustomParams
// Tests calculateJQLD_detailed() with custom parameters.
TEST_F(QCCAlphaTests, DetailedJQLD_CustomParams) {
    params.classical_base_performance_C = 1.0e9;
    NextVerse::QuantumComputationalCore::JQLDParams::AlgorithmOptimization opt = {"TestOpt", 2, 0.5, 1.5}; // N_jq=2, eta=0.5, theta=1.5
    params.algo_optimizations.push_back(opt);
    params.omega_q_divisor = 2.0;

    NextVerse::QuantumComputationalCore qcc(params);

    // Expected calculation:
    // sum_exponent_terms = N_jq * eta_task * theta_j = 2 * 0.5 * 1.5 = 1.5
    // Q = C * 2^(sum_exponent_terms) / omega_q
    //   = 1.0e9 * std::pow(2.0, 1.5) / 2.0
    //   = 1.0e9 * 2.8284271247461903 / 2.0
    double expected_value = (1.0e9 * std::pow(2.0, 1.5)) / 2.0; // Approx 1.41421356e9
    std::optional<double> result = qcc.calculateJQLD_detailed();
    ASSERT_TRUE(result.has_value());
    ASSERT_NEAR(result.value(), expected_value, 1e2); // Allow some tolerance
}

// Test Case 3: DetailedJQLD_ZeroOmegaDivisor
// Tests calculateJQLD_detailed() with omega_q_divisor = 0.0.
// Expects 0.0 as output due to error handling.
TEST_F(QCCAlphaTests, DetailedJQLD_ZeroOmegaDivisor) {
    NextVerse::QuantumComputationalCore::JQLDParams::AlgorithmOptimization opt = {"TestOpt", 1, 1.0, 1.0}; // Dummy optimization
    params.algo_optimizations.push_back(opt); // Need some dummy optimization to avoid fallback path
    params.omega_q_divisor = 0.0;

    // Now, the constructor should throw, not the calculateJQLD_detailed method returning 0.
    // This test is now similar to ConstructorThrowsOnInvalidOmega.
    EXPECT_THROW(NextVerse::QuantumComputationalCore qcc(params), std::invalid_argument);
}

// Test Case 4: QliteJQLD_DefaultParams
// Tests calculateJQLD_Qlite() with default JQLDParams.
TEST_F(QCCAlphaTests, QliteJQLD_DefaultParams) {
    NextVerse::QuantumComputationalCore qcc(params); // Default params

    // Expected calculation from defaults:
    // C = 1.1e9
    // njq_lite = 10
    // psi_task_lite = 0.6
    // free_ram_gb_lite = 4.0
    // hardware_constraint_term = 0.01 + (1.0 / free_ram_gb_lite) = 0.01 + (1.0 / 4.0) = 0.01 + 0.25 = 0.26
    // main_multiplier = 1.0 - hardware_constraint_term = 1.0 - 0.26 = 0.74
    // log_term = std::log2(static_cast<double>(njq_lite) + 1.0) = std::log2(10.0 + 1.0) = std::log2(11.0) approx 3.4594316186
    // Result = 0.25 * C * log_term * psi_task_lite * main_multiplier
    //        = 0.25 * 1.1e9 * std::log2(11.0) * 0.6 * 0.74
    double C = 1.1e9;
    double njq_lite = 10.0;
    double psi_task_lite = 0.6;
    double free_ram_gb_lite = 4.0;
    double hardware_constraint = 0.01 + (1.0 / free_ram_gb_lite);
    double main_multiplier = 1.0 - hardware_constraint;
    double log_term = std::log2(njq_lite + 1.0);
    double expected_value = 0.25 * C * log_term * psi_task_lite * main_multiplier; // Approx 4.180033e8
    std::optional<double> result = qcc.calculateJQLD_Qlite();
    ASSERT_TRUE(result.has_value());
    ASSERT_NEAR(result.value(), expected_value, 1e2); // Allow some tolerance
}

// Test Case 5: QliteJQLD_LowRamConstraint
// Tests calculateJQLD_Qlite() when low RAM causes hardware_constraint_term >= 1.0.
// Expects 0.0 as output.
TEST_F(QCCAlphaTests, QliteJQLD_LowRamConstraint) {
    params.free_ram_gb_lite = 0.5; // This will make (1 - (0.01 + 1/0.5)) negative
    NextVerse::QuantumComputationalCore qcc(params);

    // hardware_constraint_term = 0.01 + (1.0 / 0.5) = 0.01 + 2.0 = 2.01
    // main_multiplier = 1.0 - 2.01 = -1.01
    // The function should now return std::nullopt if the result would be non-positive or input params invalid.
    std::optional<double> result = qcc.calculateJQLD_Qlite();
    ASSERT_FALSE(result.has_value());
}

// Test Case 6: SimplifiedJQLD_DefaultParams (Bonus, as detailed falls back to it)
TEST_F(QCCAlphaTests, SimplifiedJQLD_DefaultParams) {
    NextVerse::QuantumComputationalCore qcc(params); // Default params
    double expected_value = 1.1e9 * std::pow(2.0, 10.0); // 1.1264e12
    std::optional<double> result = qcc.calculateJQLD_simplifiedExponent();
    ASSERT_TRUE(result.has_value());
    ASSERT_NEAR(result.value(), expected_value, 1e5);
}

// Test Case 7: SubmitAndGetTask
// Tests the interaction of submitQuantumTask() and getTaskResult().
TEST_F(QCCAlphaTests, SubmitAndGetTask) {
    NextVerse::QuantumComputationalCore qcc; // Uses default JQLDParams
    NextVerse::QuantumTaskAlpha sample_task("Test Task", NextVerse::QuantumAlgorithmTypeAlpha::GENERAL_QUANTUM_OPTIMIZATION, 0.5, 50);

    std::string task_id1 = qcc.submitQuantumTask(sample_task);
    ASSERT_FALSE(task_id1.empty());
    ASSERT_EQ(task_id1.rfind("QCC_Task_", 0), 0); // Check if starts with "QCC_Task_"

    NextVerse::QuantumTaskResultAlpha result1 = qcc.getTaskResult(task_id1);
    ASSERT_TRUE(result1.success);
    ASSERT_EQ(result1.task_id, task_id1);
    ASSERT_FALSE(result1.message.empty());
    ASSERT_NE(result1.message.find("completion"), std::string::npos); // Check for "completion" in message

    // Submit a second task to check for unique IDs
    NextVerse::QuantumTaskAlpha sample_task2("Test Task 2", NextVerse::QuantumAlgorithmTypeAlpha::GROVERS_SEARCH_LIKE, 0.3, 30);
    std::string task_id2 = qcc.submitQuantumTask(sample_task2);
    ASSERT_FALSE(task_id2.empty());
    ASSERT_NE(task_id1, task_id2);

    // Test getting result for a non-existent task
    NextVerse::QuantumTaskResultAlpha result_non_existent = qcc.getTaskResult("non_existent_id_123");
    ASSERT_FALSE(result_non_existent.success);
    ASSERT_FALSE(result_non_existent.message.empty());
    ASSERT_NE(result_non_existent.message.find("not found"), std::string::npos); // Check for "not found"
}


// It's good practice to ensure main is defined for running tests,
// especially if not linking GTest::gtest_main for some reason (though we will).
// However, when linking with GTest::gtest_main, this is not strictly necessary.
// int main(int argc, char **argv) {
//     ::testing::InitGoogleTest(&argc, argv);
//     return RUN_ALL_TESTS();
// }

// Test Case 11: ConstructorThrowsOnInvalidC
// Tests that the QCC constructor throws std::invalid_argument for non-positive classical_base_performance_C.
TEST_F(QCCAlphaTests, ConstructorThrowsOnInvalidC) {
    params.classical_base_performance_C = 0.0;
    EXPECT_THROW(NextVerse::QuantumComputationalCore qcc_invalid_c(params), std::invalid_argument);

    // Re-initialize params for the next check or use a local copy if params is modified by constructor
    params = NextVerse::QuantumComputationalCore::JQLDParams(); // Reset to default
    params.classical_base_performance_C = -1.0;
    EXPECT_THROW(NextVerse::QuantumComputationalCore qcc_negative_c(params), std::invalid_argument);
}

// Test Case 12: ConstructorThrowsOnInvalidOmega
// Tests that the QCC constructor throws std::invalid_argument for zero or near-zero omega_q_divisor.
TEST_F(QCCAlphaTests, ConstructorThrowsOnInvalidOmega) {
    params.omega_q_divisor = 0.0;
    EXPECT_THROW(NextVerse::QuantumComputationalCore qcc_zero_omega(params), std::invalid_argument);

    params = NextVerse::QuantumComputationalCore::JQLDParams(); // Reset to default
    params.omega_q_divisor = 1e-10; // Very small positive, should throw
    EXPECT_THROW(NextVerse::QuantumComputationalCore qcc_tiny_pos_omega(params), std::invalid_argument);

    params = NextVerse::QuantumComputationalCore::JQLDParams(); // Reset to default
    params.omega_q_divisor = -1e-10; // Very small negative, should throw
    EXPECT_THROW(NextVerse::QuantumComputationalCore qcc_tiny_neg_omega(params), std::invalid_argument);
}

// Test Case 13: SetterThrowsAndPreservesStateOnInvalidC
// Tests that setJQLDParams throws for invalid C and preserves the QCC's original state.
TEST_F(QCCAlphaTests, SetterThrowsAndPreservesStateOnInvalidC) {
    NextVerse::QuantumComputationalCore qcc; // Initialized with default valid params
    NextVerse::QuantumComputationalCore::JQLDParams original_params = qcc.getJQLDParams();
    NextVerse::QuantumComputationalCore::JQLDParams invalid_params = original_params;

    invalid_params.classical_base_performance_C = 0.0;
    EXPECT_THROW(qcc.setJQLDParams(invalid_params), std::invalid_argument);
    // Verify original params are unchanged
    ASSERT_DOUBLE_EQ(qcc.getJQLDParams().classical_base_performance_C, original_params.classical_base_performance_C);
    ASSERT_DOUBLE_EQ(qcc.getJQLDParams().omega_q_divisor, original_params.omega_q_divisor);


    invalid_params = original_params; // Reset invalid_params
    invalid_params.classical_base_performance_C = -10.0;
    EXPECT_THROW(qcc.setJQLDParams(invalid_params), std::invalid_argument);
    ASSERT_DOUBLE_EQ(qcc.getJQLDParams().classical_base_performance_C, original_params.classical_base_performance_C);
}

// Test Case 14: SetterThrowsAndPreservesStateOnInvalidOmega
// Tests that setJQLDParams throws for invalid omega_q_divisor and preserves QCC's original state.
TEST_F(QCCAlphaTests, SetterThrowsAndPreservesStateOnInvalidOmega) {
    NextVerse::QuantumComputationalCore qcc; // Initialized with default valid params
    NextVerse::QuantumComputationalCore::JQLDParams original_params = qcc.getJQLDParams();
    NextVerse::QuantumComputationalCore::JQLDParams invalid_params = original_params;

    invalid_params.omega_q_divisor = 0.0;
    EXPECT_THROW(qcc.setJQLDParams(invalid_params), std::invalid_argument);
    // Verify original params are unchanged
    ASSERT_DOUBLE_EQ(qcc.getJQLDParams().omega_q_divisor, original_params.omega_q_divisor);
    ASSERT_DOUBLE_EQ(qcc.getJQLDParams().classical_base_performance_C, original_params.classical_base_performance_C);

    invalid_params = original_params; // Reset invalid_params
    invalid_params.omega_q_divisor = 1e-12; // Near zero
    EXPECT_THROW(qcc.setJQLDParams(invalid_params), std::invalid_argument);
    ASSERT_DOUBLE_EQ(qcc.getJQLDParams().omega_q_divisor, original_params.omega_q_divisor);
}
