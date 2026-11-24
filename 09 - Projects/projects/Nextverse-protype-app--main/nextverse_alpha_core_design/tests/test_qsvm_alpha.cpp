#include "gtest/gtest.h"
#include "qsvm_alpha.hpp"
#include "qcc_alpha.hpp" // QSVM depends on QCC
#include "quantum_common_alpha.hpp"
#include <cmath> // For std::pow
#include <atomic> // For std::atomic in task tests
#include <thread> // For std::this_thread::sleep_for
#include <chrono> // For std::chrono::milliseconds
#include <functional> // For std::function if not already pulled in by qsvm_alpha.hpp
#include <optional> // For std::optional

// Define a test suite for QuantumSuperVM (QSVM)
class QSVMAlphaTests : public ::testing::Test {
protected:
    // Correctly scope JQLDParams as it's nested in QuantumComputationalCore
    NextVerse::QuantumComputationalCore::JQLDParams defaultJqldParams;
    NextVerse::QuantumComputationalCore qcc; // Will be initialized with default JQLDParams

    // Correctly scope LVVMParams as it's nested in QuantumSuperVM
    NextVerse::QuantumSuperVM::LVVMParams defaultLvvmsParams;

    QSVMAlphaTests() : qcc(defaultJqldParams) {} // Initialize QCC with default params for each test

    // You can add setup/teardown logic here if needed
};

// Test Case 1: DetailedLVVM_DefaultParams
// Tests getAmplifiedPerformance() (which calls calculateLVVM_detailed()) with default QCC and LVVM parameters.
TEST_F(QSVMAlphaTests, DetailedLVVM_DefaultParams) {
    NextVerse::QuantumSuperVM qsvm(qcc, defaultLvvmsParams, 0); // Instance ID 0

    // Expected Q from QCC (default JQLDParams):
    // Falls back to simplifiedExponent: C * 2^effective_Nq_exponent_factor
    // Q = 1.1e9 * std::pow(2.0, 10.0) = 1.1e9 * 1024.0 = 1.1264e12
    double expected_q_performance = 1.1e9 * 1024.0;

    // Default LVVMParams:
    // replication_factor_Rvm = 0.98
    // adaptive_boost_psi_vm = 0.15
    // virtualization_overhead_mu_vm = 0.03
    // sync_cost_tau_vm = 0.02

    // Numerator = R_vm + psi_vm * (1 - mu_vm)
    //           = 0.98 + 0.15 * (1.0 - 0.03)
    //           = 0.98 + 0.15 * 0.97
    //           = 0.98 + 0.1455 = 1.1255
    double numerator = defaultLvvmsParams.replication_factor_Rvm +
                       defaultLvvmsParams.adaptive_boost_psi_vm * (1.0 - defaultLvvmsParams.virtualization_overhead_mu_vm);

    // Denominator = 1 + tau_vm
    //             = 1.0 + 0.02 = 1.02
    double denominator = 1.0 + defaultLvvmsParams.sync_cost_tau_vm;

    double expected_lvvm_performance = expected_q_performance * (numerator / denominator);
    // expected_lvvm_performance = 1.1264e12 * (1.1255 / 1.02)
    //                           = 1.1264e12 * 1.1034313725...
    //                           = 1.2428376e12 (approx)
    std::optional<double> result = qsvm.getAmplifiedPerformance();
    ASSERT_TRUE(result.has_value());
    ASSERT_NEAR(result.value(), expected_lvvm_performance, 1e7); // Allow some tolerance
}

// Test Case 2: DetailedLVVM_NearZeroDenominator
// Tests getAmplifiedPerformance() when sync_cost_tau_vm makes the denominator very close to zero.
// Expects 0.0 as output due to error handling in calculateLVVM_detailed.
TEST_F(QSVMAlphaTests, DetailedLVVM_NearZeroDenominator) {
    NextVerse::QuantumSuperVM::LVVMParams customLvvmsParams; // Correctly scope LVVMParams
    customLvvmsParams.sync_cost_tau_vm = -0.9999999999; // Very close to -1.0

    // The constructor should throw for tau_vm that makes the denominator effectively zero or negative.
    // This test becomes similar to ConstructorThrowsOnInvalidTauVm.
    EXPECT_THROW(NextVerse::QuantumSuperVM qsvm(qcc, customLvvmsParams, 1), std::invalid_argument);

    // Test with tau_vm that makes denominator exactly zero
    customLvvmsParams.sync_cost_tau_vm = -1.0;
    EXPECT_THROW(NextVerse::QuantumSuperVM qsvm2(qcc, customLvvmsParams, 2), std::invalid_argument);
}

// Test Case 3: SimplifiedLVVM_DefaultParams
TEST_F(QSVMAlphaTests, SimplifiedLVVM_DefaultParams) {
    NextVerse::QuantumSuperVM qsvm(qcc, defaultLvvmsParams, 0);
    double expected_q_performance = 1.1e9 * 1024.0; // From QCC default
    double expected_lvvm_simplified = expected_q_performance * defaultLvvmsParams.replication_factor_Rvm; // defaultLvvmsParams is already correctly scoped
    // = 1.1264e12 * 0.98 = 1.103872e12
    std::optional<double> result = qsvm.calculateLVVM_simplified();
    ASSERT_TRUE(result.has_value());
    ASSERT_NEAR(result.value(), expected_lvvm_simplified, 1e7);
}

// Test Case 4: VMSlimLVVM_DefaultParams
TEST_F(QSVMAlphaTests, VMSlimLVVM_DefaultParams) {
    NextVerse::QuantumSuperVM qsvm(qcc, defaultLvvmsParams, 0);
    double expected_q_performance = 1.1e9 * 1024.0; // From QCC default

    // defaultLvvmsParams.swap_usage_percent_slim is now 10.0 (meaning 10%) due to changes in qsvm_alpha.hpp.

    double R_vm = defaultLvvmsParams.replication_factor_Rvm; // 0.98
    double swap_percentage_value = defaultLvvmsParams.swap_usage_percent_slim; // This is now 10.0

    // hardware_penalty = 0.01 * swap_percentage_value
    //                  = 0.01 * 10.0 = 0.1
    double hardware_penalty = 0.01 * swap_percentage_value;
    // vm_slim_factor = 1.0 - hardware_penalty
    //                = 1.0 - 0.1 = 0.9
    double vm_slim_factor = 1.0 - hardware_penalty;

    double expected_vm_slim = 0.85 * expected_q_performance * std::sqrt(R_vm) * vm_slim_factor;
    // = 0.85 * 1.1264e12 * std::sqrt(0.98) * 0.9
    // = 0.85 * 1.1264e12 * 0.98994949366 * 0.9
    // = 8.5374005e11 (approx 853,740,050,961.88)
    std::optional<double> result = qsvm.calculateLVVM_VMSlim();
    ASSERT_TRUE(result.has_value());
    ASSERT_NEAR(result.value(), expected_vm_slim, 1e6); // Tolerance 1 million
}

// Test Case 5: VMSlimLVVM_HighSwap (Negative result before clamp)
TEST_F(QSVMAlphaTests, VMSlimLVVM_HighSwap) {
    NextVerse::QuantumSuperVM::LVVMParams customLvvmsParams = defaultLvvmsParams; // Correctly scope LVVMParams
    customLvvmsParams.swap_usage_percent_slim = 150.0; // 150% swap should make (1 - penalty) negative

    NextVerse::QuantumSuperVM qsvm(qcc, customLvvmsParams, 0);
    // hardware_penalty = 0.01 * 150.0 = 1.5
    // vm_slim_factor = 1.0 - 1.5 = -0.5
    // Result will be non-positive, so calculateLVVM_VMSlim should return std::nullopt.
    std::optional<double> result = qsvm.calculateLVVM_VMSlim();
    ASSERT_FALSE(result.has_value());
}

// Test Case 6: SubmitTaskAndCheckCompletion
// Tests the task submission, status checking, and execution mechanism of QSVM.
TEST_F(QSVMAlphaTests, SubmitTaskAndCheckCompletion) {
    // Instance ID 3 for this test to avoid interference if logs were shared globally (not an issue here)
    NextVerse::QuantumSuperVM qsvm(qcc, defaultLvvmsParams, 3);

    std::atomic<bool> task1_executed_flag(false);
    std::atomic<int> task2_counter(0);

    // Submit first task
    std::string task_id1 = qsvm.submitTask([&task1_executed_flag]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(10)); // Simulate some work
        task1_executed_flag.store(true);
    });

    ASSERT_FALSE(task_id1.empty());
    ASSERT_EQ(task_id1.rfind("QSVM_Task_", 0), 0); // Check if starts with "QSVM_Task_"

    // Submit second task
    std::string task_id2 = qsvm.submitTask([&task2_counter]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(20)); // Simulate some work
        task2_counter.fetch_add(1);
    });
    ASSERT_FALSE(task_id2.empty());
    ASSERT_NE(task_id1, task_id2);

    // Allow time for tasks to process. Max simulated delay in taskProcessorLoop is ~110ms.
    // Adding a buffer for scheduling and execution.
    std::this_thread::sleep_for(std::chrono::milliseconds(350)); // Increased from 300 to be safer

    ASSERT_EQ(qsvm.getTaskStatus(task_id1), NextVerse::QuantumTaskStatusAlpha::COMPLETED);
    ASSERT_TRUE(task1_executed_flag.load());

    ASSERT_EQ(qsvm.getTaskStatus(task_id2), NextVerse::QuantumTaskStatusAlpha::COMPLETED);
    ASSERT_EQ(task2_counter.load(), 1);

    // Test getting status for a non-existent task
    NextVerse::QuantumTaskStatusAlpha status_non_existent = qsvm.getTaskStatus("non_existent_qsvm_task_id_456");
    ASSERT_EQ(status_non_existent, NextVerse::QuantumTaskStatusAlpha::UNKNOWN);
}


// Test Case 7: ManageVMInstancesStub
// Tests the stubbed manageVMInstances() method.
TEST_F(QSVMAlphaTests, ManageVMInstancesStub) {
    NextVerse::QuantumSuperVM qsvm(qcc, defaultLvvmsParams, 0); // Instance ID 0
    // Call the void method. The test passes if it doesn't crash or throw unhandled exceptions.
    ASSERT_NO_THROW(qsvm.manageVMInstances(5));
}

// Test Case 8: SyncWithOtherVMStub
// Tests the stubbed syncWithOtherVM() method.
TEST_F(QSVMAlphaTests, SyncWithOtherVMStub) {
    NextVerse::QuantumSuperVM qsvm(qcc, defaultLvvmsParams, 0); // Instance ID 0
    ASSERT_TRUE(qsvm.syncWithOtherVM(1, "test_sync_message_content"));
}


// int main(int argc, char **argv) {
//     ::testing::InitGoogleTest(&argc, argv);
//     return RUN_ALL_TESTS();
// }

// Test Case 9: ConstructorThrowsOnInvalidRvm
// Tests that the QSVM constructor throws std::invalid_argument for negative replication_factor_Rvm.
TEST_F(QSVMAlphaTests, ConstructorThrowsOnInvalidRvm) {
    NextVerse::QuantumSuperVM::LVVMParams test_params = defaultLvvmsParams;
    test_params.replication_factor_Rvm = -0.1;
    EXPECT_THROW(NextVerse::QuantumSuperVM qsvm(qcc, test_params, 0), std::invalid_argument);
}

// Test Case 10: ConstructorThrowsOnInvalidTauVm
// Tests that the QSVM constructor throws std::invalid_argument for sync_cost_tau_vm <= -1.0.
TEST_F(QSVMAlphaTests, ConstructorThrowsOnInvalidTauVm) {
    NextVerse::QuantumSuperVM::LVVMParams test_params = defaultLvvmsParams;
    test_params.sync_cost_tau_vm = -1.0;
    EXPECT_THROW(NextVerse::QuantumSuperVM qsvm(qcc, test_params, 0), std::invalid_argument);

    test_params = defaultLvvmsParams; // Reset
    test_params.sync_cost_tau_vm = -2.0;
    EXPECT_THROW(NextVerse::QuantumSuperVM qsvm2(qcc, test_params, 0), std::invalid_argument);
}

// Test Case 11: ConstructorThrowsOnInvalidMuVm
// Tests that the QSVM constructor throws std::invalid_argument for virtualization_overhead_mu_vm out of [0.0, 1.0) range.
TEST_F(QSVMAlphaTests, ConstructorThrowsOnInvalidMuVm) {
    NextVerse::QuantumSuperVM::LVVMParams test_params = defaultLvvmsParams;
    test_params.virtualization_overhead_mu_vm = 1.0; // Exactly 1.0, should throw
    EXPECT_THROW(NextVerse::QuantumSuperVM qsvm_mu_eq_1(qcc, test_params, 0), std::invalid_argument);

    test_params = defaultLvvmsParams; // Reset
    test_params.virtualization_overhead_mu_vm = 1.1;  // Greater than 1.0
    EXPECT_THROW(NextVerse::QuantumSuperVM qsvm_mu_gt_1(qcc, test_params, 0), std::invalid_argument);

    test_params = defaultLvvmsParams; // Reset
    test_params.virtualization_overhead_mu_vm = -0.1; // Less than 0.0
    EXPECT_THROW(NextVerse::QuantumSuperVM qsvm_mu_lt_0(qcc, test_params, 0), std::invalid_argument);
}

// Test Case 12: SetterThrowsAndPreservesStateOnInvalidRvm
// Tests that setLVVMParams throws for invalid R_vm and preserves QSVM's original state.
TEST_F(QSVMAlphaTests, SetterThrowsAndPreservesStateOnInvalidRvm) {
    NextVerse::QuantumSuperVM qsvm(qcc, defaultLvvmsParams, 0);
    NextVerse::QuantumSuperVM::LVVMParams original_params = qsvm.getLVVMParams();
    NextVerse::QuantumSuperVM::LVVMParams invalid_params = original_params;

    invalid_params.replication_factor_Rvm = -0.1;
    EXPECT_THROW(qsvm.setLVVMParams(invalid_params), std::invalid_argument);
    // Verify original params are unchanged by comparing a few key members
    ASSERT_DOUBLE_EQ(qsvm.getLVVMParams().replication_factor_Rvm, original_params.replication_factor_Rvm);
    ASSERT_DOUBLE_EQ(qsvm.getLVVMParams().sync_cost_tau_vm, original_params.sync_cost_tau_vm);
}

// Test Case 13: SetterThrowsAndPreservesStateOnInvalidTauVm
// Tests that setLVVMParams throws for invalid tau_vm and preserves QSVM's original state.
TEST_F(QSVMAlphaTests, SetterThrowsAndPreservesStateOnInvalidTauVm) {
    NextVerse::QuantumSuperVM qsvm(qcc, defaultLvvmsParams, 0);
    NextVerse::QuantumSuperVM::LVVMParams original_params = qsvm.getLVVMParams();
    NextVerse::QuantumSuperVM::LVVMParams invalid_params = original_params;

    invalid_params.sync_cost_tau_vm = -1.0;
    EXPECT_THROW(qsvm.setLVVMParams(invalid_params), std::invalid_argument);
    ASSERT_DOUBLE_EQ(qsvm.getLVVMParams().sync_cost_tau_vm, original_params.sync_cost_tau_vm);

    invalid_params = original_params; // Reset
    invalid_params.sync_cost_tau_vm = -1.0 - 1e-5; // Slightly more than -1.0
     EXPECT_THROW(qsvm.setLVVMParams(invalid_params), std::invalid_argument);
    ASSERT_DOUBLE_EQ(qsvm.getLVVMParams().sync_cost_tau_vm, original_params.sync_cost_tau_vm);
}

// Test Case 14: SetterThrowsAndPreservesStateOnInvalidMuVm
// Tests that setLVVMParams throws for invalid mu_vm and preserves QSVM's original state.
TEST_F(QSVMAlphaTests, SetterThrowsAndPreservesStateOnInvalidMuVm) {
    NextVerse::QuantumSuperVM qsvm(qcc, defaultLvvmsParams, 0);
    NextVerse::QuantumSuperVM::LVVMParams original_params = qsvm.getLVVMParams();
    NextVerse::QuantumSuperVM::LVVMParams invalid_params = original_params;

    invalid_params.virtualization_overhead_mu_vm = -0.1; // Negative
    EXPECT_THROW(qsvm.setLVVMParams(invalid_params), std::invalid_argument);
    ASSERT_DOUBLE_EQ(qsvm.getLVVMParams().virtualization_overhead_mu_vm, original_params.virtualization_overhead_mu_vm);

    invalid_params = original_params; // Reset
    invalid_params.virtualization_overhead_mu_vm = 1.0; // Equal to 1.0
    EXPECT_THROW(qsvm.setLVVMParams(invalid_params), std::invalid_argument);
    ASSERT_DOUBLE_EQ(qsvm.getLVVMParams().virtualization_overhead_mu_vm, original_params.virtualization_overhead_mu_vm);

    invalid_params = original_params; // Reset
    invalid_params.virtualization_overhead_mu_vm = 1.1; // Greater than 1.0
    EXPECT_THROW(qsvm.setLVVMParams(invalid_params), std::invalid_argument);
    ASSERT_DOUBLE_EQ(qsvm.getLVVMParams().virtualization_overhead_mu_vm, original_params.virtualization_overhead_mu_vm);
}
