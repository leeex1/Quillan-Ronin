#include "qsvm_alpha.hpp"
#include <iostream>
#include <stdexcept> // Now used for std::invalid_argument
#include <chrono>    // For std::this_thread::sleep_for
#include <sstream>   // For std::ostringstream
#include <random>    // For std::random_device, std::mt19937, std::uniform_real_distribution
#include <iomanip>   // For std::fixed, std::setprecision
#include <optional>  // For std::optional, std::nullopt

/**
 * @file qsvm_alpha.cpp
 * @brief Implements the Quantum Super Virtual Machine (QSVM) Alpha version.
 * @details This file contains the method definitions for the QuantumSuperVM class.
 * The QSVM simulates a layer above the QCC, applying LVVM (Layered Virtualization & Viability Metric)
 * formulas to the QCC's performance output (Q). It also includes a basic simulation of
 * task offloading and processing via a worker thread. All operations are conceptual
 * and do not represent real quantum hardware or complex VM orchestration.
 */

namespace NextVerse {

/**
 * @brief Constructor for QuantumSuperVM.
 * @param qcc Constant reference to the QuantumComputationalCore, which provides the base performance Q.
 * @param params LVVMParams struct containing parameters for the QSVM's performance formula.
 * @param instance_id Identifier for this specific VM instance, supporting concepts like the "11 Copy Principle".
 * @details Initializes the QSVM with its parameters, stores a reference to the QCC,
 * and starts a worker thread (`taskProcessorLoop`) for handling simulated task submissions.
 * Logs initialization status and parameter values, including basic validation checks.
 */
QuantumSuperVM::QuantumSuperVM(const QuantumComputationalCore& qcc, const LVVMParams& params, int instance_id)
    : qcc_(qcc), params_(params), instance_id_(instance_id), running_(false), next_task_id_(0) { // Initialize running_ to false initially

    // Perform all validations FIRST
    if (params_.replication_factor_Rvm < 0) {
        throw std::invalid_argument("Replication factor R_vm must be non-negative.");
    }
    if (params_.virtualization_overhead_mu_vm < 0) {
        throw std::invalid_argument("Virtualization overhead mu_vm must be non-negative.");
    }
    if (params_.virtualization_overhead_mu_vm >= 1.0) {
        throw std::invalid_argument("Virtualization overhead mu_vm must be less than 1.0.");
    }
    if (params_.sync_cost_tau_vm <= -1.0 + 1e-9) { // Epsilon for float comparison
        throw std::invalid_argument("Sync cost tau_vm must be greater than -1.0 to ensure positive denominator.");
    }

    // If all validations pass, then proceed with resource allocation and logging
    running_ = true; // Set running to true before starting thread
    worker_thread_ = std::thread(&QuantumSuperVM::taskProcessorLoop, this);
    std::cout << "[QSVM_Alpha Instance " << instance_id_ << "] Initialized and worker thread started." << std::endl;
    std::cout << "[QSVM_Alpha Instance " << instance_id_ << "] Params: R_vm=" << params_.replication_factor_Rvm
              << ", psi_vm=" << params_.adaptive_boost_psi_vm
              << ", mu_vm=" << params_.virtualization_overhead_mu_vm
              << ", tau_vm=" << params_.sync_cost_tau_vm
              << ", swap_slim=" << params_.swap_usage_percent_slim << std::endl;
}

/**
 * @brief Destructor for QuantumSuperVM.
 * @details Signals the worker thread to stop by setting `running_` to false.
 * Then, it waits for the `worker_thread_` to complete its execution by calling `join()`.
 * Logs shutdown status.
 */
QuantumSuperVM::~QuantumSuperVM() {
    running_ = false; // Signal worker thread to stop
    std::cout << "[QSVM_Alpha Instance " << instance_id_ << "] Shutting down worker thread..." << std::endl;
    if (worker_thread_.joinable()) {
        worker_thread_.join(); // Wait for worker thread to finish
    }
    std::cout << "[QSVM_Alpha Instance " << instance_id_ << "] Destroyed. Processed " << next_task_id_
              << " tasks via its queue during its lifetime." << std::endl;
}

/**
 * @brief Sets or updates the LVVM parameters for the QSVM.
 * @param params The new LVVMParams structure.
 */
void QuantumSuperVM::setLVVMParams(const LVVMParams& params) {
    // Validate incoming parameters before assigning them
    if (params.replication_factor_Rvm < 0) {
        throw std::invalid_argument("Input LVVMParams: Replication factor R_vm must be non-negative.");
    }
    if (params.virtualization_overhead_mu_vm < 0) {
        throw std::invalid_argument("Input LVVMParams: Virtualization overhead mu_vm must be non-negative.");
    }
    if (params.virtualization_overhead_mu_vm >= 1.0) {
        throw std::invalid_argument("Input LVVMParams: Virtualization overhead mu_vm must be less than 1.0.");
    }
    if (params.sync_cost_tau_vm <= -1.0 + 1e-9) { // Epsilon for float comparison
        throw std::invalid_argument("Input LVVMParams: Sync cost tau_vm must be greater than -1.0.");
    }

    // If validations pass, assign the new parameters
    params_ = params;
    std::cout << "[QSVM_Alpha Instance " << instance_id_ << "] LVVMParams updated by setter." << std::endl;
}

/**
 * @brief Retrieves the current LVVM parameters used by the QSVM.
 * @return A constant reference to the LVVMParams structure.
 */
const QuantumSuperVM::LVVMParams& QuantumSuperVM::getLVVMParams() const {
    return params_;
}

/**
 * @brief Gets the instance ID of this QSVM.
 * @return The integer instance ID.
 */
int QuantumSuperVM::getInstanceId() const {
    return instance_id_;
}

/**
 * @brief Calculates the effective amplified performance using the primary detailed LVVM formula.
 * @return The calculated LVVM value (VM_eff), representing simulated amplified performance.
 * This method calls `calculateLVVM_detailed()` which uses the QCC's `getEffectiveComputationalPower()`.
 * @see calculateLVVM_detailed()
 */
std::optional<double> QuantumSuperVM::getAmplifiedPerformance() const {
    std::cout << "[QSVM_Alpha Instance " << instance_id_ << "] Calculating LVVM using DETAILED formula with QCC's JQLD." << std::endl;
    return calculateLVVM_detailed();
}

/**
 * @brief Implements the detailed LVVM formula: VM_eff = Q * (R_vm + psi_vm * (1 - mu_vm)) / (1 + tau_vm).
 * @details This formula calculates the Layered Virtualization & Viability Metric (LVVM).
 * - Q: Base performance from the QuantumComputationalCore (QCC).
 * - R_vm: Replication factor.
 * - psi_vm: Adaptive boost factor.
 * - mu_vm: Virtualization overhead.
 * - tau_vm: Synchronization cost.
 * Includes checks for division by zero (denominator) and warnings for non-positive Q or numerator.
 * Logs calculation details.
 * @return The calculated LVVM value (VM_eff).
 */
std::optional<double> QuantumSuperVM::calculateLVVM_detailed() const {
    std::cout << "[QSVM_Alpha Instance " << instance_id_ << "] Called calculateLVVM_detailed()." << std::endl;

    std::optional<double> q_performance_opt = qcc_.getEffectiveComputationalPower();
    if (!q_performance_opt.has_value()) {
        std::cerr << "[QSVM_Alpha Instance " << instance_id_ << "] ERROR in calculateLVVM_detailed: QCC performance (Q) is unavailable (std::nullopt)." << std::endl;
        return std::nullopt;
    }
    double q_performance = q_performance_opt.value();

    // Numerator: R_vm + psi_vm * (1 - mu_vm)
    double numerator_boost_term = params_.adaptive_boost_psi_vm * (1.0 - params_.virtualization_overhead_mu_vm);
    double numerator = params_.replication_factor_Rvm + numerator_boost_term;
    // Denominator: 1 + tau_vm
    double denominator = 1.0 + params_.sync_cost_tau_vm;

    // Critical check: Denominator cannot be zero.
    // Constructor validation for tau_vm should prevent this, but defense-in-depth.
    if (std::abs(denominator) < 1e-9) { // Using a small epsilon for floating point comparison
        std::cerr << "[QSVM_Alpha Instance " << instance_id_ << "] ERROR in calculateLVVM_detailed: Denominator (1 + tau_vm) is " << denominator << " (tau_vm: "
                  << params_.sync_cost_tau_vm << "), which is zero or too close to zero." << std::endl;
        return std::nullopt;
    }

    // Informational warnings for unusual inputs that don't stop calculation but might indicate issues.
    if (q_performance <= 0 && !(std::abs(q_performance) < 1e-9)) { // Log warning if Q is negative, allow Q=0
        std::cerr << "[QSVM_Alpha Instance " << instance_id_ << "] Warning in calculateLVVM_detailed: QCC performance (Q) is " << q_performance
                  << ", which is non-positive. The LVVM result may be non-positive or zero." << std::endl;
    }
     if (numerator < 0) {
        std::cerr << "[QSVM_Alpha Instance " << instance_id_ << "] Warning in calculateLVVM_detailed: LVVM formula numerator (" << numerator
                  << ") is negative. This might be due to R_vm (" << params_.replication_factor_Rvm
                  << ") + psi_vm (" << params_.adaptive_boost_psi_vm << ") * (1 - mu_vm (" << params_.virtualization_overhead_mu_vm
                  << ")). The LVVM result may be negative." << std::endl;
    }

    double result = q_performance * (numerator / denominator);
    std::cout << "[QSVM_Alpha Instance " << instance_id_ << "] LVVM_detailed: Q_from_QCC=" << std::fixed << std::setprecision(2) << q_performance
              << ", R_vm=" << params_.replication_factor_Rvm
              << ", psi_vm=" << params_.adaptive_boost_psi_vm
              << ", mu_vm=" << params_.virtualization_overhead_mu_vm
              << ", tau_vm=" << params_.sync_cost_tau_vm
              << ", Numerator=" << numerator
              << ", Denominator=" << denominator
              << ", Result (VM_eff)=" << std::fixed << std::setprecision(2) << result << std::endl;
    return result;
}

/**
 * @brief Implements a simplified LVVM formula: VM_eff = Q * R_vm.
 * @details This is an alternative, much less granular calculation for LVVM.
 * - Q: Base performance from the QCC.
 * - R_vm: Replication factor.
 * Logs calculation details.
 * @return The calculated LVVM value.
 */
std::optional<double> QuantumSuperVM::calculateLVVM_simplified() const {
    std::cout << "[QSVM_Alpha Instance " << instance_id_ << "] Called calculateLVVM_simplified()." << std::endl;
    std::optional<double> q_performance_opt = qcc_.getEffectiveComputationalPower();
    if (!q_performance_opt.has_value()) {
        std::cerr << "[QSVM_Alpha Instance " << instance_id_ << "] ERROR in calculateLVVM_simplified: QCC performance (Q) is unavailable (std::nullopt)." << std::endl;
        return std::nullopt;
    }
    double q_performance = q_performance_opt.value();

    if (q_performance <= 0 && !(std::abs(q_performance) < 1e-9)) {
        std::cerr << "[QSVM_Alpha Instance " << instance_id_ << "] Warning in calculateLVVM_simplified: QCC performance (Q) is " << q_performance
                  << ", which is non-positive. The LVVM result may be non-positive or zero." << std::endl;
    }
    double result = q_performance * params_.replication_factor_Rvm;
    std::cout << "[QSVM_Alpha Instance " << instance_id_ << "] LVVM_simplified: Q_from_QCC=" << std::fixed << std::setprecision(2) << q_performance
              << ", R_vm=" << params_.replication_factor_Rvm
              << ", Result (VM_eff)=" << std::fixed << std::setprecision(2) << result << std::endl;
    return result;
}

/**
 * @brief Implements the VM_slim LVVM variant from "CrashOverrideX":
 * VM_slim = 0.85 * Q * sqrt(R_vm) * (1 - (0.01 * swap_usage_percent_slim)).
 * @details This formula is specific and includes:
 * - Q: Base performance from the QCC.
 * - R_vm: Replication factor (square root is taken).
 * - swap_usage_percent_slim: Penalty from swap memory usage.
 * Includes checks for non-negative R_vm (for sqrt) and warnings for non-positive Q or unusual swap usage.
 * Logs calculation details.
 * @return The calculated VM_slim value, or 0.0 if R_vm is negative or result is non-positive.
 */
std::optional<double> QuantumSuperVM::calculateLVVM_VMSlim() const {
    std::cout << "[QSVM_Alpha Instance " << instance_id_ << "] Called calculateLVVM_VMSlim()." << std::endl;
    std::optional<double> q_performance_opt = qcc_.getEffectiveComputationalPower();
    if (!q_performance_opt.has_value()) {
        std::cerr << "[QSVM_Alpha Instance " << instance_id_ << "] ERROR in calculateLVVM_VMSlim: QCC performance (Q) is unavailable (std::nullopt)." << std::endl;
        return std::nullopt;
    }
    double q_performance = q_performance_opt.value();

    // Constructor validation for R_vm should prevent this, but defense-in-depth.
    if (params_.replication_factor_Rvm < 0) {
        std::cerr << "[QSVM_Alpha Instance " << instance_id_ << "] ERROR in calculateLVVM_VMSlim: Replication factor R_vm (replication_factor_Rvm) is " << params_.replication_factor_Rvm
                  << ", but it must be non-negative for std::sqrt." << std::endl;
        return std::nullopt;
    }
    if (q_performance <= 0 && !(std::abs(q_performance) < 1e-9)) {
        std::cerr << "[QSVM_Alpha Instance " << instance_id_ << "] Warning in calculateLVVM_VMSlim: QCC performance (Q) is " << q_performance
                  << ", which is non-positive. The VM_slim result may be non-positive or zero." << std::endl;
    }

    // Hardware penalty factor: 0.01 * swap_usage_percent_slim
    double hardware_penalty_raw_percentage = 0.01 * params_.swap_usage_percent_slim;
    // The term (1 - (0.01 * swap_usage_percent_slim))
    double vm_slim_factor = 1.0 - hardware_penalty_raw_percentage;

    // If vm_slim_factor is not positive, the resulting performance metric is considered invalid/unavailable for this model.
    if (vm_slim_factor <= 0) {
        std::cerr << "[QSVM_Alpha Instance " << instance_id_ << "] Warning in calculateLVVM_VMSlim: The VM_slim_factor (1 - (0.01 * swap_usage_percent_slim)) is " << vm_slim_factor
                  << ", which is non-positive. This implies swap_usage_percent_slim (" << params_.swap_usage_percent_slim
                  << "%) is >= 100%. The VM_slim result is considered invalid." << std::endl;
        return std::nullopt;
    }
    // Also, if q_performance is non-positive, the result would be non-positive.
    if (q_performance <=0) {
         return std::nullopt;
    }


    double sqrt_R_vm = std::sqrt(params_.replication_factor_Rvm);
    double result = 0.85 * q_performance * sqrt_R_vm * vm_slim_factor;

    std::cout << "[QSVM_Alpha Instance " << instance_id_ << "] LVVM_VMSlim: Q_from_QCC=" << std::fixed << std::setprecision(2) << q_performance
              << ", R_vm=" << params_.replication_factor_Rvm
              << ", sqrt(R_vm)=" << sqrt_R_vm
              << ", swap_usage_slim_percent=" << params_.swap_usage_percent_slim
              << ", vm_slim_factor (1 - penalty)=" << vm_slim_factor
              << ", Result (VM_eff)=" << std::fixed << std::setprecision(2) << result << std::endl;

    // Ensure result is positive. If not, it's considered an invalid/unavailable metric.
    // This also implicitly handles cases where q_performance might have been zero or negative if not caught earlier.
    return result > 0 ? std::make_optional(result) : std::nullopt;
}

/**
 * @brief Simulates submitting a task to the QSVM for asynchronous processing.
 * @details Generates a unique task ID, adds the task (a `std::function<void()>`) to a queue,
 * and sets its initial status to PENDING. The task is later picked up by the `taskProcessorLoop`.
 * Thread safety for queue and status map is handled by `queue_mutex_`.
 * @param task_function The function to be executed as the task.
 * @return A string representing the unique ID of the submitted task.
 */
std::string QuantumSuperVM::submitTask(std::function<void()> task_function) {
    std::ostringstream oss;
    // Create a unique task ID, incorporating instance_id for clarity if multiple QSVMs exist.
    oss << "QSVM_Task_" << instance_id_ << "_" << next_task_id_++;
    std::string task_id_str = oss.str();

    { // Lock scope for thread-safe access to task_queue_ and task_statuses_
        std::lock_guard<std::mutex> lock(queue_mutex_);
        task_queue_.push(std::make_pair(task_id_str, task_function));
        task_statuses_[task_id_str] = QuantumTaskStatusAlpha::PENDING; // Set initial status
        std::cout << "[QSVM_Alpha Instance " << instance_id_ << "] Task ID: " << task_id_str
                  << " accepted and enqueued. Current queue size: " << task_queue_.size() << std::endl;
    }
    // Note: No notification to worker thread is strictly needed here as it polls.
    // For a more advanced system, a condition_variable could be used to wake the worker.
    return task_id_str;
}

/**
 * @brief Retrieves the status of a task previously submitted to the QSVM.
 * @details Accesses the `task_statuses_` map in a thread-safe manner to get the current status.
 * @param task_id The unique ID of the task.
 * @return The `QuantumTaskStatusAlpha` of the task. Returns `QuantumTaskStatusAlpha::FAILED`
 * (or a more specific `NOT_FOUND` status if defined) if the task ID is not found.
 * @note The `queue_mutex_` is marked `mutable` to allow locking in a const method.
 */
QuantumTaskStatusAlpha QuantumSuperVM::getTaskStatus(const std::string& task_id) const {
    std::lock_guard<std::mutex> lock(queue_mutex_); // queue_mutex_ is mutable for this const method
    auto it = task_statuses_.find(task_id);
    if (it != task_statuses_.end()) {
        return it->second; // Return current status
    }
    // If task_id is not found, log it and return UNKNOWN status.
    std::cerr << "[QSVM_Alpha Instance " << instance_id_ << "] INFO: Task ID '" << task_id
              << "' not found in task_statuses_ map. Returning UNKNOWN status." << std::endl;
    return QuantumTaskStatusAlpha::UNKNOWN;
}

/**
 * @brief The main loop for the worker thread that processes tasks from the queue.
 * @details This loop runs as long as the `running_` flag is true.
 * It attempts to pop tasks from the `task_queue_` (thread-safely).
 * If a task is found, it simulates its execution:
 * - Sets status to RUNNING.
 * - Simulates work with a random delay.
 * - Executes the task function.
 * - Sets status to COMPLETED on success, FAILED on exception.
 * If the queue is empty, the thread sleeps for a short duration before polling again.
 * This is a simplified model of an asynchronous task processor.
 */
void QuantumSuperVM::taskProcessorLoop() {
    std::cout << "[QSVM_Alpha Instance " << instance_id_ << "] Task processor loop started." << std::endl;
    std::random_device rd; // Source of non-deterministic random numbers
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    // Simulation: Task "complexity" factor, influences simulated processing time.
    // Generates random floating-point values between 0.1 (inclusive) and 1.0 (exclusive).
    std::uniform_real_distribution<> distrib_complexity(0.1, 1.0);

    while (running_) {
        std::pair<std::string, std::function<void()>> current_task_item;
        bool task_found = false;

        { // Lock scope for thread-safe access to task_queue_ and task_statuses_
            std::lock_guard<std::mutex> lock(queue_mutex_);
            if (!task_queue_.empty()) {
                current_task_item = task_queue_.front();
                task_queue_.pop();
                task_statuses_[current_task_item.first] = QuantumTaskStatusAlpha::RUNNING;
                task_found = true;
            }
        } // Mutex released here

        if (task_found) {
            std::cout << "[QSVM_Alpha Instance " << instance_id_ << "] Began processing offloaded Task ID: "
                      << current_task_item.first << "." << std::endl;
            try {
                // Simulate task execution time based on a random "complexity".
                // This is a placeholder for actual computational work.
                double simulated_complexity_value = distrib_complexity(gen);
                // sleep_duration_ms: Base 10ms + complexity-scaled duration (10ms to 100ms based on complexity)
                // Total simulated time: 20ms to 110ms.
                int sleep_duration_ms = 10 + static_cast<int>(simulated_complexity_value * 100);
                std::cout << "[QSVM_Alpha Instance " << instance_id_ << "] Task ID: " << current_task_item.first
                          << " (simulated complexity factor: " << std::fixed << std::setprecision(2) << simulated_complexity_value
                          << ") will run for approx. " << sleep_duration_ms << "ms (simulated delay)." << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(sleep_duration_ms));

                current_task_item.second(); // Execute the actual task function provided by the client

                { // Lock scope for updating status
                    std::lock_guard<std::mutex> lock(queue_mutex_);
                    task_statuses_[current_task_item.first] = QuantumTaskStatusAlpha::COMPLETED;
                }
                std::cout << "[QSVM_Alpha Instance " << instance_id_ << "] Successfully completed offloaded Task ID: "
                          << current_task_item.first << "." << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "[QSVM_Alpha Instance " << instance_id_ << "] EXCEPTION during task "
                          << current_task_item.first << ": " << e.what() << std::endl;
                { // Lock scope for updating status
                    std::lock_guard<std::mutex> lock(queue_mutex_);
                    task_statuses_[current_task_item.first] = QuantumTaskStatusAlpha::FAILED;
                }
            } catch (...) { // Catch-all for non-standard exceptions
                std::cerr << "[QSVM_Alpha Instance " << instance_id_ << "] UNKNOWN EXCEPTION during task "
                          << current_task_item.first << "." << std::endl;
                 { // Lock scope for updating status
                    std::lock_guard<std::mutex> lock(queue_mutex_);
                    task_statuses_[current_task_item.first] = QuantumTaskStatusAlpha::FAILED;
                }
            }
        } else {
            // If no task was found, wait a bit before checking the queue again to avoid busy-waiting.
            // Poll frequency when idle: 100 milliseconds.
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
    std::cout << "[QSVM_Alpha Instance " << instance_id_ << "] Task processor loop stopped." << std::endl;
}

/**
 * @brief Stub for managing multiple VM instances.
 * @details This is a placeholder function for the Alpha version.
 * It only logs the conceptual action of managing a number of VM instances.
 * In a real system, this could involve complex orchestration.
 * @param num_instances The number of VM instances to conceptually manage.
 * @note This is a simulation stub.
 */
void QuantumSuperVM::manageVMInstances(int num_instances) {
    std::cout << "[QSVM_Alpha Instance " << instance_id_ << "] STUB: Conceptual management of " << num_instances
              << " VM instances. (Simulated operation - no actual instance management)" << std::endl;
}

/**
 * @brief Stub for inter-VM synchronization.
 * @details This is a placeholder function for the Alpha version.
 * It only logs the conceptual action of synchronizing with another VM.
 * @param target_vm_id The ID of the target VM for conceptual synchronization.
 * @param message The message to be conceptually sent.
 * @return Always returns true in this simulated Alpha version.
 * @note This is a simulation stub.
 */
bool QuantumSuperVM::syncWithOtherVM(int target_vm_id, const std::string& message) {
    std::cout << "[QSVM_Alpha Instance " << instance_id_ << "] STUB: Conceptual sync with VM " << target_vm_id
              << " with message: \"" << message << "\". (Simulated operation - always returns true)" << std::endl;
    return true;
}

} // namespace NextVerse
