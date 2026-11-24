/**
 * @file quantum_common_alpha.hpp
 * @brief Defines common data structures, enumerations, and utility types for the NextVerse Quantum Core Alpha simulation.
 * @details This includes definitions for task representation, algorithm types, task statuses, and results
 *          within the simulated quantum environment. All elements are tailored for the Alpha version's scope.
 */
#ifndef QUANTUM_COMMON_ALPHA_HPP
#define QUANTUM_COMMON_ALPHA_HPP

#include <string>
#include <vector>
#include <variant> // For QuantumTaskResultAlpha potential results

namespace NextVerse {

    /**
     * @brief Defines categories of quantum algorithms or optimization approaches for simulation purposes.
     * @details In the Alpha version, this enum is primarily used to conceptually differentiate tasks.
     * It could be used in more advanced simulations to select specific parameters for JQLD (J-Quantum Algorithmic Logic Density)
     * calculations (e.g., different eta_j or theta_j values for different algorithm types) or to influence
     * simulated processing characteristics. The current JQLD calculation in `qcc_alpha.cpp` uses a vector of
     * `AlgorithmOptimization` structs, making this enum less directly coupled to it but still useful for task definition.
     */
    enum class QuantumAlgorithmTypeAlpha {
        /** @brief A general or default type, potentially for baseline N_q style calculations or generic tasks. */
        GENERAL_QUANTUM_OPTIMIZATION,
        /** @brief Simulates tasks akin to Grover's search algorithm, known for quadratic speedup in search problems. */
        GROVERS_SEARCH_LIKE,
        /** @brief Simulates tasks related to physics simulations or optimizations, potentially Rowen's field. */
        ROWENS_PHYSICS_LIKE,
        /** @brief A placeholder for a custom-defined algorithm type, allowing for future expansion. */
        CUSTOM_ALGORITHM_TYPE_1,
        // Future developers can add more specific algorithm types here.
        // These could correspond to different sets of parameters (eta_j, theta_j, etc.)
        // if the JQLD calculation logic were to be expanded to use this enum directly for parameter lookup.
    };

    /**
     * @brief Defines the structure for a quantum task in the Alpha simulation.
     * @details This structure encapsulates the necessary information to simulate a quantum task.
     * In this Alpha version, it does not contain actual quantum circuit data. Instead, it uses
     * parameters that can influence simulated performance metrics (like JQLD if task-specific details were used)
     * or simulated outcomes and processing time within the QSVM.
     */
    struct QuantumTaskAlpha {
        /**
         * @brief Unique identifier for the task.
         * @details Typically assigned by the QCC or QSVM upon submission.
         */
        std::string task_id;
        /**
         * @brief Human-readable description of the task.
         * @details Useful for logging and identification.
         */
        std::string description;
        /**
         * @brief The type of quantum algorithm this task represents.
         * @details Can be used to conceptually categorize the task or, in more advanced simulations,
         * to apply specific performance parameters or processing logic.
         * @see QuantumAlgorithmTypeAlpha
         */
        QuantumAlgorithmTypeAlpha algorithm_type;
        /**
         * @brief A generic factor (typically 0.0 to 1.0) to simulate task difficulty or resource intensity.
         * @details Higher complexity might translate to longer simulated processing times or affect
         * simulated success rates if such logic were implemented.
         */
        double complexity_factor;
        /**
         * @brief Conceptual number of qubits required for the task simulation.
         * @details In this Alpha version, this is a simulated parameter. It might be used by
         * `QCC::manageQubitResources` or influence simulated result calculations.
         */
        int simulated_qubits_needed;

        // Example of a potential future extension:
        // /**
        //  * @brief Key or reference to input data for the task.
        //  * @details In a more complex simulation, this could point to where actual input data is stored.
        //  */
        // std::string input_data_key;

        /**
         * @brief Constructor for QuantumTaskAlpha.
         * @param desc Human-readable description of the task. Defaults to "Default Task".
         * @param type The QuantumAlgorithmTypeAlpha for the task. Defaults to GENERAL_QUANTUM_OPTIMIZATION.
         * @param complexity The complexity factor (0.0-1.0). Defaults to 0.5.
         * @param qubits The number of simulated qubits needed. Defaults to 10.
         */
        QuantumTaskAlpha(std::string desc = "Default Task",
                         QuantumAlgorithmTypeAlpha type = QuantumAlgorithmTypeAlpha::GENERAL_QUANTUM_OPTIMIZATION,
                         double complexity = 0.5,
                         int qubits = 10)
            : description(std::move(desc)),
              algorithm_type(type),
              complexity_factor(complexity),
              simulated_qubits_needed(qubits) {}
    };

    /**
     * @brief Represents the execution status of a task submitted to the QuantumSuperVM (QSVM).
     * @details This enum tracks the lifecycle of a task within the simulated QSVM environment,
     * from its submission to its completion or failure.
     */
    enum class QuantumTaskStatusAlpha {
        /** @brief Task has been submitted but is waiting in a queue to be processed. */
        PENDING,
        /** @brief Task is currently being processed by the QSVM's simulated worker thread. */
        RUNNING,
        /** @brief Task processing has finished successfully. */
        COMPLETED,
        /** @brief Task processing failed due to an error or exception during simulation. */
        FAILED,
        /** @brief Task status is unknown, or task ID was not found.
         *  @note While `UNKNOWN` is defined, `QuantumTaskStatusAlpha::FAILED` might also be used by QSVM if a task ID is not found and an explicit error state is preferred.
         */
        UNKNOWN
    };

    /**
     * @brief Structure for representing the result of a simulated quantum task in the Alpha version.
     * @details This structure conveys whether the simulated task was successful, a message about its outcome,
     * and a placeholder for any simulated output data. The actual nature of the output is
     * flexible due to the use of `std::variant`.
     */
    struct QuantumTaskResultAlpha {
        /**
         * @brief The unique ID of the task this result corresponds to.
         */
        std::string task_id;
        /**
         * @brief Flag indicating whether the simulated task execution was successful.
         * @details True if the task completed without simulated errors, false otherwise.
         */
        bool success;
        /**
         * @brief A message providing details about the task's outcome.
         * @details Can be "Simulation complete", an error message, or other relevant information.
         */
        std::string message;

        /**
         * @brief Placeholder for the simulated output data of the task.
         * @details Uses `std::variant` to allow for different types of simple simulated results
         * (e.g., a single number, a string, or a vector of numbers).
         * For the Alpha version, this allows flexibility without defining complex data structures.
         * Example: `simulated_output = 123;` or `simulated_output = "Measurement: 0110";`
         */
        std::variant<int, double, std::string, std::vector<double>> simulated_output;

        /**
         * @brief Constructor for QuantumTaskResultAlpha.
         * @param id The task ID. Defaults to an empty string.
         * @param succ Success status. Defaults to false.
         * @param msg Result message. Defaults to an empty string.
         * @details The `simulated_output` member is default-initialized by `std::variant`.
         */
        QuantumTaskResultAlpha(std::string id = "", bool succ = false, std::string msg = "")
            : task_id(std::move(id)), success(succ), message(std::move(msg)) {}
    };

} // namespace NextVerse

#endif // QUANTUM_COMMON_ALPHA_HPP
