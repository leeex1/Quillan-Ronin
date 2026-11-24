#ifndef QUANTUM_VM_MOCK_HPP
#define QUANTUM_VM_MOCK_HPP

#include <string>
#include <iostream>
#include <vector>
#include <numeric>
#include <random>
#include <thread>
#include <chrono>
#include <ctime>

#include "game_data.hpp" // Provides Ingredient struct

class QuantumVM {
public:
    QuantumVM() {
        std::cout << "QuantumVM Mock Initialized" << std::endl;
        srand(time(nullptr));
    }

    void offload_task(const std::string& task_details) {
        std::cout << "QuantumVM Mock: Generic task offloaded - " << task_details << std::endl;
        simulated_processing_delay();
        last_task_result_ = "Generic task completed for: " + task_details;
    }

    std::string get_results() {
        return last_task_result_;
    }

    std::string processQuantumCrafting(int character_id, int recipe_id, const std::vector<Ingredient>& input_ingredients) {
        std::cout << "QuantumVM Mock: Received Quantum Crafting request for char_id " << character_id
                  << ", recipe_id " << recipe_id << std::endl;
        LogIngredients(input_ingredients);

        simulated_processing_delay();

        if (recipe_id == 901) {
            int random_roll = rand() % 100;
            if (random_roll < 10) {
                last_task_result_ = "SUCCESS:minor_stability_flux:0";
                std::cout << "QuantumVM Mock: Quantum Crafting (Recipe 901) result: Minor Stability Flux." << std::endl;
            } else if (random_roll < 70) {
                int bonus_value = (rand() % 5) + 1;
                last_task_result_ = "SUCCESS:damage_bonus:" + std::to_string(bonus_value);
                std::cout << "QuantumVM Mock: Quantum Crafting (Recipe 901) result: Damage Bonus " << bonus_value << std::endl;
            } else {
                int bonus_value = (rand() % 3) + 1;
                last_task_result_ = "SUCCESS:crit_chance_bonus:" + std::to_string(bonus_value);
                 std::cout << "QuantumVM Mock: Quantum Crafting (Recipe 901) result: Crit Chance Bonus " << bonus_value << std::endl;
            }
        } else {
            last_task_result_ = "FAILURE:recipe_not_quantum_compatible";
            std::cout << "QuantumVM Mock: Recipe " << recipe_id << " is not quantum compatible." << std::endl;
        }
        return last_task_result_;
    }

private:
    std::string last_task_result_ = "No task processed yet.";

    void simulated_processing_delay() {
        std::this_thread::sleep_for(std::chrono::milliseconds(50 + (rand() % 100)));
    }

    void LogIngredients(const std::vector<Ingredient>& ingredients) {
        std::string log_str = "  Ingredients: ";
        for(const auto& ing : ingredients) {
            log_str += "Item ID " + std::to_string(ing.item_id) + " (Qty: " + std::to_string(ing.quantity) + "), ";
        }
        if (!ingredients.empty()) {
            log_str.resize(log_str.size() - 2);
        } else {
            log_str += "None";
        }
        std::cout << log_str << std::endl;
    }
};

#endif // QUANTUM_VM_MOCK_HPP
