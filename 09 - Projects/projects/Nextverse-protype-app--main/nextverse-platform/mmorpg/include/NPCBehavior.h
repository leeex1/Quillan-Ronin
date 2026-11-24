#ifndef NPC_BEHAVIOR_H
#define NPC_BEHAVIOR_H

#include <string>
#include <iostream> // For logging
#include <utility>  // For std::move

/**
 * @struct NPCState
 * @brief Holds basic state for an NPC.
 */
struct NPCState {
    int npcId;                          ///< Unique identifier for the NPC.
    std::string name;                   ///< NPC's name.
    float x = 0.0f;                     ///< Current x-coordinate.
    float y = 0.0f;                     ///< Current y-coordinate.
    float z = 0.0f;                     ///< Current z-coordinate.
    std::string currentBehavior = "idle"; ///< String describing the current behavior state.

    /**
     * @brief Default constructor.
     */
    NPCState() : npcId(0) {}

    /**
     * @brief Constructor for NPCState.
     * @param id Unique ID for the NPC.
     * @param _name Name of the NPC.
     * @param _x Initial x-coordinate.
     * @param _y Initial y-coordinate.
     * @param _z Initial z-coordinate.
     */
    NPCState(int id, std::string _name, float _x = 0.0f, float _y = 0.0f, float _z = 0.0f)
        : npcId(id), name(std::move(_name)), x(_x), y(_y), z(_z) {}
};

/**
 * @class NPCBehavior
 * @brief Interface for defining NPC behaviors and AI.
 *
 * Derived classes will implement specific behaviors (e.g., patrolling, attacking, idling).
 */
class NPCBehavior {
public:
    virtual ~NPCBehavior() = default;

    /**
     * @brief Updates the NPC's state based on its behavior logic.
     * This is a pure virtual function to be implemented by derived classes.
     * @param currentState The current state of the NPC.
     * @param deltaTime The time elapsed since the last update, in seconds.
     * @return The updated NPCState.
     */
    virtual NPCState update(const NPCState& currentState, double deltaTime) = 0;

    /**
     * @brief Logs a message related to NPC behavior.
     * @param message The message to log.
     */
    static void log(const std::string& message) {
        std::cout << "[NPCBehavior LOG]: " << message << std::endl;
    }

    /**
     * @brief Logs an error message related to NPC behavior.
     * @param errorMessage The error message to log.
     */
    static void logError(const std::string& errorMessage) {
        std::cerr << "[NPCBehavior ERROR]: " << errorMessage << std::endl;
    }
};

/**
 * @class SimpleIdleNPCBehavior
 * @brief A very simple NPC behavior that just stays idle.
 *
 * This class is an example implementation of the NPCBehavior interface.
 * Its `update` method returns the state largely unchanged, signifying idling.
 */
class SimpleIdleNPCBehavior : public NPCBehavior {
public:
    /**
     * @brief Updates the NPC state for an idling behavior.
     * The NPC's position remains the same. The behavior string is updated.
     * @param currentState The current state of the NPC.
     * @param deltaTime The time elapsed since the last update (unused in this simple idle behavior).
     * @return The updated NPCState, with `currentBehavior` marked as "idle_processed".
     */
    NPCState update(const NPCState& currentState, double deltaTime) override {
        NPCState nextState = currentState;
        // In a real scenario, deltaTime would be used if idling had animations or subtle movements.
        // For this placeholder, the NPC does nothing but confirm it processed.
        // Static log from base class can be used if needed:
        // NPCBehavior::log("NPC " + currentState.name + " (ID: " + std::to_string(currentState.npcId) +
        //                  ") is idling at (" + std::to_string(currentState.x) + ", " +
        //                  std::to_string(currentState.y) + ", " + std::to_string(currentState.z) +
        //                  "). Delta time: " + std::to_string(deltaTime));
        nextState.currentBehavior = "idle_processed"; // Show it was processed
        return nextState;
    }
};

#endif // NPC_BEHAVIOR_H
