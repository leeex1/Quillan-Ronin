#ifndef AI_INTERFACE_H
#define AI_INTERFACE_H

#include <string>
#include <vector>
#include <iostream> // For basic logging

/**
 * @class AIInterface
 * @brief Interface for AI decision-making and processing.
 * This class defines the contract for AI components, including NLP
 * and decision tree functionalities.
 */
class AIInterface {
public:
    virtual ~AIInterface() = default;

    /**
     * @brief Processes a natural language query.
     * @param query The input string from the user.
     * @return A string representing the AI's response.
     * (Placeholder for more complex NLP logic)
     */
    virtual std::string processNLPQuery(const std::string& query) = 0;

    /**
     * @brief Evaluates a decision based on given context or input.
     * @param context A string or structured data representing the decision context.
     * @return A string representing the decision made.
     */
    virtual std::string evaluateDecision(const std::string& context) = 0;

    /**
     * @brief Logs a message.
     * @param message The message to log.
     */
    void log(const std::string& message) const {
        std::cout << "[AIInterface LOG]: " << message << std::endl;
    }

    /**
     * @brief Logs an error message.
     * @param errorMessage The error message to log.
     */
    void logError(const std::string& errorMessage) const {
        std::cerr << "[AIInterface ERROR]: " << errorMessage << std::endl;
    }
};

#endif // AI_INTERFACE_H
