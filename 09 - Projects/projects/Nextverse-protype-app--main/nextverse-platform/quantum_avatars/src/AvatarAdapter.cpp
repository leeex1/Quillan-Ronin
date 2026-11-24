#include "AvatarAdapter.h" // Assuming AvatarAdapter.h is in the include path
#include <stdexcept>      // For std::runtime_error (already in .h but good practice)

AvatarAdapter::AvatarAdapter(const std::string& avatarId) : avatarId_(avatarId) {
    log("AvatarAdapter initialized for avatar: " + avatarId_);
}

AvatarAdapter::~AvatarAdapter() {
    log("AvatarAdapter destroyed for avatar: " + avatarId_);
}

std::string AvatarAdapter::determineNewBehavior(const UserAction& action) {
    // Simplified logic for behavior adaptation
    // In a real system, this could involve more complex rules, state machines, or even ML models.
    if (action.type == "combat_move") {
        if (action.details.find("attack") != std::string::npos) {
            return "aggressive";
        } else if (action.details.find("defend") != std::string::npos) {
            return "defensive";
        } else if (action.details.find("flee") != std::string::npos) {
            return "evasive_fleeing";
        }
    } else if (action.type == "interaction") {
        if (action.details.find("friendly_greeting") != std::string::npos) {
            return "neutral_friendly";
        } else if (action.details.find("threaten") != std::string::npos) {
            return "cautious_hostile";
        } else if (action.details.find("query_trade") != std::string::npos) {
            return "neutral_trading";
        }
    } else if (action.type == "environment_change") {
        if (action.details.find("imminent_danger_aoe") != std::string::npos) {
            return "evasive_aoe";
        } else if (action.details.find("safe_zone_entered") != std::string::npos) {
            return "passive_safe";
        }
    } else if (action.type == "low_health_event") { // Example of internal state trigger
        return "defensive_wounded";
    }

    // Fallback based on current state or a general default
    if (currentBehaviorState_.find("aggressive") != std::string::npos) {
        return "aggressive_idle"; // Stay aggressive if already in that mode without new stimuli
    }
    return "idle"; // Default if no specific adaptation matched
}

std::string AvatarAdapter::adaptBehavior(const UserAction& action) {
    log("Adapting behavior for avatar " + avatarId_ + " based on action type: '" + action.type + "', details: '" + action.details + "'");
    std::string newBehavior;
    try {
        if (action.type.empty()) {
            // This check is good, but an empty type might also be a valid "general observation"
            // depending on the system design. For now, requiring a type.
            throw std::runtime_error("UserAction type cannot be empty.");
        }
        newBehavior = determineNewBehavior(action);
        if (newBehavior != currentBehaviorState_) {
            setBehaviorState(newBehavior);
            log("Behavior adapted. Old state: (was different), New state: " + currentBehaviorState_);
            return "Avatar " + avatarId_ + " new behavior: " + currentBehaviorState_;
        } else {
            log("Behavior assessed. No change from current state: " + currentBehaviorState_);
            return "Avatar " + avatarId_ + " behavior unchanged: " + currentBehaviorState_;
        }
    } catch (const std::exception& e) {
        logError("Error adapting behavior: " + std::string(e.what()));
        // Optionally, revert to a safe default behavior if the current one is problematic
        // if (currentBehaviorState_ != "passive_error_state") { // Avoid loop if already in error state
        //     setBehaviorState("passive_error_state");
        //     return "Error adapting behavior for avatar " + avatarId_ + ". Reverted to error state.";
        // }
        return "Error adapting behavior for avatar " + avatarId_ + ". Current state remains: " + currentBehaviorState_;
    }
}

bool AvatarAdapter::setBehaviorState(const std::string& newBehavior) {
    if (newBehavior.empty()) {
        logError("Cannot set an empty behavior state.");
        return false;
    }
    if (currentBehaviorState_ != newBehavior) {
        log("Behavior state changing from '" + currentBehaviorState_ + "' to '" + newBehavior + "' for avatar " + avatarId_);
        currentBehaviorState_ = newBehavior;
    } else {
        // log("Behavior state '" + newBehavior + "' is already current for avatar " + avatarId_); // Can be too verbose
    }
    return true;
}

std::string AvatarAdapter::getCurrentBehaviorState() const {
    return currentBehaviorState_;
}

void AvatarAdapter::log(const std::string& message) const {
    std::cout << "[AvatarAdapter LOG " << avatarId_ << "]: " << message << std::endl;
}

void AvatarAdapter::logError(const std::string& errorMessage) const {
    std::cerr << "[AvatarAdapter ERROR " << avatarId_ << "]: " << errorMessage << std::endl;
}
