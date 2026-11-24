#include "LoadBalancer.h" // Assuming it's in the include path
#include <algorithm> // For std::remove_if

LoadBalancer::LoadBalancer() {
    initialize();
    log("LoadBalancer initialized.");
}

LoadBalancer::~LoadBalancer() {
    cleanup();
    log("LoadBalancer destroyed.");
}

void LoadBalancer::initialize() {
    // Placeholder for any initialization logic
    log("LoadBalancer systems initialized (placeholder).");
}

void LoadBalancer::cleanup() {
    // Placeholder for any cleanup logic
    log("LoadBalancer systems cleaned up (placeholder).");
    servers_.clear();
}

void LoadBalancer::addServer(const ServerNode& server) {
    // std::lock_guard<std::mutex> lock(servers_mutex_); // If using mutex
    // Basic check to prevent duplicates by ID, can be more sophisticated
    for (const auto& s : servers_) {
        if (s.id == server.id) {
            logError("Server with ID " + server.id + " already exists.");
            return;
        }
    }
    servers_.push_back(server);
    log("Added server: " + server.id + " (" + server.address + ":" + std::to_string(server.port) + ")");
}

bool LoadBalancer::removeServer(const std::string& serverId) {
    // std::lock_guard<std::mutex> lock(servers_mutex_); // If using mutex
    auto initial_size = servers_.size();
    servers_.erase(
        std::remove_if(servers_.begin(), servers_.end(),
                       [&serverId](const ServerNode& s) { return s.id == serverId; }),
        servers_.end());

    if (servers_.size() < initial_size) {
        log("Removed server: " + serverId);
        if (current_server_index_ >= servers_.size() && !servers_.empty()) {
            current_server_index_ = 0; // Reset index if it's out of bounds
        } else if (servers_.empty()) {
            current_server_index_ = 0;
        }
        return true;
    } else {
        logError("Server not found: " + serverId);
        return false;
    }
}

ServerNode LoadBalancer::getNextServer() {
    // std::lock_guard<std::mutex> lock(servers_mutex_); // If using mutex
    if (servers_.empty()) {
        logError("No servers available.");
        // Return a default/invalid ServerNode or throw an exception
        return ServerNode("INVALID", "0.0.0.0", 0);
    }

    // Simple round-robin strategy
    ServerNode next_server = servers_[current_server_index_];
    current_server_index_ = (current_server_index_ + 1) % servers_.size();

    log("Selected server (Round-Robin): " + next_server.id);
    return next_server;
}

void LoadBalancer::log(const std::string& message) const {
    std::cout << "[LoadBalancer LOG]: " << message << std::endl;
}

void LoadBalancer::logError(const std::string& errorMessage) const {
    std::cerr << "[LoadBalancer ERROR]: " << errorMessage << std::endl;
}
