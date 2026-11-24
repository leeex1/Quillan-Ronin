#ifndef LOAD_BALANCER_H
#define LOAD_BALANCER_H

#include <iostream> // For logging
#include <string>
#include <vector>
#include <mutex>    // For std::mutex (though commented out, good for context)
#include <utility>  // For std::move

/**
 * @struct ServerNode
 * @brief Represents a server or node that can handle requests.
 * Contains identification and connection details for a server.
 */
struct ServerNode {
    std::string id;         ///< Unique identifier for the server.
    std::string address;    ///< IP address or hostname of the server.
    int port;               ///< Port number for the server.
    // Potentially other metrics like current load, capacity, etc.
    // int currentLoad = 0; ///< Optional: Current load on the server.
    // int capacity = 100;  ///< Optional: Max capacity of the server.

    /**
     * @brief Constructor for ServerNode.
     * @param _id Unique ID for the server.
     * @param _addr Address (IP/hostname) of the server.
     * @param _port Port number of the server.
     */
    ServerNode(std::string _id, std::string _addr, int _port)
        : id(std::move(_id)), address(std::move(_addr)), port(_port) {}
};

/**
 * @class LoadBalancer
 * @brief Distributes network traffic across multiple servers.
 *
 * This class manages a list of servers and provides a strategy
 * (e.g., round-robin) to select the next available server for a request.
 * (Currently placeholder logic for strategy).
 */
class LoadBalancer {
public:
    /**
     * @brief Constructor for LoadBalancer.
     */
    LoadBalancer();

    /**
     * @brief Destructor for LoadBalancer.
     */
    ~LoadBalancer();

    /**
     * @brief Adds a server to the load balancer's list.
     * @param server The ServerNode object to add.
     */
    void addServer(const ServerNode& server);

    /**
     * @brief Removes a server from the load balancer's list by its ID.
     * @param serverId The unique ID of the server to remove.
     * @return True if the server was found and removed, false otherwise.
     */
    bool removeServer(const std::string& serverId);

    /**
     * @brief Gets the next server to handle a request based on the load balancing strategy.
     * Currently implements a simple round-robin strategy.
     * @return A ServerNode object representing the selected server.
     *         Returns an invalid ServerNode if no servers are available.
     */
    ServerNode getNextServer();

    // Basic Logging
    /**
     * @brief Logs a general message.
     * @param message The message to log.
     */
    void log(const std::string& message) const;

    /**
     * @brief Logs an error message.
     * @param errorMessage The error message to log.
     */
    void logError(const std::string& errorMessage) const;

private:
    std::vector<ServerNode> servers_;
    size_t current_server_index_ = 0; // For round-robin
    // std::mutex servers_mutex_; // If servers can be added/removed concurrently

    void initialize();
    void cleanup();
};

#endif // LOAD_BALANCER_H
