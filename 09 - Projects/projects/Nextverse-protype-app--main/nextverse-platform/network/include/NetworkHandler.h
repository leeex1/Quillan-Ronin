#ifndef NETWORK_HANDLER_H
#define NETWORK_HANDLER_H

#include <iostream> // For logging (can be replaced with a proper logger)
#include <string>
#include <vector>
#include <functional> // For std::function

// Forward declaration if needed, e.g., for a Message class
// class Message;

/**
 * @class NetworkHandler
 * @brief Manages network connections and data transmission.
 *
 * This class provides basic functionalities to connect to a server,
 * send and receive messages, and manage the connection state.
 * It's intended as a foundational component for network communication.
 */
class NetworkHandler {
public:
    /**
     * @brief Constructor for NetworkHandler.
     */
    NetworkHandler();

    /**
     * @brief Destructor for NetworkHandler.
     * Ensures disconnection if still connected.
     */
    ~NetworkHandler();

    /**
     * @brief Connects to a specified host and port.
     * @param host The hostname or IP address of the server.
     * @param port The port number on the server.
     * @return True if connection is successful, false otherwise.
     */
    bool connect(const std::string& host, int port);

    /**
     * @brief Disconnects from the current server.
     */
    void disconnect();

    /**
     * @brief Checks if the handler is currently connected.
     * @return True if connected, false otherwise.
     */
    bool isConnected() const;

    /**
     * @brief Sends a message over the network.
     * @param message The string message to send.
     * @return True if the message was sent successfully, false otherwise.
     */
    bool sendMessage(const std::string& message);
    // /**
    //  * @brief Sends a structured message over the network.
    //  * @param message The Message object to send.
    //  * @return True if the message was sent successfully, false otherwise.
    //  */
    // bool sendMessage(const Message& message);

    /**
     * @brief Receives a message from the network (blocking call).
     * @param bufferSize The size of the buffer to use for receiving.
     * @return The received message as a string. Returns an empty string on error or disconnection.
     */
    std::string receiveMessage(int bufferSize = 1024);

    /**
     * @brief Defines the callback function type for asynchronous message reception.
     * @param const std::string& The received message.
     */
    using MessageCallback = std::function<void(const std::string&)>;

    /**
     * @brief Starts receiving messages asynchronously.
     * Incoming messages will be passed to the provided callback.
     * @param callback The function to call when a message is received.
     */
    void startReceiving(MessageCallback callback);

    /**
     * @brief Stops receiving messages asynchronously.
     */
    void stopReceiving();

    // Basic Logging (can be expanded)
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
    // Internal state
    bool connected_ = false;
    // Placeholder for socket/connection details
    // int socket_descriptor_ = -1;
    // Placeholder for receiver thread if using callbacks
    // std::thread receiver_thread_;
    bool receiving_ = false; // Added to track if receiving is active

    // Private helper methods
    void initializeNetwork();
    void shutdownNetwork();
};

#endif // NETWORK_HANDLER_H
