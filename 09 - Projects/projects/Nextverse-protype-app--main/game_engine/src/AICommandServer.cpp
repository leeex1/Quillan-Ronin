#include "AICommandServer.hpp"
#include <cstring> // For memset, strerror
#include <algorithm> // for std::remove

// Helper function for basic JSON parsing (MVP)
// Example: {"avatar_id": "id", "action": "move_to", "target_x": X, "target_y": Y, "target_z": Z}
// This is a very naive parser. A proper JSON library is recommended for robustness.
AICommand parse_json_command_mvp(const std::string& json_str) {
    AICommand cmd;
    std::string current_str = json_str;
    // Remove whitespace and braces for simplicity
    current_str.erase(std::remove_if(current_str.begin(), current_str.end(), ::isspace), current_str.end());
    current_str.erase(std::remove(current_str.begin(), current_str.end(), '{'), current_str.end());
    current_str.erase(std::remove(current_str.begin(), current_str.end(), '}'), current_str.end());

    // Tokenize by comma
    std::vector<std::string> tokens;
    size_t start = 0;
    size_t end = current_str.find(',');
    while (end != std::string::npos) {
        tokens.push_back(current_str.substr(start, end - start));
        start = end + 1;
        end = current_str.find(',', start);
    }
    tokens.push_back(current_str.substr(start));

    for (const auto& token : tokens) {
        size_t colon_pos = token.find(':');
        if (colon_pos == std::string::npos) continue;

        std::string key = token.substr(0, colon_pos);
        std::string value_str = token.substr(colon_pos + 1);

        // Remove quotes from key and value
        key.erase(std::remove(key.begin(), key.end(), '\"'), key.end());
        value_str.erase(std::remove(value_str.begin(), value_str.end(), '\"'), value_str.end());

        try {
            if (key == "avatar_id") cmd.avatar_id = value_str;
            else if (key == "action") cmd.action = value_str;
            else if (key == "target_x") cmd.x = std::stof(value_str);
            else if (key == "target_y") cmd.y = std::stof(value_str);
            else if (key == "target_z") cmd.z = std::stof(value_str);
        } catch (const std::invalid_argument& ia) {
            std::cerr << "AICommandServer: Invalid argument for key '" << key << "': " << value_str << std::endl;
        } catch (const std::out_of_range& oor) {
            std::cerr << "AICommandServer: Out of range for key '" << key << "': " << value_str << std::endl;
        }
    }
    return cmd;
}


AICommandServer::AICommandServer(int port, ThreadSafeQueue<AICommand>* command_queue)
    : port_(port), command_queue_(command_queue), running_(false), server_socket_fd_(-1) {
    std::cout << "AICommandServer created for port " << port_ << "." << std::endl;
}

AICommandServer::~AICommandServer() {
    stop_listening(); // Ensure server is stopped and thread is joined
    std::cout << "AICommandServer destroyed." << std::endl;
}

void AICommandServer::start_listening() {
    if (running_) {
        std::cout << "AICommandServer already running." << std::endl;
        return;
    }
    running_ = true;
    // Start the server_loop in a new thread
    server_thread_ = std::thread(&AICommandServer::server_loop, this);
    std::cout << "AICommandServer started listening on port " << port_ << "." << std::endl;
}

void AICommandServer::stop_listening() {
    if (!running_) {
        // std::cout << "AICommandServer not running or already stopped." << std::endl;
        return;
    }
    running_ = false;

    // To unblock accept() in server_loop(), we need to close the server socket.
    // This is a common way to signal the listening thread to exit.
    if (server_socket_fd_ != -1) {
        // Shut down the socket to prevent new connections and unblock accept
        // SHUT_RDWR: Disables further sends and receives.
        shutdown(server_socket_fd_, SHUT_RDWR);
        close(server_socket_fd_); // Then close the file descriptor
        server_socket_fd_ = INVALID_SOCKET;   // Mark as closed
         std::cout << "AICommandServer: Server socket " << server_socket_fd_ << " shut down and closed to interrupt accept()." << std::endl;
    }


    if (server_thread_.joinable()) {
        server_thread_.join();
        std::cout << "AICommandServer: Server thread joined." << std::endl;
    } else {
         std::cout << "AICommandServer: Server thread was not joinable on stop." << std::endl;
    }
    std::cout << "AICommandServer stopped." << std::endl;
}

void AICommandServer::server_loop() {
    server_socket_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (server_socket_fd_ == -1) {
        std::cerr << "AICommandServer: Failed to create socket: " << socket_error() << std::endl;
        running_ = false;
        return;
    }

    // Allow address reuse
    int opt = 1;
    if (setsockopt(server_socket_fd_, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&opt), sizeof(opt)) < 0) {
        std::cerr << "AICommandServer: setsockopt(SO_REUSEADDR) failed: " << socket_error() << std::endl;
        // Non-fatal, but good to log
    }


    sockaddr_in server_addr;
    std::memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY; // Listen on all available interfaces
    server_addr.sin_port = htons(port_);

    if (bind(server_socket_fd_, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        std::cerr << "AICommandServer: Failed to bind to port " << port_ << ": " << socket_error() << std::endl;
        close(server_socket_fd_);
        server_socket_fd_ = INVALID_SOCKET;
        running_ = false;
        return;
    }

    if (listen(server_socket_fd_, 5) < 0) { // Max 5 pending connections
        std::cerr << "AICommandServer: Failed to listen on socket: " << socket_error() << std::endl;
        close(server_socket_fd_);
        server_socket_fd_ = INVALID_SOCKET;
        running_ = false;
        return;
    }

    std::cout << "AICommandServer: server_loop listening on port " << port_ << std::endl;

    while (running_) {
        sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        socket_fd_t client_socket_fd = accept(server_socket_fd_, (struct sockaddr*)&client_addr, &client_len);

        if (!running_) { // Check running_ again after accept returns, in case stop_listening was called
            if (client_socket_fd != INVALID_SOCKET) close(client_socket_fd);
            break;
        }

        if (client_socket_fd == INVALID_SOCKET) {
            if (running_) { // Only log error if we weren't expecting to stop
                 // Common error when server_socket_fd_ is closed by stop_listening()
                if (socket_errno() != kSockBadFd && socket_errno() != kSockInvalid) { // EBADF/EINVAL might happen if socket closed during stop
                    std::cerr << "AICommandServer: Failed to accept connection: " << socket_error() << std::endl;
                }
            }
            // If accept failed (e.g. server socket closed by stop_listening), try to continue loop to check running_ flag
            // or break if error is persistent and not due to stopping.
            // For simplicity, if accept fails and we are supposed to be running, we might want to stop.
            // However, closing the socket in stop_listening is the primary way to exit this loop.
            if (socket_errno() != kSockAborted && socket_errno() != kSockInterrupted) { // These can be recoverable or due to signal
                 // if (running_) running_ = false; // Consider stopping if a persistent error occurs
            }
            continue; // Continue to check running_ flag
        }

        char client_ip_str[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &client_addr.sin_addr, client_ip_str, INET_ADDRSTRLEN);
        std::cout << "AICommandServer: Accepted connection from " << client_ip_str << ":" << ntohs(client_addr.sin_port) << std::endl;

        char buffer[1024] = {0};
        ssize_t bytes_read = read(client_socket_fd, buffer, sizeof(buffer) - 1);

        if (bytes_read > 0) {
            buffer[bytes_read] = '\0'; // Null-terminate
            std::string received_data(buffer);
            std::cout << "AICommandServer: Received data: " << received_data << std::endl;

            // Basic MVP JSON parsing
            AICommand cmd = parse_json_command_mvp(received_data);

            // Validate parsed command minimally
            if (!cmd.avatar_id.empty() && !cmd.action.empty()) {
                if (command_queue_) {
                    command_queue_->push(cmd);
                    std::cout << "AICommandServer: Pushed command to queue for avatar_id: " << cmd.avatar_id << ", action: " << cmd.action << std::endl;
                }
                const char* response = "OK\n";
                send(client_socket_fd, response, static_cast<int>(strlen(response)), 0);
            } else {
                std::cerr << "AICommandServer: Failed to parse command or command invalid: " << received_data << std::endl;
                const char* response = "ERROR: Invalid command format\n";
                send(client_socket_fd, response, static_cast<int>(strlen(response)), 0);
            }

        } else if (bytes_read == 0) {
            std::cout << "AICommandServer: Client disconnected." << std::endl;
        } else {
            std::cerr << "AICommandServer: Error reading from client socket: " << socket_error() << std::endl;
        }
        close(client_socket_fd);
    }

    if (server_socket_fd_ != -1) { // Ensure it's closed if loop exits for other reasons
        close(server_socket_fd_);
        server_socket_fd_ = INVALID_SOCKET;
    }
    std::cout << "AICommandServer: server_loop ended." << std::endl;
}
