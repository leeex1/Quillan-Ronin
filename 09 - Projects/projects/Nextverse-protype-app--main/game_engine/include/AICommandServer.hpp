#ifndef AICOMMANDSERVER_HPP
#define AICOMMANDSERVER_HPP

#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <queue>
#include <atomic>
#include <iostream> // For cerr, cout

// Portable socket layer (POSIX on Linux, Winsock on Windows)
#include "SocketCompat.h"

// Forward declaration if Game needs to know about AICommand, or include a common header
struct AICommand {
    std::string avatar_id;
    std::string action;
    float x = 0.0f, y = 0.0f, z = 0.0f; // For move_to, target_x, target_y, target_z
    // Add other potential fields like animation_name, speed_factor etc. as needed
};

template <typename T>
class ThreadSafeQueue {
public:
    void push(T value) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(std::move(value));
    }

    bool try_pop(T& value) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) {
            return false;
        }
        value = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

private:
    std::queue<T> queue_;
    mutable std::mutex mutex_; // mutable to allow locking in const empty()
};


class AICommandServer {
public:
    AICommandServer(int port, ThreadSafeQueue<AICommand>* command_queue);
    ~AICommandServer(); // Important to handle thread and socket cleanup

    void start_listening();
    void stop_listening();

private:
    void server_loop();

    int port_;
    ThreadSafeQueue<AICommand>* command_queue_; // Pointer to the Game's queue
    std::thread server_thread_;
    std::atomic<bool> running_;
    socket_fd_t server_socket_fd_; // Server's listening socket
};

#endif // AICOMMANDSERVER_HPP
