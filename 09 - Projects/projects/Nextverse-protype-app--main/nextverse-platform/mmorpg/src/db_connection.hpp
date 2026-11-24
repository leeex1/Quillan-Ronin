#ifndef DB_CONNECTION_HPP
#define DB_CONNECTION_HPP

#include <pqxx/pqxx> // C++ PostgreSQL library
#include <string>
#include <iostream>
#include <stdexcept> // For std::runtime_error

class DBConnection {
public:
    DBConnection(const std::string& conn_string) : C(nullptr) {
        try {
            C = new pqxx::connection(conn_string);
            if (C->is_open()) {
                std::cout << "Successfully connected to database: " << C->dbname() << std::endl;
            } else {
                std::cerr << "Can't open database (connection pointer is null after construction or is_open is false)" << std::endl;
                delete C;
                C = nullptr;
                throw std::runtime_error("Database connection failed: is_open() is false.");
            }
        } catch (const pqxx::broken_connection &e) {
            std::cerr << "Database connection failed (pqxx::broken_connection): " << e.what() << std::endl;
            delete C;
            C = nullptr;
            throw;
        } catch (const std::exception &e) {
            std::cerr << "Database connection failed (std::exception): " << e.what() << std::endl;
            delete C;
            C = nullptr;
            throw;
        }
    }

    ~DBConnection() {
        if (C && C->is_open()) {
            C->close();
            std::cout << "Disconnected from database." << std::endl;
        }
        delete C;
        C = nullptr;
    }

    pqxx::connection* getConnection() {
        return C;
    }

    bool isValid() const {
        return C != nullptr && C->is_open();
    }

private:
    pqxx::connection *C;
};

#endif // DB_CONNECTION_HPP
