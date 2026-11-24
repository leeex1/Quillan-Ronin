#ifndef SOCKETCOMPAT_H
#define SOCKETCOMPAT_H

// Portable socket surface for the AI command server.
// Maps the POSIX calls used by the engine onto Winsock when compiling on Windows.

#include <cstdio>
#include <cstring>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#include <BaseTsd.h>

using ssize_t = SSIZE_T;
using socket_fd_t = SOCKET;

#define close(fd) closesocket(fd)
#define SHUT_RDWR SD_BOTH
#define read(fd, buf, len) recv((fd), (buf), (int)(len), 0)

inline constexpr int kSockBadFd = WSAEBADF;
inline constexpr int kSockInvalid = WSAEINVAL;
inline constexpr int kSockAborted = WSAECONNABORTED;
inline constexpr int kSockInterrupted = WSAEINTR;

inline int socket_errno() { return WSAGetLastError(); }
inline const char* socket_error() {
    static char buffer[80];
    std::snprintf(buffer, sizeof(buffer), "WSA error %d", WSAGetLastError());
    return buffer;
}
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <cerrno>

using socket_fd_t = int;
#define INVALID_SOCKET (-1)

inline constexpr int kSockBadFd = EBADF;
inline constexpr int kSockInvalid = EINVAL;
inline constexpr int kSockAborted = ECONNABORTED;
inline constexpr int kSockInterrupted = EINTR;

inline int socket_errno() { return errno; }
inline const char* socket_error() { return std::strerror(errno); }
#endif

#endif // SOCKETCOMPAT_H
