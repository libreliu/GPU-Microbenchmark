// Imiatation of https://www.fftw.org/cycle.h

#include <cstdio>
#if defined(_MSC_VER)
#include <windows.h>
#elif defined(__GNUC__)
#include <unistd.h>
#endif

inline bool bind_to_first_cpu() {
#if defined(_MSC_VER)
    DWORD affinityMask = 0x00000001L;
    BOOL ret = SetProcessAffinityMask(GetCurrentProcess(), affinityMask);
    if (!ret) {
        DWORD errorCode = GetLastError();
        fprintf(stderr, "Error while setting affinity mask: %x", errorCode);
        return false;
    }
#elif defined(__GNUC__)
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    sched_setaffinity (getpid(), sizeof(cpuset), &cpuset);

    // TODO: error check
#endif

    return true;
}

