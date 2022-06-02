#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <chrono>
#include <random>
#include <unordered_map>

#include "json.hpp"

#define KiB 1024
#define MiB (1024 * KiB)

#if defined(_MSC_VER)
#define NOINLINE __declspec(noinline)

#include <windows.h>

#define ALIGNED_ALLOC(align, size) _aligned_malloc(size, align)
#define ALIGNED_FREE(x) _aligned_free(x)
#elif defined(__GNUC__)
// TODO: implement me
#define NOINLINE __attribute__ ((noinline))

#include <unistd.h>

#define ALIGNED_ALLOC(align, size) aligned_alloc(align, size)
#define ALIGNED_FREE(x) free(x)
#else
#error "Implement platform macros"

#endif


// template<int _npad>
// struct Node {
//     Node* next;
//     unsigned int npad[_npad];
// };

struct Node {
    Node* next;
};

size_t get_page_size() {
    static size_t pageSize = 0;

    if (pageSize != 0) {
        return pageSize;
    }

#if defined(_MSC_VER)
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    pageSize = si.dwPageSize;
#elif defined(__GNUC__)
    long sz = sysconf(_SC_PAGESIZE);
    pageSize = sz;
#endif

    printf("The page size for this system is %zu bytes.\n", pageSize);
    return pageSize;
}

// return time measured in sec
template<auto _Func, typename ..._Args>
double timeit(_Args&&... args) {
    auto start = std::chrono::high_resolution_clock::now();

    _Func(std::forward<_Args>(args)...);

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = end-start;
    return diff.count();
}

template<typename NodeT, size_t numLoop>
NOINLINE void do_chase(NodeT *start, NodeT *&res) {
    NodeT* p = start;
    for (size_t i = 0; i < numLoop; i++) {
        p = p->next;
    }
    res = p;
}

// workingSetSize: 
template<typename NodeT>
void prepare_full_random(NodeT *&arr, size_t nodeSize, size_t workingSetSize) {
    std::random_device rd;
    std::mt19937 gen(rd());

    size_t numElements = workingSetSize / nodeSize;
    assert(numElements >= 1);

    arr = (NodeT*)ALIGNED_ALLOC(get_page_size(), workingSetSize);

    if (numElements == 1) {
        arr[0].next = arr;
    } else {
        // shuffle [0, ..., numElements - 1]
        std::vector<size_t> vec(numElements, 0);
        for (size_t i = 0; i < numElements; i++) {
            vec[i] = i;
        }

        std::shuffle(vec.begin(), vec.end(), gen);

        std::unordered_map<size_t, size_t> permuteKV;
        for (size_t idx = 0; idx < numElements; idx++) {
            permuteKV.insert({vec[idx], idx});
        }

        for (size_t i = 0; i < numElements; i++) {
            size_t realIdx = vec[i];
            size_t nextIdx = (realIdx + 1) % numElements;
            NodeT* nextAddr = (NodeT *)((size_t)arr + nodeSize * permuteKV[nextIdx]);
            NodeT* thisAddr = (NodeT *)((size_t)arr + nodeSize * i);

            thisAddr->next = nextAddr;
        }
    }
}

template<typename NodeT>
void verify_node_list(NodeT *arr, size_t nodeSize, size_t workingSetSize, bool &success) {
    size_t expectedElements = workingSetSize / nodeSize;

    NodeT *start = arr;
    NodeT *p = start;

    if (expectedElements > 1) {
        for (size_t i = 0; i < expectedElements; i++) {
            if (p == p->next) {
                success = false;
                return;
            }
            p = p->next;
        }
    }

    if (p == start) {
        success = true;
    } else {
        success = false;
    }
}

template<typename NodeT>
void cleanup_node_list(NodeT* arr) {
    ALIGNED_FREE(arr);
}

std::string pretty_size(size_t size) {
    if (size <= 1024) {
        return std::to_string(size) + " B";
    } else if (size <= 1024 * 1024) {
        return std::to_string((double)size / 1024) + " KB";
    } else {
        return std::to_string((double)size / (1024 * 1024)) + " MB";
    }
}

int main(int argc, char *argv[]) {

    if (argc != 3) {
        fprintf(stderr, "Usage: %s output_json_path num_repeats\n", argv[0]);
        return 1;
    }

    std::string outputJsonPath = argv[1];
    int numRepeats = std::atoi(argv[2]);

    double prepareTime, chaseTime, verifyTime;
    constexpr int chaseLoopCountBase = 1 << 26;
    double cpuFreq = 2.9 * 1000 * 1000 * 1000; // 2.9GHz
    double cpuCycleTime = 1.0 / cpuFreq;

    nlohmann::json result = {
        {"type", "pchase-cpu"},
        {"repeats", nlohmann::json::array()}, /* p */
        {"strides", nlohmann::json::array()}, /* q */
        {"workingSets", nlohmann::json::array()}, /* r */
        {"avgChaseTime", nlohmann::json::array()}
    };

    std::vector<size_t> repeats, strides, workingSets;

    for (int stride = 8; stride <= 128; stride *= 2) {
        strides.push_back(stride);
        result["strides"].push_back(stride);
    }

    for (int repeatIdx = 0; repeatIdx < numRepeats; repeatIdx++) {
        repeats.push_back(repeatIdx);
        result["repeats"].push_back(repeatIdx);
    }

    for (int setSize = 128; setSize <= 256 * MiB; setSize *= 2) {
        workingSets.push_back(setSize);
        result["workingSets"].push_back(setSize);
    }

    auto &chaseTimeArray = result["avgChaseTime"];
    Node *arr = nullptr, *lastPtr = nullptr, *lastPtrWarmUp = nullptr;
    for (auto repeatIdx: repeats) {
        for (auto stride: strides) {
            for (auto setSize: workingSets) {
                
                bool success = false;
                prepareTime = timeit<prepare_full_random<Node>>(arr, stride, setSize);
                verifyTime = timeit<verify_node_list<Node>>(arr, stride, setSize, success);
                assert(success);

                // warmup
                do_chase<Node, (chaseLoopCountBase << 3)>(arr, lastPtrWarmUp);

                double avgChaseTime;
                if (setSize <= 256 * KiB) {
                    chaseTime = timeit<do_chase<Node, (chaseLoopCountBase << 4)>>(arr, lastPtr);
                    avgChaseTime = chaseTime / (chaseLoopCountBase << 4);
                } else {
                    chaseTime = timeit<do_chase<Node, chaseLoopCountBase>>(arr, lastPtr);
                    avgChaseTime = chaseTime / chaseLoopCountBase;
                }
                
                cleanup_node_list<Node>(arr);
                
                
                std::string prettySize = pretty_size(setSize);
                printf("[%s]: Stride: %d, Avg-Chase: %lf us (~%lf cycles) (Prepare: %lf us, Chase: %lf us, lastPtr: %lld, lastPtrWarmUp: %lld)\n",
                    prettySize.c_str(), stride, avgChaseTime * 1e6, avgChaseTime / cpuCycleTime, prepareTime * 1e6, chaseTime * 1e6, lastPtr - arr, lastPtrWarmUp - arr);

                chaseTimeArray.push_back(avgChaseTime);
            }
        }
    }

    std::ofstream outStream(outputJsonPath);
    outStream << result;

    outStream.close();

    return 0;
}