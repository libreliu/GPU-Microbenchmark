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

#ifdef _MSC_VER
#define NOINLINE __declspec(noinline)

#include <windows.h>

#define ALIGNED_ALLOC(align, size) _aligned_malloc(size, align)
#define ALIGNED_FREE(x) _aligned_free(x)
#else if
// TODO: implement me
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

#ifdef _MSC_VER
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    pageSize = si.dwPageSize;
#else if 
    // TODO: implement me
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
    constexpr int chaseLoopCount = 1 << 24;
    double cpuFreq = 3.0 * 1000 * 1000 * 1000; // 3GHz
    double cpuCycleTime = 1.0 / cpuFreq;

    nlohmann::json result = {
        {"type", "pchase-cpu"},
        {"numRepeats", numRepeats},
        {"data", nlohmann::json::array()}
    };

    for (int repeatIdx = 0; repeatIdx < numRepeats; repeatIdx++) {
        result["data"].push_back(nlohmann::json::object());
        result["data"].back()["avgChaseTime"] = nlohmann::json::array();
        result["data"].back()["lastIdx"] = nlohmann::json::array();

        if (repeatIdx == 0) {
            result["data"].back()["size"] = nlohmann::json::array();
        }

        auto &dataArray = result["data"].back()["avgChaseTime"];
        auto &lastIdxArray = result["data"].back()["lastIdx"];
        auto &sizeArray = result["data"][0]["size"];

        Node *arr = nullptr, *lastPtr = nullptr;
        for (int i = 64; i <= 128 * MiB; i *= 2) {
            bool success = false;
            prepareTime = timeit<prepare_full_random<Node>>(arr, 64, i);
            verifyTime = timeit<verify_node_list<Node>>(arr, 64, i, success);
            assert(success);

            chaseTime = timeit<do_chase<Node, chaseLoopCount>>(arr, lastPtr);
            cleanup_node_list<Node>(arr);
            
            double avgChaseTime = chaseTime / chaseLoopCount;
            std::string prettySize = pretty_size(i);
            printf("[%s]: Avg-Chase: %lf us - about %lf cycles (Prepare: %lf us, Chase: %lf us, lastPtr: %lld)\n",
                prettySize.c_str(), avgChaseTime * 1e6, avgChaseTime / cpuCycleTime, prepareTime * 1e6, chaseTime * 1e6, lastPtr - arr);

            dataArray.push_back(avgChaseTime);
            lastIdxArray.push_back((size_t)(lastPtr - arr));

            if (repeatIdx == 0) {
                sizeArray.push_back(i);
            }
        }
    }
    
    std::ofstream outStream(outputJsonPath);
    result >> outStream;

    outStream.close();

    return 0;
}