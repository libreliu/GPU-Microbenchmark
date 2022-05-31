#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <chrono>
#include <random>

#include "json.hpp"

#define KiB 1024
#define MiB (1024 * KiB)

#ifdef __MSC_VER
#define NOINLINE __declspec(noinline)

#include <windows.h>
#endif


template<int _npad>
struct Node {
    Node* next;
    unsigned int npad[_npad];
};

size_t get_page_size() {
    static size_t pageSize = 0;

    if (pageSize != 0) {
        return pageSize;
    }

#ifdef __MSC_VER
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    pageSize = si.dwPageSize;
#endif

    printf("The page size for this system is %u bytes.\n", pageSize);
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
NOINLINE
void do_chase(NodeT *start, NodeT *&res) {
    NodeT* p = start;
    for (size_t i = 0; i < numLoop; i++) {
        p = start->next;
    }
    res = p;
}

// workingSetSize: 
template<typename NodeT>
void prepare_full_random(NodeT *&arr, size_t workingSetSize) {
    std::random_device rd;
    std::mt19937 gen(rd());

    size_t numElements = workingSetSize / sizeof(NodeT);
    arr = (NodeT*)std::aligned_alloc(get_page_size(), workingSetSize);

    // shuffle [0, ..., numElements - 1]
    std::vector<size_t> vec(numElements, 0);
    for (size_t i = 0; i < numElements; i++) {
        vec[i] = i + (size_t)arr;
    }

    std::shuffle(vec.begin(), vec.end(), gen);
    for (size_t i = 0; i < numElements; i++) {
        arr[i].next = (NodeT*)vec[i];
    }
}

template<typename NodeT>
void verify_node_list(NodeT *arr, size_t workingSetSize) {
    size_t expectedElements = workingSetSize / sizeof(NodeT);
}


template<typename _Obj_t>
_Obj_t* prepare_chase(_Obj_t *&arr, size_t size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, size - 1);

    arr = (_Obj_t*)malloc(size * sizeof(_Obj_t));
    for (size_t i = 0; i < size; i++) {
        arr[i] = distrib(gen);
    }

    return arr;
}

template<typename _Obj_t>
void cleanup_chase(_Obj_t* arr) {
    free(arr);
}

int main(int argc, char *argv[]) {

    if (argc != 3) {
        fprintf(stderr, "Usage: %s output_json_path num_repeats\n", argv[0]);
        return 1;
    }

    std::string outputJsonPath = argv[1];
    int numRepeats = std::atoi(argv[2]);

    double prepareTime, chaseTime;
    int chaseLoopCount = 100;

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

        int *arr = nullptr, lastIdx;
        for (int i = 1  * KiB; i <= 8 * MiB; i += 128 * KiB) {
            
            prepareTime = timeit<prepare_full_random<int>>(arr, i);
            chaseTime = timeit<do_chase<int, int>>(lastIdx, arr, chaseLoopCount);
            cleanup_chase<int>(arr);
            
            double avgChaseTime = chaseTime / chaseLoopCount;

            printf("[%d KB]: Avg-Chase: %lf us (Prepare: %lf us, Chase: %lf us, lastIdx: %d)\n",
                i, chaseTime * 1e6 / chaseLoopCount, prepareTime * 1e6, chaseTime * 1e6, lastIdx);

            dataArray.push_back(avgChaseTime);
            lastIdxArray.push_back(lastIdx);

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