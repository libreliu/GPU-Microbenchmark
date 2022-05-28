#include <cstdio>
#include <iostream>
#include <fstream>
#include <chrono>
#include <random>

#include "json.hpp"

#define KiB 1024
#define MiB (1024 * KiB)

// return time measured in sec
template<auto _Func, typename ..._Args>
double timeit(_Args&&... args) {

    auto start = std::chrono::high_resolution_clock::now();

    _Func(std::forward<_Args>(args)...);

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = end-start;
    return diff.count();
}

template<typename _Obj_t, typename _LoopCount_t>
void do_chase(_Obj_t &res, _Obj_t* arr, _LoopCount_t numLoop) {
    _Obj_t idx = 0;
    for (_LoopCount_t i = 0; i < numLoop; i++) {
        idx = arr[idx];
    }
    res = idx;
}

template<typename _Obj_t>
_Obj_t* prepare_full_random(_Obj_t *&arr, size_t size) {
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
    int chaseLoopCount = 1000000;

    nlohmann::json result = {
        {"type", "pchase-cpu"},
        {"numRepeats", numRepeats},
        {"data", nlohmann::json::array()}
    };

    for (int repeatIdx = 0; repeatIdx < numRepeats; repeatIdx++) {
        result["data"].push_back(nlohmann::json::object());
        result["data"].back()["avgChaseTime"] = nlohmann::json::array();

        if (repeatIdx == 0) {
            result["data"].back()["size"] = nlohmann::json::array();
        }
        

        auto &dataArray = result["data"].back()["avgChaseTime"];
        auto &sizeArray = result["data"][0]["size"];

        for (int i = 1  * KiB; i <= 1 * MiB; i += 1 * KiB) {
            int *arr, lastIdx;
            prepareTime = timeit<prepare_full_random<int>>(arr, i);
            chaseTime = timeit<do_chase<int, int>>(lastIdx, arr, chaseLoopCount);
            cleanup_chase<int>(arr);
            
            double avgChaseTime = chaseTime / chaseLoopCount;

            printf("[%d KB]: Avg-Chase: %lf us (Prepare: %lf us, Chase: %lf us, lastIdx: %d)\n",
                i, chaseTime * 1e6 / chaseLoopCount, prepareTime * 1e6, chaseTime * 1e6, lastIdx);

            dataArray.push_back(avgChaseTime);

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