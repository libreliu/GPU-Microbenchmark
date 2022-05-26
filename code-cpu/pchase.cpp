#include <cstdio>
#include <iostream>
#include <chrono>

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
void do_chase(_Obj_t* arr, _LoopCount_t numLoop) {
    _Obj_t idx;
    for (_LoopCount_t i = 0; i < numLoop; i++) {
        idx = arr[idx];
    }
}

template<typename _Obj_t>
_Obj_t* prepare_chase() {
    
}

void cleanup_chase() {

}

int main(int argc, char *argv[]) {
    
}