# 指针追逐

指针追逐 (Pointer Chasing) 用如下结构的访问来实现对 Cache 大小的探测：

```
for (int i = 0; i < loopCount; i++) {
    idx = arr[idx];
}
```

或者想节约掉一次整数加法的话，可以这样

```
for (size_t i = 0; i < numLoop; i++) {
    p = *(void**)(p);
}
```

可以用这个结构来搞
```
struct l {
    struct l *n;
    long int pad[NPAD];
};
```

> TODO: 验证一下没有区别

## 原理分析

指针追逐技术起效只需要如下假设成立：


1. 某一级 Cache 的大小是有上限的
2. 当经过**足够多次**固定的访问模式的循环之后
   - 某个元素没有一直呆在该级 Cache 的原因只可能是由于 Cache 容量限制
   - 元素位于 L1 - L2 - L3 - Memory 的位置由其稳态的位置决定
3. 处理器不能预见到下次访问的元素的地址：
   - 关闭预取器 (prefetcher)，或者
   - 让预取器无法预测下次 Load

> 缓存工作的基本单位是缓存行 (Cache Line) 而不是机器字，这会带来什么影响，如何建模？
> 
> 由于 Cache 按缓存行工作，这相当于说取一个数据时，Cache 有概率取到相邻的数据。
> 这些相邻的数据会影响指针追逐过程探索到的 Cache 大小数据
> （具体的说，会探测到一个比 真实大小 / 缓存行 大，比 真实大小 / 元素 小 的值）

预取的特点：
- 预取无法跨越页边界

所以，仔细设计内存访问顺序就成了测试的关键部分。

### 方案一

这里采用一个来自 https://github.com/ucb-bar/ccbench 的方案。

方案对每个页内的访问顺序，最小间隔 stride，且进行 shuffle；页间元素直接连接。

间隔 stride 取缓存行大小，为了测试效率考虑（否则

### 高精度计时

https://docs.microsoft.com/en-us/cpp/intrinsics/rdtsc?view=msvc-170

https://stackoverflow.com/questions/9887839/how-to-count-clock-cycles-with-rdtsc-in-gcc-x86

https://stackoverflow.com/questions/13772567/how-to-get-the-cpu-cycle-count-in-x86-64-from-c/51907627#51907627

(arm) https://stackoverflow.com/questions/40454157/is-there-an-equivalent-instruction-to-rdtsc-in-arm