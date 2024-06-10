# Benchmarks

This code is repurposed from the `hoomd-benchmarks` repo.

## State of benchmarks

Using an Intel 12700 and RTX 3060Ti, we find a moderate speedup once we surpass ~1,000 particles. These benchmarks are compiled in (mostly) 64-bit precision mode, so the 3060Ti suffers a bit there. Though otherwise, there are a handful of opportunities to improve performance further.

![Example Image](./perf-12700-3060Ti.png)