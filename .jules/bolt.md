## 2024-05-24 - Slice Hoisting inside Hot Paths
**Learning:** In Go, memory allocations within a tight inner loop (e.g. `searchLayerParallel`) can create huge GC overhead even when overall heap space is not fully constrained. Slices can be easily allocated outside the hot loop and reused via `slice = slice[:0]`.
**Action:** Always inspect the tightest loops for dynamically sized slices or buffers. Instead of fresh allocations, hoist the `make` call outside and re-slice for each iteration. For slices where capacity can grow, manually track capacity and only reallocate `if cap(slice) < required_len`.

## 2024-08-07 - Object Pooling for Cross-Iteration Buffers
**Learning:** Even when slice buffers are hoisted out of inner loops and zeroed inside a function (e.g. `neighbors`, `candidateNodes`, `flatData` in `searchLayerParallel`), the high frequency of the function calls themselves (like `BatchSearch` querying repeatedly) creates massive GC pressure because a new set of buffers is allocated on every query. Moving from function-scoped allocation to global `sync.Pool` for these recurring buffers can slash memory allocations by 90% and increase throughput by 4-5x in high-concurrency scenarios like parallel searches.
**Action:** Identify buffers that can't be statically sized across all concurrent goroutines but are heavily requested per-operation. Lift these out of the hot-path function into a `sync.Pool`. Ensure slices are cleared (e.g. `slice[:0]`, niled-out pointers) before returning to the pool to prevent memory leaks.

## 2026-07-28 - Redundant Tracking Sets in Graph Search
**Learning:** During HNSW graph search operations (e.g. `searchLayer`, `searchLayerParallel`), keeping a separate `visitedResults` map to track nodes added to the result set is often completely redundant if there is already a primary `visited` map guaranteeing that each neighbor is only explored and evaluated once. Re-adding the same node to the heap multiple times is implicitly prevented because the node is only ever processed once from its parent's neighbor list.
**Action:** When auditing search logic or BFS/DFS traversals for performance, always double-check if multiple "seen" sets can be collapsed into one. Eliminating redundant sets saves significant overhead (e.g., sync.Pool map allocations, clearing maps, and multiple map lookups per query).

## 2024-05-18 - Caching Distances During Sort
**Learning:** During the final sort of candidates in `SearchWithConfig`, the distance function was being evaluated multiple times for the same pairs inside `sort.Slice`. While `searchLayer` returns sorted results, keeping them in `[]*Node` forces re-computing distances if any sort is applied.
**Action:** By packaging the candidates into a `[]nodeDist` slice before sorting, the distance is calculated exactly once per candidate and cached, resulting in fewer distance function evaluations.
