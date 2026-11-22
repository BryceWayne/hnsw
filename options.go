package hnsw

import (
	"context"
	"fmt"
	"math"
	"runtime"
)

type config struct {
	m              int
	mmax           int
	mmaxSet        bool
	efConstruction int
	efSearch       int
	efSearchSet    bool
	metric         Metric
	metricSet      bool
	customDist     DistanceFunc
	customDistSet  bool
	quant          Quantization
	quantSet       bool
	seed           int64
	hasSeed        bool
	heuristic      bool
}

func newDefaultConfig() config {
	return config{
		m:              16,
		mmax:           32,
		efConstruction: 200,
		efSearch:       50,
		metric:         MetricEuclidean,
		quant:          QuantNone,
		heuristic:      true,
	}
}

func (c *config) validate() error {
	if !c.mmaxSet {
		c.mmax = c.m * 2
	}
	if c.m < 2 || c.m > math.MaxUint16 {
		return fmt.Errorf("%w: M must be in [2, %d]", ErrInvalidConfig, math.MaxUint16)
	}
	if c.mmax < c.m || c.mmax <= 0 || c.mmax > math.MaxUint16 {
		return fmt.Errorf("%w: Mmax must be in [M, %d]", ErrInvalidConfig, math.MaxUint16)
	}
	if c.efConstruction <= 0 {
		return fmt.Errorf("%w: EfConstruction must be positive", ErrInvalidConfig)
	}
	if c.efSearch <= 0 {
		return fmt.Errorf("%w: EfSearch must be positive", ErrInvalidConfig)
	}
	if c.metric == metricCustom {
		if c.customDist == nil {
			return fmt.Errorf("%w: custom distance function must not be nil", ErrInvalidConfig)
		}
	} else if c.metric > MetricDotProduct {
		return fmt.Errorf("%w: unknown metric %d", ErrInvalidConfig, c.metric)
	}
	if c.quant > QuantFloat16 {
		return fmt.Errorf("%w: unknown quantization %d", ErrInvalidConfig, c.quant)
	}
	return nil
}

// Option configures an [Index] created by [New] or loaded by [ReadFrom] and [LoadFile].
// Invalid option values cause [New], [ReadFrom], or [LoadFile] to return an error wrapping [ErrInvalidConfig].
type Option func(*config) error

// WithM sets the target number of bi-directional links per node on upper layers.
// Valid values are in [2, 65535]; the default is 16.
func WithM(m int) Option {
	return func(c *config) error {
		if m < 2 || m > math.MaxUint16 {
			return fmt.Errorf("%w: M must be in [2, %d]", ErrInvalidConfig, math.MaxUint16)
		}
		c.m = m
		return nil
	}
}

// WithMmax sets the maximum number of bi-directional links per node at layer 0.
// Valid values are in [M, 65535]; when omitted, Mmax defaults to 2*M.
func WithMmax(mmax int) Option {
	return func(c *config) error {
		if mmax <= 0 || mmax > math.MaxUint16 {
			return fmt.Errorf("%w: Mmax must be in [1, %d]", ErrInvalidConfig, math.MaxUint16)
		}
		c.mmax = mmax
		c.mmaxSet = true
		return nil
	}
}

// WithEfConstruction sets the beam search width (candidate list size) used during index construction.
// Must be positive; the default is 200. Higher values improve graph quality at the cost of slower inserts.
func WithEfConstruction(ef int) Option {
	return func(c *config) error {
		if ef <= 0 {
			return fmt.Errorf("%w: EfConstruction must be positive", ErrInvalidConfig)
		}
		c.efConstruction = ef
		return nil
	}
}

// WithEfSearch sets the default beam search width (candidate list size) used during queries.
// Must be positive; the default is 50. May also be passed to [ReadFrom] or [LoadFile] to override
// the serialized default for subsequent searches.
func WithEfSearch(ef int) Option {
	return func(c *config) error {
		if ef <= 0 {
			return fmt.Errorf("%w: EfSearch must be positive", ErrInvalidConfig)
		}
		c.efSearch = ef
		c.efSearchSet = true
		return nil
	}
}

// WithMetric selects a built-in distance [Metric] ([MetricEuclidean], [MetricSquaredEuclidean],
// [MetricCosine], or [MetricDotProduct]). The default is [MetricEuclidean].
func WithMetric(m Metric) Option {
	return func(c *config) error {
		if m > MetricDotProduct {
			return fmt.Errorf("%w: invalid metric %d", ErrInvalidConfig, m)
		}
		c.metric = m
		c.metricSet = true
		c.customDist = nil
		c.customDistSet = false
		return nil
	}
}

// WithCustomDistance configures a user-supplied [DistanceFunc] for the index.
// Smaller values returned by fn indicate closer vectors. Indexes persisted with a custom distance
// require supplying WithCustomDistance again to [ReadFrom] or [LoadFile].
func WithCustomDistance(fn DistanceFunc) Option {
	return func(c *config) error {
		if fn == nil {
			return fmt.Errorf("%w: custom distance function cannot be nil", ErrInvalidConfig)
		}
		c.metric = metricCustom
		c.metricSet = false
		c.customDist = fn
		c.customDistSet = true
		return nil
	}
}

// WithSeed sets the PRNG seed used for deterministic HNSW level assignment.
// When omitted, the index is seeded from the current wall-clock time.
// May also be passed to [ReadFrom] or [LoadFile] to re-seed subsequent insertions.
func WithSeed(seed int64) Option {
	return func(c *config) error {
		c.seed = seed
		c.hasSeed = true
		return nil
	}
}

// WithHeuristic enables or disables HNSW Algorithm 4 diversity-aware neighbor selection.
// Enabled by default; disabling falls back to simple nearest-neighbor pruning.
func WithHeuristic(enabled bool) Option {
	return func(c *config) error {
		c.heuristic = enabled
		return nil
	}
}

// WithQuantization selects the vector storage encoding ([QuantNone] or [QuantFloat16]).
// The default is [QuantNone] (full float32). [QuantFloat16] halves vector memory and
// memory bandwidth using IEEE-754 binary16 storage (AVX2+F16C+FMA accelerated on amd64).
func WithQuantization(q Quantization) Option {
	return func(c *config) error {
		if q > QuantFloat16 {
			return fmt.Errorf("%w: invalid quantization %d", ErrInvalidConfig, q)
		}
		c.quant = q
		c.quantSet = true
		return nil
	}
}

type searchConfig struct {
	ef         int
	efSet      bool
	ctx        context.Context
	workers    int
	workersSet bool
	maxDist    float32
	maxDistSet bool
}

func defaultSearchConfig() searchConfig {
	return searchConfig{}
}

func (sc *searchConfig) apply(opts []SearchOption) error {
	for _, opt := range opts {
		if opt != nil {
			opt(sc)
		}
	}
	if sc.efSet && sc.ef <= 0 {
		return fmt.Errorf("%w: search ef must be positive", ErrInvalidConfig)
	}
	if sc.workersSet && sc.workers <= 0 {
		return fmt.Errorf("%w: worker count must be positive", ErrInvalidConfig)
	}
	if sc.maxDistSet && math.IsNaN(float64(sc.maxDist)) {
		return fmt.Errorf("%w: max distance cannot be NaN", ErrInvalidConfig)
	}
	if sc.ctx != nil {
		if err := sc.ctx.Err(); err != nil {
			return err
		}
	}
	return nil
}

func (sc *searchConfig) effectiveWorkers() int {
	if sc.workersSet {
		return sc.workers
	}
	w := runtime.GOMAXPROCS(0)
	if w < 1 {
		return 1
	}
	return w
}

// SearchOption configures per-call behavior for [Index.Search], [Index.SearchFilter],
// [Index.SearchBatch], and [Index.InsertBatch].
type SearchOption func(*searchConfig)

// WithSearchEf overrides the index default efSearch parameter for a single search call.
// Must be positive (otherwise returns [ErrInvalidConfig]); effective ef is always clamped to >= k.
func WithSearchEf(ef int) SearchOption {
	return func(sc *searchConfig) {
		sc.ef = ef
		sc.efSet = true
	}
}

// WithContext attaches a cancellation or deadline [context.Context] to a search or batch operation.
// If ctx is already canceled or expires during traversal, the operation aborts and returns ctx.Err().
func WithContext(ctx context.Context) SearchOption {
	return func(sc *searchConfig) {
		sc.ctx = ctx
	}
}

// WithWorkers sets the bounded worker pool size for [Index.InsertBatch] and [Index.SearchBatch].
// Must be positive (otherwise returns [ErrInvalidConfig]); defaults to runtime.GOMAXPROCS(0).
func WithWorkers(n int) SearchOption {
	return func(sc *searchConfig) {
		sc.workers = n
		sc.workersSet = true
	}
}

// WithMaxDistance filters search results to include only neighbors with final Distance <= maxDist.
// Applies across [Index.Search], [Index.SearchFilter], [Index.SearchInto], and [Index.SearchBatch].
// Non-matching nodes are still traversed so graph connectivity is preserved.
// Returns [ErrInvalidConfig] if maxDist is NaN.
func WithMaxDistance(maxDist float32) SearchOption {
	return func(sc *searchConfig) {
		sc.maxDist = maxDist
		sc.maxDistSet = true
	}
}
