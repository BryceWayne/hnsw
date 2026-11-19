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
	return nil
}

// Option configures an Index created by New or loaded by ReadFrom and LoadFile.
type Option func(*config) error

// WithM sets the target number of bi-directional links per node on upper layers (default 16, minimum 2).
func WithM(m int) Option {
	return func(c *config) error {
		if m < 2 || m > math.MaxUint16 {
			return fmt.Errorf("%w: M must be in [2, %d]", ErrInvalidConfig, math.MaxUint16)
		}
		c.m = m
		return nil
	}
}

// WithMmax sets the maximum number of links per node at layer 0 (default 2*M).
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

// WithEfConstruction sets the candidate list size used during index construction (default 200).
func WithEfConstruction(ef int) Option {
	return func(c *config) error {
		if ef <= 0 {
			return fmt.Errorf("%w: EfConstruction must be positive", ErrInvalidConfig)
		}
		c.efConstruction = ef
		return nil
	}
}

// WithEfSearch sets the default candidate list size used during search (default 50).
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

// WithMetric selects a built-in distance metric (default MetricEuclidean).
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

// WithCustomDistance configures a user-supplied distance function for the index.
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

// WithSeed sets the random seed used for HNSW level assignment.
func WithSeed(seed int64) Option {
	return func(c *config) error {
		c.seed = seed
		c.hasSeed = true
		return nil
	}
}

// WithHeuristic enables or disables Algorithm 4 diversity-aware neighbor selection (default true).
func WithHeuristic(enabled bool) Option {
	return func(c *config) error {
		c.heuristic = enabled
		return nil
	}
}

type searchConfig struct {
	ef         int
	efSet      bool
	ctx        context.Context
	workers    int
	workersSet bool
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

// SearchOption configures per-call behavior for Search, SearchFilter, SearchBatch, and InsertBatch.
type SearchOption func(*searchConfig)

// WithSearchEf overrides the index default efSearch parameter for a single search call.
func WithSearchEf(ef int) SearchOption {
	return func(sc *searchConfig) {
		sc.ef = ef
		sc.efSet = true
	}
}

// WithContext attaches a cancellation context to a search or batch operation.
func WithContext(ctx context.Context) SearchOption {
	return func(sc *searchConfig) {
		sc.ctx = ctx
	}
}

// WithWorkers sets the bounded worker pool size for batch operations.
func WithWorkers(n int) SearchOption {
	return func(sc *searchConfig) {
		sc.workers = n
		sc.workersSet = true
	}
}
