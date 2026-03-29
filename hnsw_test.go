package hnswgo

import (
	"errors"
	"math"
	"math/rand"
	"os"
	"testing"
)

const testVectorDB = "./test.db"
const (
	dim            = 400
	M              = 20
	efConstruction = 10

	batchSize = 100
)

func newTestIndex(batch int, allowRepaceDeleted bool) *HnswIndex {
	maxElements := batch * batchSize

	index := New(dim, M, efConstruction, 55, uint64(maxElements), Cosine, allowRepaceDeleted)

	for i := 0; i < batch; i++ {
		points, labels := randomPoints(dim, i*batchSize, batchSize)
		index.AddPoints(points, labels, 1, false)
	}

	return index
}

func TestNewIndex(t *testing.T) {
	var maxElements uint64 = batchSize * 1

	idx := newTestIndex(1, true)
	defer idx.Free()

	if idx.GetMaxElements() != maxElements {
		t.Fail()
	}

	if idx.GetAllowReplaceDeleted() != true {
		t.Fail()
	}

	if idx.GetCurrentCount() != maxElements {
		t.Fail()
	}

}

func TestLoadAndSaveIndex(t *testing.T) {
	var maxElements uint64 = batchSize * 1

	// setup
	idx := newTestIndex(1, true)
	idx.Save(testVectorDB)
	idx.Free()

	index := Load(testVectorDB, Cosine, dim, uint64(maxElements), true)
	index.SetEf(efConstruction)
	defer index.Free()

	index.Save(testVectorDB)
	t.Cleanup(func() {
		deleteDB()
	})
}

func TestResizeIndex(t *testing.T) {
	var maxElements uint64 = batchSize * 1

	idx := newTestIndex(1, false)
	defer idx.Free()

	if idx.GetMaxElements() != maxElements {
		t.Fail()
	}

	if idx.GetCurrentCount() != maxElements {
		t.Fail()
	}

	if idx.GetAllowReplaceDeleted() != false {
		t.Fail()
	}

	points, labels := randomPoints(dim, 1*batchSize, batchSize)
	err := idx.AddPoints(points, labels, 1, false)
	if err == nil {
		t.Log(err)
		t.FailNow()
	}

	idx.ResizeIndex(maxElements * 2)
	if idx.GetMaxElements() != maxElements*2 {
		t.Fail()
	}

	if idx.GetCurrentCount() != maxElements {
		t.Fail()
	}

	err = idx.AddPoints(points, labels, 1, false)
	if err != nil {
		t.Log(err)
		t.Fail()
	}
}

func TestReplacePoint(t *testing.T) {
	allowRepaceDeleted := true
	maxElements := 100
	index := New(dim, M, efConstruction, 505, uint64(maxElements), Cosine, allowRepaceDeleted)
	defer index.Free()

	if !index.GetAllowReplaceDeleted() {
		t.Fail()
	}

	points, labels := randomPoints(dim, 0, maxElements)
	index.AddPoints(points, labels, 1, false)

	index.MarkDeleted(labels[len(labels)-1])

	err := index.AddPoints([][]float32{randomPoint(dim)}, []uint64{math.MaxUint64 - 1}, 1, false)
	if err == nil {
		t.Fail()
	}

	err = index.AddPoints([][]float32{randomPoint(dim)}, []uint64{math.MaxUint64 - 1}, 1, true)
	if err != nil {
		t.Fail()
	}

}

func TestVectorSearch(t *testing.T) {
	// Test 1: Basic search with valid index
	t.Run("BasicSearch", func(t *testing.T) {
		index := newTestIndex(1, false)
		index.SetEf(efConstruction)
		defer index.Free()

		query := genQuery(dim, 10)
		topK := 5

		result, err := index.SearchKNN(query, topK, 1)
		if err != nil {
			t.Errorf("SearchKNN failed: %v", err)
			return
		}

		if len(result) != len(query) {
			t.Errorf("expected %d results, got %d", len(query), len(result))
		}

		for i, rv := range result {
			if len(rv) != topK {
				t.Errorf("query %d: expected %d results, got %d", i, topK, len(rv))
			}
		}
	})

	// Test 2: Verify distances are in ascending order
	t.Run("SortedDistances", func(t *testing.T) {
		index := newTestIndex(1, false)
		index.SetEf(efConstruction)
		defer index.Free()

		query := genQuery(dim, 1)
		result, err := index.SearchKNN(query, 5, 1)
		if err != nil {
			t.Errorf("SearchKNN failed: %v", err)
			return
		}

		for i, rv := range result {
			for j := 1; j < len(rv); j++ {
				if rv[j].Distance < rv[j-1].Distance {
					t.Errorf("query %d: distances not sorted at position %d: %f > %f",
						i, j, rv[j-1].Distance, rv[j].Distance)
				}
			}
		}
	})

	// Test 3: Edge case - k larger than maxElements returns error (expected behavior)
	t.Run("KExceedsElements", func(t *testing.T) {
		index := newTestIndex(1, false) // 100 elements
		index.SetEf(efConstruction)
		defer index.Free()

		query := genQuery(dim, 1)
		// Request more than available - library returns error
		_, err := index.SearchKNN(query, 200, 1)
		if err == nil {
			t.Errorf("expected error when k > maxElements, got nil")
		}
	})

	// Test 4: Edge case - search with k=1
	t.Run("SingleK", func(t *testing.T) {
		index := newTestIndex(1, false)
		index.SetEf(efConstruction)
		defer index.Free()

		query := genQuery(dim, 1)
		result, err := index.SearchKNN(query, 1, 1)
		if err != nil {
			t.Errorf("SearchKNN failed: %v", err)
			return
		}

		if len(result[0]) != 1 {
			t.Errorf("expected 1 result, got %d", len(result[0]))
		}
	})

	// Test 5: Verify results are labeled (not empty labels)
	t.Run("ValidLabels", func(t *testing.T) {
		index := newTestIndex(1, false)
		index.SetEf(efConstruction)
		defer index.Free()

		query := genQuery(dim, 1)
		result, err := index.SearchKNN(query, 5, 1)
		if err != nil {
			t.Errorf("SearchKNN failed: %v", err)
			return
		}

		for i, rv := range result {
			for j, r := range rv {
				if r.Label == 0 {
					t.Errorf("query %d: result %d has empty label", i, j)
				}
			}
		}
	})

	// Test 6: Multiple queries in a single call
	t.Run("MultipleQueries", func(t *testing.T) {
		index := newTestIndex(3, false) // 300 elements
		index.SetEf(efConstruction)
		defer index.Free()

		query := genQuery(dim, 50)
		result, err := index.SearchKNN(query, 10, 1)
		if err != nil {
			t.Errorf("SearchKNN failed: %v", err)
			return
		}

		if len(result) != 50 {
			t.Errorf("expected 50 results, got %d", len(result))
		}
	})

}

func TestGetVectorData(t *testing.T) {

}

func randomPoints(dim int, startLabel int, batchSize int) ([][]float32, []uint64) {
	points := make([][]float32, batchSize)
	labels := make([]uint64, 0)

	for i := 0; i < batchSize; i++ {
		v := make([]float32, dim)
		for i := range v {
			v[i] = rand.Float32()
		}
		points[i] = v
		labels = append(labels, uint64(startLabel+i))
	}

	return points, labels
}

func genQuery(dim int, size int) [][]float32 {
	points := make([][]float32, size)

	for i := 0; i < size; i++ {
		v := make([]float32, dim)
		for i := range v {
			v[i] = rand.Float32()
		}
		points[i] = v
	}

	return points
}

func pathExists(path string) bool {
	stat, err := os.Stat(path)
	if errors.Is(err, os.ErrNotExist) {
		return false
	}

	if err == nil || stat != nil {
		return true
	}

	return false
}

func deleteDB() error {
	if pathExists(testVectorDB) {
		return os.Remove(testVectorDB)
	}

	return nil
}

func randomPoint(dim int) []float32 {
	v := make([]float32, dim)
	for i := range v {
		v[i] = rand.Float32()
	}
	return v
}
