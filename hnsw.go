package hnswgo

// #cgo CXXFLAGS: -fPIC -pthread -Wall -std=c++11 -O2 -march=native -I.
// #cgo LDFLAGS: -pthread
// #cgo CFLAGS: -I./
// #include <stdlib.h>
// #include "hnsw_wrapper.h"
import "C"
import (
	"errors"
	"unsafe"
)

type SpaceType int

const (
	L2 SpaceType = iota
	IP
	Cosine
)

type HnswIndex struct {
	index *C.HnswIndex
}

type SearchResult struct {
	Label    uint64
	Distance float32
}

func New(dim, M, efConstruction, randSeed int, maxElements uint64, spaceType SpaceType, allowReplaceDeleted bool) *HnswIndex {
	var allowReplace int = 0
	if allowReplaceDeleted {
		allowReplace = 1
	}

	var sType C.spaceType = C.l2
	switch spaceType {
	case L2:
		sType = C.l2
	case IP:
		sType = C.ip
	case Cosine:
		sType = C.cosine
	}

	cindex := C.newIndex(sType, C.int(dim), C.size_t(maxElements), C.int(M), C.int(efConstruction), C.int(randSeed), C.int(allowReplace))

	return &HnswIndex{
		index: cindex,
	}
}

func Load(location string, spaceType SpaceType, dim int, maxElements uint64, allowReplaceDeleted bool) *HnswIndex {
	var allowReplace int = 0
	if allowReplaceDeleted {
		allowReplace = 1
	}

	var sType C.spaceType = C.l2
	switch spaceType {
	case L2:
		sType = C.l2
	case IP:
		sType = C.ip
	case Cosine:
		sType = C.cosine
	}

	cloc := C.CString(location)
	defer C.free(unsafe.Pointer(cloc))

	cindex := C.loadIndex(cloc, sType, C.int(dim), C.size_t(maxElements), C.int(allowReplace))

	return &HnswIndex{
		index: cindex,
	}
}

func (idx *HnswIndex) SetEf(ef int) {
	C.setEf(idx.index, C.size_t(ef))
}

func (idx *HnswIndex) IndexFileSize() uint64 {
	sz := C.indexFileSize(idx.index)

	return uint64(sz)
}

func (idx *HnswIndex) Save(location string) {
	cloc := C.CString(location)
	defer C.free(unsafe.Pointer(cloc))

	C.saveIndex(idx.index, cloc)
}

// Adds points. Updates the point if it is already in the index.
// If replacement of deleted elements is enabled: replaces previously deleted point if any, updating it with new point.
func (idx *HnswIndex) AddPoints(vectors [][]float32, labels []uint64, concurrency int, replaceDeleted bool) error {
	var replace int = 0
	if replaceDeleted {
		replace = 1
	}

	if len(vectors) <= 0 || len(labels) <= 0 {
		return errors.New("invalid vector data")
	}

	if len(labels) != len(vectors) {
		return errors.New("unmatched vectors size and labels size")
	}

	if len(vectors[0]) != int(idx.index.dim) {
		return errors.New("unmatched dimensions of vector and index")
	}

	rows := len(vectors)
	flatVectors := flatten2DArray(vectors)

	//as a Go []float32 is layout-compatible with a C float[] so we can pass  Go slice directly to the C function as a pointer to its first element.
	errCode := C.addPoints(idx.index,
		(*C.float)(unsafe.Pointer(&flatVectors[0])),
		C.int(rows),
		(*C.size_t)(unsafe.Pointer(&labels[0])),
		C.int(concurrency),
		C.int(replace))

	if int(errCode) != 0 {
		return errors.New("add point failed, check logged error to see details")
	}

	return nil
}

// flatten the vectors to prevent the "cgo argument has Go pointer to unpinned Go pointer" issue.
func flatten2DArray(vectors [][]float32) []float32 {
	rows := len(vectors)
	dim := len(vectors[0])
	flatVectors := make([]float32, 0, rows*dim)

	for _, vector := range vectors {
		flatVectors = append(flatVectors, vector...)
	}

	return flatVectors
}

func (idx *HnswIndex) SearchKNN(vectors [][]float32, topK int, concurrency int) ([][]*SearchResult, error) {
	if len(vectors) <= 0 {
		return nil, errors.New("invalid vector data")
	}

	if len(vectors[0]) != int(idx.index.dim) {
		return nil, errors.New("unmatched dimensions of vector and index")
	}

	if uint64(topK) > uint64(C.getMaxElements(idx.index)) {
		return nil, errors.New("topK is larger than maxElements")
	}

	rows := len(vectors)
	flatVectors := flatten2DArray(vectors)
	cResult := C.searchKnn(idx.index,
		(*C.float)(unsafe.Pointer(&flatVectors[0])),
		C.int(rows),
		C.int(topK),
		C.int(concurrency),
	)

	defer C.freeResult(cResult)

	results := make([][]*SearchResult, rows) //the resulting slice
	for rowID := range results {
		rowTopk := make([]*SearchResult, topK)
		for j := 0; j < topK; j++ {
			r := SearchResult{}
			r.Label = *(*uint64)(unsafe.Add(unsafe.Pointer(cResult.label), (rowID*topK+j)*C.sizeof_ulong))
			r.Distance = *(*float32)(unsafe.Add(unsafe.Pointer(cResult.dist), (rowID*topK+j)*C.sizeof_float))
			rowTopk[j] = &r
		}
		results[rowID] = rowTopk
	}

	return results, nil

}

func (idx *HnswIndex) GetDataByLabel(label uint64) []float32 {
	var vec []float32 = make([]float32, idx.index.dim)

	C.getDataByLabel(idx.index, C.size_t(label), (*C.float)(unsafe.Pointer(&vec)))
	return vec
}

func (idx *HnswIndex) GetAllowReplaceDeleted() bool {
	return C.getAllowReplaceDeleted(idx.index) > 0
}

func (idx *HnswIndex) MarkDeleted(label uint64) {
	C.markDeleted(idx.index, C.size_t(label))
}

func (idx *HnswIndex) UnmarkDeleted(label uint64) {
	C.unmarkDeleted(idx.index, C.size_t(label))
}

func (idx *HnswIndex) ResizeIndex(newSize uint64) {
	C.resizeIndex(idx.index, C.size_t(newSize))
}

func (idx *HnswIndex) GetMaxElements() uint64 {
	return uint64(C.getMaxElements(idx.index))
}

func (idx *HnswIndex) GetCurrentCount() uint64 {
	return uint64(C.getCurrentCount(idx.index))
}

func (idx *HnswIndex) Free() {
	C.freeHNSW(idx.index)
}
