package numgo

import (
	"bytes"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"testing"
)

// ---------- Флаги ----------
var benchDir = flag.String("benchdir", "", "base directory for benchmark temporary files (will be cleaned up after each benchmark)")

// ---------- TestMain: генерация testdata и запуск тестов ----------
func TestMain(m *testing.M) {
	flag.Parse()

	// Если Python доступен, пытаемся сгенерировать эталонные файлы
	if _, err := exec.LookPath("python3"); err == nil {
		// Проверяем, существует ли уже папка testdata (можно не перегенерировать)
		if _, err := os.Stat("testdata"); os.IsNotExist(err) {
			cmd := exec.Command("python3", "gen_testdata.py")
			if out, err := cmd.CombinedOutput(); err != nil {
				fmt.Printf("Warning: failed to generate testdata: %v\nOutput: %s\n", err, out)
			}
		}
	}

	os.Exit(m.Run())
}

// ---------- Генерация тестовых данных ----------

func randomShape(rng *rand.Rand) []int {
	ndim := rng.Intn(4)
	if ndim == 0 {
		return []int{}
	}
	shape := make([]int, ndim)
	total := 1
	for i := 0; i < ndim; i++ {
		dim := rng.Intn(5) + 1
		if total*dim > 1000 {
			dim = 1
		}
		shape[i] = dim
		total *= dim
	}
	return shape
}

func randomSliceOfType(kind reflect.Kind, shape []int, rng *rand.Rand) interface{} {
	total := 1
	for _, d := range shape {
		total *= d
	}
	if total == 0 {
		total = 1
	}

	switch kind {
	case reflect.Float32:
		slice := make([]float32, total)
		for i := range slice {
			slice[i] = rng.Float32()
		}
		return slice
	case reflect.Float64:
		slice := make([]float64, total)
		for i := range slice {
			slice[i] = rng.Float64()
		}
		return slice
	case reflect.Int8:
		slice := make([]int8, total)
		for i := range slice {
			slice[i] = int8(rng.Intn(256) - 128)
		}
		return slice
	case reflect.Int16:
		slice := make([]int16, total)
		for i := range slice {
			slice[i] = int16(rng.Intn(65536) - 32768)
		}
		return slice
	case reflect.Int32:
		slice := make([]int32, total)
		for i := range slice {
			slice[i] = rng.Int31()
		}
		return slice
	case reflect.Int64:
		slice := make([]int64, total)
		for i := range slice {
			slice[i] = rng.Int63()
		}
		return slice
	case reflect.Uint8:
		slice := make([]uint8, total)
		for i := range slice {
			slice[i] = uint8(rng.Intn(256))
		}
		return slice
	case reflect.Uint16:
		slice := make([]uint16, total)
		for i := range slice {
			slice[i] = uint16(rng.Intn(65536))
		}
		return slice
	case reflect.Uint32:
		slice := make([]uint32, total)
		for i := range slice {
			slice[i] = rng.Uint32()
		}
		return slice
	case reflect.Uint64:
		slice := make([]uint64, total)
		for i := range slice {
			slice[i] = rng.Uint64()
		}
		return slice
	case reflect.Bool:
		slice := make([]bool, total)
		for i := range slice {
			slice[i] = rng.Intn(2) == 1
		}
		return slice
	case reflect.Complex64:
		slice := make([]complex64, total)
		for i := range slice {
			slice[i] = complex(rng.Float32(), rng.Float32())
		}
		return slice
	case reflect.Complex128:
		slice := make([]complex128, total)
		for i := range slice {
			slice[i] = complex(rng.Float64(), rng.Float64())
		}
		return slice
	default:
		panic("unsupported kind")
	}
}

// ---------- Тесты корректности ----------

func TestLoadSaveRoundTrip(t *testing.T) {
	kinds := []reflect.Kind{
		reflect.Float32, reflect.Float64,
		reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64,
		reflect.Bool,
		reflect.Complex64, reflect.Complex128,
	}

	rng := rand.New(rand.NewSource(42))

	for _, kind := range kinds {
		t.Run(kind.String(), func(t *testing.T) {
			shape := randomShape(rng)
			fortranOrder := rng.Intn(2) == 0

			originalData := randomSliceOfType(kind, shape, rng)
			original := &Array{
				Shape:        shape,
				Data:         originalData,
				FortranOrder: fortranOrder,
			}

			var buf bytes.Buffer
			if err := Write(&buf, original); err != nil {
				t.Fatalf("Write failed: %v", err)
			}

			loaded, err := Read(&buf)
			if err != nil {
				t.Fatalf("Read failed: %v", err)
			}

			if !reflect.DeepEqual(original.Shape, loaded.Shape) {
				t.Errorf("Shape mismatch: expected %v, got %v", original.Shape, loaded.Shape)
			}
			if original.FortranOrder != loaded.FortranOrder {
				t.Errorf("FortranOrder mismatch: expected %v, got %v", original.FortranOrder, loaded.FortranOrder)
			}
			if !reflect.DeepEqual(original.Data, loaded.Data) {
				t.Errorf("Data mismatch for kind %v", kind)
			}
		})
	}
}

func TestScalar(t *testing.T) {
	original := &Array{
		Shape:        []int{},
		Data:         []float64{3.1415},
		FortranOrder: false,
	}

	var buf bytes.Buffer
	if err := Write(&buf, original); err != nil {
		t.Fatal(err)
	}

	loaded, err := Read(&buf)
	if err != nil {
		t.Fatal(err)
	}

	if len(loaded.Shape) != 0 {
		t.Errorf("expected empty shape, got %v", loaded.Shape)
	}
	if !reflect.DeepEqual(original.Data, loaded.Data) {
		t.Errorf("scalar data mismatch")
	}
}

func Test1DArray(t *testing.T) {
	original := &Array{
		Shape:        []int{5},
		Data:         []int32{10, 20, 30, 40, 50},
		FortranOrder: false,
	}

	var buf bytes.Buffer
	if err := Write(&buf, original); err != nil {
		t.Fatal(err)
	}

	loaded, err := Read(&buf)
	if err != nil {
		t.Fatal(err)
	}

	if !reflect.DeepEqual(original.Shape, loaded.Shape) {
		t.Errorf("shape mismatch: expected %v, got %v", original.Shape, loaded.Shape)
	}
	if !reflect.DeepEqual(original.Data, loaded.Data) {
		t.Errorf("data mismatch")
	}
}

func TestNPZRoundTrip(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "numgo_test")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	npzPath := filepath.Join(tmpDir, "test.npz")

	arrays := map[string]*Array{
		"arr1": {
			Shape: []int{2, 3},
			Data:  []float64{1, 2, 3, 4, 5, 6},
		},
		"arr2": {
			Shape: []int{4},
			Data:  []int32{10, 20, 30, 40},
		},
		"scalar": {
			Shape: []int{},
			Data:  []complex128{1 + 2i},
		},
	}

	if err := SaveNPZ(npzPath, arrays); err != nil {
		t.Fatalf("SaveNPZ failed: %v", err)
	}

	loaded, err := LoadNPZ(npzPath)
	if err != nil {
		t.Fatalf("LoadNPZ failed: %v", err)
	}

	if len(loaded) != len(arrays) {
		t.Errorf("expected %d arrays, got %d", len(arrays), len(loaded))
	}

	for name, orig := range arrays {
		got, ok := loaded[name]
		if !ok {
			t.Errorf("missing array %s", name)
			continue
		}
		if !reflect.DeepEqual(orig.Shape, got.Shape) {
			t.Errorf("%s shape mismatch: expected %v, got %v", name, orig.Shape, got.Shape)
		}
		if !reflect.DeepEqual(orig.Data, got.Data) {
			t.Errorf("%s data mismatch", name)
		}
	}
}

func TestNPZCompressedRoundTrip(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "numgo_test")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	npzPath := filepath.Join(tmpDir, "test_compressed.npz")

	arrays := map[string]*Array{
		"arr1": {
			Shape: []int{2, 3},
			Data:  []float64{1, 2, 3, 4, 5, 6},
		},
		"arr2": {
			Shape: []int{4},
			Data:  []int32{10, 20, 30, 40},
		},
		"scalar": {
			Shape: []int{},
			Data:  []complex128{1 + 2i},
		},
	}

	if err := SaveNPZCompressed(npzPath, arrays); err != nil {
		t.Fatalf("SaveNPZCompressed failed: %v", err)
	}

	loaded, err := LoadNPZ(npzPath)
	if err != nil {
		t.Fatalf("LoadNPZ failed: %v", err)
	}

	if len(loaded) != len(arrays) {
		t.Errorf("expected %d arrays, got %d", len(arrays), len(loaded))
	}

	for name, orig := range arrays {
		got, ok := loaded[name]
		if !ok {
			t.Errorf("missing array %s", name)
			continue
		}
		if !reflect.DeepEqual(orig.Shape, got.Shape) {
			t.Errorf("%s shape mismatch: expected %v, got %v", name, orig.Shape, got.Shape)
		}
		if !reflect.DeepEqual(orig.Data, got.Data) {
			t.Errorf("%s data mismatch", name)
		}
	}
}

func TestLoadFile(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "numgo_test")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	filePath := filepath.Join(tmpDir, "test.npy")

	original := &Array{
		Shape:        []int{2, 2},
		Data:         []uint16{100, 200, 300, 400},
		FortranOrder: true,
	}

	if err := Save(filePath, original); err != nil {
		t.Fatalf("Save failed: %v", err)
	}

	loaded, err := Load(filePath)
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}

	if !reflect.DeepEqual(original.Shape, loaded.Shape) {
		t.Errorf("shape mismatch")
	}
	if original.FortranOrder != loaded.FortranOrder {
		t.Errorf("fortran_order mismatch")
	}
	if !reflect.DeepEqual(original.Data, loaded.Data) {
		t.Errorf("data mismatch")
	}
}

// ---------- Кросс-языковые тесты с Python ----------

func TestLoadFromPythonNpy(t *testing.T) {
	if _, err := os.Stat("testdata"); os.IsNotExist(err) {
		t.Skip("testdata directory not found, generate it with 'python3 gen_testdata.py'")
	}

	tests := []struct {
		file     string
		expected *Array
	}{
		{
			file:     "testdata/scalar_float64.npy",
			expected: &Array{Shape: []int{}, Data: []float64{3.1415}},
		},
		{
			file:     "testdata/1d_int32.npy",
			expected: &Array{Shape: []int{3}, Data: []int32{10, 20, 30}},
		},
		{
			file: "testdata/2d_float32_F.npy",
			expected: &Array{
				Shape:        []int{2, 2},
				Data:         []float32{1.0, 3.0, 2.0, 4.0}, // Fortran order
				FortranOrder: true,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.file, func(t *testing.T) {
			arr, err := Load(tt.file)
			if err != nil {
				t.Fatalf("Load failed: %v", err)
			}
			if !reflect.DeepEqual(arr.Shape, tt.expected.Shape) {
				t.Errorf("shape mismatch: got %v, want %v", arr.Shape, tt.expected.Shape)
			}
			if arr.FortranOrder != tt.expected.FortranOrder {
				t.Errorf("fortran_order mismatch: got %v, want %v", arr.FortranOrder, tt.expected.FortranOrder)
			}
			if !reflect.DeepEqual(arr.Data, tt.expected.Data) {
				t.Errorf("data mismatch: got %v, want %v", arr.Data, tt.expected.Data)
			}
		})
	}
}

func TestLoadFromPythonNpz(t *testing.T) {
	if _, err := os.Stat("testdata/multi_arrays.npz"); os.IsNotExist(err) {
		t.Skip("testdata not found")
	}

	loaded, err := LoadNPZ("testdata/multi_arrays.npz")
	if err != nil {
		t.Fatal(err)
	}

	expected := map[string]*Array{
		"a": {Shape: []int{5}, Data: []uint16{0, 1, 2, 3, 4}},
		"b": {Shape: []int{4}, Data: []float64{0.0, 0.3333333333333333, 0.6666666666666666, 1.0}},
		"c": {Shape: []int{3}, Data: []bool{true, false, true}},
	}

	for name, want := range expected {
		got, ok := loaded[name]
		if !ok {
			t.Errorf("missing array %q", name)
			continue
		}
		if !reflect.DeepEqual(got.Shape, want.Shape) {
			t.Errorf("%q shape: got %v, want %v", name, got.Shape, want.Shape)
		}
		if !reflect.DeepEqual(got.Data, want.Data) {
			t.Errorf("%q data: got %v, want %v", name, got.Data, want.Data)
		}
	}
}

func TestSaveReadByPython(t *testing.T) {
	if _, err := exec.LookPath("python3"); err != nil {
		t.Skip("python3 not found in PATH")
	}

	tmpDir, err := os.MkdirTemp("", "numgo_py_test")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	data := []float64{1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9}
	arr := &Array{Shape: []int{3, 3}, Data: data, FortranOrder: false}
	npyPath := filepath.Join(tmpDir, "from_go.npy")

	if err := Save(npyPath, arr); err != nil {
		t.Fatal(err)
	}

	script := `
import sys
import numpy as np
arr = np.load(sys.argv[1])
expected_shape = (3, 3)
expected_data = [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]
if arr.shape != expected_shape:
    sys.exit(f"Shape mismatch: {arr.shape} != {expected_shape}")
if not np.allclose(arr.flatten(), expected_data):
    sys.exit("Data mismatch")
print("OK")
`
	cmd := exec.Command("python3", "-c", script, npyPath)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("Python verification failed: %s\nOutput: %s", err, out)
	}
}

func TestSaveNPZReadByPython(t *testing.T) {
	if _, err := exec.LookPath("python3"); err != nil {
		t.Skip("python3 not found")
	}

	tmpDir, _ := os.MkdirTemp("", "numgo_py_test")
	defer os.RemoveAll(tmpDir)

	arrays := map[string]*Array{
		"x": {Shape: []int{2}, Data: []int64{-100, 200}},
		"y": {Shape: []int{}, Data: []complex128{3 + 4i}},
	}
	npzPath := filepath.Join(tmpDir, "from_go.npz")
	if err := SaveNPZ(npzPath, arrays); err != nil {
		t.Fatal(err)
	}

	script := `
import sys, numpy as np
data = np.load(sys.argv[1])
assert 'x' in data and 'y' in data
assert data['x'].shape == (2,)
assert data['x'].dtype == np.int64
assert data['x'].tolist() == [-100, 200]
assert data['y'].shape == ()
assert data['y'].dtype == np.complex128
assert data['y'] == 3+4j
print("OK")
`
	cmd := exec.Command("python3", "-c", script, npzPath)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("Python NPZ verification failed: %s\nOutput: %s", err, out)
	}
}

// ---------- Бенчмарки ----------

func benchTempDir(b *testing.B) (string, func()) {
	base := *benchDir
	if base == "" {
		base = os.TempDir()
	} else {
		if err := os.MkdirAll(base, 0755); err != nil {
			b.Fatalf("failed to create base dir %s: %v", base, err)
		}
	}
	dir, err := os.MkdirTemp(base, "numgo_bench")
	if err != nil {
		b.Fatalf("failed to create temp dir: %v", err)
	}
	return dir, func() { os.RemoveAll(dir) }
}

func randomFloat64Array(size int) *Array {
	data := make([]float64, size)
	for i := range data {
		data[i] = rand.Float64()
	}
	return &Array{Shape: []int{size}, Data: data, FortranOrder: false}
}

var benchmarkSizesBytes = []int64{
	4 * 1024,
	40 * 1024,
	1 * 1024 * 1024,
	128 * 1024 * 1024,
}

func BenchmarkSave(b *testing.B) {
	dir, cleanup := benchTempDir(b)
	defer cleanup()

	for _, sizeBytes := range benchmarkSizesBytes {
		sizeElems := int(sizeBytes / 8)
		b.Run(fmt.Sprintf("size=%.0fKB", float64(sizeBytes)/1024), func(b *testing.B) {
			arr := randomFloat64Array(sizeElems)
			path := filepath.Join(dir, "bench.npy")
			b.SetBytes(sizeBytes)
			b.ReportMetric(float64(sizeBytes)/1e6, "MB")
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if err := Save(path, arr); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkLoad(b *testing.B) {
	dir, cleanup := benchTempDir(b)
	defer cleanup()

	for _, sizeBytes := range benchmarkSizesBytes {
		sizeElems := int(sizeBytes / 8)
		arr := randomFloat64Array(sizeElems)
		path := filepath.Join(dir, fmt.Sprintf("bench_%d.npy", sizeBytes))
		if err := Save(path, arr); err != nil {
			b.Fatal(err)
		}

		b.Run(fmt.Sprintf("size=%.0fKB", float64(sizeBytes)/1024), func(b *testing.B) {
			b.SetBytes(sizeBytes)
			b.ReportMetric(float64(sizeBytes)/1e6, "MB")
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := Load(path)
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkWrite(b *testing.B) {
	for _, sizeBytes := range benchmarkSizesBytes {
		sizeElems := int(sizeBytes / 8)
		arr := randomFloat64Array(sizeElems)
		b.Run(fmt.Sprintf("size=%.0fKB", float64(sizeBytes)/1024), func(b *testing.B) {
			var buf bytes.Buffer
			b.SetBytes(sizeBytes)
			b.ReportMetric(float64(sizeBytes)/1e6, "MB")
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				buf.Reset()
				if err := Write(&buf, arr); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkRead(b *testing.B) {
	for _, sizeBytes := range benchmarkSizesBytes {
		sizeElems := int(sizeBytes / 8)
		arr := randomFloat64Array(sizeElems)
		var buf bytes.Buffer
		if err := Write(&buf, arr); err != nil {
			b.Fatal(err)
		}
		data := buf.Bytes()

		b.Run(fmt.Sprintf("size=%.0fKB", float64(sizeBytes)/1024), func(b *testing.B) {
			b.SetBytes(sizeBytes)
			b.ReportMetric(float64(sizeBytes)/1e6, "MB")
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				reader := bytes.NewReader(data)
				_, err := Read(reader)
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func measureCompressionRatio(path string, arrays map[string]*Array, compress bool) (dataSize, fileSize int64, ratio float64, err error) {
	os.Remove(path)

	var saveErr error
	if compress {
		saveErr = SaveNPZCompressed(path, arrays)
	} else {
		saveErr = SaveNPZ(path, arrays)
	}
	if saveErr != nil {
		return 0, 0, 0, saveErr
	}

	dataSize = 0
	for _, arr := range arrays {
		dataSize += int64(reflect.ValueOf(arr.Data).Len()) * int64(reflect.TypeOf(arr.Data).Elem().Size())
	}

	info, err := os.Stat(path)
	if err != nil {
		return 0, 0, 0, err
	}
	fileSize = info.Size()

	if fileSize > 0 {
		ratio = float64(dataSize) / float64(fileSize)
	}
	return dataSize, fileSize, ratio, nil
}

func createNPZArrays(sizeBytes int64) (map[string]*Array, int64) {
	numArrays := 3
	elemsTotal := int(sizeBytes / 8)
	elemsPerArray := elemsTotal / numArrays
	rem := elemsTotal % numArrays

	arrays := make(map[string]*Array)
	totalDataBytes := int64(0)
	for i := 0; i < numArrays; i++ {
		name := fmt.Sprintf("arr_%d", i)
		size := elemsPerArray
		if i == 0 {
			size += rem
		}
		arr := randomFloat64Array(size)
		arrays[name] = arr
		totalDataBytes += int64(size) * 8
	}
	return arrays, totalDataBytes
}

func BenchmarkSaveNPZ(b *testing.B) {
	dir, cleanup := benchTempDir(b)
	defer cleanup()

	for _, sizeBytes := range benchmarkSizesBytes {
		b.Run(fmt.Sprintf("size=%.0fKB", float64(sizeBytes)/1024), func(b *testing.B) {
			arrays, totalDataBytes := createNPZArrays(sizeBytes)
			path := filepath.Join(dir, "bench.npz")

			dataSize, fileSize, ratio, err := measureCompressionRatio(path, arrays, false)
			if err != nil {
				b.Fatalf("failed to measure compression ratio: %v", err)
			}
			_ = dataSize

			b.Logf("[NPZ store] Size: %.0f KB, Data: %.3f MB, File: %.3f MB, Ratio: %.2f",
				float64(sizeBytes)/1024, float64(totalDataBytes)/1e6, float64(fileSize)/1e6, ratio)

			b.SetBytes(totalDataBytes)
			b.ReportMetric(float64(totalDataBytes)/1e6, "MB")
			b.ReportMetric(ratio, "ratio")
			b.ReportMetric(float64(fileSize)/1e6, "file_MB")

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if err := SaveNPZ(path, arrays); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkSaveNPZCompressed(b *testing.B) {
	dir, cleanup := benchTempDir(b)
	defer cleanup()

	for _, sizeBytes := range benchmarkSizesBytes {
		b.Run(fmt.Sprintf("size=%.0fKB", float64(sizeBytes)/1024), func(b *testing.B) {
			arrays, totalDataBytes := createNPZArrays(sizeBytes)
			path := filepath.Join(dir, "bench_compressed.npz")

			dataSize, fileSize, ratio, err := measureCompressionRatio(path, arrays, true)
			if err != nil {
				b.Fatalf("failed to measure compression ratio: %v", err)
			}
			_ = dataSize

			b.Logf("[NPZ deflate] Size: %.0f KB, Data: %.3f MB, File: %.3f MB, Ratio: %.2f",
				float64(sizeBytes)/1024, float64(totalDataBytes)/1e6, float64(fileSize)/1e6, ratio)

			b.SetBytes(totalDataBytes)
			b.ReportMetric(float64(totalDataBytes)/1e6, "MB")
			b.ReportMetric(ratio, "ratio")
			b.ReportMetric(float64(fileSize)/1e6, "file_MB")

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if err := SaveNPZCompressed(path, arrays); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkLoadNPZ(b *testing.B) {
	dir, cleanup := benchTempDir(b)
	defer cleanup()

	for _, sizeBytes := range benchmarkSizesBytes {
		b.Run(fmt.Sprintf("size=%.0fKB", float64(sizeBytes)/1024), func(b *testing.B) {
			arrays, totalDataBytes := createNPZArrays(sizeBytes)
			path := filepath.Join(dir, fmt.Sprintf("bench_%d.npz", sizeBytes))

			if err := SaveNPZ(path, arrays); err != nil {
				b.Fatal(err)
			}

			b.Logf("[NPZ load] Size: %.0f KB, Data: %.3f MB",
				float64(sizeBytes)/1024, float64(totalDataBytes)/1e6)

			b.SetBytes(totalDataBytes)
			b.ReportMetric(float64(totalDataBytes)/1e6, "MB")

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := LoadNPZ(path)
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
