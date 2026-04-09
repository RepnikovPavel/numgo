package numgo

import (
	"archive/zip"
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
	"reflect"
	"regexp"
	"strconv"
	"strings"
)

type Array struct {
	Shape        []int
	Data         interface{}
	FortranOrder bool
}

var magicPrefix = []byte{0x93, 'N', 'U', 'M', 'P', 'Y'}

type dtypeInfo struct {
	descr string
	size  int
	kind  reflect.Kind
}

var dtypeMap = map[string]dtypeInfo{
	"<f4":  {"<f4", 4, reflect.Float32},
	"<f8":  {"<f8", 8, reflect.Float64},
	"<i1":  {"<i1", 1, reflect.Int8},
	"<i2":  {"<i2", 2, reflect.Int16},
	"<i4":  {"<i4", 4, reflect.Int32},
	"<i8":  {"<i8", 8, reflect.Int64},
	"<u1":  {"<u1", 1, reflect.Uint8},
	"<u2":  {"<u2", 2, reflect.Uint16},
	"<u4":  {"<u4", 4, reflect.Uint32},
	"<u8":  {"<u8", 8, reflect.Uint64},
	"<c8":  {"<c8", 8, reflect.Complex64},
	"<c16": {"<c16", 16, reflect.Complex128},
	"<b1":  {"<b1", 1, reflect.Bool},

	"=f4":  {"<f4", 4, reflect.Float32},
	"=f8":  {"<f8", 8, reflect.Float64},
	"=i1":  {"<i1", 1, reflect.Int8},
	"=i2":  {"<i2", 2, reflect.Int16},
	"=i4":  {"<i4", 4, reflect.Int32},
	"=i8":  {"<i8", 8, reflect.Int64},
	"=u1":  {"<u1", 1, reflect.Uint8},
	"=u2":  {"<u2", 2, reflect.Uint16},
	"=u4":  {"<u4", 4, reflect.Uint32},
	"=u8":  {"<u8", 8, reflect.Uint64},
	"=c8":  {"<c8", 8, reflect.Complex64},
	"=c16": {"<c16", 16, reflect.Complex128},
	"=b1":  {"<b1", 1, reflect.Bool},

	"|b1": {"|b1", 1, reflect.Bool},
	"|u1": {"|u1", 1, reflect.Uint8},
	"|i1": {"|i1", 1, reflect.Int8},
}

func getDtypeInfo(descr string) (dtypeInfo, error) {
	descr = strings.TrimSpace(strings.ToLower(descr))
	info, ok := dtypeMap[descr]
	if !ok {
		if strings.HasPrefix(descr, ">") {
			return dtypeInfo{}, fmt.Errorf("big endian not supported: %s", descr)
		}
		return dtypeInfo{}, fmt.Errorf("unsupported dtype: %s", descr)
	}
	return info, nil
}

func parseHeader(header string) (shape []int, descr string, fortranOrder bool, err error) {
	// descr: 'descr': '<f8'
	reDescr := regexp.MustCompile(`'descr'\s*:\s*'([^']*)'`)
	matches := reDescr.FindStringSubmatch(header)
	if len(matches) < 2 {
		return nil, "", false, errors.New("could not find 'descr' in header")
	}
	descr = matches[1]

	// shape: 'shape': (2, 3) или ()
	reShape := regexp.MustCompile(`'shape'\s*:\s*\(([^)]*)\)`)
	matches = reShape.FindStringSubmatch(header)
	if len(matches) < 2 {
		return nil, "", false, errors.New("could not find 'shape' in header")
	}

	shapeStr := matches[1]
	if strings.TrimSpace(shapeStr) != "" {
		parts := strings.Split(shapeStr, ",")
		for _, p := range parts {
			p = strings.TrimSpace(p)
			if p == "" {
				continue
			}
			val, err := strconv.Atoi(p)
			if err != nil {
				return nil, "", false, fmt.Errorf("invalid shape integer: %s", p)
			}
			shape = append(shape, val)
		}
	}

	reOrder := regexp.MustCompile(`'fortran_order'\s*:\s*(False|True)`)
	matches = reOrder.FindStringSubmatch(header)
	if len(matches) > 1 {
		fortranOrder = matches[1] == "True"
	}

	if shape == nil {
		shape = []int{}
	}
	return shape, descr, fortranOrder, nil
}

func Load(filename string) (*Array, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return Read(f)
}

func Read(r io.Reader) (*Array, error) {
	magic := make([]byte, 6)
	if _, err := io.ReadFull(r, magic); err != nil {
		return nil, err
	}
	if !bytes.Equal(magic, magicPrefix) {
		return nil, errors.New("invalid npy file (magic string mismatch)")
	}

	ver := make([]byte, 2)
	if _, err := io.ReadFull(r, ver); err != nil {
		return nil, err
	}

	var headerLen int
	if ver[0] == 1 && ver[1] == 0 {
		lenBytes := make([]byte, 2)
		if _, err := io.ReadFull(r, lenBytes); err != nil {
			return nil, err
		}
		headerLen = int(binary.LittleEndian.Uint16(lenBytes))
	} else if ver[0] == 2 && ver[1] == 0 {
		lenBytes := make([]byte, 4)
		if _, err := io.ReadFull(r, lenBytes); err != nil {
			return nil, err
		}
		headerLen = int(binary.LittleEndian.Uint32(lenBytes))
	} else {
		return nil, fmt.Errorf("unsupported npy version: %d.%d", ver[0], ver[1])
	}

	headerBytes := make([]byte, headerLen)
	if _, err := io.ReadFull(r, headerBytes); err != nil {
		return nil, err
	}
	header := string(headerBytes)
	header = strings.TrimSpace(header)

	shape, descr, fortranOrder, err := parseHeader(header)
	if err != nil {
		return nil, err
	}
	if shape == nil {
		shape = []int{}
	}
	dtype, err := getDtypeInfo(descr)
	if err != nil {
		return nil, err
	}

	totalElements := 1
	for _, dim := range shape {
		totalElements *= dim
	}
	if len(shape) == 0 {
		totalElements = 1
	}

	dataBytes := make([]byte, totalElements*dtype.size)
	if _, err := io.ReadFull(r, dataBytes); err != nil {
		return nil, err
	}

	dataSlice, err := bytesToSlice(dataBytes, dtype.kind)
	if err != nil {
		return nil, err
	}

	return &Array{
		Shape:        shape,
		Data:         dataSlice,
		FortranOrder: fortranOrder,
	}, nil
}

func bytesToSlice(b []byte, kind reflect.Kind) (interface{}, error) {
	switch kind {
	case reflect.Float32:
		slice := make([]float32, len(b)/4)
		buf := bytes.NewReader(b)
		if err := binary.Read(buf, binary.LittleEndian, &slice); err != nil {
			return nil, err
		}
		return slice, nil
	case reflect.Float64:
		slice := make([]float64, len(b)/8)
		buf := bytes.NewReader(b)
		if err := binary.Read(buf, binary.LittleEndian, &slice); err != nil {
			return nil, err
		}
		return slice, nil
	case reflect.Int8:
		slice := make([]int8, len(b))
		buf := bytes.NewReader(b)
		if err := binary.Read(buf, binary.LittleEndian, &slice); err != nil {
			return nil, err
		}
		return slice, nil
	case reflect.Int16:
		slice := make([]int16, len(b)/2)
		buf := bytes.NewReader(b)
		binary.Read(buf, binary.LittleEndian, &slice)
		return slice, nil
	case reflect.Int32:
		slice := make([]int32, len(b)/4)
		buf := bytes.NewReader(b)
		binary.Read(buf, binary.LittleEndian, &slice)
		return slice, nil
	case reflect.Int64:
		slice := make([]int64, len(b)/8)
		buf := bytes.NewReader(b)
		binary.Read(buf, binary.LittleEndian, &slice)
		return slice, nil
	case reflect.Uint8:
		return b, nil
	case reflect.Uint16:
		slice := make([]uint16, len(b)/2)
		buf := bytes.NewReader(b)
		binary.Read(buf, binary.LittleEndian, &slice)
		return slice, nil
	case reflect.Uint32:
		slice := make([]uint32, len(b)/4)
		buf := bytes.NewReader(b)
		binary.Read(buf, binary.LittleEndian, &slice)
		return slice, nil
	case reflect.Uint64:
		slice := make([]uint64, len(b)/8)
		buf := bytes.NewReader(b)
		binary.Read(buf, binary.LittleEndian, &slice)
		return slice, nil
	case reflect.Bool:
		slice := make([]bool, len(b))
		for i, v := range b {
			slice[i] = v != 0
		}
		return slice, nil
	case reflect.Complex64:
		slice := make([]complex64, len(b)/8)
		buf := bytes.NewReader(b)
		binary.Read(buf, binary.LittleEndian, &slice)
		return slice, nil
	case reflect.Complex128:
		slice := make([]complex128, len(b)/16)
		buf := bytes.NewReader(b)
		binary.Read(buf, binary.LittleEndian, &slice)
		return slice, nil
	default:
		return nil, fmt.Errorf("unsupported kind: %v", kind)
	}
}

func LoadNPZ(filename string) (map[string]*Array, error) {
	r, err := zip.OpenReader(filename)
	if err != nil {
		return nil, err
	}
	defer r.Close()

	result := make(map[string]*Array)
	for _, f := range r.File {
		rc, err := f.Open()
		if err != nil {
			return nil, err
		}
		arr, err := Read(rc)
		rc.Close()
		if err != nil {
			return nil, fmt.Errorf("error reading %s inside npz: %v", f.Name, err)
		}
		name := strings.TrimSuffix(f.Name, ".npy")
		result[name] = arr
	}
	return result, nil
}

func Save(filename string, arr *Array) error {
	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer f.Close()
	return Write(f, arr)
}

func Write(w io.Writer, arr *Array) error {
	descr, size, err := getDtypeFromData(arr.Data)
	if err != nil {
		return err
	}

	var shapeStr string
	if len(arr.Shape) == 0 {
		shapeStr = "()"
	} else if len(arr.Shape) == 1 {
		shapeStr = fmt.Sprintf("(%d,)", arr.Shape[0])
	} else {
		parts := make([]string, len(arr.Shape))
		for i, v := range arr.Shape {
			parts[i] = strconv.Itoa(v)
		}
		shapeStr = "(" + strings.Join(parts, ", ") + ")"
	}

	headerDict := fmt.Sprintf("{'descr': '%s', 'fortran_order': %s, 'shape': %s}",
		descr, formatBool(arr.FortranOrder), shapeStr)

	headerBytes := []byte(headerDict)
	prefixLen := 6 + 2 + 2                 // magic + version + len_field
	totalHeaderLen := len(headerBytes) + 1 // +1 for newline

	rem := (prefixLen + totalHeaderLen) % 64
	padding := 0
	if rem != 0 {
		padding = 64 - rem
	}
	finalHeaderLen := totalHeaderLen + padding
	if finalHeaderLen > 65535 {
		return errors.New("header too large for npy version 1.0")
	}

	w.Write(magicPrefix)
	w.Write([]byte{0x01, 0x00})
	lenBuf := make([]byte, 2)
	binary.LittleEndian.PutUint16(lenBuf, uint16(finalHeaderLen))
	w.Write(lenBuf)

	w.Write(headerBytes)
	for i := 0; i < padding; i++ {
		w.Write([]byte{' '})
	}
	w.Write([]byte{'\n'})

	return writeData(w, arr.Data, size)
}

func formatBool(b bool) string {
	if b {
		return "True"
	}
	return "False"
}

func getDtypeFromData(data interface{}) (string, int, error) {
	switch data.(type) {
	case []float32:
		return "<f4", 4, nil
	case []float64:
		return "<f8", 8, nil
	case []int8:
		return "<i1", 1, nil
	case []int16:
		return "<i2", 2, nil
	case []int32:
		return "<i4", 4, nil
	case []int64:
		return "<i8", 8, nil
	case []uint8:
		return "|u1", 1, nil
	case []uint16:
		return "<u2", 2, nil
	case []uint32:
		return "<u4", 4, nil
	case []uint64:
		return "<u8", 8, nil
	case []bool:
		return "|b1", 1, nil
	case []complex64:
		return "<c8", 8, nil
	case []complex128:
		return "<c16", 16, nil
	default:
		return "", 0, fmt.Errorf("unsupported go slice type for saving: %T", data)
	}
}

func writeData(w io.Writer, data interface{}, size int) error {
	return binary.Write(w, binary.LittleEndian, data)
}

func SaveNPZ(filename string, arrays map[string]*Array) error {
	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer f.Close()

	zipw := zip.NewWriter(f)
	defer zipw.Close()

	for name, arr := range arrays {
		w, err := zipw.Create(name + ".npy")
		if err != nil {
			return err
		}
		if err := Write(w, arr); err != nil {
			return err
		}
	}
	return nil
}

func SaveNPZCompressed(filename string, arrays map[string]*Array) error {
	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer f.Close()

	zipw := zip.NewWriter(f)
	defer zipw.Close()

	for name, arr := range arrays {
		header := &zip.FileHeader{
			Name:   name + ".npy",
			Method: zip.Deflate,
		}
		w, err := zipw.CreateHeader(header)
		if err != nil {
			return err
		}
		if err := Write(w, arr); err != nil {
			return err
		}
	}
	return nil
}
