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
	descr     string
	size      int // размер элемента в байтах
	kind      reflect.Kind
	isUnicode bool // true для U-типов, false для S-типов
}

var dtypeMap = map[string]dtypeInfo{
	"<f4":  {"<f4", 4, reflect.Float32, false},
	"<f8":  {"<f8", 8, reflect.Float64, false},
	"<i1":  {"<i1", 1, reflect.Int8, false},
	"<i2":  {"<i2", 2, reflect.Int16, false},
	"<i4":  {"<i4", 4, reflect.Int32, false},
	"<i8":  {"<i8", 8, reflect.Int64, false},
	"<u1":  {"<u1", 1, reflect.Uint8, false},
	"<u2":  {"<u2", 2, reflect.Uint16, false},
	"<u4":  {"<u4", 4, reflect.Uint32, false},
	"<u8":  {"<u8", 8, reflect.Uint64, false},
	"<c8":  {"<c8", 8, reflect.Complex64, false},
	"<c16": {"<c16", 16, reflect.Complex128, false},
	"<b1":  {"<b1", 1, reflect.Bool, false},

	"=f4":  {"<f4", 4, reflect.Float32, false},
	"=f8":  {"<f8", 8, reflect.Float64, false},
	"=i1":  {"<i1", 1, reflect.Int8, false},
	"=i2":  {"<i2", 2, reflect.Int16, false},
	"=i4":  {"<i4", 4, reflect.Int32, false},
	"=i8":  {"<i8", 8, reflect.Int64, false},
	"=u1":  {"<u1", 1, reflect.Uint8, false},
	"=u2":  {"<u2", 2, reflect.Uint16, false},
	"=u4":  {"<u4", 4, reflect.Uint32, false},
	"=u8":  {"<u8", 8, reflect.Uint64, false},
	"=c8":  {"<c8", 8, reflect.Complex64, false},
	"=c16": {"<c16", 16, reflect.Complex128, false},
	"=b1":  {"<b1", 1, reflect.Bool, false},

	"|b1": {"|b1", 1, reflect.Bool, false},
	"|u1": {"|u1", 1, reflect.Uint8, false},
	"|i1": {"|i1", 1, reflect.Int8, false},
}

func getDtypeInfo(descr string) (dtypeInfo, error) {
	descr = strings.TrimSpace(strings.ToLower(descr))

	// Сначала проверяем стандартные типы из dtypeMap
	if info, ok := dtypeMap[descr]; ok {
		return info, nil
	}

	// Unicode строки: <U5, |U10, =U8
	if matched, _ := regexp.MatchString(`^[|<=]?u\d+$`, descr); matched {
		re := regexp.MustCompile(`u(\d+)`)
		matches := re.FindStringSubmatch(descr)
		if len(matches) < 2 {
			return dtypeInfo{}, fmt.Errorf("invalid unicode dtype: %s", descr)
		}
		n, err := strconv.Atoi(matches[1])
		if err != nil {
			return dtypeInfo{}, err
		}
		size := n * 4 // UTF-32
		return dtypeInfo{descr: descr, size: size, kind: reflect.String, isUnicode: true}, nil
	}

	// Байтовые строки: |S5, <S10
	if matched, _ := regexp.MatchString(`^[|<=]?s\d+$`, descr); matched {
		re := regexp.MustCompile(`s(\d+)`)
		matches := re.FindStringSubmatch(descr)
		if len(matches) < 2 {
			return dtypeInfo{}, fmt.Errorf("invalid string dtype: %s", descr)
		}
		size, err := strconv.Atoi(matches[1])
		if err != nil {
			return dtypeInfo{}, err
		}
		return dtypeInfo{descr: descr, size: size, kind: reflect.String, isUnicode: false}, nil
	}

	return dtypeInfo{}, fmt.Errorf("unsupported dtype: %s", descr)
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

	dataSlice, err := bytesToSlice(dataBytes, dtype)
	if err != nil {
		return nil, err
	}

	return &Array{
		Shape:        shape,
		Data:         dataSlice,
		FortranOrder: fortranOrder,
	}, nil
}

func bytesToSlice(b []byte, info dtypeInfo) (interface{}, error) {
	switch info.kind {
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
		// Преобразуем []byte в []uint8, чтобы сохранить тип
		s := make([]uint8, len(b))
		copy(s, b)
		return s, nil
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
	case reflect.String:
		count := len(b) / info.size
		if info.isUnicode {
			slice := make([]string, count)
			for i := 0; i < count; i++ {
				start := i * info.size
				end := start + info.size
				raw := b[start:end]

				// Ищем первый нулевой символ UTF-32 (4 байта), выровненный по границе
				nullIndex := -1
				for j := 0; j+4 <= len(raw); j += 4 {
					if binary.LittleEndian.Uint32(raw[j:j+4]) == 0 {
						nullIndex = j
						break
					}
				}
				if nullIndex >= 0 {
					raw = raw[:nullIndex]
				}
				// Теперь len(raw) гарантированно кратен 4
				runes := make([]rune, len(raw)/4)
				for j := 0; j < len(raw); j += 4 {
					r := binary.LittleEndian.Uint32(raw[j : j+4])
					runes[j/4] = rune(r)
				}
				slice[i] = string(runes)
			}
			return slice, nil
		} else {
			// байтовые строки (S) – без изменений
			slice := make([]string, count)
			for i := 0; i < count; i++ {
				start := i * info.size
				end := start + info.size
				raw := b[start:end]
				trimmed := bytes.TrimRight(raw, "\x00")
				slice[i] = string(trimmed)
			}
			return slice, nil
		}
	default:
		return nil, fmt.Errorf("unsupported kind: %v", info.kind)
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
	info, err := getDtypeFromData(arr.Data)
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
		info.descr, formatBool(arr.FortranOrder), shapeStr)

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

	return writeData(w, arr.Data, info)
}

func formatBool(b bool) string {
	if b {
		return "True"
	}
	return "False"
}

func getDtypeFromData(data interface{}) (dtypeInfo, error) {
	switch v := data.(type) {
	case []float32:
		return dtypeInfo{"<f4", 4, reflect.Float32, false}, nil
	case []float64:
		return dtypeInfo{"<f8", 8, reflect.Float64, false}, nil
	case []int8:
		return dtypeInfo{"<i1", 1, reflect.Int8, false}, nil
	case []int16:
		return dtypeInfo{"<i2", 2, reflect.Int16, false}, nil
	case []int32:
		return dtypeInfo{"<i4", 4, reflect.Int32, false}, nil
	case []int64:
		return dtypeInfo{"<i8", 8, reflect.Int64, false}, nil
	case []uint8:
		return dtypeInfo{"|u1", 1, reflect.Uint8, false}, nil
	case []uint16:
		return dtypeInfo{"<u2", 2, reflect.Uint16, false}, nil
	case []uint32:
		return dtypeInfo{"<u4", 4, reflect.Uint32, false}, nil
	case []uint64:
		return dtypeInfo{"<u8", 8, reflect.Uint64, false}, nil
	case []bool:
		return dtypeInfo{"|b1", 1, reflect.Bool, false}, nil
	case []complex64:
		return dtypeInfo{"<c8", 8, reflect.Complex64, false}, nil
	case []complex128:
		return dtypeInfo{"<c16", 16, reflect.Complex128, false}, nil
	case []string:
		// Определяем, есть ли не-ASCII символы
		hasUnicode := false
		maxBytes := 0
		maxRunes := 0
		for _, s := range v {
			b := []byte(s)
			if len(b) > maxBytes {
				maxBytes = len(b)
			}
			runes := []rune(s)
			if len(runes) > maxRunes {
				maxRunes = len(runes)
			}
			for _, r := range runes {
				if r > 127 {
					hasUnicode = true
					break
				}
			}
		}
		if hasUnicode {
			descr := fmt.Sprintf("<U%d", maxRunes)
			size := maxRunes * 4
			return dtypeInfo{descr, size, reflect.String, true}, nil
		} else {
			descr := fmt.Sprintf("|S%d", maxBytes)
			return dtypeInfo{descr, maxBytes, reflect.String, false}, nil
		}
	default:
		return dtypeInfo{}, fmt.Errorf("unsupported go slice type for saving: %T", data)
	}
}

func writeData(w io.Writer, data interface{}, info dtypeInfo) error {
	switch v := data.(type) {
	case []string:
		for _, s := range v {
			if info.isUnicode {
				runes := []rune(s)
				if len(runes) > info.size/4 {
					return fmt.Errorf("string %q has %d runes, exceeds capacity %d", s, len(runes), info.size/4)
				}
				buf := make([]byte, info.size)
				for i, r := range runes {
					binary.LittleEndian.PutUint32(buf[i*4:], uint32(r))
				}
				if _, err := w.Write(buf); err != nil {
					return err
				}
			} else {
				b := []byte(s)
				if len(b) > info.size {
					return fmt.Errorf("string %q exceeds fixed length %d bytes", s, info.size)
				}
				padded := make([]byte, info.size)
				copy(padded, b)
				if _, err := w.Write(padded); err != nil {
					return err
				}
			}
		}
		return nil
	default:
		return binary.Write(w, binary.LittleEndian, data)
	}
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
