just test
```sh
go test
```

read/write benchmark
```sh
go test -bench=NPZ -benchdir=/tmp/numgo_bench -v
```

read numpy file format using go
```go
package main

import (
    "fmt"
    "github.com/RepnikovPavel/numgo"
)

func main() {
    arr, err := numgo.Load("data.npy")
    if err != nil {
        panic(err)
    }

    fmt.Printf("Shape: %v\n", arr.Shape)
    fmt.Printf("Data: %v\n", arr.Data)
}
```

write numpy file format using go
```go
data := []float64{1.1, 2.2, 3.3, 4.4, 5.5, 6.6}
arr := &numgo.Array{
    Shape: []int{2, 3},
    Data:  data,
}

err := numgo.Save("output.npy", arr)
```

npz not compressed
```go
arrays := map[string]*numgo.Array{
    "a": {Shape: []int{3}, Data: []int32{10, 20, 30}},
    "b": {Shape: []int{2,2}, Data: []float64{1,2,3,4}},
}
err := numgo.SaveNPZ("bundle.npz", arrays)
data, err := numgo.LoadNPZ("bundle.npz")
a := data["a"]
b := data["b"]
```

npz compressed
```go
arrays := map[string]*numgo.Array{
    "a": {Shape: []int{3}, Data: []int32{10, 20, 30}},
    "b": {Shape: []int{2, 2}, Data: []float64{1, 2, 3, 4}},
}
err := numgo.SaveNPZCompressed("bundle_compressed.npz", arrays)

data, err := numgo.LoadNPZ("bundle.npz")
if err != nil {
    panic(err)
}
a := data["a"]
b := data["b"]
fmt.Println(a.Shape, a.Data)
fmt.Println(b.Shape, b.Data)

```

