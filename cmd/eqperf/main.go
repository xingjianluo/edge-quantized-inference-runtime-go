// Command eqperf generates a local quantized model package and runs
// one-by-one vs batch inference against it, demonstrating that the edge
// runtime rejects unsupported packages at load time and produces identical,
// reproducible predictions either way.
//
// Usage:
//
//	eqperf generate -dir ./mymodel
//	eqperf infer    -dir ./mymodel -inputs "1,0;0,1;0.7,0.3"
//	eqperf infer    -dir ./mymodel -latency 100ms -timeout 20ms   # timeout demo
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/xingjianluo/edge-quantized-inference-runtime-go/pkg/merrors"
	"github.com/xingjianluo/edge-quantized-inference-runtime-go/pkg/model"
	"github.com/xingjianluo/edge-quantized-inference-runtime-go/pkg/runtime"

	pio "github.com/xingjianluo/edge-quantized-inference-runtime-go/internal/packageio"
)

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintf(os.Stderr, "error [%s]: %v\n", merrors.ReasonOf(err), err)
		os.Exit(1)
	}
}

func run(args []string) error {
	if len(args) == 0 {
		return errors.New("subcommand required: generate | infer")
	}
	switch args[0] {
	case "generate":
		return cmdGenerate(args[1:])
	case "infer":
		return cmdInfer(args[1:])
	default:
		return fmt.Errorf("unknown subcommand %q", args[0])
	}
}

func cmdGenerate(args []string) error {
	fs := flag.NewFlagSet("generate", flag.ContinueOnError)
	dir := fs.String("dir", "demo-model", "output package directory")
	precision := fs.String("precision", "int8", "hidden-layer weight precision: int8 | int4")
	if err := fs.Parse(args); err != nil {
		return err
	}

	var wp model.Precision
	switch model.Precision(*precision) {
	case model.PrecisionINT8:
		wp = model.PrecisionINT8
	case model.PrecisionINT4:
		wp = model.PrecisionINT4
	default:
		return merrors.New(merrors.ReasonUnsupportedPrecision,
			"demo generator supports int8|int4, got %q", *precision)
	}

	// 2 -> 2 -> 2 classifier. With int4 hidden weights values stay in [-8,7].
	w0 := [][]int{{10, 0}, {0, 10}}
	if wp == model.PrecisionINT4 {
		w0 = [][]int{{7, 0}, {0, 7}}
	}
	layers := []model.LayerSpec{
		{
			Name: "fc0",
			Weights: model.TensorSpec{
				Name: "fc0/w", Precision: wp,
				Rows: 2, Cols: 2, ZeroPoint: 0, Scale: 0.1,
			},
			Bias: []float64{0, 0},
		},
		{
			Name: "fc1",
			Weights: model.TensorSpec{
				Name: "fc1/w", Precision: model.PrecisionINT8,
				Rows: 2, Cols: 2, ZeroPoint: 0, Scale: 0.1,
			},
			Bias: []float64{0, 0},
		},
	}
	m := &model.Manifest{
		FormatVersion:  model.FormatV1,
		ModelID:        "demo-classifier-" + string(wp),
		ModelVersion:   "1.0.0",
		InputPrecision: model.PrecisionINT8,
		InputScale:     0.1,
		InputZeroPoint: 0,
		InputDim:       2,
		OutputLabels:   []string{"a", "b"},
		Layers:         layers,
	}
	q := [][]int{flatten(w0), flatten([][]int{{10, 0}, {0, 10}})}
	weights, err := pio.EncodeWeights(layers, q)
	if err != nil {
		return err
	}
	bias, err := pio.EncodeBias(layers)
	if err != nil {
		return err
	}
	if err := pio.WritePackage(*dir, m, weights, bias); err != nil {
		return err
	}
	fmt.Printf("wrote %s package %s (input int8, hidden weights %s)\n",
		model.FormatV1, *dir, wp)
	return nil
}

func cmdInfer(args []string) error {
	fs := flag.NewFlagSet("infer", flag.ContinueOnError)
	dir := fs.String("dir", "demo-model", "model package directory")
	raw := fs.String("inputs", "1,0;0,1;0.7,0.3",
		"rows separated by ';' and features by ','")
	latency := fs.Duration("latency", 0, "simulated per-row device compute latency")
	timeout := fs.Duration("timeout", 0, "inference deadline (0 = none)")
	maxBatch := fs.Int("max-batch", 16, "session max batch size")
	if err := fs.Parse(args); err != nil {
		return err
	}

	rt := runtime.New(runtime.Config{})
	defer rt.Close()
	loaded, err := rt.LoadModel(*dir)
	if err != nil {
		return err
	}
	sess, err := rt.NewSession(loaded.Manifest.ModelID,
		runtime.SessionOptions{MaxBatchSize: *maxBatch, ComputeLatency: *latency})
	if err != nil {
		return err
	}
	defer sess.Close()

	rows, err := parseRows(*raw)
	if err != nil {
		return err
	}

	ctx := context.Background()
	if *timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, *timeout)
		defer cancel()
	}

	fmt.Println("one-by-one inference:")
	for i, row := range rows {
		p, err := sess.Infer(ctx, row)
		if err != nil {
			return fmt.Errorf("row %d: %w", i, err)
		}
		fmt.Printf("  %v -> label=%s confidence=%.4f logits=%v precision=%s\n",
			row, p.Label, p.Confidence, round(p.Logits), p.Precision)
	}

	fmt.Println("batch inference:")
	batch, err := sess.InferBatch(ctx, rows)
	if err != nil {
		return err
	}
	for i, p := range batch {
		fmt.Printf("  %v -> label=%s confidence=%.4f logits=%v\n",
			rows[i], p.Label, p.Confidence, round(p.Logits))
	}
	return nil
}

func parseRows(s string) ([][]float64, error) {
	var rows [][]float64
	for _, line := range strings.Split(s, ";") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		var row []float64
		for _, f := range strings.Split(line, ",") {
			v, err := strconv.ParseFloat(strings.TrimSpace(f), 64)
			if err != nil {
				return nil, merrors.New(merrors.ReasonInputValue,
					"bad input number %q: %v", f, err)
			}
			row = append(row, v)
		}
		rows = append(rows, row)
	}
	if len(rows) == 0 {
		return nil, errors.New("no input rows")
	}
	return rows, nil
}

func flatten(m [][]int) []int {
	var out []int
	for _, r := range m {
		out = append(out, r...)
	}
	return out
}

func round(xs []float64) []float64 {
	out := make([]float64, len(xs))
	for i, v := range xs {
		out[i] = float64(int(v*10000+0.5)) / 10000
		if v < 0 {
			out[i] = float64(int(v*10000-0.5)) / 10000
		}
	}
	return out
}
