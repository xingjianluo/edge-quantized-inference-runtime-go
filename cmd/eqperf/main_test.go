package main

import (
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func eqperfBin(t *testing.T) string {
	t.Helper()
	bin := filepath.Join(t.TempDir(), "eqperf")
	out, err := exec.Command("go", "build", "-o", bin, ".").CombinedOutput()
	if err != nil {
		t.Fatalf("build eqperf: %v\n%s", err, out)
	}
	return bin
}

func TestCLI_GenerateInferAndTimeout(t *testing.T) {
	bin := eqperfBin(t)
	dir := filepath.Join(t.TempDir(), "model")

	gen := exec.Command(bin, "generate", "-dir", dir)
	if out, err := gen.CombinedOutput(); err != nil {
		t.Fatalf("generate: %v\n%s", err, out)
	}
	t.Logf("判定依据: generate 在 %s 写出 int8 模型包", dir)

	infer := exec.Command(bin, "infer", "-dir", dir,
		"-inputs", "1,0;0,1;0.7,0.3")
	out, err := infer.CombinedOutput()
	if err != nil {
		t.Fatalf("infer: %v\n%s", err, out)
	}
	text := string(out)
	t.Logf("eqperf 输出:\n%s", text)
	if strings.Count(text, "label=a") != 4 { // 2 from single + 2 from batch
		t.Fatalf("输入 1,0 与 0.7,0.3 在逐条与批量下都应判为 a（共 4 次）")
	}
	if strings.Count(text, "label=b") != 2 {
		t.Fatalf("输入 0,1 在逐条与批量下都应判为 b（共 2 次）")
	}
	t.Logf("判定依据: 逐条与批量两条路径结论完全一致（a 出现 4 次、b 出现 2 次）")

	// Timeout must fail with a distinguishable reason and non-zero exit.
	timed := exec.Command(bin, "infer", "-dir", dir,
		"-latency", "200ms", "-timeout", "20ms")
	out, err = timed.CombinedOutput()
	if err == nil {
		t.Fatalf("超时命令应非零退出，输出:\n%s", out)
	}
	if !strings.Contains(string(out), "[timeout]") {
		t.Fatalf("超时失败必须带 [timeout] 原因，输出:\n%s", out)
	}
	t.Logf("判定依据: latency=200ms 超过 timeout=20ms，stderr 以 [timeout] 区分退出:\n%s", string(out))
}

func TestCLI_RejectsUnsupportedPrecisionAtGenerate(t *testing.T) {
	bin := eqperfBin(t)
	dir := filepath.Join(t.TempDir(), "bad")
	out, err := exec.Command(bin, "generate", "-dir", dir,
		"-precision", "fp32").CombinedOutput()
	if err == nil {
		t.Fatalf("fp32 精度必须被拒绝，输出: %s", out)
	}
	if !strings.Contains(string(out), "[unsupported_precision]") {
		t.Fatalf("拒绝必须带 [unsupported_precision] 原因，输出:\n%s", out)
	}
	t.Logf("判定依据: fp32 在生成阶段即以 [unsupported_precision] 拒绝:\n%s", string(out))
}
