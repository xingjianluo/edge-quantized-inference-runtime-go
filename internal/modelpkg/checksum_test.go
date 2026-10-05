package modelpkg

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"testing"
	"testing/fstest"
)

// signedFS 基于 pkgBuilder 生成内存 FS，可选择为每层注入正确校验和，
// 并可翻转指定权重文件首字节（注入的校验和始终按“未翻转”字节计算）。
func signedFS(t *testing.T, b *pkgBuilder, withChecksums bool, flip map[string]bool) fstest.MapFS {
	t.Helper()
	ffsys := b.fs(t)
	mfs, ok := ffsys.(fstest.MapFS)
	if !ok {
		t.Fatalf("夹具 FS 类型非 fstest.MapFS: %T", ffsys)
	}
	checksums := map[string]string{}
	for _, l := range b.layers {
		wf, _ := l["weight_file"].(string)
		if wf == "" {
			continue
		}
		data := mfs[b.dir+"/"+wf].Data
		sum := sha256.Sum256(data)
		checksums[wf] = hex.EncodeToString(sum[:])
		if flip[wf] {
			tampered := make([]byte, len(data))
			copy(tampered, data)
			tampered[0] ^= 0xFF
			mfs[b.dir+"/"+wf] = &fstest.MapFile{Data: tampered}
		}
	}
	if withChecksums {
		root := map[string]interface{}{
			"name":          "toy",
			"version":       string(b.version),
			"precision":     string(b.precision),
			"input_dim":     b.inputDim,
			"input_quant":   b.inputQP,
			"layers":        b.layers,
			"weight_sha256": checksums,
		}
		raw := mustMarshal(t, root)
		mfs[b.dir+"/model.json"] = &fstest.MapFile{Data: raw}
	}
	return mfs
}

// signedFSWith 允许直接指定 model.json 中的 weight_sha256 内容（构造非法引用）。
func signedFSWith(t *testing.T, b *pkgBuilder, declared map[string]string) fstest.MapFS {
	t.Helper()
	mfs := b.fs(t).(fstest.MapFS)
	root := map[string]interface{}{
		"name":          "toy",
		"version":       string(b.version),
		"precision":     string(b.precision),
		"input_dim":     b.inputDim,
		"input_quant":   b.inputQP,
		"layers":        b.layers,
		"weight_sha256": declared,
	}
	mfs[b.dir+"/model.json"] = &fstest.MapFile{Data: mustMarshal(t, root)}
	return mfs
}

func mustMarshal(t *testing.T, v interface{}) []byte {
	t.Helper()
	raw, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

// TestLoad_ChecksumMismatchRejected 权重内容与声明校验和不符 -> ErrChecksumMismatch。
func TestLoad_ChecksumMismatchRejected(t *testing.T) {
	b := validBuilder()
	fsys := signedFS(t, b, true, map[string]bool{"fc1.bin": true})
	pkg, err := Load(fsys, b.effectiveDir())
	if !errors.Is(err, ErrChecksumMismatch) {
		t.Fatalf("权重被改动应报 ErrChecksumMismatch，实际 pkg=%v err=%v", pkg != nil, err)
	}
	if pkg != nil {
		t.Fatal("校验失败不得返回包")
	}
	t.Logf("判定依据: fc1.bin 首字节翻转但 model.json 校验和为原值 -> %v", err)
}

// TestLoadForUpdate_RequiresChecksums 更新加载强制每一层都有校验和。
func TestLoadForUpdate_RequiresChecksums(t *testing.T) {
	// 无任何校验和声明 -> ErrChecksumRequired。
	b := validBuilder()
	if _, err := LoadForUpdate(b.fs(t), b.effectiveDir()); !errors.Is(err, ErrChecksumRequired) {
		t.Fatalf("无校验和应 ErrChecksumRequired，实际 %v", err)
	}
	// 只声明一层（且为正确摘要）-> 仍 ErrChecksumRequired（要求每层都声明）。
	bOne := validBuilder()
	mfsOne := bOne.fs(t).(fstest.MapFS)
	sumFC1 := sha256.Sum256(mfsOne["model/fc1.bin"].Data)
	one := signedFSWith(t, bOne, map[string]string{"fc1.bin": hex.EncodeToString(sumFC1[:])})
	if _, err := LoadForUpdate(one, "model"); !errors.Is(err, ErrChecksumRequired) {
		t.Fatalf("只声明部分层也应 ErrChecksumRequired，实际 %v", err)
	}
	t.Logf("判定依据: 更新包无校验和/缺层校验和 -> ErrChecksumRequired")

	// 两层都声明正确校验和 -> 成功；相同内容两次加载指纹一致。
	p1, err := LoadForUpdate(signedFS(t, validBuilder(), true, nil), "model")
	if err != nil {
		t.Fatalf("完整校验和应加载成功: %v", err)
	}
	p2, err := LoadForUpdate(signedFS(t, validBuilder(), true, nil), "model")
	if err != nil {
		t.Fatal(err)
	}
	if p1.Revision() != p2.Revision() {
		t.Fatalf("相同内容两次加载指纹应一致: %s vs %s", p1.Revision()[:12], p2.Revision()[:12])
	}
	t.Logf("判定依据: 完整正确校验和加载成功；相同内容指纹=%s 可复现", p1.Revision()[:12])

	// 校验和引用未知权重文件 / 摘要长度非法 -> ErrMalformedPackage（结构问题）。
	ghost := hex.EncodeToString(make([]byte, sha256.Size))
	badRef := signedFSWith(t, validBuilder(), map[string]string{"ghost.bin": ghost})
	if _, err := Load(badRef, "model"); !errors.Is(err, ErrMalformedPackage) {
		t.Fatalf("引用未知权重文件应 ErrMalformedPackage，实际 %v", err)
	}
	badHex := signedFSWith(t, validBuilder(), map[string]string{"fc1.bin": ghost, "fc2.bin": "abc"})
	if _, err := Load(badHex, "model"); !errors.Is(err, ErrMalformedPackage) {
		t.Fatalf("非法长度校验和应 ErrMalformedPackage，实际 %v", err)
	}
	t.Logf("判定依据: 校验和引用未知文件/非法长度 -> ErrMalformedPackage")
}

// TestRevision_DifferentiatesWeightContent 结构/版本完全一致、仅权重不同，
// 指纹必须不同，支撑“不能悄悄用错版本”。
func TestRevision_DifferentiatesWeightContent(t *testing.T) {
	p1, err := Load(signedFS(t, validBuilder(), false, nil), "model")
	if err != nil {
		t.Fatal(err)
	}
	p2, err := Load(signedFS(t, validBuilder(), false, map[string]bool{"fc2.bin": true}), "model")
	if err != nil {
		t.Fatal(err)
	}
	if p1.Revision() == p2.Revision() {
		t.Fatal("权重内容不同但指纹相同")
	}
	t.Logf("判定依据: 仅 fc2 首字节不同，revision %s != %s",
		p1.Revision()[:12], p2.Revision()[:12])
}
