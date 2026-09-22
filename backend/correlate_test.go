package main

import (
	"math/rand"
	"reflect"
	"testing"
)

// bruteForce 逐姿态逐平移直接计数，是 FFT 相关搜索的精确参照。
func bruteForce(ref, rec [][]uint8, n int) searchResult {
	res := searchResult{maxOverlap: -1}
	for p := 0; p < 8; p++ {
		for dy := -(n - 1); dy <= n-1; dy++ {
			for dx := -(n - 1); dx <= n-1; dx++ {
				v := countOverlapDirect(ref, rec, n, p, dy, dx)
				if v > res.maxOverlap {
					res = searchResult{maxOverlap: v, tieCount: 1, poseIndex: p, dy: dy, dx: dx}
				} else if v == res.maxOverlap {
					res.tieCount++
				}
			}
		}
	}
	return res
}

func randomMatrix(rng *rand.Rand, n int, density float64) [][]uint8 {
	m := make([][]uint8, n)
	for r := range m {
		m[r] = make([]uint8, n)
		for c := range m[r] {
			if rng.Float64() < density {
				m[r][c] = 1
			}
		}
	}
	return m
}

func pointsToMatrix(n int, pts [][2]int) [][]uint8 {
	m := make([][]uint8, n)
	for r := range m {
		m[r] = make([]uint8, n)
	}
	for _, p := range pts {
		m[p[0]][p[1]] = 1
	}
	return m
}

// TestSparseFourWayTie 复现审计台缺陷：稀疏 16×16 点阵存在 4 组并列最优，
// 旧实现用「行投影最优 dy × 列投影最优 dx」一维剪枝，漏掉了其中 3 组，
// 错误报告并列最优数量为 1（最大重合 4 与规范解 rot90/(4,0) 本身正确）。
func TestSparseFourWayTie(t *testing.T) {
	const n = 16
	refPts := [][2]int{
		{2, 2}, {6, 7}, {6, 8}, {7, 2}, {7, 3}, {8, 5}, {9, 4}, {9, 9},
		{10, 2}, {10, 10}, {10, 13}, {11, 12}, {12, 13}, {13, 4},
	}
	recPts := [][2]int{
		{2, 2}, {2, 6}, {2, 8}, {2, 13}, {3, 5}, {3, 11}, {5, 2}, {5, 3},
		{8, 8}, {9, 6}, {10, 10}, {11, 5}, {12, 3}, {12, 12},
	}
	ref := pointsToMatrix(n, refPts)
	rec := pointsToMatrix(n, recPts)

	got := searchMaxOverlap(ref, rec, n)
	want := bruteForce(ref, rec, n)
	if got != want {
		t.Fatalf("FFT 搜索与独立整数穷举不一致：%+v，暴力参照 %+v", got, want)
	}
	if got.maxOverlap != 4 {
		t.Fatalf("最大重合应为 4，得到 %d", got.maxOverlap)
	}
	if got.tieCount != 4 {
		t.Fatalf("并列最优数量应为 4，得到 %d", got.tieCount)
	}
	if got.poseIndex != 1 || got.dy != 4 || got.dx != 0 {
		t.Fatalf("规范解应为 rot90/(dy=4,dx=0)，得到 pose=%d dy=%d dx=%d",
			got.poseIndex, got.dy, got.dx)
	}

	// 独立整数穷举逐个列出四组最优解，明确防止回归。
	type opt struct {
		pose, dy, dx int
	}
	wantOpts := []opt{{1, 4, 0}, {2, -4, -5}, {5, -3, 0}, {7, 3, 0}}
	gotOpts := []opt{}
	for p := 0; p < 8; p++ {
		for dy := -(n - 1); dy <= n-1; dy++ {
			for dx := -(n - 1); dx <= n-1; dx++ {
				if countOverlapDirect(ref, rec, n, p, dy, dx) == 4 {
					gotOpts = append(gotOpts, opt{p, dy, dx})
				}
			}
		}
	}
	if !reflect.DeepEqual(gotOpts, wantOpts) {
		t.Fatalf("四组最优解枚举不一致：得到 %+v，期望 %+v", gotOpts, wantOpts)
	}

	// 红蓝叠加证据：规范解下 matched 恰为指定四点，缺陷总数不变；
	// 2 个复检缺陷变换后移出画布。
	resp, err := runAudit(ref, rec, n)
	if err != nil {
		t.Fatal(err)
	}
	wantMatched := map[point]bool{
		{7, 3}: true, {9, 4}: true, {10, 13}: true, {12, 13}: true,
	}
	if len(resp.Overlay.Matched) != 4 {
		t.Fatalf("matched 应为 4 点，得到 %d", len(resp.Overlay.Matched))
	}
	for _, q := range resp.Overlay.Matched {
		if !wantMatched[q] {
			t.Fatalf("matched 出现非期望重合点 %v", q)
		}
	}
	if resp.ReferenceCount != 14 || resp.RecheckCount != 14 {
		t.Fatalf("缺陷总数不得改变：ref=%d rec=%d", resp.ReferenceCount, resp.RecheckCount)
	}
	if len(resp.Overlay.ReferenceOnly) != 10 || len(resp.Overlay.RecheckOnly) != 10 {
		t.Fatalf("referenceOnly/recheckOnly 应为 10/10，得到 %d/%d",
			len(resp.Overlay.ReferenceOnly), len(resp.Overlay.RecheckOnly))
	}
	if resp.Overlay.RecheckOut != 2 {
		t.Fatalf("移出画布缺陷应为 2，得到 %d", resp.Overlay.RecheckOut)
	}
}

func TestAllZeros(t *testing.T) {
	n := 16
	empty := make([][]uint8, n)
	for r := range empty {
		empty[r] = make([]uint8, n)
	}
	nonEmpty := randomMatrix(rand.New(rand.NewSource(9)), n, 0.1)

	span := 2*n - 1
	allTies := 8 * span * span
	for _, tc := range []struct {
		name     string
		ref, rec [][]uint8
	}{
		{"两图全 0", empty, empty},
		{"参考图全 0", empty, nonEmpty},
		{"复检图全 0", nonEmpty, empty},
	} {
		got := searchMaxOverlap(tc.ref, tc.rec, n)
		if got.maxOverlap != 0 || got.tieCount != allTies {
			t.Fatalf("%s：应 max=0 ties=%d，得到 %+v", tc.name, allTies, got)
		}
		if got.poseIndex != 0 || got.dy != -(n-1) || got.dx != -(n-1) {
			t.Fatalf("%s：规范解应为 identity/(%d,%d)，得到 %+v",
				tc.name, -(n - 1), -(n - 1), got)
		}
		if got != bruteForce(tc.ref, tc.rec, n) {
			t.Fatalf("%s：与暴力穷举不一致：%+v", tc.name, got)
		}
	}

	// 经完整审计链路：全 0 对全 0 的叠加证据应为空、无画布外缺陷。
	resp, err := runAudit(empty, empty, n)
	if err != nil {
		t.Fatal(err)
	}
	if resp.MaxOverlap != 0 || resp.TieCount != allTies ||
		len(resp.Overlay.Matched) != 0 || len(resp.Overlay.ReferenceOnly) != 0 ||
		len(resp.Overlay.RecheckOnly) != 0 || resp.Overlay.RecheckOut != 0 ||
		resp.ReferenceCount != 0 || resp.RecheckCount != 0 {
		t.Fatalf("全 0 审计响应异常：%+v", resp)
	}
}

func TestUniqueOptimum(t *testing.T) {
	// 无对称结构的稀疏点阵：复检图按 rot270 平移 (+3,+5) 拍摄，
	// 审计要恢复参考图需施加其逆姿态 rot90 与反向旋转后的平移 (-5,+3)，
	// 该最优必须唯一且被精确识别（不能被任何剪枝漏掉）。
	rng := rand.New(rand.NewSource(11))
	n := 32
	ref := make([][]uint8, n)
	for r := range ref {
		ref[r] = make([]uint8, n)
	}
	pts := 0
	for pts < 25 {
		r := 6 + rng.Intn(20)
		c := 6 + rng.Intn(20)
		if ref[r][c] == 0 {
			ref[r][c] = 1
			pts++
		}
	}
	const misPose, misDy, misDx = 3, 3, 5 // 装片方向（rot270）
	rec := make([][]uint8, n)
	for r := range rec {
		rec[r] = make([]uint8, n)
	}
	for r := 0; r < n; r++ {
		for c := 0; c < n; c++ {
			if ref[r][c] == 0 {
				continue
			}
			pr, pc := applyPose(misPose, r, c, n)
			tr, tc := pr+misDy, pc+misDx
			if tr < 0 || tr >= n || tc < 0 || tc >= n {
				t.Fatalf("构造点 (%d,%d) 移出画布", tr, tc)
			}
			rec[tr][tc] = 1
		}
	}
	got := searchMaxOverlap(ref, rec, n)
	want := bruteForce(ref, rec, n)
	if got != want {
		t.Fatalf("唯一最优用例与暴力穷举不一致：%+v，参照 %+v", got, want)
	}
	if got.maxOverlap != 25 || got.tieCount != 1 {
		t.Fatalf("应唯一重合 25 点，得到 max=%d ties=%d", got.maxOverlap, got.tieCount)
	}
	// 恢复变换：rot270 的逆姿态是 rot90；f1(f3(x)+(3,5)) = x+(5,-3)，故配准平移 (-5,+3)。
	if got.poseIndex != 1 || got.dy != -5 || got.dx != 3 {
		t.Fatalf("规范解应为 rot90/(dy=-5,dx=3)，得到 %+v", got)
	}
}

func TestSearchMatchesBruteForce(t *testing.T) {
	sizes := []int{16, 17, 24, 33}
	densities := []float64{0.05, 0.3, 0.8}
	rng := rand.New(rand.NewSource(42))
	for _, n := range sizes {
		for _, d := range densities {
			ref := randomMatrix(rng, n, d)
			rec := randomMatrix(rng, n, d)
			got := searchMaxOverlap(ref, rec, n)
			want := bruteForce(ref, rec, n)
			if got != want {
				t.Fatalf("n=%d d=%.2f: FFT 搜索 %+v，暴力参照 %+v", n, d, got, want)
			}
		}
	}
}

func TestAllOnesDense(t *testing.T) {
	n := 16
	ref := randomMatrix(rand.New(rand.NewSource(1)), n, 1.0)
	rec := randomMatrix(rand.New(rand.NewSource(2)), n, 1.0)
	got := searchMaxOverlap(ref, rec, n)
	if got.maxOverlap != n*n {
		t.Fatalf("全 1 矩阵最大重合应为 %d，得到 %d", n*n, got.maxOverlap)
	}
	if got.tieCount != 8 {
		t.Fatalf("全 1 矩阵每种姿态仅在 (0,0) 达到最大，并列数应为 8，得到 %d", got.tieCount)
	}
	if got.poseIndex != 0 || got.dy != 0 || got.dx != 0 {
		t.Fatalf("规范解应为 identity/(0,0)，得到 pose=%d dy=%d dx=%d", got.poseIndex, got.dy, got.dx)
	}
}

func TestConstructedTransformRecovered(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	n := 64
	// 在中央区域布点，保证姿态+平移后全部留在画布内。
	ref := make([][]uint8, n)
	for r := range ref {
		ref[r] = make([]uint8, n)
	}
	pts := 0
	for pts < 200 {
		r := 8 + rng.Intn(48)
		c := 8 + rng.Intn(48)
		if ref[r][c] == 0 {
			ref[r][c] = 1
			pts++
		}
	}
	const pose, dy, dx = 3, 5, -7
	rec := make([][]uint8, n)
	for r := range rec {
		rec[r] = make([]uint8, n)
	}
	for r := 0; r < n; r++ {
		for c := 0; c < n; c++ {
			if ref[r][c] == 0 {
				continue
			}
			pr, pc := applyPose(pose, r, c, n)
			tr, tc := pr+dy, pc+dx
			if tr < 0 || tr >= n || tc < 0 || tc >= n {
				t.Fatalf("构造点 (%d,%d) 移出画布", tr, tc)
			}
			rec[tr][tc] = 1
		}
	}
	got := searchMaxOverlap(ref, rec, n)
	if got.maxOverlap != 200 {
		t.Fatalf("构造 200 点应全部重合，得到 %d", got.maxOverlap)
	}
	// 用规范解把复检图映射回去，必须与参考图完全一致。
	if countOverlapDirect(ref, rec, n, got.poseIndex, got.dy, got.dx) != 200 {
		t.Fatal("规范解整数复核失败")
	}
}

func TestTieBreakOrder(t *testing.T) {
	// 两个关于中心对称的点：identity 与 rot180 都能以相反平移达到 2，
	// 规范解必须落在姿态下标更小的 identity 上。
	n := 16
	ref := make([][]uint8, n)
	for r := range ref {
		ref[r] = make([]uint8, n)
	}
	ref[4][5] = 1
	ref[11][10] = 1
	rec := make([][]uint8, n)
	for r := range rec {
		rec[r] = make([]uint8, n)
	}
	rec[4][5] = 1
	rec[11][10] = 1
	got := searchMaxOverlap(ref, rec, n)
	want := bruteForce(ref, rec, n)
	if got != want {
		t.Fatalf("平局裁决不一致：FFT %+v，暴力 %+v", got, want)
	}
	if got.poseIndex != 0 || got.dy != 0 || got.dx != 0 {
		t.Fatalf("规范解应为 identity/(0,0)，得到 %+v", got)
	}
	if got.tieCount < 2 {
		t.Fatalf("对称点阵应存在并列最优，得到 tieCount=%d", got.tieCount)
	}
}

func TestOutOfCanvasStillCounted(t *testing.T) {
	// 复检图边缘点经规范平移后移出画布：不计入重合，但计入复检图总数。
	n := 16
	ref := make([][]uint8, n)
	rec := make([][]uint8, n)
	for r := 0; r < n; r++ {
		ref[r] = make([]uint8, n)
		rec[r] = make([]uint8, n)
	}
	cluster := [][2]int{{2, 3}, {3, 9}, {4, 4}, {6, 12}, {7, 7}, {9, 11}, {11, 5}, {12, 10}, {13, 13}, {5, 2}, {10, 8}, {8, 6}}
	for _, p := range cluster {
		rec[p[0]][p[1]] = 1
		ref[p[0]+2][p[1]+1] = 1 // 参考图 = 复检图平移 (+2,+1)
	}
	rec[15][15] = 1 // 边缘噪声点，平移后移出画布
	resp, err := runAudit(ref, rec, n)
	if err != nil {
		t.Fatal(err)
	}
	if resp.RecheckCount != len(cluster)+1 {
		t.Fatalf("复检图总数应含移出画布的缺陷：期望 %d，得到 %d", len(cluster)+1, resp.RecheckCount)
	}
	if resp.MaxOverlap != len(cluster) {
		t.Fatalf("最大重合应为 %d，得到 %d", len(cluster), resp.MaxOverlap)
	}
	if resp.Overlay.RecheckOut != 1 {
		t.Fatalf("应有 1 个移出画布的复检缺陷，得到 %d", resp.Overlay.RecheckOut)
	}
	if len(resp.Overlay.Matched) != resp.MaxOverlap {
		t.Fatalf("matched 数 %d 与最大重合 %d 不一致", len(resp.Overlay.Matched), resp.MaxOverlap)
	}
	if len(resp.Overlay.RecheckOnly) != resp.RecheckCount-resp.MaxOverlap {
		t.Fatal("recheckOnly 计数不一致")
	}
}

func TestLargeDense512(t *testing.T) {
	if testing.Short() {
		t.Skip("short 模式跳过满尺寸稠密用例")
	}
	n := 512
	ref := make([][]uint8, n)
	rec := make([][]uint8, n)
	for r := 0; r < n; r++ {
		ref[r] = make([]uint8, n)
		rec[r] = make([]uint8, n)
		for c := 0; c < n; c++ {
			ref[r][c] = 1
			rec[r][c] = 1
		}
	}
	got := searchMaxOverlap(ref, rec, n)
	if got.maxOverlap != n*n || got.tieCount != 8 || got.poseIndex != 0 || got.dy != 0 || got.dx != 0 {
		t.Fatalf("512 稠密结果异常：%+v", got)
	}
}
