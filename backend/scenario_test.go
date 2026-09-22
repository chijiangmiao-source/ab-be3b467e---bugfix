package main

import (
	"reflect"
	"testing"
)

// 题述 16×16 审计场景：
// 参考图 14 个缺陷、复检图 14 个缺陷；最大重合 4，且有 4 组并列最优，
// 二维最优平移 (dy,dx) 在行/列一维投影上都不是最优——投影剪枝会漏掉其中 3 组。
var scenarioRefPts = []point{
	{2, 2}, {6, 7}, {6, 8}, {7, 2}, {7, 3}, {8, 5}, {9, 4},
	{9, 9}, {10, 2}, {10, 10}, {10, 13}, {11, 12}, {12, 13}, {13, 4},
}

var scenarioRecPts = []point{
	{2, 2}, {2, 6}, {2, 8}, {2, 13}, {3, 5}, {3, 11}, {5, 2},
	{5, 3}, {8, 8}, {9, 6}, {10, 10}, {11, 5}, {12, 3}, {12, 12},
}

func matrixFromPoints(n int, pts []point) [][]uint8 {
	m := make([][]uint8, n)
	for r := range m {
		m[r] = make([]uint8, n)
	}
	for _, p := range pts {
		m[p[0]][p[1]] = 1
	}
	return m
}

// allOptima 独立整数穷举：返回达到最大重合的全部 (姿态,dy,dx) 与最大重合数。
func allOptima(ref, rec [][]uint8, n int) (int, []struct{ p, dy, dx int }) {
	best := -1
	var winners []struct{ p, dy, dx int }
	for p := 0; p < 8; p++ {
		for dy := -(n - 1); dy <= n-1; dy++ {
			for dx := -(n - 1); dx <= n-1; dx++ {
				v := countOverlapDirect(ref, rec, n, p, dy, dx)
				if v > best {
					best = v
					winners = winners[:0]
					winners = append(winners, struct{ p, dy, dx int }{p, dy, dx})
				} else if v == best {
					winners = append(winners, struct{ p, dy, dx int }{p, dy, dx})
				}
			}
		}
	}
	return best, winners
}

func TestSparseFourWayTie(t *testing.T) {
	const n = 16
	ref := matrixFromPoints(n, scenarioRefPts)
	rec := matrixFromPoints(n, scenarioRecPts)

	got := searchMaxOverlap(ref, rec, n)
	want := bruteForce(ref, rec, n)
	if got != want {
		t.Fatalf("搜索结果与整数穷举不一致：%+v，期望 %+v", got, want)
	}
	if got.maxOverlap != 4 {
		t.Fatalf("最大重合应为 4，得到 %d", got.maxOverlap)
	}
	if got.tieCount != 4 {
		t.Fatalf("并列最优数量应为 4，得到 %d", got.tieCount)
	}
	// 规范裁决：按既定姿态顺序取第一组——顺时针旋转 90°、纵移 4、横移 0。
	if got.poseIndex != 1 || got.dy != 4 || got.dx != 0 {
		t.Fatalf("规范解应为 rot90/(4,0)，得到 pose=%d dy=%d dx=%d", got.poseIndex, got.dy, got.dx)
	}

	max, winners := allOptima(ref, rec, n)
	wantWinners := []struct{ p, dy, dx int }{
		{1, 4, 0},   // 顺时针旋转 90°
		{2, -4, -5}, // 旋转 180°
		{5, -3, 0},  // 上下镜像
		{7, 3, 0},   // 副对角线翻转
	}
	if max != 4 || !reflect.DeepEqual(winners, wantWinners) {
		t.Fatalf("独立穷举最优集合不符：max=%d winners=%+v", max, winners)
	}
}

func TestSparseFourWayTieOverlay(t *testing.T) {
	const n = 16
	ref := matrixFromPoints(n, scenarioRefPts)
	rec := matrixFromPoints(n, scenarioRecPts)
	resp, err := runAudit(ref, rec, n)
	if err != nil {
		t.Fatal(err)
	}
	if resp.ReferenceCount != 14 || resp.RecheckCount != 14 {
		t.Fatalf("缺陷总数不得改变：ref=%d rec=%d", resp.ReferenceCount, resp.RecheckCount)
	}
	if resp.MaxOverlap != 4 || resp.TieCount != 4 {
		t.Fatalf("最大重合/并列数异常：max=%d ties=%d", resp.MaxOverlap, resp.TieCount)
	}
	if resp.Transform.PoseIndex != 1 || resp.Transform.Dy != 4 || resp.Transform.Dx != 0 {
		t.Fatalf("规范变换异常：%+v", resp.Transform)
	}
	// 红蓝叠加证据：四个重合点逐一复核。
	wantMatched := map[point]bool{
		{7, 3}: true, {9, 4}: true, {10, 13}: true, {12, 13}: true,
	}
	gotMatched := map[point]bool{}
	for _, q := range resp.Overlay.Matched {
		gotMatched[q] = true
	}
	if !reflect.DeepEqual(gotMatched, wantMatched) {
		t.Fatalf("重合点集合不符：%+v", gotMatched)
	}
	if len(resp.Overlay.ReferenceOnly) != 14-4 {
		t.Fatalf("referenceOnly 应为 10，得到 %d", len(resp.Overlay.ReferenceOnly))
	}
	if len(resp.Overlay.RecheckOnly) != 14-4 {
		t.Fatalf("recheckOnly 应为 10，得到 %d", len(resp.Overlay.RecheckOnly))
	}
	if resp.Overlay.RecheckOut != 2 {
		t.Fatalf("规范变换下应有 2 个复检缺陷移出画布，得到 %d", resp.Overlay.RecheckOut)
	}
}

func TestAllZeroSparse(t *testing.T) {
	const n = 16
	zero := matrixFromPoints(n, nil)
	got := searchMaxOverlap(zero, zero, n)
	span := 2*n - 1
	if got.maxOverlap != 0 || got.tieCount != 8*span*span {
		t.Fatalf("全 0：所有平移均并列 0，期望 ties=%d，得到 %+v", 8*span*span, got)
	}
	if got.poseIndex != 0 || got.dy != -(n-1) || got.dx != -(n-1) {
		t.Fatalf("全 0 规范解应为 identity/(-15,-15)，得到 %+v", got)
	}
	resp, err := runAudit(zero, zero, n)
	if err != nil {
		t.Fatal(err)
	}
	if resp.MaxOverlap != 0 || resp.ReferenceCount != 0 || resp.RecheckCount != 0 ||
		len(resp.Overlay.Matched) != 0 || resp.Overlay.RecheckOut != 0 {
		t.Fatalf("全 0 叠加证据异常：%+v", resp.Overlay)
	}

	// 单边为空（参考图空、复检图非空）：所有变换点都计为画布外或仅复检图。
	rec := matrixFromPoints(n, scenarioRecPts)
	resp2, err := runAudit(zero, rec, n)
	if err != nil {
		t.Fatal(err)
	}
	if resp2.MaxOverlap != 0 || resp2.TieCount != 8*span*span || resp2.RecheckCount != 14 {
		t.Fatalf("单边为空结果异常：%+v", resp2)
	}
}

func TestSparseUniqueOptimum(t *testing.T) {
	const n = 16
	// 一组无对称性的稀疏点；复检图 = 参考图经 rot270 + (3,-2) 构造，
	// 映回参考图的规范对齐姿态是其逆姿态 rot90（关于画布中心的线性部分把 (3,-2) 映为 (-2,-3)，
	// 故平移为 (2,3)），整数穷举确认该最优唯一。
	ref := matrixFromPoints(n, []point{
		{2, 3}, {3, 6}, {4, 4}, {6, 8}, {9, 5}, {11, 9}, {12, 7},
	})
	rec := matrixFromPoints(n, nil)
	const buildPose, buildDy, buildDx = 3, 3, -2
	for r := 0; r < n; r++ {
		for c := 0; c < n; c++ {
			if ref[r][c] == 0 {
				continue
			}
			pr, pc := applyPose(buildPose, r, c, n)
			rec[pr+buildDy][pc+buildDx] = 1
		}
	}
	got := searchMaxOverlap(ref, rec, n)
	want := bruteForce(ref, rec, n)
	if got != want {
		t.Fatalf("唯一最优场景与穷举不符：%+v，期望 %+v", got, want)
	}
	if got.tieCount != 1 {
		t.Fatalf("应唯一最优，得到 tieCount=%d", got.tieCount)
	}
	if got.maxOverlap != 7 || got.poseIndex != 1 || got.dy != 2 || got.dx != 3 {
		t.Fatalf("唯一最优解异常：%+v", got)
	}
}

func TestSparseAndDensePathsAgree(t *testing.T) {
	// 阈值两侧的中等密度输入：稀疏点对路径与稠密 FFT 路径都必须与整数穷举一致。
	n := 24
	mk := func(k int, seed uint64) [][]uint8 {
		m := make([][]uint8, n)
		for r := range m {
			m[r] = make([]uint8, n)
		}
		state := seed
		for placed := 0; placed < k; {
			state = state*6364136223846793005 + 1442695040888963407
			r := int((state >> 33) % uint64(n))
			state = state*6364136223846793005 + 1442695040888963407
			c := int((state >> 33) % uint64(n))
			if m[r][c] == 0 {
				m[r][c] = 1
				placed++
			}
		}
		return m
	}
	for _, k := range []int{3, 12, 40, 130} { // 130^2=16900 > 4*64^2=16384，切换到 FFT 路径
		ref := mk(k, uint64(1+k))
		rec := mk(k, uint64(99-k))
		got := searchMaxOverlap(ref, rec, n)
		want := bruteForce(ref, rec, n)
		if got != want {
			t.Fatalf("k=%d 搜索与穷举不符：%+v，期望 %+v", k, got, want)
		}
	}
}
