package main

import "math"

// 正方形的八种旋转/镜像姿态（二面体群 D4），顺序固定。
// 规范解裁决时按此顺序优先，姿态内再按纵移 dy、横移 dx 升序。
var poses = []struct {
	Name  string
	Label string
}{
	{"identity", "恒等（不旋转不镜像）"},
	{"rot90", "顺时针旋转 90°"},
	{"rot180", "旋转 180°"},
	{"rot270", "顺时针旋转 270°"},
	{"flipLR", "左右镜像"},
	{"flipTB", "上下镜像"},
	{"transpose", "主对角线翻转"},
	{"antiTranspose", "副对角线翻转"},
}

// applyPose 把 N×N 方阵中的坐标 (r,c) 按第 p 种姿态映射到新坐标。
func applyPose(p, r, c, n int) (int, int) {
	switch p {
	case 0:
		return r, c
	case 1:
		return c, n - 1 - r
	case 2:
		return n - 1 - r, n - 1 - c
	case 3:
		return n - 1 - c, r
	case 4:
		return r, n - 1 - c
	case 5:
		return n - 1 - r, c
	case 6:
		return c, r
	default: // 7
		return n - 1 - c, n - 1 - r
	}
}

// fftRoots 预计算单位根，避免蝶形运算内重复复乘带来的漂移与开销。
// 这里的一维/二维 FFT 均为本项目自实现，不调用任何现成配准或二维相关接口。
type fftRoots struct {
	n     int
	roots []complex128 // roots[k] = e^(2πi·k/n)，0 <= k < n/2
}

func newFFTRoots(n int) *fftRoots {
	f := &fftRoots{n: n, roots: make([]complex128, n/2)}
	for k := 0; k < n/2; k++ {
		ang := 2 * math.Pi * float64(k) / float64(n)
		f.roots[k] = complex(math.Cos(ang), math.Sin(ang))
	}
	return f
}

// fft1 迭代基-2 一维 FFT，invert 时为逆变换并除以 n。
func (f *fftRoots) fft1(a []complex128, invert bool) {
	n := len(a)
	for i, j := 1, 0; i < n; i++ {
		bit := n >> 1
		for ; j&bit != 0; bit >>= 1 {
			j ^= bit
		}
		j ^= bit
		if i < j {
			a[i], a[j] = a[j], a[i]
		}
	}
	for length := 2; length <= n; length <<= 1 {
		step := f.n / length
		half := length >> 1
		for i := 0; i < n; i += length {
			k := 0
			for j := i; j < i+half; j++ {
				w := f.roots[k]
				if invert {
					w = complex(real(w), -imag(w))
				}
				v := a[j+half] * w
				a[j+half] = a[j] - v
				a[j] += v
				k += step
			}
		}
	}
	if invert {
		inv := complex(1/float64(n), 0)
		for i := range a {
			a[i] *= inv
		}
	}
}

// fft2 在 l×l 的平坦数组上做二维 FFT（先行后列）。
func (f *fftRoots) fft2(m []complex128, l int, invert bool) {
	for r := 0; r < l; r++ {
		f.fft1(m[r*l:(r+1)*l], invert)
	}
	buf := make([]complex128, l)
	for c := 0; c < l; c++ {
		for r := 0; r < l; r++ {
			buf[r] = m[r*l+c]
		}
		f.fft1(buf, invert)
		for r := 0; r < l; r++ {
			m[r*l+c] = buf[r]
		}
	}
}

// searchResult 是穷举 8 种姿态 × 全部整数平移后的规范解与统计。
type searchResult struct {
	maxOverlap int // 最大重合数
	tieCount   int // 达到最大重合的 (姿态, dy, dx) 总数
	poseIndex  int // 规范解姿态下标
	dy, dx     int // 规范解纵移、横移
}

// sparsePairCutoff 是稀疏点对路径与稠密 FFT 路径的选择阈值：
// 当 Kref*Krec（点对计数代价）超过 4*L^2（单姿态二维 FFT 的量级）时走 FFT。
const sparsePairCutoff = 4

// searchMaxOverlap 精确穷举 8 种姿态 × 横纵各 -(N-1)..(N-1) 全部 (2N-1)^2 个整数平移。
// 两条路径都不得用任何启发式剪枝——二维最优平移在任一维投影上都可能不是最优，
// 按一维最佳偏移取笛卡尔积会漏掉真正的二维最优（稀疏点阵并列解漏报即源于此）。
//
// 稀疏路径：对每个 (参考点, 姿态变换后的复检点) 对，平移 (rr-pr, cc-pc) 命中一次，
// 用位移直方图直接累计每个整数平移的重合数，复杂度 O(Kref*Krec)。
//
// 稠密路径：用自实现的二维 FFT 做精确整数互相关。对每种姿态，一次二维相关即得到
// 全部 (2N-1)^2 个整数平移的重合数，复杂度 O(N^2 log N)，满尺寸 512 稠密输入也不退化为逐点尝试。
//
// 精确性：矩阵元素为 0/1，相关值上界为 N^2 <= 262144，float64 尾数 53 位，
// L<=1024 的 FFT 舍入误差上界约 1e-10 量级，远低于 0.5，四舍五入后即为精确整数；
// 规范解还会被 countOverlapDirect 用纯整数逐一复核。
func searchMaxOverlap(ref, rec [][]uint8, n int) searchResult {
	span := 2*n - 1
	if countOnes(ref) == 0 || countOnes(rec) == 0 {
		return searchResult{
			maxOverlap: 0,
			tieCount:   8 * span * span,
			poseIndex:  0,
			dy:         -(n - 1),
			dx:         -(n - 1),
		}
	}

	refPts := gatherPoints(ref, n)
	recPts := gatherPoints(rec, n)

	// L 为不小于 2N-1 的最小 2 的幂，仅 FFT 路径需要。
	l := 1
	for l < span {
		l <<= 1
	}
	useFFT := len(refPts)*len(recPts) >= sparsePairCutoff*l*l

	var roots *fftRoots
	var refFreq, corr, fftBuf []complex128
	var hist []int // 仅稀疏路径使用：按姿态复用、按命中下标清零
	if useFFT {
		roots = newFFTRoots(l)
		refFreq = make([]complex128, l*l)
		for _, p := range refPts {
			refFreq[p[0]*l+p[1]] = 1
		}
		roots.fft2(refFreq, l, false)
		corr = make([]complex128, l*l)
		fftBuf = make([]complex128, l*l)
	} else {
		hist = make([]int, span*span)
	}

	traverse := func(p int, visit func(dy, dx, v int)) {
		if useFFT {
			// Z = IFFT(FFT(ref) · conj(FFT(posedRec)))，
			// 则 Z[dy,dx]（负偏移回绕到 +L）恰为平移 (dy,dx) 的重合数：
			// Z[d]=Σ_i A[i]B[i-d]，令 i=b+d 即 Σ_b A[b+d]B[b]。
			clear(fftBuf)
			for _, q := range recPts {
				pr, pc := applyPose(p, q[0], q[1], n)
				fftBuf[pr*l+pc] = 1
			}
			roots.fft2(fftBuf, l, false)
			for i := range fftBuf {
				corr[i] = refFreq[i] * complex(real(fftBuf[i]), -imag(fftBuf[i]))
			}
			roots.fft2(corr, l, true)
			for dy := -(n - 1); dy <= n-1; dy++ {
				ry := dy
				if ry < 0 {
					ry += l
				}
				for dx := -(n - 1); dx <= n-1; dx++ {
					cx := dx
					if cx < 0 {
						cx += l
					}
					v := int(math.Round(real(corr[ry*l+cx])))
					visit(dy, dx, v)
				}
			}
		} else {
			// 每个 (参考点, 变换后复检点) 对恰好命中平移 (rr-pr, cc-pc) 一次。
			touched := make([]int, 0, len(refPts)*len(recPts))
			for _, q := range recPts {
				pr, pc := applyPose(p, q[0], q[1], n)
				for _, a := range refPts {
					dy := a[0] - pr
					dx := a[1] - pc
					idx := (dy+n-1)*span + (dx + n - 1)
					if hist[idx] == 0 {
						touched = append(touched, idx)
					}
					hist[idx]++
				}
			}
			// 未被任何点对命中的平移重合数为 0。
			for dy := -(n - 1); dy <= n-1; dy++ {
				for dx := -(n - 1); dx <= n-1; dx++ {
					visit(dy, dx, hist[(dy+n-1)*span+(dx+n-1)])
				}
			}
			for _, idx := range touched {
				hist[idx] = 0
			}
		}
	}

	// 严格大于才替换：姿态顺序、dy、dx 均升序遍历，首个最大值即规范解。
	res := searchResult{maxOverlap: -1}
	for p := 0; p < 8; p++ {
		traverse(p, func(dy, dx, v int) {
			if v > res.maxOverlap {
				res.maxOverlap = v
				res.tieCount = 1
				res.poseIndex = p
				res.dy, res.dx = dy, dx
			} else if v == res.maxOverlap {
				res.tieCount++
			}
		})
	}
	return res
}

// gatherPoints 收集二值方阵中全部缺陷坐标。
func gatherPoints(m [][]uint8, n int) []point {
	pts := make([]point, 0)
	for r := 0; r < n; r++ {
		for c := 0; c < n; c++ {
			if m[r][c] != 0 {
				pts = append(pts, point{r, c})
			}
		}
	}
	return pts
}

// countOverlapDirect 纯整数直接计数：复检图经姿态 p 与平移 (dy,dx) 后落在参考图缺陷上的点数。
// 移出画布的缺陷不计入重合（但仍属于原图缺陷总数，由调用方统计）。
func countOverlapDirect(ref, rec [][]uint8, n, p, dy, dx int) int {
	cnt := 0
	for r := 0; r < n; r++ {
		for c := 0; c < n; c++ {
			if rec[r][c] == 0 {
				continue
			}
			pr, pc := applyPose(p, r, c, n)
			tr, tc := pr+dy, pc+dx
			if tr >= 0 && tr < n && tc >= 0 && tc < n && ref[tr][tc] != 0 {
				cnt++
			}
		}
	}
	return cnt
}
