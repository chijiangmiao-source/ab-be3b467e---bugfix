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

// poseMatrix 返回 m 经第 p 种 D4 姿态变换后的 N×N 点阵（尚未平移）。
func poseMatrix(m [][]uint8, n, p int) [][]uint8 {
	out := make([][]uint8, n)
	for r := range out {
		out[r] = make([]uint8, n)
	}
	for r := 0; r < n; r++ {
		for c := 0; c < n; c++ {
			if m[r][c] == 0 {
				continue
			}
			pr, pc := applyPose(p, r, c, n)
			out[pr][pc] = 1
		}
	}
	return out
}

// correlationPose 计算一种姿态下复检图相对参考图的线性二维互相关：
// 对横纵各 -(N-1)..(N-1) 的全部 (2N-1)^2 个整数平移 (dy,dx)，返回
// 复检点经姿态变换并平移后落在参考图缺陷上的数量（移出画布者自然不计入）。
//
// 在 L×L（L 为 >= 2N-1 的最小 2 的幂）补零网格上做频域循环相关：
//
//	corr = IFFT2(FFT2(ref) · conj(FFT2(recPose)))
//
// 两个点集都位于 N×N 内而 L >= 2N-1，跨周期卷绕不可能把点对到一起，
// 循环相关即线性相关；负偏移 a 对应频率下标 a+L。一次二维 FFT 即得到
// 全部整数平移的重合数，满尺寸稠密输入复杂度为 O(N² log N)，
// 不会退化为逐平移尝试。
//
// 精确性：矩阵元素为 0/1，相关值上界为 N² <= 262144，float64 尾数 53 位，
// L<=1024 的自实现 FFT 舍入误差远低于 0.5，四舍五入后即为精确整数；
// 规范解还会被 countOverlapDirect 用纯整数逐一复核。
//
// 注意：不能用「行投影最优 dy × 列投影最优 dx」的一维剪枝替代二维相关——
// 二维最优 (dy,dx) 的行投影分数与列投影分数都不必各自取到一维最大值，
// 稀疏点阵下会漏掉绝大多数并列最优（实测 4 组并列最优只留下 1 组）。
func correlationPose(refFreq, recPose []complex128, n, l int, roots *fftRoots) []int {
	roots.fft2(recPose, l, false)
	for i := range refFreq {
		recPose[i] = refFreq[i] * complex(real(recPose[i]), -imag(recPose[i]))
	}
	roots.fft2(recPose, l, true)

	span := 2*n - 1
	out := make([]int, span*span)
	for dy := -(n - 1); dy <= n-1; dy++ {
		fy := dy
		if fy < 0 {
			fy += l
		}
		for dx := -(n - 1); dx <= n-1; dx++ {
			fx := dx
			if fx < 0 {
				fx += l
			}
			out[(dy+n-1)*span+(dx+n-1)] = int(math.Round(real(recPose[fy*l+fx])))
		}
	}
	return out
}

// searchMaxOverlap 穷举 8 种姿态 × 横纵各 -(N-1)..(N-1) 全部整数平移，
// 用自实现的二维 FFT 精确整数互相关求最大重合数与并列最优数量，
// 并按姿态顺序 → dy 升序 → dx 升序给出规范解。
func searchMaxOverlap(ref, rec [][]uint8, n int) searchResult {
	span := 2*n - 1
	if countOnes(ref) == 0 || countOnes(rec) == 0 {
		// 任一点集为空时所有平移的重合数都是 0，全部互为并列最优。
		return searchResult{
			maxOverlap: 0,
			tieCount:   8 * span * span,
			poseIndex:  0,
			dy:         -(n - 1),
			dx:         -(n - 1),
		}
	}

	l := 1
	for l < span {
		l <<= 1
	}
	roots := newFFTRoots(l)

	// 参考图的二维 FFT 对八种姿态共用，只计算一次。
	refFreq := make([]complex128, l*l)
	for r := 0; r < n; r++ {
		for c := 0; c < n; c++ {
			if ref[r][c] != 0 {
				refFreq[r*l+c] = 1
			}
		}
	}
	roots.fft2(refFreq, l, false)

	res := searchResult{maxOverlap: -1}
	for p := 0; p < 8; p++ {
		recPose := poseMatrix(rec, n, p)
		buf := make([]complex128, l*l)
		for r := 0; r < n; r++ {
			for c := 0; c < n; c++ {
				if recPose[r][c] != 0 {
					buf[r*l+c] = 1
				}
			}
		}
		corr := correlationPose(refFreq, buf, n, l, roots)
		// corr 按 dy 升序、dx 升序排列；姿态本身也升序遍历，
		// 严格大于才替换：首个达到最大值的 (姿态,dy,dx) 即规范解。
		for i, v := range corr {
			if v > res.maxOverlap {
				res.maxOverlap = v
				res.tieCount = 1
				res.poseIndex = p
				res.dy = -(n - 1) + i/span
				res.dx = -(n - 1) + i%span
			} else if v == res.maxOverlap {
				res.tieCount++
			}
		}
	}
	return res
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
