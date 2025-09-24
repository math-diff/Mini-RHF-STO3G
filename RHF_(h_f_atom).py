import numpy as np
from math import comb, factorial, erf, sqrt, exp, pi
from typing import List, Dict, Tuple
from dataclasses import dataclass
import argparse
import sys
import os

ANGSTROM_TO_BOHR = 1.8897259886

# ========== 基组 ( STO-3G 数据) ==========
sto3g_data: Dict[str, Dict] = {
    'H': {'S': {'exponents': np.array([3.42525091, 0.62391373, 0.16885540]),
                'coeffs':    np.array([0.15432897, 0.53532814, 0.44463454])}},
    'He': {'S': {'exponents': np.array([6.36242139, 1.15892300, 0.31364979]),
                 'coeffs':    np.array([0.15432897, 0.53532814, 0.44463454])}},
    'Li': {'S1': {'exponents': np.array([16.1195750, 2.9362007, 0.7946505]),
                  'coeffs':    np.array([0.15432897, 0.53532814, 0.44463454])},
           'SP': {'exponents': np.array([0.6362897, 0.1478601, 0.0480887]),
                  'coeffs_s':  np.array([-0.09996723, 0.39951283, 0.70011547]),
                  'coeffs_p':  np.array([ 0.15591627, 0.60768372, 0.39195739])}},
    'Be': {'S1': {'exponents': np.array([30.1678710, 5.4951153, 1.4871927]),
                  'coeffs':    np.array([0.15432897, 0.53532814, 0.44463454])},
           'SP': {'exponents': np.array([1.3148331, 0.3055389, 0.0993707]),
                  'coeffs_s':  np.array([-0.09996723, 0.39951283, 0.70011547]),
                  'coeffs_p':  np.array([ 0.15591627, 0.60768372, 0.39195739])}},
    'B': {'S1': {'exponents': np.array([48.7911130, 8.8873622, 2.4052670]),
                 'coeffs':    np.array([0.15432897, 0.53532814, 0.44463454])},
          'SP': {'exponents': np.array([2.2369561, 0.5198205, 0.1690618]),
                 'coeffs_s':  np.array([-0.09996723, 0.39951283, 0.70011547]),
                 'coeffs_p':  np.array([ 0.15591627, 0.60768372, 0.39195739])}},
    'C': {'S1': {'exponents': np.array([71.6168370, 13.0450960, 3.5305122]),
                 'coeffs':    np.array([0.15432897, 0.53532814, 0.44463454])},
          'SP': {'exponents': np.array([2.9412494, 0.6834831, 0.2222899]),
                 'coeffs_s':  np.array([-0.09996723, 0.39951283, 0.70011547]),
                 'coeffs_p':  np.array([ 0.15591627, 0.60768372, 0.39195739])}},
    'N': {'S1': {'exponents': np.array([99.1061690, 18.0523120, 4.8856602]),
                 'coeffs':    np.array([0.15432897, 0.53532814, 0.44463454])},
          'SP': {'exponents': np.array([3.7804559, 0.8784966, 0.2857144]),
                 'coeffs_s':  np.array([-0.09996723, 0.39951283, 0.70011547]),
                 'coeffs_p':  np.array([ 0.15591627, 0.60768372, 0.39195739])}},
    'O': {'S1': {'exponents': np.array([130.7093200, 23.8088610, 6.4436083]),
                 'coeffs':    np.array([0.15432897, 0.53532814, 0.44463454])},
          'SP': {'exponents': np.array([5.0331513, 1.1695961, 0.3803890]),
                 'coeffs_s':  np.array([-0.09996723, 0.39951283, 0.70011547]),
                 'coeffs_p':  np.array([ 0.15591627, 0.60768372, 0.39195739])}},
    'F': {'S1': {'exponents': np.array([166.6791300, 30.3608120, 8.2168207]),
                 'coeffs':    np.array([0.15432897, 0.53532814, 0.44463454])},
          'SP': {'exponents': np.array([6.4648032, 1.5022812, 0.4885885]),
                 'coeffs_s':  np.array([-0.09996723, 0.39951283, 0.70011547]),
                 'coeffs_p':  np.array([ 0.15591627, 0.60768372, 0.39195739])}}
}

periodic_table = {1:'H',2:'He',3:'Li',4:'Be',5:'B',6:'C',7:'N',8:'O',9:'F'}
symbol_to_Z = {v:k for k,v in periodic_table.items()}

# ========== PDB 解析函数 ==========
def _extract_element(line: str) -> str:
    if len(line) < 78:
        atom_name = line[12:16].strip()
        letters = ''.join([c for c in atom_name if c.isalpha()])
        if not letters:
            return ''
        if len(letters) >= 2 and letters[1].islower():
            return letters[:2].capitalize()
        return letters[0].upper()
    elem = line[76:78].strip()
    if elem:
        if len(elem) == 1:
            return elem.upper()
        return elem[0].upper() + (elem[1].lower() if elem[1].isalpha() else '')
    atom_name = line[12:16].strip()
    letters = ''.join([c for c in atom_name if c.isalpha()])
    if not letters:
        return ''
    if len(letters) >= 2 and letters[1].islower():
        return letters[:2].capitalize()
    return letters[0].upper()

def parse_pdb(pdb_path: str,
              select_model: int = 1,
              drop_h: bool = False,
              center: bool = False,
              allowed_Z = set(range(1,10))) -> Tuple[List[int], np.ndarray]:
    if not os.path.isfile(pdb_path):
       raise FileNotFoundError(f"PDB 文件不存在: {pdb_path}")
    atomic_numbers = []
    coords = []
    current_model = 0
    in_model_block = False
    with open(pdb_path,'r') as f:
        for line in f:
            rec = line[0:6].strip()
            if rec == 'MODEL':
                current_model += 1
                in_model_block = (current_model == select_model)
            elif rec == 'ENDMDL':
                if in_model_block:
                    break
                in_model_block = False
            if rec not in ('ATOM','HETATM'):
                continue
            if current_model == 0:
                in_model_block = True
                current_model = 1
            if not in_model_block:
                continue
            try:
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            except ValueError:
                continue
            elem = _extract_element(line)
            if elem == '':
                continue
            # 标准化元素：首字母大写其余小写
            elem = elem[0].upper() + (elem[1:].lower() if len(elem) > 1 else '')
            # 去掉氢
            if drop_h and elem == 'H':
                continue
            # 只支持 1-9
            if elem not in symbol_to_Z:
                raise ValueError(f"元素 {elem} 不在当前基组支持范围(H-F)内。可扩展 sto3g_data 后再用。")
            Z = symbol_to_Z[elem]
            if Z not in allowed_Z:
                raise ValueError(f"元素 Z={Z} 不在 allowed_Z 列表中。")
            atomic_numbers.append(Z)
            coords.append([x,y,z])
    if len(atomic_numbers) == 0:
        raise ValueError("未在 PDB 中解析到任何支持的原子。")
    coords = np.array(coords, dtype=float)
    if center:
        coords -= coords.mean(axis=0)
    # 转为 Bohr
    coords_bohr = coords * ANGSTROM_TO_BOHR
    return atomic_numbers, coords_bohr

# ========== 工具函数 ==========
def double_factorial(n: int) -> int:
    if n <= 0: return 1
    r = 1
    while n > 1:
        r *= n
        n -= 2
    return r

def primitive_norm(alpha, l, m, n):
    pre = (2 * alpha / np.pi) ** 0.75
    lmn = l + m + n
    num = (4 * alpha) ** (lmn / 2.0)
    denom = (double_factorial(2 * l - 1) *
             double_factorial(2 * m - 1) *
             double_factorial(2 * n - 1))
    return pre * (num / np.sqrt(denom))

def hermite_1d(l1, l2, PA, PB, p):
    E = np.zeros((l1 + 1, l2 + 1), dtype=float)
    E[0, 0] = 1.0
    for i in range(1, l1 + 1):
        term_prev2 = E[i - 2, 0] if i - 2 >= 0 else 0.0
        E[i, 0] = PA * E[i - 1, 0] + ((i - 1) / (2.0 * p)) * term_prev2
    for j in range(1, l2 + 1):
        term_prev2 = E[0, j - 2] if j - 2 >= 0 else 0.0
        E[0, j] = PB * E[0, j - 1] + ((j - 1) / (2.0 * p)) * term_prev2
    for i in range(1, l1 + 1):
        for j in range(1, l2 + 1):
            term_i2 = E[i - 2, j] if i - 2 >= 0 else 0.0
            E[i, j] = (PA * E[i - 1, j]
                       + ((i - 1) / (2.0 * p)) * term_i2
                       + (j / (2.0 * p)) * E[i - 1, j - 1])
    return E

def primitive_overlap(alpha, beta, A, B, lmn1, lmn2):
    l1, m1, n1 = lmn1
    l2, m2, n2 = lmn2
    p = alpha + beta
    mu = alpha * beta / p
    R_AB = A - B
    R2 = np.dot(R_AB, R_AB)
    P = (alpha * A + beta * B) / p
    K = np.exp(-mu * R2)
    pref = (np.pi / p) ** 1.5 * K
    PA = P - A
    PB = P - B
    Ex = hermite_1d(l1, l2, PA[0], PB[0], p)[l1, l2]
    Ey = hermite_1d(m1, m2, PA[1], PB[1], p)[m1, m2]
    Ez = hermite_1d(n1, n2, PA[2], PB[2], p)[n1, n2]
    return pref * Ex * Ey * Ez

def primitive_kinetic(alpha, beta, A, B, lmn1, lmn2):
    l1, m1, n1 = lmn1
    l2, m2, n2 = lmn2
    S_same = primitive_overlap(alpha, beta, A, B, lmn1, lmn2)
    total = 0.0
    l2_list = [l2, m2, n2]
    for dim in range(3):
        lb = l2_list[dim]
        lmn2_plus = [l2, m2, n2]
        lmn2_plus[dim] = lb + 2
        S_plus = primitive_overlap(alpha, beta, A, B, lmn1, tuple(lmn2_plus))
        if lb >= 2:
            lmn2_minus = [l2, m2, n2]
            lmn2_minus[dim] = lb - 2
            S_minus = primitive_overlap(alpha, beta, A, B, lmn1, tuple(lmn2_minus))
            term_minus = -0.5 * lb * (lb - 1) * S_minus
        else:
            term_minus = 0.0
        term_same = beta * (2 * lb + 1) * S_same
        term_plus = -2.0 * (beta ** 2) * S_plus
        total += term_minus + term_same + term_plus
    return total

def gaussian_product(alpha, beta, A, B):
    A = np.asarray(A); B = np.asarray(B)
    p = alpha + beta
    P = (alpha * A + beta * B) / p
    mu = alpha * beta / p
    diff = A - B
    c = np.exp(-mu * np.dot(diff, diff))
    return P, c

def boys(nu, x):
    if x < 1e-8:
        return 1.0 / (2 * nu + 1) - x / (2 * nu + 3) + x * x / (2 * (2 * nu + 5))
    if nu == 0:
        rt = sqrt(x)
        return 0.5 * sqrt(pi) * erf(rt) / rt
    F0 = boys(0, x)
    F_prev = F0
    for n in range(0, nu):
        F_next = ((2 * n + 1) * F_prev - exp(-x)) / (2 * x)
        F_prev = F_next
    return F_prev

def f_poly(j, l, m, a, b):
    s = 0.0
    kmin = max(0, j - m)
    kmax = min(j, l)
    for k in range(kmin, kmax + 1):
        s += comb(l, k) * comb(m, j - k) * (a ** (l - k)) * (b ** (m + k - j))
    return s

def primitive_nuclear(alpha, beta, A, B, lmn1, lmn2, C, Z):
    l1, m1, n1 = lmn1
    l2, m2, n2 = lmn2
    p = alpha + beta
    P, c = gaussian_product(alpha, beta, A, B)
    PC = P - C
    g = p
    eps = 1.0 / (4.0 * g)
    RPC2 = np.dot(PC, PC)

    def A_coef(l, r, i, l_a, l_b, A_coord, B_coord, C_coord, P_coord):
        power = l - 2 * r - 2 * i
        if power < 0: return 0.0
        term = ((-1) ** l) * f_poly(l, l_a, l_b, P_coord - A_coord, P_coord - B_coord)
        term *= ((-1) ** i) * factorial(l) * ((P_coord - C_coord) ** power) * (eps ** (r + i))
        denom = factorial(r) * factorial(i) * factorial(power)
        term /= denom
        return term

    Vsum = 0.0
    for lx in range(0, l1 + l2 + 1):
        for rx in range(0, lx // 2 + 1):
            for ix in range(0, (lx - 2 * rx) // 2 + 1):
                Ax = A_coef(lx, rx, ix, l1, l2, A[0], B[0], C[0], P[0])
                if Ax == 0.0: continue
                for ly in range(0, m1 + m2 + 1):
                    for ry in range(0, ly // 2 + 1):
                        for iy in range(0, (ly - 2 * ry) // 2 + 1):
                            Ay = A_coef(ly, ry, iy, m1, m2, A[1], B[1], C[1], P[1])
                            if Ay == 0.0: continue
                            for lz in range(0, n1 + n2 + 1):
                                for rz in range(0, lz // 2 + 1):
                                    for iz in range(0, (lz - 2 * rz) // 2 + 1):
                                        Az = A_coef(lz, rz, iz, n1, n2, A[2], B[2], C[2], P[2])
                                        if Az == 0.0: continue
                                        nu = (lx + ly + lz
                                              - 2 * (rx + ry + rz)
                                              - (ix + iy + iz))
                                        Fv = boys(nu, g * RPC2)
                                        Vsum += Ax * Ay * Az * Fv
    V = - Z * (2.0 * np.pi / g) * c * Vsum
    return V

def primitive_eri(alpha, beta, gamma, delta,
                  A, B, C, D,
                  lmn1, lmn2, lmn3, lmn4):
    la, ma, na = lmn1
    lb, mb, nb = lmn2
    lc, mc, nc = lmn3
    ld, md, nd = lmn4

    g1 = alpha + beta
    g2 = gamma + delta
    Rp, c1 = gaussian_product(alpha, beta, A, B)
    Rq, c2 = gaussian_product(gamma, delta, C, D)
    Delta = 1.0 / (4.0 * g1) + 1.0 / (4.0 * g2)

    def theta(l_, l1_, l2_, a_, b_, r_, g_):
        t = f_poly(l_, l1_, l2_, a_, b_)
        t *= factorial(l_) * (g_ ** (r_ - l_))
        t /= (factorial(r_) * factorial(l_ - 2 * r_))
        return t

    def B_coef(l_, ll_, r_, rr_, i_,
               l1_, l2_, Acoord, Bcoord, Rpcoord, g1_,
               l3_, l4_, Ccoord, Dcoord, Rqcoord, g2_):
        bval = ((-1) ** l_) * theta(l_, l1_, l2_, Rpcoord - Acoord, Rpcoord - Bcoord, r_, g1_)
        bval *= theta(ll_, l3_, l4_, Rqcoord - Ccoord, Rqcoord - Dcoord, rr_, g2_)
        bval *= ((-1) ** i_) * (2 * Delta) ** (2 * (r_ + rr_))
        bval *= factorial(l_ + ll_ - 2 * r_ - 2 * rr_)
        power = l_ + ll_ - 2 * (r_ + rr_ + i_)
        bval *= (Delta ** i_) * ((Rpcoord - Rqcoord) ** power)
        denom = (4 * Delta) ** (l_ + ll_) * factorial(i_) * factorial(power)
        bval /= denom
        return bval

    G = 0.0
    diff = Rp - Rq
    Tval = np.dot(diff, diff) / (4.0 * Delta)

    for lx in range(0, la + lb + 1):
        for rx in range(0, lx // 2 + 1):
            for lcx in range(0, lc + ld + 1):
                for rcx in range(0, lcx // 2 + 1):
                    max_i = (lx + lcx - 2 * rx - 2 * rcx) // 2
                    for ix in range(0, max_i + 1):
                        Bx = B_coef(lx, lcx, rx, rcx, ix,
                                    la, lb, A[0], B[0], Rp[0], g1,
                                    lc, ld, C[0], D[0], Rq[0], g2)
                        if Bx == 0.0: continue
                        for ly in range(0, ma + mb + 1):
                            for ry in range(0, ly // 2 + 1):
                                for lcy in range(0, mc + md + 1):
                                    for rcy in range(0, lcy // 2 + 1):
                                        max_j = (ly + lcy - 2 * ry - 2 * rcy) // 2
                                        for iy in range(0, max_j + 1):
                                            By = B_coef(ly, lcy, ry, rcy, iy,
                                                        ma, mb, A[1], B[1], Rp[1], g1,
                                                        mc, md, C[1], D[1], Rq[1], g2)
                                            if By == 0.0: continue
                                            for lz in range(0, na + nb + 1):
                                                for rz in range(0, lz // 2 + 1):
                                                    for lcz in range(0, nc + nd + 1):
                                                        for rcz in range(0, lcz // 2 + 1):
                                                            max_k = (lz + lcz - 2 * rz - 2 * rcz) // 2
                                                            for iz in range(0, max_k + 1):
                                                                Bz = B_coef(lz, lcz, rz, rcz, iz,
                                                                            na, nb, A[2], B[2], Rp[2], g1,
                                                                            nc, nd, C[2], D[2], Rq[2], g2)
                                                                if Bz == 0.0: continue
                                                                nu = (lx + lcx + ly + lcy + lz + lcz
                                                                      - 2 * (rx + rcx + ry + rcy + rz + rcz)
                                                                      - (ix + iy + iz))
                                                                Fv = boys(nu, Tval)
                                                                G += Bx * By * Bz * Fv

    G *= c1 * c2 * 2 * (pi ** 2) / (g1 * g2) * sqrt(pi / (g1 + g2))
    return G

# ========== 基函数类 ==========
@dataclass
class ContractedGaussian:
    center: np.ndarray
    ang_mom: Tuple[int, int, int]
    exponents: np.ndarray
    coeffs: np.ndarray
    normalize_contraction: bool = True

    def __post_init__(self):
        self.center = np.asarray(self.center, dtype=float)
        self.l, self.m, self.n = self.ang_mom
        self.prim_norms = np.array([primitive_norm(a, self.l, self.m, self.n)
                                    for a in self.exponents])
        if self.normalize_contraction:
            self.normalize()
        self.cn = self.coeffs * self.prim_norms

    def contraction_self_overlap(self):
        S = 0.0
        lmn = (self.l, self.m, self.n)
        A = self.center
        for i, ai in enumerate(self.exponents):
            Ni = self.prim_norms[i]
            di = self.coeffs[i]
            for j, aj in enumerate(self.exponents):
                Nj = self.prim_norms[j]
                dj = self.coeffs[j]
                S += di * dj * Ni * Nj * primitive_overlap(ai, aj, A, A, lmn, lmn)
        return S

    def normalize(self):
        Sself = self.contraction_self_overlap()
        self.coeffs *= 1.0 / np.sqrt(Sself)

    def __repr__(self):
        label = { (0,0,0):"s", (1,0,0):"p_x", (0,1,0):"p_y", (0,0,1):"p_z" }.get(self.ang_mom,"?" )
        return f"<CGTO {label} at {self.center}>"

# ========== 构建基函数 ==========
def build_basis(atom_Z_list: List[int], coords: np.ndarray,
                sto3g: Dict[str, Dict]) -> List[ContractedGaussian]:
    basis = []
    for idx, Z in enumerate(atom_Z_list):
        sym = periodic_table[Z]
        data = sto3g[sym]
        c = coords[idx]
        if sym in ('H', 'He'):
            sblock = data['S']
            basis.append(ContractedGaussian(c, (0,0,0),
                                            sblock['exponents'], sblock['coeffs']))
        else:
            s1 = data['S1']
            basis.append(ContractedGaussian(c, (0,0,0), s1['exponents'], s1['coeffs']))
            sp = data['SP']
            basis.append(ContractedGaussian(c, (0,0,0), sp['exponents'], sp['coeffs_s']))
            for ang in [(1,0,0),(0,1,0),(0,0,1)]:
                basis.append(ContractedGaussian(c, ang, sp['exponents'], sp['coeffs_p']))
    return basis

# ========== 收缩双中心 ==========
def contracted_two_center(bf1: ContractedGaussian, bf2: ContractedGaussian, primitive_func, *p_args):
    lmn1 = (bf1.l, bf1.m, bf1.n)
    lmn2 = (bf2.l, bf2.m, bf2.n)
    A = bf1.center; B = bf2.center
    val = 0.0
    exp1 = bf1.exponents; exp2 = bf2.exponents
    cn1 = bf1.cn; cn2 = bf2.cn
    for i, a in enumerate(exp1):
        Ni_di = cn1[i]
        for j, b in enumerate(exp2):
            Nj_dj = cn2[j]
            val += Ni_di * Nj_dj * primitive_func(a, b, A, B, lmn1, lmn2, *p_args)
    return val

def contracted_overlap(bf1, bf2):
    return contracted_two_center(bf1, bf2, primitive_overlap)

def contracted_kinetic(bf1, bf2):
    return contracted_two_center(bf1, bf2, primitive_kinetic)

def contracted_nuclear(bf1, bf2, C, Z):
    return contracted_two_center(bf1, bf2, primitive_nuclear, C, Z)

# ========== ERI (收缩) ==========
def contracted_eri(bf1, bf2, bf3, bf4):
    lmn1 = (bf1.l, bf1.m, bf1.n)
    lmn2 = (bf2.l, bf2.m, bf2.n)
    lmn3 = (bf3.l, bf3.m, bf3.n)
    lmn4 = (bf4.l, bf4.m, bf4.n)
    A = bf1.center; B = bf2.center; C = bf3.center; D = bf4.center
    e1 = bf1.exponents; e2 = bf2.exponents; e3 = bf3.exponents; e4 = bf4.exponents
    cn1 = bf1.cn; cn2 = bf2.cn; cn3 = bf3.cn; cn4 = bf4.cn
    val = 0.0
    for i, a in enumerate(e1):
        ci = cn1[i]
        for j, b in enumerate(e2):
            cj = cn2[j]
            for k, c in enumerate(e3):
                ck = cn3[k]
                for l, d in enumerate(e4):
                    cl = cn4[l]
                    prim = primitive_eri(a, b, c, d, A, B, C, D, lmn1, lmn2, lmn3, lmn4)
                    val += ci * cj * ck * cl * prim
    return val

def compute_eri_8fold(basis: List[ContractedGaussian]):
    n = len(basis)
    eri = np.zeros((n, n, n, n), dtype=float)
    pair_list = [(i, j) for i in range(n) for j in range(i + 1)]
    total_unique = 0
    for idx_p, (i, j) in enumerate(pair_list):
        for idx_q in range(idx_p + 1):
            k, l = pair_list[idx_q]
            v = contracted_eri(basis[i], basis[j], basis[k], basis[l])
            total_unique += 1
            perms = (
                (i, j, k, l), (j, i, k, l), (i, j, l, k), (j, i, l, k),
                (k, l, i, j), (l, k, i, j), (k, l, j, i), (l, k, j, i)
            )
            for a,b,c,d in perms:
                eri[a,b,c,d] = v
    print(f"\nERI 计算完成：唯一积分数 = {total_unique}, 张量元素 = {n**4}")
    return eri

def print_unique_eri(eri, thresh=1e-12):
    n = eri.shape[0]
    pair_list = [(i, j) for i in range(n) for j in range(i + 1)]
    print("\n--- Unique ERI (ij|kl) index 0-based ---")
    count = 0
    for idx_p, (i, j) in enumerate(pair_list):
        for idx_q in range(idx_p + 1):
            k, l = pair_list[idx_q]
            val = eri[i, j, k, l]
            if abs(val) >= thresh:
                print(f"({i}{j}|{k}{l}) = {val: .10f}")
                count += 1
    print(f"打印非零（|val| >= {thresh:g}）唯一 ERI 数: {count}")

# ========== 一电子矩阵 ==========
def compute_pair_matrix(basis, contracted_func):
    n = len(basis)
    M = np.zeros((n, n))
    for i in range(n):
        bi = basis[i]
        for j in range(i + 1):
            bj = basis[j]
            v = contracted_func(bi, bj)
            M[i, j] = v
            M[j, i] = v
    return M

def compute_overlap_matrix(basis):
    return compute_pair_matrix(basis, contracted_overlap)

def compute_kinetic_matrix(basis):
    return compute_pair_matrix(basis, contracted_kinetic)

def compute_nuclear_matrix(basis, atom_coords, atom_Z):
    n = len(basis)
    V = np.zeros((n, n))
    for A, Z in enumerate(atom_Z):
        C = atom_coords[A]
        for i in range(n):
            bi = basis[i]
            for j in range(i + 1):
                bj = basis[j]
                val = contracted_nuclear(bi, bj, C, Z)
                V[i, j] += val
                if i != j:
                    V[j, i] += val
    return V

def nuclear_repulsion(atom_coords, atom_Z):
    e = 0.0
    n = len(atom_Z)
    for i in range(n):
        Zi = atom_Z[i]
        for j in range(i + 1, n):
            e += Zi * atom_Z[j] / np.linalg.norm(atom_coords[i] - atom_coords[j])
    return e

def print_matrix(mat, name, fmt="{: >12.6f}"):
    print(f"\n--- {name} (size={mat.shape[0]}) ---")
    header = "     " + "".join([f"   AO{j:>3}    " for j in range(mat.shape[1])])
    print(header)
    print("-"*len(header))
    for i in range(mat.shape[0]):
        row = f"AO{ i:>3} |"
        for j in range(mat.shape[1]):
            row += fmt.format(mat[i,j])
        print(row)
    print()

# ========== SCF ==========
def symmetric_orthogonalization(S):
    eigvals, U = np.linalg.eigh(S)
    thresh = 1e-12
    s_inv_sqrt = np.array([1.0/np.sqrt(v) if v > thresh else 0.0 for v in eigvals])
    X = (U * s_inv_sqrt) @ U.T
    return X

def build_initial_density(Hcore, S, nelec):
    X = symmetric_orthogonalization(S)
    Fp = X.T @ Hcore @ X
    eps, Cp = np.linalg.eigh(Fp)
    nocc = nelec // 2
    C = X @ Cp
    C_occ = C[:, :nocc]
    P = 2.0 * (C_occ @ C_occ.T)
    return P, eps, C

def build_fock(Hcore, eri, P):
    J = np.einsum('mnls,ls->mn', eri, P, optimize=True)
    K = np.einsum('mlns,ls->mn', eri, P, optimize=True)
    G = J - 0.5 * K
    return Hcore + G

def electronic_energy(P, Hcore, F):
    return 0.5 * np.sum(P * (Hcore + F))

def rms_density_change(P, P_old):
    return np.sqrt(np.mean((P - P_old) ** 2))

def scf(Hcore, S, eri, nelec,
        max_iter=100, e_conv=1e-10, d_conv=1e-6, damping=None, print_level=1):
    if nelec % 2 != 0:
        raise ValueError("当前实现为 RHF（闭壳层），电子数为奇数不支持。")
    nocc = nelec // 2
    P, eps_core, C_core = build_initial_density(Hcore, S, nelec)
    E_old = 0.0
    X = symmetric_orthogonalization(S)
    if print_level:
        print("\nSCF 开始:")
        print(f"  电子数 = {nelec}, 占据轨道数 = {nocc}")
        print(f"  初始（Core guess）最低 5 个 MO 能量 (Hartree): {eps_core[:5]}")
    for it in range(1, max_iter + 1):
        F = build_fock(Hcore, eri, P)
        Fp = X.T @ F @ X
        eps, Cp = np.linalg.eigh(Fp)
        C = X @ Cp
        C_occ = C[:, :nocc]
        P_new = 2.0 * (C_occ @ C_occ.T)
        if damping is not None and 0.0 < damping < 1.0:
            P_new = (1.0 - damping) * P_new + damping * P
        E_elec = electronic_energy(P_new, Hcore, F)
        dE = E_elec - E_old if it > 1 else E_elec
        dP = rms_density_change(P_new, P)
        if print_level:
            print(f" Iter {it:3d}: E_elec = {E_elec: .12f}  dE = {dE: .3e}  RMS_D = {dP: .3e}")
        if it > 1 and abs(dE) < e_conv and dP < d_conv:
            if print_level:
                print(" SCF 收敛。")
            return {"converged": True,"iterations": it,"E_elec": E_elec,"eps": eps,"C": C,"P": P_new}
        P = P_new
        E_old = E_elec
    if print_level:
        print(" SCF 未在最大迭代内收敛。")
    return {"converged": False,"iterations": max_iter,"E_elec": E_old,"eps": eps,"C": C,"P": P}

# ========== 对外运行接口 ==========
def run_scf(atom_Z, atom_coords_bohr, basis_data=sto3g_data,
            print_integrals=True, scf_print=1,
            max_iter=100, e_conv=1e-10, d_conv=1e-6, damping=None):
    basis_functions = build_basis(atom_Z, atom_coords_bohr, basis_data)
    S = compute_overlap_matrix(basis_functions)
    T = compute_kinetic_matrix(basis_functions)
    V = compute_nuclear_matrix(basis_functions, atom_coords_bohr, atom_Z)
    Hcore = T + V
    if print_integrals:
        print_matrix(S, "Overlap Matrix S")
        print(f"最大重叠对角偏差: {np.max(np.abs(np.diag(S) - 1.0)):.3e}")
        print_matrix(T, "Kinetic Energy Matrix T")
        print_matrix(V, "Nuclear Attraction Matrix V")
        print_matrix(Hcore, "Core Hamiltonian H = T + V")
    eri = compute_eri_8fold(basis_functions)
    if print_integrals:
        print_unique_eri(eri, thresh=1e-12)
    E_nuc = nuclear_repulsion(atom_coords_bohr, atom_Z)
    nelec = sum(atom_Z)
    scf_res = scf(Hcore, S, eri, nelec,
                  max_iter=max_iter, e_conv=e_conv, d_conv=d_conv,
                  damping=damping, print_level=scf_print)
    E_total = scf_res["E_elec"] + E_nuc
    return {
        "basis": basis_functions,
        "S": S, "T": T, "V": V, "Hcore": Hcore,
        "eri": eri,
        "E_nuc": E_nuc,
        "SCF": scf_res,
        "E_total": E_total
    }

# ========== CLI 主程序 ==========
def main():
    parser = argparse.ArgumentParser(description="RHF STO-3G (H-F) 基于 PDB 输入")
    parser.add_argument("--pdb", required=True, help="输入 PDB 文件路径")
    parser.add_argument("--model", type=int, default=1, help="选择 MODEL 序号 (默认 1)")
    parser.add_argument("--center", action="store_true", help="是否平移几何到几何中心")
    parser.add_argument("--drop-h", action="store_true", help="是否删除氢原子")
    parser.add_argument("--no-print-integrals", action="store_true", help="不打印积分矩阵/ERI")
    parser.add_argument("--damping", type=float, default=None, help="SCF 简单密度阻尼因子 (0-1 之间，例如 0.2)")
    parser.add_argument("--max-iter", type=int, default=100, help="SCF 最大迭代数")
    parser.add_argument("--e-conv", type=float, default=1e-10, help="SCF 能量收敛阈值")
    parser.add_argument("--d-conv", type=float, default=1e-6, help="SCF 密度 RMS 收敛阈值")
    args = parser.parse_args()

    print("通用 STO-3G (H–F) 版本: 读取 PDB + 积分 + RHF SCF")
    print(f"读取 PDB: {args.pdb}")

    atom_Z, coords_bohr = parse_pdb(args.pdb,
                                    select_model=args.model,
                                    drop_h=args.drop_h,
                                    center=args.center)
    print(f"解析得到原子数: {len(atom_Z)}  元素序列: {[periodic_table[Z] for Z in atom_Z]}")
    print("开始积分与 SCF 计算 ...")

    res = run_scf(atom_Z, coords_bohr, sto3g_data,
                  print_integrals=not args.no_print_integrals,
                  scf_print=1,
                  max_iter=args.max_iter,
                  e_conv=args.e_conv,
                  d_conv=args.d_conv,
                  damping=args.damping)

    scf_res = res["SCF"]
    print("\n===== RHF 结果 =====")
    print(f" SCF 收敛: {scf_res['converged']}")
    print(f" 迭代次数: {scf_res['iterations']}")
    print(f" 电子能量 E_elec = {scf_res['E_elec']:.12f} Hartree")
    print(f" 核斥能量 E_nuc  = {res['E_nuc']:.12f} Hartree")
    print(f" 总能量   E_tot  = {res['E_total']:.12f} Hartree")
    eps = scf_res["eps"]
    nelec = sum(atom_Z)
    print("\n 分子轨道能量 (Hartree):")
    for i, e in enumerate(eps):
        occ = "2" if i < (nelec//2) else "0"
        print(f"  MO {i:2d}  eps = {e: .8f}   occ = {occ}")

if __name__ == "__main__":
    main()
