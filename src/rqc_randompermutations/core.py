"""
Core simulation routines for "Error accumulation in dissipative random quantum circuits".

"""
import copy
import random
import numpy as np
import scipy as sp
from scipy.stats import unitary_group
_V_cache = {}
_S_cache = {}

def random_permutation(L):
    pos = np.arange(0, L)
    for i in range(L - 1, 0, -1):
        j = random.randint(0, i)
        pos[i], pos[j] = (pos[j], pos[i])
    return pos

def swap_composition(L, perm):
    order = list(range(L))
    swaps = []
    for target_pos, label in enumerate(perm):
        i = order.index(label)
        for a in range(i, target_pos, -1):
            order[a - 1], order[a] = (order[a], order[a - 1])
            swaps.append((a - 1, a))
    return swaps

def perm_decomposition(perm, L):
    visited = [False] * L
    container = []
    for i in range(L):
        if not visited[i]:
            j = i
            l = 0
            cycle = []
            while not visited[j]:
                visited[j] = True
                pos = copy.copy(j)
                j = int(perm[j])
                l += 1
                cycle.append([pos, j])
            container.append([cycle[i][1] for i in range(l)])
    return container

def cycle_decompositionv1(cycle):
    l = len(cycle)
    if l <= 2:
        return cycle
    else:
        layer1 = []
        layer2 = []
        for i in range(l // 2):
            swap1 = [cycle[i], cycle[l - 1 - i]]
            swap2 = [cycle[i + 1], cycle[l - 1 - i]]
            layer1.append(swap1)
            layer2.append(swap2)
        if l % 2 == 0:
            layer2 = layer2[:-1]
        return (layer1, layer2)

def optimal_swaps(cycles, L):
    container_layer1 = []
    container_layer2 = []
    for i in range(len(cycles)):
        cycle = cycles[i]
        l = len(cycle)
        if l == 1:
            continue
        if l == 2:
            container_layer1.append(cycle)
        else:
            layer1, layer2 = cycle_decompositionv1(cycle)
            container_layer1.extend(layer1)
            container_layer2.extend(layer2)
    return (container_layer1, container_layer2)

def cf_to_majos(N):
    if N not in _V_cache:
        A = np.eye(N, dtype=np.complex128)
        V = np.block([[A, 1j * A], [A, -1j * A]]) / np.sqrt(2)
        _V_cache[N] = V
    return _V_cache[N]

def dagger(M):
    return np.conjugate(np.transpose(M))

def index_to_binary1(i, L, *, dtype=np.uint8):
    if i < 0 or i >= 1 << L:
        raise ValueError(f'index {i} not in [0, 2**{L})')
    return (i >> np.arange(L - 1, -1, -1) & 1).astype(dtype)

def count_minuses(word, k, l):
    word = np.asarray(word)
    a, b = sorted((k, l))
    return int(word[a:b].sum())

def array_stringops(L, arrayindices, pos1, pos2):
    l = len(arrayindices)
    signstring = np.zeros(l)
    for i in range(len(arrayindices)):
        signstring[i] = (-1) ** count_minuses(index_to_binary1(arrayindices[i], L), pos1, pos2)
    return signstring

def sgn_star(x):
    return np.where(x >= 0, 1, -1)

def find_indices_2fixpositions(L, q1, q2):
    b1, pos1 = q1
    b2, pos2 = q2
    indexing = [2 ** (L - 1 - i) for i in range(L)]
    common = indexing[pos1] * b1 + indexing[pos2] * b2
    remaining = [indexing[p] for p in range(L) if p not in (pos1, pos2)]
    n = 2 ** (L - 2)
    indices = [0] * n
    for m in range(n):
        x = common
        for j, w in enumerate(remaining):
            if m // 2 ** j % 2 == 1:
                x += w
        indices[m] = x
    return sorted(indices)

def find_indices_fixposition(L, q1):
    b1, pos1 = q1
    indexing = [2 ** (L - 1 - i) for i in range(L)]
    common = indexing[pos1] * b1
    remaining = [indexing[p] for p in range(L) if p != pos1]
    n = 2 ** (L - 1)
    indices = [0] * n
    for m in range(n):
        x = common
        for j, w in enumerate(remaining):
            if m // 2 ** j % 2 == 1:
                x += w
        indices[m] = x
    return sorted(indices)

def HmanybodyFF_permutation(Hnambu, L, permutation):
    h = Hnambu[:L, :L]
    Delta = Hnambu[:L, L:]
    d = 2 ** L
    hmany_body = np.zeros((d, d), dtype=np.complex128)
    for i in range(L):
        pi = permutation[i]
        aii = sgn_star(i - pi)
        j = i + 1
        indexket = find_indices_fixposition(L, [0, i])
        hii = h[i, i]
        hmany_body[indexket, indexket] -= hii
        indexket = [element + 2 ** (L - 1 - i) for element in indexket]
        hmany_body[indexket, indexket] += hii
        while j < L:
            hij = h[i, j]
            Dij = Delta[i, j]
            pj = permutation[j]
            ajj = sgn_star(j - pj)
            aij = sgn_star(i - pj)
            aji = sgn_star(j - pi)
            A = hij * (aii * aij - ajj * aji)
            indexket = find_indices_2fixpositions(L, [0, i], [1, j])
            signkets = array_stringops(L, indexket, pi, pj) * A
            indexbra = find_indices_2fixpositions(L, [1, i], [0, j])
            hmany_body[indexbra, indexket] += signkets
            A = -np.conjugate(hij) * (aii * aij - ajj * aji)
            indexket = find_indices_2fixpositions(L, [1, i], [0, j])
            signkets = array_stringops(L, indexket, pi, pj) * A
            indexbra = find_indices_2fixpositions(L, [0, i], [1, j])
            hmany_body[indexbra, indexket] += signkets
            A = np.conjugate(Dij) * aii * aij - Dij * ajj * aji
            indexket = find_indices_2fixpositions(L, [0, i], [0, j])
            signkets = array_stringops(L, indexket, pi, pj) * A
            indexbra = find_indices_2fixpositions(L, [1, i], [1, j])
            hmany_body[indexbra, indexket] += signkets
            A = -(Dij * aii * aij - np.conjugate(Dij) * ajj * aji)
            indexket = find_indices_2fixpositions(L, [1, i], [1, j])
            signkets = array_stringops(L, indexket, pi, pj) * A
            indexbra = find_indices_2fixpositions(L, [0, i], [0, j])
            hmany_body[indexbra, indexket] += signkets
            j += 1
    return hmany_body

def random_uff(sites):
    V = cf_to_majos(sites)
    u = sp.stats.ortho_group.rvs(2 * sites)
    Hmajos = 1j * sp.linalg.logm(u)
    Hcffrommajos = np.dot(V, np.dot(Hmajos, dagger(V)))
    Hmbrandom = HmanybodyFF_permutation(Hcffrommajos, sites, np.arange(sites))
    Uff = sp.linalg.expm(-1j * Hmbrandom)
    return Uff

def GUE(n, sigma=1.0):
    M = np.zeros((n, n), dtype=np.complex128)
    for i in range(n):
        M[i, i] = np.random.normal(scale=sigma)
    off_scale = sigma / np.sqrt(2)
    for i in range(n):
        for j in range(i + 1, n):
            re = np.random.normal(scale=off_scale)
            im = np.random.normal(scale=off_scale)
            z = re + 1j * im
            M[i, j] = z
            M[j, i] = np.conj(z)
    return M

def random_egue(alpha, d=4):
    return sp.linalg.expm(-1j * GUE(d) * alpha)

def ff_rank(rank: int) -> int:
    return rank if rank <= 2 else 4 if rank <= 5 else 8

def kraus_from_ensemble(rank, ensemble, beta=1):
    krauses = []
    d = 4
    if ensemble == 'CUE':
        U = unitary_group.rvs(d * rank)
    elif ensemble == 'eGUE':
        U = random_egue(beta, d * rank)
    elif ensemble == 'FF':
        ntilde = 2 + int(np.log2(rank))
        U = random_uff(ntilde)
    for i in range(rank):
        krauses.append(U[i * d:(i + 1) * d, :d])
    return krauses

def layer_of_kraus(L, rank, ensemble, beta=1):
    l = L // 2
    layer_kraus = []
    for i in range(l):
        layer_kraus.append(kraus_from_ensemble(rank, ensemble, beta))
    return layer_kraus

def unitaries_from_ensemble(L, ensemble, alpha=1):
    d = 4
    unitaries = []
    l = L // 2
    if ensemble == 'CUE':
        for i in range(l):
            unitaries.append(unitary_group.rvs(d))
    elif ensemble == 'eGUE':
        for i in range(l):
            unitaries.append(random_egue(alpha, d))
    elif ensemble == 'FF':
        for i in range(l):
            unitaries.append(random_uff(int(np.log2(d))))
    return unitaries

def update_Psi_layerRQCperm(psi, L, permutation, unitaries):
    l = L // 2
    psi = np.transpose(psi, permutation)
    psi = psi.reshape([4] * l)
    for i, u in enumerate(unitaries):
        psiT = np.moveaxis(psi, i, 0)
        psiT = psiT.reshape(4, -1)
        psiT = u @ psiT
        psiT = psiT.reshape([4] * l)
        psi = np.moveaxis(psiT, 0, i)
    psi = np.transpose(psi.reshape([2] * L), np.argsort(permutation))
    return psi

def update_rho_layerRQCperm(rhovec, L, permutation, unitariesvec, krausvec, kappa):
    l = L // 2
    rhovec = np.transpose(rhovec, permutation)
    rhovec = rhovec.reshape([16] * l)
    S = build_shuffle()
    for i in range(l):
        rhoT = np.moveaxis(rhovec, i, 0)
        rhoT = rhoT.reshape(16, -1)
        diluted = (1 - kappa) * unitariesvec[i] + kappa * krausvec[i]
        dilutedprime = S @ diluted @ S.T
        rhoT = dilutedprime @ rhoT
        rhoT = rhoT.reshape([16] * l)
        rhovec = np.moveaxis(rhoT, 0, i)
    rhovec = rhovec.reshape([4] * L)
    rhovec = np.transpose(rhovec, np.argsort(permutation))
    return rhovec

def update_Psi_layerRQCpermFF(psi, L, permutation, unitariesFF):
    swaps = swap_composition(L, permutation)
    for swap in swaps:
        psi = np.moveaxis(psi, swap, (0, 1))
        psi = psi.reshape(4, -1)
        psi = FSWAP @ psi
        psi = psi.reshape([2] * L)
        psi = np.moveaxis(psi, (0, 1), swap)
    l = L // 2
    psi = psi.reshape([4] * l)
    for i, u in enumerate(unitariesFF):
        psiT = np.moveaxis(psi, i, 0)
        psiT = psiT.reshape(4, -1)
        psiT = u @ psiT
        psiT = psiT.reshape([4] * l)
        psi = np.moveaxis(psiT, 0, i)
    psi = psi.reshape([2] * L)
    for swap in reversed(swaps):
        psi = np.moveaxis(psi, swap, (0, 1))
        psi = psi.reshape(4, -1)
        psi = FSWAP @ psi
        psi = psi.reshape([2] * L)
        psi = np.moveaxis(psi, (0, 1), swap)
    return psi

def update_rho_layerRQCpermFF(rhovec, L, permutation, unitariesvec, krausvec, kappa):
    swaps = swap_composition(L, permutation)
    for swap in swaps:
        rhovec = np.moveaxis(rhovec, swap, (0, 1))
        rhovec = rhovec.reshape(16, -1)
        rhovec = SUPERFSWAP @ rhovec
        rhovec = rhovec.reshape([4] * L)
        rhovec = np.moveaxis(rhovec, (0, 1), swap)
    l = L // 2
    rhovec = rhovec.reshape([16] * l)
    S = build_shuffle()
    for i in range(l):
        rhoT = np.moveaxis(rhovec, i, 0)
        rhoT = rhoT.reshape(16, -1)
        diluted = (1 - kappa) * unitariesvec[i] + kappa * krausvec[i]
        dilutedprime = S @ diluted @ S.T
        rhoT = dilutedprime @ rhoT
        rhoT = rhoT.reshape([16] * l)
        rhovec = np.moveaxis(rhoT, 0, i)
    rhovec = rhovec.reshape([4] * L)
    for swap in reversed(swaps):
        rhovec = np.moveaxis(rhovec, swap, (0, 1))
        rhovec = rhovec.reshape(16, -1)
        rhovec = SUPERFSWAP @ rhovec
        rhovec = rhovec.reshape([4] * L)
        rhovec = np.moveaxis(rhovec, (0, 1), swap)
    return rhovec

def build_shuffle(N=2):
    if N not in _S_cache:
        'Permutation matrix mapping |i1,j1,i2,j2> ↔ |i1,i2,j1,j2>.'
        P = np.zeros((16, 16), dtype=int)
        for i1 in (0, 1):
            for j1 in (0, 1):
                for i2 in (0, 1):
                    for j2 in (0, 1):
                        idx_from = 8 * i1 + 4 * j1 + 2 * i2 + j2
                        idx_to = 8 * i1 + 4 * i2 + 2 * j1 + j2
                        P[idx_to, idx_from] = 1
        _S_cache[N] = P
    return _S_cache[N]

FSWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, -1]], dtype=np.complex128)
S = build_shuffle()
SUPERFSWAP = S @ np.kron(FSWAP, FSWAP) @ S.T

def compute_overlap_pure_dilutedRQC(L, T, kappa, rank, ensemble_unitaries='CUE', ensemble_kraus='CUE', alpha=1, beta=1):
    return compute_overlap_pure_dilutedRQCv1(L=L, T=T, kappa=kappa, rank=rank, ensemble_unitaries=ensemble_unitaries, ensemble_kraus=ensemble_kraus, gamma=0, p=0, alpha=alpha, beta=beta)

def compute_overlap_pure_dilutedRQCFF(L, T, kappa, rank, ensemble_unitaries='FF', ensemble_kraus='FF', alpha=1, beta=1):
    d = 2 ** L
    psi = np.zeros(d, dtype=np.complex128)
    rhovec = np.zeros(d ** 2, dtype=np.complex128)
    m = np.random.randint(0, d)
    psi[m] = 1
    rhovec[m * (d + 1)] = 1
    rhovec = rhovec.reshape([2] * 2 * L)
    axes = [None] * (2 * L)
    axes[::2] = list(range(L))
    axes[1::2] = list(range(L, 2 * L))
    rhovec = np.transpose(rhovec, axes)
    rhovec = rhovec.reshape([4] * L)
    psi = psi.reshape([2] * L)
    for t in range(T):
        unitaries, unitariesvec, krausvec = prepare_layer_dilutedunitary(L, rank, ensemble_unitaries, ensemble_kraus, alpha, beta)
        permutation = random_permutation(L)
        psi = update_Psi_layerRQCpermFF(psi, L, permutation, unitaries)
        rhovec = update_rho_layerRQCpermFF(rhovec, L, permutation, unitariesvec, krausvec, kappa)
    rhovec = rhovec.reshape([2, 2] * L)
    rhovec = np.transpose(rhovec, list(range(0, 2 * L, 2)) + list(range(1, 2 * L, 2)))
    rho = rhovec.reshape(d, d)
    psi = psi.reshape(d)
    overlap = np.vdot(psi, rho @ psi)
    return overlap

def compute_overlap_pure_dilutedRQCv1(L, T, kappa, rank, ensemble_unitaries='CUE', ensemble_kraus='CUE', gamma=0, p=0, alpha=1, beta=1):
    d = 2 ** L
    psi = np.zeros(d, dtype=np.complex128)
    rhovec = np.zeros(d ** 2, dtype=np.complex128)
    m = np.random.randint(0, d)
    psi[m] = 1
    psi = psi.reshape([2] * L)
    rhovec[m * (d + 1)] = 1
    rhovec = rhovec.reshape([2] * 2 * L)
    axes = [None] * (2 * L)
    axes[::2] = list(range(L))
    axes[1::2] = list(range(L, 2 * L))
    rhovec = np.transpose(rhovec, axes)
    rhovec = rhovec.reshape([4] * L)
    for t in range(T):
        unitaries, unitariesvec, krausvec = prepare_layer_dilutedunitaryv1(L, rank, ensemble_unitaries, ensemble_kraus, gamma, alpha, beta)
        permutation = random_permutation(L)
        faultypermutation = faulty_permutation(permutation, L, p)
        psi = update_Psi_layerRQCperm(psi, L, permutation, unitaries)
        rhovec = update_rho_layerRQCperm(rhovec, L, faultypermutation, unitariesvec, krausvec, kappa)
    rhovec = rhovec.reshape([2, 2] * L)
    rhovec = np.transpose(rhovec, list(range(0, 2 * L, 2)) + list(range(1, 2 * L, 2)))
    rho = rhovec.reshape(d, d)
    psi = psi.reshape(d)
    overlap = np.vdot(psi, rho @ psi)
    return overlap

def fidtheor(kapas, L, T):
    l = len(kapas)
    fidtheoor = np.zeros(l)
    for i in range(l):
        fidtheoor[i] = 1 / 2 ** L + (((16 - 15 * kapas[i]) ** (L // 2) - 1) / (4 ** L - 1)) ** T * (1 - 1 / 2 ** L)
    return fidtheoor

def find_corresponding_index(descriptors, description):
    i = 0
    while i < len(descriptors):
        if descriptors[i] == description:
            return i
        i += 1
    if i == len(descriptors):
        raise TypeError(f'No descriptor in data')

def prepare_layer_dilutedunitary(L, rank, ensemble_unitaries, ensemble_kraus, alpha=1, beta=1):
    unitaries = unitaries_from_ensemble(L, ensemble_unitaries, alpha)
    krauses = layer_of_kraus(L, rank, ensemble_kraus, beta)
    unitaries_vectorized = []
    krauses_vectorized = []
    l = L // 2
    for i in range(l):
        u = unitaries[i]
        unitaries_vectorized.append(np.kron(u, np.conjugate(u)))
        ksum = np.zeros((16, 16), dtype=complex)
        for j in range(rank):
            k = krauses[i][j]
            ksum += np.kron(k, np.conjugate(k))
        krauses_vectorized.append(ksum)
    return (unitaries, unitaries_vectorized, krauses_vectorized)

def prepare_layer_dilutedunitaryv1(L, rank, ensemble_unitaries, ensemble_kraus, gamma=0, alpha=1, beta=1):
    unitaries = unitaries_from_ensemble(L, ensemble_unitaries, alpha)
    krauses = layer_of_kraus(L, rank, ensemble_kraus, beta)
    unitaries_vectorized = []
    krauses_vectorized = []
    faulty_unitaries = unitaries_from_ensemble(L, 'eGUE', gamma)
    l = L // 2
    for i in range(l):
        u = unitaries[i]
        v = faulty_unitaries[i]
        utilde = v @ u
        unitaries_vectorized.append(np.kron(utilde, np.conjugate(utilde)))
        ksum = np.zeros((16, 16), dtype=complex)
        for j in range(rank):
            k = krauses[i][j]
            ksum += np.kron(k, np.conjugate(k))
        krauses_vectorized.append(ksum)
    return (unitaries, unitaries_vectorized, krauses_vectorized)

def faulty_permutation(permutation, L, p, nsuaps=False):
    cycles = perm_decomposition(permutation, L)
    layer1, layer2 = optimal_swaps(cycles, L)
    faultypermutation = list(range(0, L))
    l1 = len(layer1)
    l2 = len(layer2)
    nswaps = 0
    for i in range(l1):
        coin = np.random.rand()
        if coin < p:
            continue
        else:
            nswaps += 1
            swap = layer1[i]
            faultypermutation[swap[0]], faultypermutation[swap[1]] = (faultypermutation[swap[1]], faultypermutation[swap[0]])
    for i in range(l2):
        coin = np.random.rand()
        if coin < p:
            continue
        else:
            swap = layer2[i]
            nswaps += 1
            index0 = faultypermutation.index(swap[0])
            index1 = faultypermutation.index(swap[1])
            faultypermutation[index0], faultypermutation[index1] = (faultypermutation[index1], faultypermutation[index0])
    if nsuaps == False:
        return np.asarray(faultypermutation)
    else:
        return (np.asarray(faultypermutation), nswaps)

def fidtheorfull(kappa, alpha, p, L, time):
    time = int(time)
    f = np.exp(-alpha ** 2) * (1 - 4 * alpha ** 2 + 23 / 6 * alpha ** 4 - 2 * alpha ** 6 + 25 / 72 * alpha ** 8 - 1 / 36 * alpha ** 10)
    deltap = 4 ** L * np.exp(-3 / 4 * p * L * (1 - (np.log(L) + 0.5772157) / L))
    fidtheor = np.zeros(time)
    d = 2 ** L
    for t in range(time):
        T = t + 1
        fidtheor[t] = 1 / d + np.power((deltap - 1) / (d ** 2 - 1), T) * np.power((np.power(4 + 12 * f * (1 - kappa) - 3 * kappa, L / 2) - 1) / (d ** 2 - 1), T) * (1 - 1 / d)
    return fidtheor

def kapatilde(kapas, L):
    l = len(kapas)
    ktildes = np.zeros(l)
    for i, kapa in enumerate(kapas):
        ktildes[i] = (4 ** L - (16 - 15 * kapa) ** (L / 2)) / (4 ** L - 1)
    return ktildes
