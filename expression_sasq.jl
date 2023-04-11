using SpinAdaptedSecondQuantization

enable_color_translated()

gt(inds...) = psym_tensor("g", inds...)

hF = ∑((
        rsym_tensor("F", 1, 2) +
        ∑((-2 // 1 * gt(1, 2, 3, 3) + gt(1, 3, 3, 2)) * occupied(3), [3])
    ) * E(1, 2), 1:2)

g = 1 // 2 * ∑(gt(1:4...) * e(1:4...), 1:4) |> simplify

HF = hF + g |> simplify

E_hf = hf_expectation_value(HF) |> simplify_heavy

a, i, b, j = 1:4

trans = translate(VirtualOrbital => [a, b], OccupiedOrbital => [i, j])

occ = occupied(i, j) * virtual(a, b)

H_offdiag = 1 // 2 * hf_expectation_value(
    E(i, a) * HF * E(b, j) * occ
) |> simplify_heavy

H_diag = δ(i, j) * δ(a, b) * E_hf * occ |> simplify_heavy

A_aibj = H_offdiag - H_diag |> simplify_heavy

println("A_aibj = ", (A_aibj, trans))
