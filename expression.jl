using ExcitationOperators
using ExcitationOperators.BasicStuff.StandardIndices
using ExcitationOperators.BasicStuff.StandardOperators

o = ind(occ, "o")
hF = summation(
    (
        real_tensor("F", p, q) +
        summation(-2 // 1 * psym_tensor("g", p, q, o, o) +
                  psym_tensor("g", p, o, o, q), [o])
    ) * E(p, q),
    [p, q]
)
HF = hF + g

println(simplify(
    exval(E(i, a) * HF * E(b, j)) * 1 // 2 - δ(i, j) * δ(a, b) * exval(HF)
))
