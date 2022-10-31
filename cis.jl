using PyCall
using LinearAlgebra
pyscf = pyimport("pyscf")
geomopt = pyimport("pyscf.geomopt.geometric_solver")

function construct_cis_matrix(mol, hf)
    g = pyscf.ao2mo.incore.full(mol.intor("int2e"), hf.mo_coeff)

    nao = mol.nao
    nocc = mol.nelectron ÷ 2
    nvir = nao - nocc

    occ = 1:nocc
    vir = (nocc+1):nao

    cis = [2g[a, i, j, b] - g[a, b, j, i]
           for i in occ, a in vir, j in occ, b in vir]

    for i in occ, (na, a) in enumerate(vir)
        cis[i, na, i, na] += hf.mo_energy[a] - hf.mo_energy[i]
    end

    reshape(cis, (nocc * nvir, nocc * nvir))
end

function construct_cis_matrix_smart(mol, hf)
    nao = py"int"(mol.nao)
    nocc = mol.nelectron ÷ 2
    nvir = nao - nocc

    occ = 1:nocc
    vir = (nocc+1):nao

    c_occ = hf.mo_coeff[:, occ]
    c_vir = hf.mo_coeff[:, vir]

    g_ao = mol.intor("int2e", aosym="s8")
    g_col = reshape(
        pyscf.ao2mo.incore.general(g_ao, (c_vir, c_vir, c_occ, c_occ),
            compact=false),
        (nvir, nvir, nocc, nocc)
    )
    g_exc = reshape(
        pyscf.ao2mo.incore.general(g_ao, (c_occ, c_vir, c_vir, c_occ),
            compact=false),
        (nvir, nocc, nocc, nvir)
    )

    cis = [2g_exc[a, i, j, b] - g_col[a, b, j, i]
           for i in occ, a in 1:nvir, j in occ, b in 1:nvir]

    for i in occ, (na, a) in enumerate(vir)
        cis[i, na, i, na] += hf.mo_energy[a] - hf.mo_energy[i]
    end

    Symmetric(reshape(cis, (nocc * nvir, nocc * nvir)))
end

function transition_dipole(mol, hf, cis_vec, proj_norm=true)
    nao = mol.nao
    nocc = mol.nelectron ÷ 2
    nvir = nao - nocc

    occ = 1:nocc
    vir = (nocc+1):nao

    c_occ = hf.mo_coeff[:, occ]
    c_vir = hf.mo_coeff[:, vir]

    d_ov = reshape(pyscf.lib.einsum("qxy,xi,yj->qij",
            mol.intor("int1e_r"), c_occ, c_vir), (3, nocc * nvir))
    
    td = pyscf.lib.einsum("qi,ij->qj", d_ov, cis_vec)

    if proj_norm
        [norm(r) for r in eachcol(td)]
    else
        td
    end
end
