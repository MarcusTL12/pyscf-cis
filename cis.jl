using PyCall
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
