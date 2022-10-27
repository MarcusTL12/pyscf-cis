using LinearAlgebra

include("cis.jl")

const au2ev = 27.21138624598853

# Comparing CIS, CCSD and TDHF excitation energies
function h2o()
    mol = pyscf.M(atom="O 0 0 0; H 1 0 0; H 0 1 0"; basis="ccpvdz", verbose=0)
    mol = geomopt.optimize(mol.RHF())

    hf = mol.RHF().run()

    cis = @time construct_cis_matrix(mol, hf)

    n_ex = 3

    cis_ee, cis_c = eigen(cis)

    cis_ee = cis_ee[1:n_ex] * au2ev

    tdhf = hf.TDHF()
    tdhf.nstates = n_ex
    tdhf.run()
    tdhf_ee = tdhf.e * au2ev

    ccsd_ee = hf.CCSD().run().eomee_ccsd_singlet(nroots=n_ex)[1] * au2ev

    println(lpad("n", 3), "    ",
        rpad("CIS", 20), "    ",
        rpad("TDHF", 20), "    ",
        rpad("CCSD", 20))

    for (i, (cis, tdhf, ccsd)) in enumerate(zip(cis_ee, tdhf_ee, ccsd_ee))
        println(lpad(i, 3), "    ",
            rpad(cis, 20), "    ",
            rpad(tdhf, 20), "    ",
            rpad(ccsd, 20))
    end
end
