using LinearAlgebra

include("cis.jl")

const au2ev = 27.21138624598853

# Comparing CIS, CCSD and TDHF excitation energies
function h2o()
    mol = pyscf.M(atom="O 0 0 0; H 1 0 0; H 0 1 0"; basis="ccpvdz")
    # mol = geomopt.optimize(mol.RHF())

    hf = mol.RHF().run()

    cis = @time construct_cis_matrix_smart(mol, hf)

    n_ex = 3

    cis_ee = eigvals(cis, 1:n_ex)

    cis_ee = cis_ee * au2ev

    tdhf = hf.TDHF()
    tdhf.nstates = n_ex
    tdhf.run()
    tdhf.analyze()
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

function h2o_transition()
    mol = pyscf.M(atom="O 0 0 0; H 1 0 0; H 0 1 0"; basis="ccpvdz")
    # mol = geomopt.optimize(mol.RHF())

    hf = mol.RHF().run()

    cis = @time construct_cis_matrix_smart(mol, hf)

    n_ex = 3

    cis_ee, cis_v = eigen(cis, 1:n_ex)

    cis_ee = cis_ee * au2ev
    @show cis_ee

    tdhf = hf.TDHF()
    tdhf.nstates = n_ex
    tdhf.run()
    tdhf.analyze()

    transition_dipole(mol, hf, cis_v)
end

function benzene()
    mol = pyscf.M(atom="""
    C  -1.664244  -0.360809   0.000000
    C  -0.275880  -0.334582   0.000000
    C   0.395745   0.880796   0.000000
    C  -0.321200   2.070000   0.000000
    C  -1.709559   2.043702  -0.000000
    C  -2.381247   0.828355  -0.000000
    H  -3.463284   0.807970  -0.000000
    H  -2.187521  -1.308116   0.000000
    H   0.282794  -1.261455   0.000000
    H   1.477780   0.901181   0.000000
    H   0.202080   3.017307   0.000000
    H  -2.268234   2.970572  -0.000000
    """; basis="ccpvdz")

    hf = @time mol.RHF().run()

    cis = @time construct_cis_matrix_smart(mol, hf)

    n_ex = 6

    cis_ee, cis_v = @time eigen(cis, 1:n_ex)

    cis_ee = cis_ee * au2ev
    @show cis_ee

    nm = 1239.84193 ./ cis_ee
    @show nm

    @time transition_dipole(mol, hf, cis_v)
end

function benzene_speed()
    mol = pyscf.M(atom="""
    C  -1.664244  -0.360809   0.000000
    C  -0.275880  -0.334582   0.000000
    C   0.395745   0.880796   0.000000
    C  -0.321200   2.070000   0.000000
    C  -1.709559   2.043702  -0.000000
    C  -2.381247   0.828355  -0.000000
    H  -3.463284   0.807970  -0.000000
    H  -2.187521  -1.308116   0.000000
    H   0.282794  -1.261455   0.000000
    H   1.477780   0.901181   0.000000
    H   0.202080   3.017307   0.000000
    H  -2.268234   2.970572  -0.000000
    """; basis="ccpvdz")

    hf = @time mol.RHF().run()

    nocc, nvir = mol.nelectron ÷ 2 - 6, 5

    cis = @time construct_reduced_cis_matrix_smart(mol, hf, nocc, nvir)

    n_ex = 6

    cis_ee, cis_v = @time eigen(cis, 1:n_ex)

    cis_ee = cis_ee * au2ev
    @show cis_ee

    nm = 1239.84193 ./ cis_ee
    @show nm

    @time transition_dipole_reduced(mol, hf, cis_v, nocc, nvir)
end
