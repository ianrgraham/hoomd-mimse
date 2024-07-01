import hoomd
# import hoomd.pair_plugin.pair as p_pair
import numpy as np

def KA_LJ(nlist: hoomd.md.nlist.NeighborList) -> hoomd.md.pair.Pair:
    '''Kob-Anderson Lennard-Jones potential
    '''
    r_cutoff = 2.5
    eps_AA = 1
    eps_AB = 1.5
    eps_BB = 0.5
    sig_AA = 1
    sig_AB = 0.8
    sig_BB = 0.88

    lj = hoomd.md.pair.LJ(nlist=nlist, mode="shift")
    lj.params[('A', 'A')] = dict(epsilon=eps_AA, sigma=sig_AA)
    lj.r_cut[('A', 'A')] = r_cutoff * sig_AA
    lj.params[('A', 'B')] = dict(epsilon=eps_AB, sigma=sig_AB)
    lj.r_cut[('A', 'B')] = r_cutoff * sig_AB
    lj.params[('B', 'B')] = dict(epsilon=eps_BB, sigma=sig_BB)
    lj.r_cut[('B', 'B')] = r_cutoff * sig_BB

    return lj

def FS_ES_LJ(nlist: hoomd.md.nlist.NeighborList) -> hoomd.md.pair.Pair:
    '''Force and Energy Lennard-Jones potential
        hoomd only does the force shift at the moment,
        changed pair.py to include the energy shift
    '''
    r_cutoff = 2.5
    eps_AA = 1
    eps_AB = 1
    eps_AC = 1
    eps_BB = 1
    eps_BC = 1
    eps_CC = 1
    sig_AA = 1
    sig_AB = 1
    sig_AC = 1
    sig_BB = 1
    sig_BC = 1
    sig_CC = 1

    lj = hoomd.md.pair.ForceShiftedLJ(nlist=nlist,mode='shift')
    lj.params[('A', 'A')] = dict(epsilon=eps_AA, sigma=sig_AA)
    lj.r_cut[('A', 'A')] = r_cutoff * sig_AA
    lj.params[('A', 'B')] = dict(epsilon=eps_AB, sigma=sig_AB)
    lj.r_cut[('A', 'B')] = r_cutoff * sig_AB
    lj.params[('A', 'C')] = dict(epsilon=eps_AC, sigma=sig_AC)
    lj.r_cut[('A', 'C')] = r_cutoff * sig_AC
    lj.params[('B', 'B')] = dict(epsilon=eps_BB, sigma=sig_BB)
    lj.r_cut[('B', 'B')] = r_cutoff * sig_BB
    lj.params[('B', 'C')] = dict(epsilon=eps_BC, sigma=sig_BC)
    lj.r_cut[('B', 'C')] = r_cutoff * sig_BC
    lj.params[('C', 'C')] = dict(epsilon=eps_CC, sigma=sig_CC)
    lj.r_cut[('C', 'C')] = r_cutoff * sig_CC

    return lj

def SG_sLJ(nlist: hoomd.md.nlist.NeighborList) -> hoomd.md.pair.Pair:
    '''Kob-Anderson Lennard-Jones potential
    '''
    r_cutoff = 2.5
    eps_AA = 1
    eps_AB = 1
    eps_AC = 1
    eps_BB = 1
    eps_BC = 1
    eps_CC = 1
    sig_AA = 1
    sig_AB = 1
    sig_AC = 1
    sig_BB = 1
    sig_BC = 1
    sig_CC = 1

    lj = hoomd.md.pair.LJ(nlist=nlist, mode="shift")
    lj.params[('A', 'A')] = dict(epsilon=eps_AA, sigma=sig_AA)
    lj.r_cut[('A', 'A')] = r_cutoff * sig_AA
    lj.params[('A', 'B')] = dict(epsilon=eps_AB, sigma=sig_AB)
    lj.r_cut[('A', 'B')] = r_cutoff * sig_AB
    lj.params[('A', 'C')] = dict(epsilon=eps_AC, sigma=sig_AC)
    lj.r_cut[('A', 'C')] = r_cutoff * sig_AC
    lj.params[('B', 'B')] = dict(epsilon=eps_BB, sigma=sig_BB)
    lj.r_cut[('B', 'B')] = r_cutoff * sig_BB
    lj.params[('B', 'C')] = dict(epsilon=eps_BC, sigma=sig_BC)
    lj.r_cut[('B', 'C')] = r_cutoff * sig_BC
    lj.params[('C', 'C')] = dict(epsilon=eps_CC, sigma=sig_CC)
    lj.r_cut[('C', 'C')] = r_cutoff * sig_CC

    return lj

def SG_LJ(nlist: hoomd.md.nlist.NeighborList) -> hoomd.md.pair.Pair:
    '''Kob-Anderson Lennard-Jones potential
    '''
    r_cutoff = 2.5
    eps_AA = 1
    eps_AB = 1
    eps_AC = 1
    eps_BB = 1
    eps_BC = 1
    eps_CC = 1
    sig_AA = 1
    sig_AB = 1
    sig_AC = 1
    sig_BB = 1
    sig_BC = 1
    sig_CC = 1

    lj = hoomd.md.pair.LJ(nlist=nlist)
    lj.params[('A', 'A')] = dict(epsilon=eps_AA, sigma=sig_AA)
    lj.r_cut[('A', 'A')] = r_cutoff * sig_AA
    lj.params[('A', 'B')] = dict(epsilon=eps_AB, sigma=sig_AB)
    lj.r_cut[('A', 'B')] = r_cutoff * sig_AB
    lj.params[('A', 'C')] = dict(epsilon=eps_AC, sigma=sig_AC)
    lj.r_cut[('A', 'C')] = r_cutoff * sig_AC
    lj.params[('B', 'B')] = dict(epsilon=eps_BB, sigma=sig_BB)
    lj.r_cut[('B', 'B')] = r_cutoff * sig_BB
    lj.params[('B', 'C')] = dict(epsilon=eps_BC, sigma=sig_BC)
    lj.r_cut[('B', 'C')] = r_cutoff * sig_BC
    lj.params[('C', 'C')] = dict(epsilon=eps_CC, sigma=sig_CC)
    lj.r_cut[('C', 'C')] = r_cutoff * sig_CC

    return lj

def SG_bonds():

    k_A = 500*2
    r0_A = 1.0
    k_B = 500*2
    r0_B = 0.666667

    harmonic = hoomd.md.bond.Harmonic()
    harmonic.params['A'] = dict(k=k_A, r0=r0_A)
    harmonic.params['B'] = dict(k=k_B, r0=r0_B)

    return harmonic

def SG_angles():

    k_A = 100*2
    theta0_A = 90*np.pi/180
    k_B = 100*2
    theta0_B = 120*np.pi/180
    k_C = 100*2
    theta0_C = 180*np.pi/180

    harmonic = hoomd.md.angle.Harmonic()
    harmonic.params['A'] = dict(k=k_A, t0=theta0_A)
    harmonic.params['B'] = dict(k=k_B, t0=theta0_B)
    harmonic.params['C'] = dict(k=k_C, t0=theta0_C)

    return harmonic

# def calculate_corrected_energy(sim, integrator, lj, slj, sslj, f_bonds, f_angles):
#     lj_energy = sslj.energy
#     integrator.forces = [lj, f_bonds, f_angles]
#     sim.run(0)
#     lj_energy -= lj.energy
#     integrator.forces = [slj, f_bonds, f_angles]
#     sim.run(0)
#     lj_energy += slj.energy
#     total_energy = lj_energy + f_bonds.energy + f_angles.energy
    
#     return lj_energy, f_bonds.energy, f_angles.energy, total_energy