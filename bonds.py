

def wavelength_from_bond(bond_energy_kj):
    avogardos_num = 6.023 * 10**23

    j = (bond_energy_kj * 1000) / avogardos_num
    c = 299792458 # m/s
    h = 6.62607015 * 10 **-34
    # set up fundamental and overtones: up to 5
    for n in range(1,6):
        hz = j / (n*h)
        wvl = (c / hz) * 1000000000

        print(f"n={n}, wvl (nm) = {wvl}")


wavelength_from_bond(bond_energy_kj=467)

