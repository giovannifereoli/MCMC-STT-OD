import math
import json
import numpy as np


def normalization_constants(n, m):
    """
    Compute the normalization constant N(n, m) for spherical harmonic coefficients.

    Reference:
        McMahon et al. (2018), "The OSIRIS-REx Radio Science Experiment at Bennu",
        Space Science Reviews, Vol. 214, Issue 1, Springer Netherlands.
        https://doi.org/10.1007/s11214-018-0480-y

    Parameters
    ----------
    n : int
        Degree of the spherical harmonic.
    m : int
        Order of the spherical harmonic.

    Returns
    -------
    float
        Normalization constant sqrt((n-m)! * (2n+1) * (2 - delta(0,m)) / (n+m)!)
    """
    numerator = math.factorial(n - m) * (2 * n + 1) * (2 - int(m == 0))
    denominator = math.factorial(n + m)
    return math.sqrt(numerator / denominator)


def bennu_10_10(generate_json=False):
    """
    Return the unnormalized spherical harmonic gravity coefficients for Bennu
    up to degree and order 10, derived from the OSIRIS-REx radio science solution.

    Parameters
    ----------
    generate_json : bool, optional
        If True, writes unnormalized coefficients to 'bennu_10_10_unnormalized.json'.
        Default is False.

    Returns
    -------
    GM : float
        Gravitational parameter [km^3/s^2].
    Cterms : np.ndarray, shape (11, 11)
        Unnormalized cosine coefficients C[n, m].
    Sterms : np.ndarray, shape (11, 11)
        Unnormalized sine coefficients S[n, m].
    """
    NAMES = [
        "GM2101955  ",
        "BENNU_J2   ",
        "BENNU_C0201",
        "BENNU_S0201",
        "BENNU_C0202",
        "BENNU_S0202",
        "BENNU_J3   ",
        "BENNU_C0301",
        "BENNU_S0301",
        "BENNU_C0302",
        "BENNU_S0302",
        "BENNU_C0303",
        "BENNU_S0303",
        "BENNU_J4   ",
        "BENNU_C0401",
        "BENNU_S0401",
        "BENNU_C0402",
        "BENNU_S0402",
        "BENNU_C0403",
        "BENNU_S0403",
        "BENNU_C0404",
        "BENNU_S0404",
        "BENNU_J5   ",
        "BENNU_C0501",
        "BENNU_S0501",
        "BENNU_C0502",
        "BENNU_S0502",
        "BENNU_C0503",
        "BENNU_S0503",
        "BENNU_C0504",
        "BENNU_S0504",
        "BENNU_C0505",
        "BENNU_S0505",
        "BENNU_J6   ",
        "BENNU_C0601",
        "BENNU_S0601",
        "BENNU_C0602",
        "BENNU_S0602",
        "BENNU_C0603",
        "BENNU_S0603",
        "BENNU_C0604",
        "BENNU_S0604",
        "BENNU_C0605",
        "BENNU_S0605",
        "BENNU_C0606",
        "BENNU_S0606",
        "BENNU_J7   ",
        "BENNU_C0701",
        "BENNU_S0701",
        "BENNU_C0702",
        "BENNU_S0702",
        "BENNU_C0703",
        "BENNU_S0703",
        "BENNU_C0704",
        "BENNU_S0704",
        "BENNU_C0705",
        "BENNU_S0705",
        "BENNU_C0706",
        "BENNU_S0706",
        "BENNU_C0707",
        "BENNU_S0707",
        "BENNU_J8   ",
        "BENNU_C0801",
        "BENNU_S0801",
        "BENNU_C0802",
        "BENNU_S0802",
        "BENNU_C0803",
        "BENNU_S0803",
        "BENNU_C0804",
        "BENNU_S0804",
        "BENNU_C0805",
        "BENNU_S0805",
        "BENNU_C0806",
        "BENNU_S0806",
        "BENNU_C0807",
        "BENNU_S0807",
        "BENNU_C0808",
        "BENNU_S0808",
        "BENNU_J9   ",
        "BENNU_C0901",
        "BENNU_S0901",
        "BENNU_C0902",
        "BENNU_S0902",
        "BENNU_C0903",
        "BENNU_S0903",
        "BENNU_C0904",
        "BENNU_S0904",
        "BENNU_C0905",
        "BENNU_S0905",
        "BENNU_C0906",
        "BENNU_S0906",
        "BENNU_C0907",
        "BENNU_S0907",
        "BENNU_C0908",
        "BENNU_S0908",
        "BENNU_C0909",
        "BENNU_S0909",
        "BENNU_J10  ",
        "BENNU_C1001",
        "BENNU_S1001",
        "BENNU_C1002",
        "BENNU_S1002",
        "BENNU_C1003",
        "BENNU_S1003",
        "BENNU_C1004",
        "BENNU_S1004",
        "BENNU_C1005",
        "BENNU_S1005",
        "BENNU_C1006",
        "BENNU_S1006",
        "BENNU_C1007",
        "BENNU_S1007",
        "BENNU_C1008",
        "BENNU_S1008",
        "BENNU_C1009",
        "BENNU_S1009",
        "BENNU_C1010",
        "BENNU_S1010",
    ]

    VALUES = [
        4.89044967462e-09,
        0.019261012209376163,
        -2.1782173147855912e-14,
        3.0009695268217895e-15,
        0.00306499464152612,
        -0.00109450399573948,
        -0.0012219404640668086,
        0.0008148921217387432,
        -0.0005434579977478096,
        -0.000934922673655136,
        -0.0005377851962265501,
        0.0011710305387050103,
        -0.00031001193429507437,
        -0.006496001836889563,
        -0.0008821561290796273,
        -0.0005752155149983148,
        -0.0008707051950524519,
        -8.400092905051768e-05,
        -7.621121755654963e-05,
        -0.0003878715548433681,
        0.0007748481150705528,
        0.0022464919895245237,
        6.72886604884985e-05,
        -0.00035156263428227734,
        0.00016090884095563233,
        -3.742917272290789e-05,
        -0.00026801937425613214,
        -2.1852446688049215e-06,
        -9.462855909167245e-06,
        0.0003221412731748062,
        5.04640787496407e-05,
        -2.2767879074397064e-05,
        0.00029573700474979004,
        0.0013718245820512308,
        0.0003332494000416658,
        0.00019808121908086352,
        0.0002592382555926419,
        -0.000180395865997385,
        0.0001036155694799452,
        -3.5437195345475506e-05,
        -0.00020998848165819636,
        -0.0005296201444343653,
        6.348119245915396e-05,
        -0.0001814795188487908,
        0.0003040938642307484,
        5.726088595072154e-05,
        -7.908007377818328e-05,
        0.0003098125626813934,
        -3.7507949085619243e-05,
        -2.2328132337592748e-05,
        -0.0001014389516068962,
        0.00015781060218211183,
        2.657692226176869e-05,
        -0.00012783690092299058,
        8.377287000739878e-05,
        -0.00011470777236382891,
        -1.763186790110652e-05,
        -3.559587496287331e-05,
        -0.00011504366027871391,
        0.0002265368233611586,
        -7.72339390179355e-05,
        -0.0006815784392478206,
        -2.9030001676895305e-05,
        -0.00010946365118487296,
        -0.00010233555011655501,
        2.9189788934857738e-05,
        -8.738528818348646e-05,
        -1.484033298268544e-05,
        0.00014549556945248513,
        0.00031606243027352724,
        7.937279533002994e-05,
        2.6009886437901692e-06,
        -0.00019981302771948577,
        0.00016268712266082272,
        8.55049523744826e-05,
        -0.00010092843513236085,
        -0.000129667463555973,
        -0.00018349696665281707,
        -3.503022185911064e-05,
        -3.7303765509286255e-05,
        9.66489722196155e-05,
        8.788409745128699e-05,
        -6.260338739528384e-05,
        1.7266221518784266e-05,
        -5.02795249853691e-05,
        9.230716427255675e-05,
        8.17707148606606e-06,
        -2.949083528288675e-05,
        -2.0809201003593117e-05,
        -4.801703774680022e-05,
        2.4998090425702476e-05,
        -0.0001892587787490265,
        3.387013269102119e-05,
        0.00011617203467506834,
        1.223914290309503e-05,
        0.00010464014931255371,
        0.0002421973026110386,
        0.00020241448751601255,
        3.918626162623062e-05,
        3.9837096806635365e-05,
        0.00020522697455208243,
        0.00011312810270752221,
        1.123551097298677e-05,
        7.953831995465936e-05,
        3.6066995875417386e-05,
        -0.00010451197366417888,
        -2.5548362884603077e-05,
        3.339448009016222e-05,
        6.0316974021461546e-05,
        -2.246540565153352e-05,
        6.985268946923399e-05,
        -8.598254468068706e-05,
        4.8226367377983336e-05,
        3.5495594997370517e-06,
        -1.5167437784847748e-05,
        -4.801254471296015e-05,
        -3.242272984657309e-05,
        6.423728830556502e-05,
    ]

    size = 10
    n_entries = len(VALUES)

    GM = VALUES[0]
    Cterms = np.zeros((1 + size, 1 + size))
    Sterms = np.zeros((1 + size, 1 + size))
    Cterms[0, 0] = 1.0

    for i in range(1, n_entries):
        tag = NAMES[i][6]
        if tag == "J":
            n = int(NAMES[i][7:9])
            m = 0
            Cterms[n, m] = -VALUES[i] * normalization_constants(n, m)
        elif tag == "C":
            n = int(NAMES[i][7:9])
            m = int(NAMES[i][9:11])
            Cterms[n, m] = VALUES[i] * normalization_constants(n, m)
        elif tag == "S":
            n = int(NAMES[i][7:9])
            m = int(NAMES[i][9:11])
            Sterms[n, m] = VALUES[i] * normalization_constants(n, m)

    if generate_json:
        data = {NAMES[0].strip(): GM}
        for i in range(1, len(NAMES)):
            name = NAMES[i].strip()
            tag = NAMES[i][6]
            if tag == "J":
                ni = int(NAMES[i][7:9])
                data[name] = Cterms[ni, 0]
            elif tag == "C":
                ni = int(NAMES[i][7:9])
                mi = int(NAMES[i][9:11])
                data[name] = Cterms[ni, mi]
            elif tag == "S":
                ni = int(NAMES[i][7:9])
                mi = int(NAMES[i][9:11])
                data[name] = Sterms[ni, mi]
        with open("bennu_10_10_unnormalized.json", "w") as f:
            json.dump(data, f, indent=2)
        print("Wrote bennu_10_10_unnormalized.json")

    return GM, Cterms, Sterms


if __name__ == "__main__":
    GM, C, S = bennu_10_10(generate_json=True)

    print(f"GM       = {GM:.6e} km^3/s^2")
    print(f"C[2,0]   = {C[2,0]:.6e}   (J2 zonal)")
    print(f"C[2,2]   = {C[2,2]:.6e}   (sectoral)")
    print(f"S[2,2]   = {S[2,2]:.6e}")
    print(f"C[3,0]   = {C[3,0]:.6e}   (J3 zonal)")
    print(f"\nFull C matrix (degree 2-4):")
    print(C[2:5, :5])
