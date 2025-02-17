import pandas as pd
import os
import rainflow as rf
import math


# ############################################### GLOBAL PARAMS ########################################################

DGT = 3  # n decimal rounding
FCT = 1.1  # factor to calculate
# LST_LOADCASE = [50077, 50079, 50081, 50082] # [50001 + i for i in range(16)]
LST_LOADCASE = [50001, 50002, 50003, 50004, 50005, 50006, 50007, 50008, 50009, 50010, 50011, 50012, 50013, 50014,
                50015, 50016, 50071, 50072, 50073, 50074, 50075, 50076, 50077, 50078, 50079, 50080, 50081, 50082]
# LST_LOADCASE = [50001]
FAST_LANES = [3, 4]
LANES = [2, 3, 4, 5]

# ############################################## IMPORT FILES #########################################################

dir_name = os.path.dirname(__file__)

dir_file_main = dir_name + '\\input\\SLS_FRE_perm.xlsx'
dir_file_main_pkl = dir_name + '\\input\\SLS_FRE_perm.pkl'

if os.path.exists(dir_file_main_pkl):
    df_main = pd.read_pickle(dir_file_main_pkl)
else:
    df_main = pd.read_excel(dir_file_main, sheet_name="forces")
    df_main.to_pickle(dir_file_main_pkl)


dir_file_truck_a = dir_name + '\\input\\truck_typeA.xlsx'
dir_file_truck_a_pkl = dir_name + '\\input\\truck_typeA.pkl'

if os.path.exists(dir_file_truck_a_pkl):
    df_to_add_1 = pd.read_pickle(dir_file_truck_a_pkl)
else:
    df_to_add_1 = pd.read_excel(dir_file_truck_a, sheet_name="forces")
    df_to_add_1.to_pickle(dir_file_truck_a_pkl)


dir_file_truck_b = dir_name + '\\input\\truck_typeB.xlsx'
dir_file_truck_b_pkl = dir_name + '\\input\\truck_typeB.pkl'

if os.path.exists(dir_file_truck_b_pkl):
    df_to_add_2 = pd.read_pickle(dir_file_truck_b_pkl)
else:
    df_to_add_2 = pd.read_excel(dir_file_truck_b, sheet_name="forces")
    df_to_add_2.to_pickle(dir_file_truck_b_pkl)


dir_RC = dir_name + '\\input\\RC_for_FAT_QUADs.xlsx'
dir_RC_pkl = dir_name + '\\input\\RC_for_FAT_QUADs.pkl'

if os.path.exists(dir_RC_pkl):
    df_RC = pd.read_pickle(dir_RC_pkl)
else:
    df_RC = pd.read_excel(dir_RC, sheet_name="Sheet1")
    df_RC.to_pickle(dir_RC_pkl)

# INITIAL PROCESS - filter only given LCs

df_main = df_main[df_main['LC '].isin(LST_LOADCASE)]

# CHECK PRINT #####################################################

print(df_main.columns)
print(df_to_add_1.columns)
print(df_RC.columns)
print(df_to_add_2.columns)
print(df_main)
print(df_to_add_1)
print(df_RC)
print(df_to_add_2)

# ########################################### CLASSES #################################################################


class Quad:
    def __init__(self, number, loadcase, vehicle):
        self.number = number
        self.loadcase = loadcase
        self.vehicle = vehicle

        self.factor = 0.8 if self.vehicle == "A" else 0.2

        self.range_sig_x = {1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
        self.range_sig_y = {1: [], 2: [], 3: [], 4: [], 5: [], 6: []}

        self.range_rainflow_x = {1: None, 2: None, 3: None, 4: None, 5: None, 6: None}
        self.range_rainflow_y = {1: None, 2: None, 3: None, 4: None, 5: None, 6: None}

        self.D_x = 0
        self.D_y = 0

    def rainflow(self):
        for key in self.range_sig_x:
            self.range_rainflow_x[key] = rf.extract_cycles(series=self.range_sig_x[key])
            self.range_rainflow_y[key] = rf.extract_cycles(series=self.range_sig_y[key])

    def calculate_D(self):
        for key in self.range_sig_x:
            n_life = 0.1 * 2 * 10 ** 6 if key in FAST_LANES else 2 * 10 ** 6
            for rng, mean, count, i_start, i_end in self.range_rainflow_x[key]:
                if rng == 0: continue
                else:
                    n = count * n_life * self.factor * 120
                    res = calculate_N(i1=i_start, i2=i_end, main_range=self.range_sig_x[key])
                    N = res[0]
                    if res[1] and N != 0: self.D_x += n / N
                    else: continue

            for rng, mean, count, i_start, i_end in self.range_rainflow_y[key]:
                if rng == 0: continue
                else:
                    n = count * n_life * self.factor * 120
                    res = calculate_N(i1=i_start, i2=i_end, main_range=self.range_sig_y[key])
                    N = res[0]
                    if res[1] and N != 0: self.D_y += n / N
                    else: continue


def calculate_N(i1, i2, main_range):
    lst = [main_range[i1], main_range[i2]]
    k1 = 0.85  # recommend value
    fck = 40.0
    fcd = fck / (1.45 * 1.1)
    s = 0.20
    t = 5 * 365
    beta = 1.0  # math.exp(s * (1 - (28 / t) ** 0.5))
    fcd_fat = k1 * beta * fcd * (1 - fck / 250)

    E_cd_min = min(lst) / fcd_fat
    E_cd_max = max(lst) / fcd_fat
    if E_cd_min != E_cd_max and E_cd_max != 0:
        res = 10 ** (14 * (1 - E_cd_max) / (1 - E_cd_min / E_cd_max) ** 0.5)
        if res == 0:
            res = 10 ** (-30)
            print(f"res=0:E_cd_max: {E_cd_max}, E_cd_min: {E_cd_min}, lst: {lst}")
        return [res, True]
    else:
        print(f"E_cd_max: {E_cd_max}, E_cd_min: {E_cd_min}, lst: {lst}")
        return [10 ** 14, False]


def compression_range(b, h, alfa, A_sd, A_sg, a1, a2, Nsd, Msd):
    d = h - a1
    # =============compression is POSITIVE=======================
    # =============compression is POSITIVE=======================
    # =============compression is POSITIVE=======================

    # Check if compression zone exists in cross-section
    # # Assume neutral axis on top x=0, check stress on top
    x = 0
    Ax0 = b * x + alfa * (A_sd + A_sg)
    sx = b * x ** 2 / 2 + alfa * (A_sg * a2 + A_sd * d)
    x0 = sx / Ax0
    Jx0 = b * x ** 3 / 12 + b * x * (x0 - 0.5 * x) ** 2 + alfa * (A_sd * (d - x0) ** 2 + A_sg * (x0 - a2) ** 2)
    Mx0 = Msd + Nsd * (x0 - 0.5 * h)
    sig0 = Nsd / Ax0 + Mx0 * (x0 - x) / Jx0

    # # Assume neutral axis on bottom x=h, check stress on bottom
    x = h
    Ax0 = b * x + alfa * (A_sd + A_sg)
    sx = b * x ** 2 / 2 + alfa * (A_sg * a2 + A_sd * d)
    x0 = sx / Ax0
    Jx0 = b * x ** 3 / 12 + b * x * (x0 - 0.5 * x) ** 2 + alfa * (A_sd * (d - x0) ** 2 + A_sg * (x0 - a2) ** 2)
    Mx0 = Msd + Nsd * (x0 - 0.5 * h)
    sigh = Nsd / Ax0 + Mx0 * (x0 - x) / Jx0

    if sig0 * sigh == 0:
        # Unlikely case to be implemented later
        pass
    else:
        if sig0 * sigh > 0:
            # The same sign of stresses whole section in tension or in compression
            if Nsd < 0:
                # Whole section in tension
                x = "NO compr zone"
                sig_c = "WHOLE TENSION"
                Ax0 = alfa * (A_sd + A_sg)
                sx = alfa * (A_sg * a2 + A_sd * d)
                x0 = sx / Ax0
                Mx0 = Nsd * (x0 - 0.5 * h) + Msd
                e = abs(Mx0 / Nsd)
                es1 = d - x0 - e
                es2 = x0 + e - a2
                ds = d - a2
                sig_s1 = Nsd / A_sd * es2 / ds / 1000
                sig_s2 = Nsd / A_sg * es1 / ds / 1000
                sig_c = 0
            else:
                # Whole section in compression, so Nsd > 0
                x = "WHOLE compr zone"
                Ax0 = b * h + alfa * (A_sd + A_sg)
                sx = b * h ** 2 / 2 + alfa * (A_sg * a2 + A_sd * d)
                x0 = sx / Ax0
                Jx0 = b * h ** 3 / 12 + b * h * (x0 - 0.5 * h) ** 2 + alfa * (A_sd * (d - x0) ** 2 + A_sg * (x0 - a2) ** 2)
                Mx0 = Msd + Nsd * (x0 - 0.5 * h)
                sig_c = (Nsd / Ax0 + Mx0 * x0 / Jx0) / 1000
                # sig_bottom_c = (Nsd / Ax0 - Mx0 * (h - x0) / Jx0) / 1000
                sig_s1 = alfa * (Nsd / Ax0 + Mx0 * (x0 - d) / Jx0) / 1000
                sig_s2 = alfa * (Nsd / Ax0 + Mx0 * (x0 - a2) / Jx0) / 1000
        else:
            # Different sign of stresses between top and bottom, neutral axis in the cross section
            dokladnosc = 0.0000001
            # i = 0
            x_1, x_2 = 0, h
            wartosc_dla_x = 1

            while abs(wartosc_dla_x) > dokladnosc and abs(x_1 - x_2) > dokladnosc:
                x = 0.5 * (x_1 + x_2)

                # Calculate value for x
                Ax0 = b * x + alfa * (A_sd + A_sg)
                sx = b * x ** 2 / 2 + alfa * (A_sg * a2 + A_sd * d)
                x0 = sx / Ax0
                Jx0 = b * x ** 3 / 12 + b * x * (x0 - 0.5 * x) ** 2 + alfa * (A_sd * (d - x0) ** 2 + A_sg * (x0 - a2) ** 2)
                Mx0 = Msd + Nsd * (x0 - 0.5 * h)
                sigx0 = Nsd / Ax0 + Mx0 * (x0 - x) / Jx0
                wartosc_dla_x = sigx0

                # Calculate value for x_1
                Ax0 = b * x_1 + alfa * (A_sd + A_sg)
                sx = b * x_1 ** 2 / 2 + alfa * (A_sg * a2 + A_sd * d)
                x0 = sx / Ax0
                Jx0 = b * x_1 ** 3 / 12 + b * x_1 * (x0 - 0.5 * x_1) ** 2 + alfa * (A_sd * (d - x0) ** 2 + A_sg * (x0 - a2) ** 2)
                Mx0 = Msd + Nsd * (x0 - 0.5 * h)
                sigx0 = Nsd / Ax0 + Mx0 * (x0 - x_1) / Jx0
                wartosc_dla_x1 = sigx0

                if wartosc_dla_x * wartosc_dla_x1 > 0:
                    x_1 = x
                else:
                    x_2 = x
                # i += 1

            sig_c = (Nsd / Ax0 + Mx0 * x0 / Jx0) / 1000
            sig_s1 = alfa * (Nsd / Ax0 + Mx0 * (x0 - d) / Jx0) / 1000
            sig_s2 = alfa * (Nsd / Ax0 + Mx0 * (x0 - a2) / Jx0) / 1000

    if sig_c < 0: sig_c = 0
    return [x, sig_c, sig_s1, sig_s2] # 4 - Jx0, 5 - i


def calc_CONC_stress(lane, order, df_to_add, df_main=df_main):
    df_fltr = df_to_add[(df_to_add["LANE"] == lane) & (df_to_add["ORDER"] == order)]
    dict_fltr = dict()
    for index, row_to_add in df_fltr.iterrows():
        dict_fltr[int(row_to_add["NR "])] = [row_to_add["mxx [kNm/m]"],
                                        row_to_add["myy [kNm/m]"],
                                        row_to_add["mxy [kNm/m]"],
                                        row_to_add["nx [kN/m]"],
                                        row_to_add["ny [kN/m]"],
                                        row_to_add["nxy [kN/m]"]]

    for row_main in df_main.itertuples(index=True, name='Pandas'):
        mxx = row_main._4+FCT*dict_fltr[int(row_main._3)][0]
        myy = row_main._5+FCT*dict_fltr[int(row_main._3)][1]
        mxy = row_main._6+FCT*dict_fltr[int(row_main._3)][2]
        nxx = row_main._7+FCT*dict_fltr[int(row_main._3)][3]
        nyy = row_main._8+FCT*dict_fltr[int(row_main._3)][4]
        nxy = row_main._9+FCT*dict_fltr[int(row_main._3)][5]

        asx_upp = DICT_RC[int(row_main._3)][0]  # in direction longitudinal of tunnel
        asx_low = DICT_RC[int(row_main._3)][1]  # in direction longitudinal of tunnel
        asy_upp = DICT_RC[int(row_main._3)][2]  # in direction transverse of tunnel
        asy_low = DICT_RC[int(row_main._3)][3]  # in direction transverse of tunnel

        a1x_upp = DICT_RC[int(row_main._3)][4]  # in direction longitudinal of tunnel
        a1x_low = DICT_RC[int(row_main._3)][5]  # in direction longitudinal of tunnel
        a1y_upp = DICT_RC[int(row_main._3)][6]  # in direction transverse of tunnel
        a1y_low = DICT_RC[int(row_main._3)][7]  # in direction transverse of tunnel

        thk = DICT_RC[int(row_main._3)][8]

        # Update into WA forces for bending 10.02.25 =====GZGH=====
        nxd = nxx + abs(nxy)
        nyd = nyy + abs(nxy)

        # =========LOW forces (positive)======
        mxd_LOW = mxx + abs(mxy)
        myd_LOW = myy + abs(mxy)

        if mxd_LOW < 0 and myd_LOW < 0:
            mxd_LOW = 0
            myd_LOW = 0
        else:
            if mxd_LOW < 0:
                mxd_LOW = 0
                myd_LOW = myy + abs(mxy * mxy / mxx)
            elif myd_LOW < 0:
                mxd_LOW = mxx + abs(mxy * mxy / myy)
                myd_LOW = 0

        # =========UPP forces (negative)======
        mxd_UPP = mxx - abs(mxy)
        myd_UPP = myy - abs(mxy)

        if mxd_UPP > 0 and myd_UPP > 0:
            mxd_UPP = 0
            myd_UPP = 0
        else:
            if mxd_UPP > 0:
                mxd_UPP = 0
                myd_UPP = myy - abs(mxy * mxy / mxx)
            elif myd_UPP > 0:
                mxd_UPP = mxx - abs(mxy * mxy / myy)
                myd_UPP = 0

        alfa = 200 / 40
        b = 1

        # In direction of X stresses longitudinal
        # LOW face (positive design bending moments)
        if mxd_LOW > 0:
            Asd = asx_low
            a1 = a1x_low
            Asg = asx_upp
            a2 = a1x_upp
            stress = compression_range(b, thk / 1000, alfa, Asd / 10000, Asg / 10000, a1 / 1000, a2 / 1000, -nxd,
                                       mxd_LOW)
            sig_c_x_upp = stress[1]
            sig_c_x = sig_c_x_upp

        else:
            # UPP face (negative design bending moments)
            Asd = asx_upp
            a1 = a1x_upp
            Asg = asx_low
            a2 = a1x_low
            stress = compression_range(b, thk / 1000, alfa, Asd / 10000, Asg / 10000, a1 / 1000, a2 / 1000, -nxd,
                                       abs(mxd_UPP))
            sig_c_x_low = stress[1]
            sig_c_x = sig_c_x_low

        # In direction of Y stresses transversal
        # LOW face (positive design bending moments)
        if myd_LOW > 0:
            Asd = asy_low
            a1 = a1y_low
            Asg = asy_upp
            a2 = a1y_upp
            stress = compression_range(b, thk / 1000, alfa, Asd / 10000, Asg / 10000, a1 / 1000, a2 / 1000, -nyd,
                                       myd_LOW)
            sig_c_y_upp = stress[1]
            sig_c_y = sig_c_y_upp

        else:
            # UPP face (negative design bending moments)
            Asd = asy_upp
            a1 = a1y_upp
            Asg = asy_low
            a2 = a1y_low
            stress = compression_range(b, thk / 1000, alfa, Asd / 10000, Asg / 10000, a1 / 1000, a2 / 1000, -nyd,
                                       abs(myd_UPP))
            sig_c_y_low = stress[1]
            sig_c_y = sig_c_y_low

        DICT_QUAD_CLS[f"{int(row_main._1)}_{int(row_main._3)}"].range_sig_x[lane].append(round(sig_c_x, DGT))
        DICT_QUAD_CLS[f"{int(row_main._1)}_{int(row_main._3)}"].range_sig_y[lane].append(round(sig_c_y, DGT))


def calc_begining_CONC_stress(lane, df_main=df_main):
    for row_main in df_main.itertuples(index=True, name='Pandas'):
        mxx = row_main._4
        myy = row_main._5
        mxy = row_main._6
        nxx = row_main._7
        nyy = row_main._8
        nxy = row_main._9

        asx_upp = DICT_RC[int(row_main._3)][0]  # in direction longitudinal of tunnel
        asx_low = DICT_RC[int(row_main._3)][1]  # in direction longitudinal of tunnel
        asy_upp = DICT_RC[int(row_main._3)][2]  # in direction transverse of tunnel
        asy_low = DICT_RC[int(row_main._3)][3]  # in direction transverse of tunnel

        a1x_upp = DICT_RC[int(row_main._3)][4]  # in direction longitudinal of tunnel
        a1x_low = DICT_RC[int(row_main._3)][5]  # in direction longitudinal of tunnel
        a1y_upp = DICT_RC[int(row_main._3)][6]  # in direction transverse of tunnel
        a1y_low = DICT_RC[int(row_main._3)][7]  # in direction transverse of tunnel

        thk = DICT_RC[int(row_main._3)][8]

        # Update into WA forces for bending 10.02.25 =====GZGH=====
        nxd = nxx + abs(nxy)
        nyd = nyy + abs(nxy)

        # =========LOW forces (positive)======
        mxd_LOW = mxx + abs(mxy)
        myd_LOW = myy + abs(mxy)

        if mxd_LOW < 0 and myd_LOW < 0:
            mxd_LOW = 0
            myd_LOW = 0
        else:
            if mxd_LOW < 0:
                mxd_LOW = 0
                myd_LOW = myy + abs(mxy * mxy / mxx)
            elif myd_LOW < 0:
                mxd_LOW = mxx + abs(mxy * mxy / myy)
                myd_LOW = 0

        # =========UPP forces (negative)======
        mxd_UPP = mxx - abs(mxy)
        myd_UPP = myy - abs(mxy)

        if mxd_UPP > 0 and myd_UPP > 0:
            mxd_UPP = 0
            myd_UPP = 0
        else:
            if mxd_UPP > 0:
                mxd_UPP = 0
                myd_UPP = myy - abs(mxy * mxy / mxx)
            elif myd_UPP > 0:
                mxd_UPP = mxx - abs(mxy * mxy / myy)
                myd_UPP = 0

        alfa = 200 / 40
        b = 1

        # In direction of X stresses longitudinal
        # LOW face (positive design bending moments)
        if mxd_LOW > 0:
            Asd = asx_low
            a1 = a1x_low
            Asg = asx_upp
            a2 = a1x_upp
            stress = compression_range(b, thk / 1000, alfa, Asd / 10000, Asg / 10000, a1 / 1000, a2 / 1000, -nxd,
                                       mxd_LOW)
            sig_c_x_upp = stress[1]
            sig_c_x = sig_c_x_upp

        else:
            # UPP face (negative design bending moments)
            Asd = asx_upp
            a1 = a1x_upp
            Asg = asx_low
            a2 = a1x_low
            stress = compression_range(b, thk / 1000, alfa, Asd / 10000, Asg / 10000, a1 / 1000, a2 / 1000, -nxd,
                                       abs(mxd_UPP))
            sig_c_x_low = stress[1]
            sig_c_x = sig_c_x_low

        # In direction of Y stresses transversal
        # LOW face (positive design bending moments)
        if myd_LOW > 0:
            Asd = asy_low
            a1 = a1y_low
            Asg = asy_upp
            a2 = a1y_upp
            stress = compression_range(b, thk / 1000, alfa, Asd / 10000, Asg / 10000, a1 / 1000, a2 / 1000, -nyd,
                                       myd_LOW)
            sig_c_y_upp = stress[1]
            sig_c_y = sig_c_y_upp

        else:
            # UPP face (negative design bending moments)
            Asd = asy_upp
            a1 = a1y_upp
            Asg = asy_low
            a2 = a1y_low
            stress = compression_range(b, thk / 1000, alfa, Asd / 10000, Asg / 10000, a1 / 1000, a2 / 1000, -nyd,
                                       abs(myd_UPP))
            sig_c_y_low = stress[1]
            sig_c_y = sig_c_y_low

        DICT_QUAD_CLS[f"{int(row_main._1)}_{int(row_main._3)}"].range_sig_x[lane].append(round(sig_c_x, DGT))
        DICT_QUAD_CLS[f"{int(row_main._1)}_{int(row_main._3)}"].range_sig_y[lane].append(round(sig_c_y, DGT))


# create dict for key-QUAD, value - data of RC and quad

DICT_RC = dict()
for index, row_RC in df_RC.iterrows():
    DICT_RC[int(row_RC["QUAD"])] = [row_RC["as_LONG_upp"], row_RC["as_LONG_low"], row_RC["as_TRANS_upp"], row_RC["as_TRANS_low"],
                                    row_RC["a1_LONG_upp"], row_RC["a1_LONG_low"], row_RC["a1_TRANS_upp"], row_RC["a1_TRANS_low"],
                                    row_RC["THICK [mm]"]]

# create dictionary of classes quad for truck A ############################################ LANE 1-6 ORDER 1-25
DICT_QUAD_CLS = dict()

for index, row in df_main.iterrows():
    DICT_QUAD_CLS[f"{int(row['LC '])}_{int(row['NR '])}"] = Quad(number=int(row['NR ']), loadcase=int(row['LC ']), vehicle="A")

for l in LANES:
    calc_begining_CONC_stress(lane=l, df_main=df_main)
    for o in range(1, 26):
        calc_CONC_stress(lane=l, order=o, df_to_add=df_to_add_1)  # create data
        print(l, o)
    calc_begining_CONC_stress(lane=l, df_main=df_main)

DICT_QUAD_CLS_A = DICT_QUAD_CLS.copy()

# create dictionary of classes quad for truck B ############################################ LANE 1-6 ORDER 1-21
DICT_QUAD_CLS = dict()

for index, row in df_main.iterrows():
    DICT_QUAD_CLS[f"{int(row['LC '])}_{int(row['NR '])}"] = Quad(number=int(row['NR ']), loadcase=int(row['LC ']), vehicle="B")

for l in LANES:
    calc_begining_CONC_stress(lane=l, df_main=df_main)
    for o in range(1, 22):
        calc_CONC_stress(lane=l, order=o, df_to_add=df_to_add_2)  # create data
        print(l, o)
    calc_begining_CONC_stress(lane=l, df_main=df_main)
    
DICT_QUAD_CLS_B = DICT_QUAD_CLS.copy()

set_nr_quad = set()
dct_quad = dict()

# rainflow calculating and calculating D ########################################################################

for quad_instance in DICT_QUAD_CLS_A.values():
    quad_instance.rainflow()
    quad_instance.calculate_D()
    dct_quad[f"{int(quad_instance.loadcase)}_{int(quad_instance.number)}"] = [quad_instance]
    set_nr_quad.add(int(quad_instance.number))

for quad_instance in DICT_QUAD_CLS_B.values():
    quad_instance.rainflow()
    quad_instance.calculate_D()
    dct_quad[f"{int(quad_instance.loadcase)}_{int(quad_instance.number)}"].append(quad_instance)


# FINAL STAGE - Summing parameter D for given LCs ################################################################

df_result = pd.DataFrame(columns=["QUAD_NR", "D_x_SUM_LCs", "D_y_SUM_LCs"])

i = 0
for quad_nr in set_nr_quad:
    d1 = 0
    d2 = 0
    for lc in LST_LOADCASE:
        val_A = dct_quad[f"{int(lc)}_{int(quad_nr)}"][0]
        val_B = dct_quad[f"{int(lc)}_{int(quad_nr)}"][1]

        d1 += val_A.D_x
        d1 += val_B.D_x

        d2 += val_A.D_y
        d2 += val_B.D_y

    df_result.loc[i] = [quad_nr, d1, d2]

    i += 1
    if i % 1000 == 0: print(i)

df_result = df_result.sort_values(by=["QUAD_NR"])
print(df_result)

dir_file_output = dir_name + '\\output\\RESULT_C.csv'
df_result.to_csv(dir_file_output, index=False)

print("Done")
