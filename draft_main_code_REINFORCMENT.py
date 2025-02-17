import pandas as pd
import os
import rainflow as rf
import math
import gc
import logging
import time

# ############################################### GLOBAL PARAMS ########################################################

START_TIME = time.time()
DGT = 6  # n decimal rounding
FCT = 1.1  # factor to calculate
# LST_LOADCASE = [50077, 50079, 50081, 50082] # [50001 + i for i in range(16)]
LST_LOADCASE = [50001, 50002, 50003, 50004, 50005, 50006, 50007, 50008, 50009, 50010, 50011, 50012, 50013, 50014,
                50015, 50016, 50071, 50072, 50073, 50074, 50075, 50076, 50077, 50078, 50079, 50080, 50081, 50082]
# LST_LOADCASE = [50080]
FAST_LANES = [3, 4]
LANES = [2, 3, 4, 5]
# LST_QUAD = [420913, 420914, 420915, 420916, 420917]


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
# df_main = df_main[df_main['NR '].isin(LST_QUAD)]

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
    def __init__(self, number, loadcase, vehicle, bgn_forces, vhl_forces):
        self.number = number
        self.loadcase = loadcase
        self.vehicle = vehicle

        self.bgn_forces = bgn_forces # to deallocate after calculating stresses
        self.vhl_forces = vhl_forces # to deallocate after calculating stresses

        self.factor = 0.8 if self.vehicle == "A" else 0.2
        self.orders = 26 if self.vehicle == "A" else 22

        self.range_sig_x_low = {1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
        self.range_sig_x_upp = {1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
        self.range_sig_y_low = {1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
        self.range_sig_y_upp = {1: [], 2: [], 3: [], 4: [], 5: [], 6: []}

        self.range_rainflow_x_low = {1: None, 2: None, 3: None, 4: None, 5: None, 6: None}
        self.range_rainflow_x_upp = {1: None, 2: None, 3: None, 4: None, 5: None, 6: None}
        self.range_rainflow_y_low = {1: None, 2: None, 3: None, 4: None, 5: None, 6: None}
        self.range_rainflow_y_upp = {1: None, 2: None, 3: None, 4: None, 5: None, 6: None}

        self.D_x_low = 0
        self.D_x_upp = 0
        self.D_y_low = 0
        self.D_y_upp = 0

    def calculate_stress(self):
        asx_upp = DICT_RC[int(self.number)][0]  # in direction longitudinal of tunnel
        asx_low = DICT_RC[int(self.number)][1]  # in direction longitudinal of tunnel
        asy_upp = DICT_RC[int(self.number)][2]  # in direction transverse of tunnel
        asy_low = DICT_RC[int(self.number)][3]  # in direction transverse of tunnel

        a1x_upp = DICT_RC[int(self.number)][4]  # in direction longitudinal of tunnel
        a1x_low = DICT_RC[int(self.number)][5]  # in direction longitudinal of tunnel
        a1y_upp = DICT_RC[int(self.number)][6]  # in direction transverse of tunnel
        a1y_low = DICT_RC[int(self.number)][7]  # in direction transverse of tunnel

        thk = DICT_RC[int(self.number)][8]  # thickness of quad

        mxx_0 = round(self.bgn_forces[0], DGT)
        myy_0 = round(self.bgn_forces[1], DGT)
        mxy_0 = round(self.bgn_forces[2], DGT)
        nxx_0 = round(self.bgn_forces[3], DGT)
        nyy_0 = round(self.bgn_forces[4], DGT)
        nxy_0 = round(self.bgn_forces[5], DGT)

        #begin/end stress

        begin_forces = calculate_forces_d(mxx=mxx_0, myy=myy_0, mxy=mxy_0, nxx=nxx_0, nyy=nyy_0, nxy=nxy_0)

        alfa = 200 / 40
        b = 1

        # In direction of X stresses longitudinal
        # LOW face (positive design bending moments)
        if begin_forces["mxd_LOW"] > 0:
            Asd = asx_low
            a1 = a1x_low
            Asg = asx_upp
            a2 = a1x_upp
            stress = compression_range(b, thk / 1000, alfa, Asd / 10000, Asg / 10000, a1 / 1000, a2 / 1000,
                                       -begin_forces["nxd"], begin_forces["mxd_LOW"])
            # sig_c_x_upp_0 = stress[1]
            sig_x_low_0 = stress[2]
            sig_x_upp_0 = stress[3]
        else:
            # UPP face (negative design bending moments)
            Asd = asx_upp
            a1 = a1x_upp
            Asg = asx_low
            a2 = a1x_low
            stress = compression_range(b, thk / 1000, alfa, Asd / 10000, Asg / 10000, a1 / 1000, a2 / 1000,
                                       -begin_forces["nxd"], abs(begin_forces["mxd_UPP"]))
            # sig_c_x_low_0 = stress[1]
            sig_x_upp_0 = stress[2]
            sig_x_low_0 = stress[3]

        # In direction of Y stresses transversal
        # LOW face (positive design bending moments)
        if begin_forces["myd_LOW"] > 0:
            Asd = asy_low
            a1 = a1y_low
            Asg = asy_upp
            a2 = a1y_upp
            stress = compression_range(b, thk / 1000, alfa, Asd / 10000, Asg / 10000, a1 / 1000, a2 / 1000,
                                       -begin_forces["nyd"], begin_forces["myd_LOW"])
            # sig_c_y_upp_0 = stress[1]
            sig_y_low_0 = stress[2]
            sig_y_upp_0 = stress[3]
        else:
            # UPP face (negative design bending moments)
            Asd = asy_upp
            a1 = a1y_upp
            Asg = asy_low
            a2 = a1y_low
            stress = compression_range(b, thk / 1000, alfa, Asd / 10000, Asg / 10000, a1 / 1000, a2 / 1000,
                                       -begin_forces["nyd"], abs(begin_forces["myd_UPP"]))
            # sig_c_y_low_0 = stress[1]
            sig_y_upp_0 = stress[2]
            sig_y_low_0 = stress[3]

        for l in LANES:
            # add begin stress to start with the same point as the end
            self.range_sig_x_low[l].append(round(sig_x_low_0, DGT))
            self.range_sig_x_upp[l].append(round(sig_x_upp_0, DGT))
            self.range_sig_y_low[l].append(round(sig_y_low_0, DGT))
            self.range_sig_y_upp[l].append(round(sig_y_upp_0, DGT))

            for o in range(1, self.orders):
                row_l_o = self.vhl_forces[(self.vhl_forces["LANE"] == l) & (self.vhl_forces["ORDER"] == o)]
                vhl_fs = [round(row_l_o.iloc[0]["mxx [kNm/m]"], DGT),
                          round(row_l_o.iloc[0]["myy [kNm/m]"], DGT),
                          round(row_l_o.iloc[0]["mxy [kNm/m]"], DGT),
                          round(row_l_o.iloc[0]["nx [kN/m]"], DGT),
                          round(row_l_o.iloc[0]["ny [kN/m]"], DGT),
                          round(row_l_o.iloc[0]["nxy [kN/m]"], DGT)]

                mxx = round(mxx_0 + FCT * vhl_fs[0], DGT)
                myy = round(myy_0 + FCT * vhl_fs[1], DGT)
                mxy = round(mxy_0 + FCT * vhl_fs[2], DGT)
                nxx = round(nxx_0 + FCT * vhl_fs[3], DGT)
                nyy = round(nyy_0 + FCT * vhl_fs[4], DGT)
                nxy = round(nxy_0 + FCT * vhl_fs[5], DGT)

                forces = calculate_forces_d(mxx=mxx, myy=myy, mxy=mxy, nxx=nxx, nyy=nyy, nxy=nxy)

                # In direction of X stresses longitudinal
                # LOW face (positive design bending moments)
                if forces["mxd_LOW"] > 0:
                    Asd = asx_low
                    a1 = a1x_low
                    Asg = asx_upp
                    a2 = a1x_upp
                    stress = compression_range(b, thk / 1000, alfa, Asd / 10000, Asg / 10000, a1 / 1000, a2 / 1000,
                                               -forces["nxd"], forces["mxd_LOW"])
                    # sig_c_x_upp_0 = stress[1]
                    sig_x_low = stress[2]
                    sig_x_upp = stress[3]
                else:
                    # UPP face (negative design bending moments)
                    Asd = asx_upp
                    a1 = a1x_upp
                    Asg = asx_low
                    a2 = a1x_low
                    stress = compression_range(b, thk / 1000, alfa, Asd / 10000, Asg / 10000, a1 / 1000, a2 / 1000,
                                               -forces["nxd"], abs(forces["mxd_UPP"]))
                    # sig_c_x_low_0 = stress[1]
                    sig_x_upp = stress[2]
                    sig_x_low = stress[3]

                # In direction of Y stresses transversal
                # LOW face (positive design bending moments)
                if forces["myd_LOW"] > 0:
                    Asd = asy_low
                    a1 = a1y_low
                    Asg = asy_upp
                    a2 = a1y_upp
                    stress = compression_range(b, thk / 1000, alfa, Asd / 10000, Asg / 10000, a1 / 1000, a2 / 1000,
                                               -forces["nyd"], forces["myd_LOW"])
                    # sig_c_y_upp_0 = stress[1]
                    sig_y_low = stress[2]
                    sig_y_upp = stress[3]
                else:
                    # UPP face (negative design bending moments)
                    Asd = asy_upp
                    a1 = a1y_upp
                    Asg = asy_low
                    a2 = a1y_low
                    stress = compression_range(b, thk / 1000, alfa, Asd / 10000, Asg / 10000, a1 / 1000, a2 / 1000,
                                               -forces["nyd"], abs(forces["myd_UPP"]))
                    # sig_c_y_low_0 = stress[1]
                    sig_y_upp = stress[2]
                    sig_y_low = stress[3]

                self.range_sig_x_low[l].append(round(sig_x_low, DGT))
                self.range_sig_x_upp[l].append(round(sig_x_upp, DGT))
                self.range_sig_y_low[l].append(round(sig_y_low, DGT))
                self.range_sig_y_upp[l].append(round(sig_y_upp, DGT))

            # add end stress to close in the same point as the begin
            self.range_sig_x_low[l].append(round(sig_x_low_0, DGT))
            self.range_sig_x_upp[l].append(round(sig_x_upp_0, DGT))
            self.range_sig_y_low[l].append(round(sig_y_low_0, DGT))
            self.range_sig_y_upp[l].append(round(sig_y_upp_0, DGT))

    def rainflow(self):
        for key in self.range_sig_x_low:
            self.range_rainflow_x_low[key] = rf.count_cycles(series=self.range_sig_x_low[key], ndigits=3)
            self.range_rainflow_x_upp[key] = rf.count_cycles(series=self.range_sig_x_upp[key], ndigits=3)
            self.range_rainflow_y_low[key] = rf.count_cycles(series=self.range_sig_y_low[key], ndigits=3)
            self.range_rainflow_y_upp[key] = rf.count_cycles(series=self.range_sig_y_upp[key], ndigits=3)

    def calculate_D(self):
        for key in self.range_sig_x_low:
            n_life = 0.1 * 2 * 10 ** 6 if key in FAST_LANES else 2 * 10 ** 6
            val = 0
            for delta_sigma, n_sing in self.range_rainflow_x_low[key]:
                if delta_sigma == 0:
                    continue
                else:
                    n = n_sing * n_life * self.factor * 120
                    N = calculate_N(delta_sigma_st=delta_sigma)
                    self.D_x_low += n / N

            for delta_sigma, n_sing in self.range_rainflow_x_upp[key]:
                if delta_sigma == 0:
                    continue
                else:
                    n = n_sing * n_life * self.factor * 120
                    N = calculate_N(delta_sigma_st=delta_sigma)
                    self.D_x_upp += n / N

            for delta_sigma, n_sing in self.range_rainflow_y_low[key]:
                if delta_sigma == 0:
                    continue
                else:
                    n = n_sing * n_life * self.factor * 120
                    N = calculate_N(delta_sigma_st=delta_sigma)
                    self.D_y_low += n / N
                    val += n / N

            for delta_sigma, n_sing in self.range_rainflow_y_upp[key]:
                if delta_sigma == 0:
                    continue
                else:
                    n = n_sing * n_life * self.factor * 120
                    N = calculate_N(delta_sigma_st=delta_sigma)
                    self.D_y_upp += n / N

    def deallocate_memory(self):
        # del self.vhl_forces
        self.bgn_forces = None
        self.vhl_forces = None

        self.range_sig_x_low = None
        self.range_sig_x_upp = None
        self.range_sig_y_low = None
        self.range_sig_y_upp = None

        self.range_rainflow_x_low = None
        self.range_rainflow_x_upp = None
        self.range_rainflow_y_low = None
        self.range_rainflow_y_upp = None

        gc.collect()


def calculate_N(delta_sigma_st):
    delta_sigma_A = 500 / (1.1 * 1.2)
    delta_sigma_Rsk = 162.5/(1.1 * 1.2)
    N_Rsk = 10**6
    k1 = 5
    k2 = 9
    # if delta_sigma_st > delta_sigma_A:
    #     return 10 ** (math.log10(N_Rsk)-k1*(math.log10(delta_sigma_A) - math.log10(delta_sigma_Rsk)))
    if delta_sigma_st > delta_sigma_Rsk:
        return 10 ** (math.log10(N_Rsk)-k1*(math.log10(delta_sigma_A) - math.log10(delta_sigma_st)))
    else:
        return 10 ** (math.log10(N_Rsk)+k2*(math.log10(delta_sigma_Rsk) - math.log10(delta_sigma_st)))


def calculate_forces_d(mxx, myy, mxy, nxx, nyy, nxy):
    # Update into WA forces for bending 10.02.25 =====GZGH=====
    nxd = round(nxx + abs(nxy), DGT)
    nyd = round(nyy + abs(nxy), DGT)

    # =========LOW forces (positive)======
    mxd_LOW = round(mxx + abs(mxy), DGT)
    myd_LOW = round(myy + abs(mxy), DGT)

    if mxd_LOW < 0 and myd_LOW < 0:
        mxd_LOW = 0
        myd_LOW = 0
    else:
        if mxd_LOW < 0:
            mxd_LOW = 0
            myd_LOW = round(myy + abs(mxy * mxy / mxx), DGT)
        elif myd_LOW < 0:
            mxd_LOW = round(mxx + abs(mxy * mxy / myy), DGT)
            myd_LOW = 0

    # =========UPP forces (negative)======
    mxd_UPP = round(mxx - abs(mxy), DGT)
    myd_UPP = round(myy - abs(mxy), DGT)

    if mxd_UPP > 0 and myd_UPP > 0:
        mxd_UPP = 0
        myd_UPP = 0
    else:
        if mxd_UPP > 0:
            mxd_UPP = 0
            myd_UPP = round(myy - abs(mxy * mxy / mxx), DGT)
        elif myd_UPP > 0:
            mxd_UPP = round(mxx - abs(mxy * mxy / myy), DGT)
            myd_UPP = 0

    return {"mxd_LOW": mxd_LOW, "myd_LOW": myd_LOW, "mxd_UPP": mxd_UPP, "myd_UPP": myd_UPP, "nxd": nxd, "nyd": nyd}


def compression_range(b, h, alfa, A_sd, A_sg, a1, a2, Nsd, Msd):
    d = h - a1
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

    return [x, sig_c, sig_s1, sig_s2] # 4 - Jx0, 5 - i

# ############################################ MAIN_CODE ############################################

# create dict for key-QUAD, value - data of RC and quad

DICT_RC = dict()
for index, row_RC in df_RC.iterrows():
    DICT_RC[int(row_RC["QUAD"])] = [row_RC["as_LONG_upp"], row_RC["as_LONG_low"], row_RC["as_TRANS_upp"], row_RC["as_TRANS_low"],
                                    row_RC["a1_LONG_upp"], row_RC["a1_LONG_low"], row_RC["a1_TRANS_upp"], row_RC["a1_TRANS_low"],
                                    row_RC["THICK [mm]"]]

dict_to_add_1 = {nr: df_to_add_1[df_to_add_1['NR '] == nr] for nr in df_main['NR '].unique()}
dict_to_add_2 = {nr: df_to_add_2[df_to_add_2['NR '] == nr] for nr in df_main['NR '].unique()}

# params for checking progress
i = 0
percent = 0
len_rows = len(df_main)

set_nr_quad = set()
dct_quad = dict()
for index, row in df_main.iterrows():
    bgn_forces = [row['mxx [kNm/m]'], row['myy [kNm/m]'], row['mxy [kNm/m]'],
                  row['nx [kN/m]'], row['ny [kN/m]'], row['nxy [kN/m]']]

    df_fltr_A = dict_to_add_1[row['NR ']]
    df_fltr_B = dict_to_add_2[row['NR ']]

    quad_A = Quad(number=int(row['NR ']), loadcase=int(row['LC ']), vehicle="A", bgn_forces=bgn_forces, vhl_forces=df_fltr_A)
    quad_B = Quad(number=int(row['NR ']), loadcase=int(row['LC ']), vehicle="B", bgn_forces=bgn_forces, vhl_forces=df_fltr_B)

    # CALCULATING OF RANGE OF STRESS

    quad_A.calculate_stress()
    quad_B.calculate_stress()

    # RAINFLOW CALCULATION AND CALCULATION OF D

    quad_A.rainflow()
    quad_A.calculate_D()
    quad_B.rainflow()
    quad_B.calculate_D()

    # RELEASING MEMORY

    quad_A.deallocate_memory()
    quad_B.deallocate_memory()

    # CREATION OF DICTIONARY OF QUADS A/B TO GETTING FINAL RESULTS

    dct_quad[f"{int(quad_A.loadcase)}_{int(quad_A.number)}"] = [quad_A]
    set_nr_quad.add(int(quad_A.number))

    dct_quad[f"{int(quad_B.loadcase)}_{int(quad_B.number)}"].append(quad_B)

    # CHECKING THE PROGRESS OF WORK
    i += 1
    logging.basicConfig(level=logging.INFO)
    if i / len_rows > percent:
        logging.info(f"{float(100 * percent)} % completed")
        percent += 0.01


# FINAL STAGE - Summing parameter D for given LCs

df_result = pd.DataFrame(columns=["QUAD_NR", "D_xupp_SUM_LCs", "D_xlow_SUM_LCs", "D_yupp_SUM_LCs", "D_ylow_SUM_LCs"])

i = 0
for quad_nr in set_nr_quad:
    d1 = 0
    d2 = 0
    d3 = 0
    d4 = 0

    for lc in LST_LOADCASE:
        val_A = dct_quad[f"{int(lc)}_{int(quad_nr)}"][0]
        val_B = dct_quad[f"{int(lc)}_{int(quad_nr)}"][1]

        d1 += val_A.D_x_upp
        d1 += val_B.D_x_upp

        d2 += val_A.D_x_low
        d2 += val_B.D_x_low

        d3 += val_A.D_y_upp
        d3 += val_B.D_y_upp

        d4 += val_A.D_y_low
        d4 += val_B.D_y_low

    df_result.loc[i] = [quad_nr, d1, d2, d3, d4]

    i += 1
    if i % 1000 == 0: print(i)

df_result = df_result.sort_values(by=["QUAD_NR"])
print(df_result)

# dir_file_output = dir_name + '\\output\\RESULT_RC.csv'
# df_result.to_csv(dir_file_output, index=False)

print("Done")

# TIME REPORT
end_time = time.time()

exec_time = end_time - START_TIME

print(f"Czas wykonania całego skryptu: {exec_time} sekund")
