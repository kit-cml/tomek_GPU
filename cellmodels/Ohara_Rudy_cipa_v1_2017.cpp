/*
   There are a total of 200 entries in the algebraic variable array.
   There are a total of 49 entries in each of the rate and state variable arrays.
   There are a total of 206 entries in the constant variable array.
 */

#include "ohara_rudy_cipa_v1_2017.hpp"
#include <cmath>
#include <cstdlib>
// #include "../../functions/inputoutput.hpp"
#include <cstdio>
#include "../modules/glob_funct.hpp"
#include <cuda_runtime.h>
#include <cuda.h>

/*
 * TIME is time in component environment (millisecond).
 * CONSTANTS[celltype] is celltype in component environment (dimensionless).
 * CONSTANTS[nao] is nao in component extracellular (millimolar).
 * CONSTANTS[cao] is cao in component extracellular (millimolar).
 * CONSTANTS[ko] is ko in component extracellular (millimolar).
 * CONSTANTS[R] is R in component physical_constants (joule_per_kilomole_kelvin).
 * CONSTANTS[T] is T in component physical_constants (kelvin).
 * CONSTANTS[F] is F in component physical_constants (coulomb_per_mole).
 * CONSTANTS[zna] is zna in component physical_constants (dimensionless).
 * CONSTANTS[zca] is zca in component physical_constants (dimensionless).
 * CONSTANTS[zk] is zk in component physical_constants (dimensionless).
 * CONSTANTS[L] is L in component cell_geometry (centimeter).
 * CONSTANTS[rad] is rad in component cell_geometry (centimeter).
 * CONSTANTS[vcell] is vcell in component cell_geometry (microliter).
 * CONSTANTS[Ageo] is Ageo in component cell_geometry (centimeter_squared).
 * CONSTANTS[Acap] is Acap in component cell_geometry (centimeter_squared).
 * CONSTANTS[vmyo] is vmyo in component cell_geometry (microliter).
 * CONSTANTS[vnsr] is vnsr in component cell_geometry (microliter).
 * CONSTANTS[vjsr] is vjsr in component cell_geometry (microliter).
 * CONSTANTS[vss] is vss in component cell_geometry (microliter).
 * STATES[V] is v in component membrane (millivolt).
 * ALGEBRAIC[vfrt] is vfrt in component membrane (dimensionless).
 * CONSTANTS[ffrt] is ffrt in component membrane (coulomb_per_mole_millivolt).
 * CONSTANTS[frt] is frt in component membrane (per_millivolt).
 * ALGEBRAIC[INa] is INa in component INa (microA_per_microF).
 * ALGEBRAIC[INaL] is INaL in component INaL (microA_per_microF).
 * ALGEBRAIC[Ito] is Ito in component Ito (microA_per_microF).
 * ALGEBRAIC[ICaL] is ICaL in component ICaL (microA_per_microF).
 * ALGEBRAIC[ICaNa] is ICaNa in component ICaL (microA_per_microF).
 * ALGEBRAIC[ICaK] is ICaK in component ICaL (microA_per_microF).
 * ALGEBRAIC[IKr] is IKr in component IKr (microA_per_microF).
 * ALGEBRAIC[IKs] is IKs in component IKs (microA_per_microF).
 * ALGEBRAIC[IK1] is IK1 in component IK1 (microA_per_microF).
 * ALGEBRAIC[INaCa_i] is INaCa_i in component INaCa_i (microA_per_microF).
 * ALGEBRAIC[INaCa_ss] is INaCa_ss in component INaCa_i (microA_per_microF).
 * ALGEBRAIC[INaK] is INaK in component INaK (microA_per_microF).
 * ALGEBRAIC[INab] is INab in component INab (microA_per_microF).
 * ALGEBRAIC[IKb] is IKb in component IKb (microA_per_microF).
 * ALGEBRAIC[IpCa] is IpCa in component IpCa (microA_per_microF).
 * ALGEBRAIC[ICab] is ICab in component ICab (microA_per_microF).
 * ALGEBRAIC[Istim] is Istim in component membrane (microA_per_microF).
 * CONSTANTS[stim_start] is stim_start in component membrane (millisecond).
 * CONSTANTS[stim_end] is stim_end in component membrane (millisecond).
 * CONSTANTS[amp] is amp in component membrane (microA_per_microF).
 * CONSTANTS[BCL] is BCL in component membrane (millisecond).
 * CONSTANTS[duration] is duration in component membrane (millisecond).
 * CONSTANTS[KmCaMK] is KmCaMK in component CaMK (millimolar).
 * CONSTANTS[aCaMK] is aCaMK in component CaMK (per_millimolar_per_millisecond).
 * CONSTANTS[bCaMK] is bCaMK in component CaMK (per_millisecond).
 * CONSTANTS[CaMKo] is CaMKo in component CaMK (dimensionless).
 * CONSTANTS[KmCaM] is KmCaM in component CaMK (millimolar).
 * ALGEBRAIC[CaMKb] is CaMKb in component CaMK (millimolar).
 * ALGEBRAIC[CaMKa] is CaMKa in component CaMK (millimolar).
 * STATES[CaMKt] is CaMKt in component CaMK (millimolar).
 * STATES[cass] is cass in component intracellular_ions (millimolar).
 * CONSTANTS[cmdnmax_b] is cmdnmax_b in component intracellular_ions (millimolar).
 * CONSTANTS[cmdnmax] is cmdnmax in component intracellular_ions (millimolar).
 * CONSTANTS[kmcmdn] is kmcmdn in component intracellular_ions (millimolar).
 * CONSTANTS[trpnmax] is trpnmax in component intracellular_ions (millimolar).
 * CONSTANTS[kmtrpn] is kmtrpn in component intracellular_ions (millimolar).
 * CONSTANTS[BSRmax] is BSRmax in component intracellular_ions (millimolar).
 * CONSTANTS[KmBSR] is KmBSR in component intracellular_ions (millimolar).
 * CONSTANTS[BSLmax] is BSLmax in component intracellular_ions (millimolar).
 * CONSTANTS[KmBSL] is KmBSL in component intracellular_ions (millimolar).
 * CONSTANTS[csqnmax] is csqnmax in component intracellular_ions (millimolar).
 * CONSTANTS[kmcsqn] is kmcsqn in component intracellular_ions (millimolar).
 * STATES[nai] is nai in component intracellular_ions (millimolar).
 * STATES[nass] is nass in component intracellular_ions (millimolar).
 * STATES[ki] is ki in component intracellular_ions (millimolar).
 * STATES[kss] is kss in component intracellular_ions (millimolar).
 * STATES[cansr] is cansr in component intracellular_ions (millimolar).
 * STATES[cajsr] is cajsr in component intracellular_ions (millimolar).
 * STATES[cai] is cai in component intracellular_ions (millimolar).
 * ALGEBRAIC[JdiffNa] is JdiffNa in component diff (millimolar_per_millisecond).
 * ALGEBRAIC[Jdiff] is Jdiff in component diff (millimolar_per_millisecond).
 * ALGEBRAIC[Jup] is Jup in component SERCA (millimolar_per_millisecond).
 * ALGEBRAIC[JdiffK] is JdiffK in component diff (millimolar_per_millisecond).
 * ALGEBRAIC[Jrel] is Jrel in component ryr (millimolar_per_millisecond).
 * ALGEBRAIC[Jtr] is Jtr in component trans_flux (millimolar_per_millisecond).
 * ALGEBRAIC[Bcai] is Bcai in component intracellular_ions (dimensionless).
 * ALGEBRAIC[Bcajsr] is Bcajsr in component intracellular_ions (dimensionless).
 * ALGEBRAIC[Bcass] is Bcass in component intracellular_ions (dimensionless).
 * CONSTANTS[cm] is cm in component intracellular_ions (microF_per_centimeter_squared).
 * CONSTANTS[PKNa] is PKNa in component reversal_potentials (dimensionless).
 * ALGEBRAIC[ENa] is ENa in component reversal_potentials (millivolt).
 * ALGEBRAIC[EK] is EK in component reversal_potentials (millivolt).
 * ALGEBRAIC[EKs] is EKs in component reversal_potentials (millivolt).
 * ALGEBRAIC[mss] is mss in component INa (dimensionless).
 * ALGEBRAIC[tm] is tm in component INa (millisecond).
 * CONSTANTS[mssV1] is mssV1 in component INa (millivolt).
 * CONSTANTS[mssV2] is mssV2 in component INa (millivolt).
 * CONSTANTS[mtV1] is mtV1 in component INa (millivolt).
 * CONSTANTS[mtV2] is mtV2 in component INa (millivolt).
 * CONSTANTS[mtD1] is mtD1 in component INa (dimensionless).
 * CONSTANTS[mtD2] is mtD2 in component INa (dimensionless).
 * CONSTANTS[mtV3] is mtV3 in component INa (millivolt).
 * CONSTANTS[mtV4] is mtV4 in component INa (millivolt).
 * STATES[m] is m in component INa (dimensionless).
 * ALGEBRAIC[hss] is hss in component INa (dimensionless).
 * ALGEBRAIC[thf] is thf in component INa (millisecond).
 * ALGEBRAIC[ths] is ths in component INa (millisecond).
 * CONSTANTS[hssV1] is hssV1 in component INa (millivolt).
 * CONSTANTS[hssV2] is hssV2 in component INa (millivolt).
 * CONSTANTS[Ahs] is Ahs in component INa (dimensionless).
 * CONSTANTS[Ahf] is Ahf in component INa (dimensionless).
 * STATES[hf] is hf in component INa (dimensionless).
 * STATES[hs] is hs in component INa (dimensionless).
 * ALGEBRAIC[h] is h in component INa (dimensionless).
 * CONSTANTS[GNa] is GNa in component INa (milliS_per_microF).
 * CONSTANTS[shift_INa_inact] is shift_INa_inact in component INa (millivolt).
 * ALGEBRAIC[jss] is jss in component INa (dimensionless).
 * ALGEBRAIC[tj] is tj in component INa (millisecond).
 * STATES[j] is j in component INa (dimensionless).
 * ALGEBRAIC[hssp] is hssp in component INa (dimensionless).
 * ALGEBRAIC[thsp] is thsp in component INa (millisecond).
 * STATES[hsp] is hsp in component INa (dimensionless).
 * ALGEBRAIC[hp] is hp in component INa (dimensionless).
 * ALGEBRAIC[tjp] is tjp in component INa (millisecond).
 * STATES[jp] is jp in component INa (dimensionless).
 * ALGEBRAIC[fINap] is fINap in component INa (dimensionless).
 * ALGEBRAIC[mLss] is mLss in component INaL (dimensionless).
 * ALGEBRAIC[tmL] is tmL in component INaL (millisecond).
 * STATES[mL] is mL in component INaL (dimensionless).
 * CONSTANTS[thL] is thL in component INaL (millisecond).
 * ALGEBRAIC[hLss] is hLss in component INaL (dimensionless).
 * STATES[hL] is hL in component INaL (dimensionless).
 * ALGEBRAIC[hLssp] is hLssp in component INaL (dimensionless).
 * CONSTANTS[thLp] is thLp in component INaL (millisecond).
 * STATES[hLp] is hLp in component INaL (dimensionless).
 * CONSTANTS[GNaL_b] is GNaL_b in component INaL (milliS_per_microF).
 * CONSTANTS[GNaL] is GNaL in component INaL (milliS_per_microF).
 * ALGEBRAIC[fINaLp] is fINaLp in component INaL (dimensionless).
 * CONSTANTS[Gto_b] is Gto_b in component Ito (milliS_per_microF).
 * ALGEBRAIC[ass] is ass in component Ito (dimensionless).
 * ALGEBRAIC[ta] is ta in component Ito (millisecond).
 * STATES[a] is a in component Ito (dimensionless).
 * ALGEBRAIC[iss] is iss in component Ito (dimensionless).
 * ALGEBRAIC[delta_epi] is delta_epi in component Ito (dimensionless).
 * ALGEBRAIC[tiF_b] is tiF_b in component Ito (millisecond).
 * ALGEBRAIC[tiS_b] is tiS_b in component Ito (millisecond).
 * ALGEBRAIC[tiF] is tiF in component Ito (millisecond).
 * ALGEBRAIC[tiS] is tiS in component Ito (millisecond).
 * ALGEBRAIC[AiF] is AiF in component Ito (dimensionless).
 * ALGEBRAIC[AiS] is AiS in component Ito (dimensionless).
 * STATES[iF] is iF in component Ito (dimensionless).
 * STATES[iS] is iS in component Ito (dimensionless).
 * ALGEBRAIC[i] is i in component Ito (dimensionless).
 * ALGEBRAIC[assp] is assp in component Ito (dimensionless).
 * STATES[ap] is ap in component Ito (dimensionless).
 * ALGEBRAIC[dti_develop] is dti_develop in component Ito (dimensionless).
 * ALGEBRAIC[dti_recover] is dti_recover in component Ito (dimensionless).
 * ALGEBRAIC[tiFp] is tiFp in component Ito (millisecond).
 * ALGEBRAIC[tiSp] is tiSp in component Ito (millisecond).
 * STATES[iFp] is iFp in component Ito (dimensionless).
 * STATES[iSp] is iSp in component Ito (dimensionless).
 * ALGEBRAIC[ip] is ip in component Ito (dimensionless).
 * CONSTANTS[Gto] is Gto in component Ito (milliS_per_microF).
 * ALGEBRAIC[fItop] is fItop in component Ito (dimensionless).
 * CONSTANTS[Kmn] is Kmn in component ICaL (millimolar).
 * CONSTANTS[k2n] is k2n in component ICaL (per_millisecond).
 * CONSTANTS[PCa_b] is PCa_b in component ICaL (dimensionless).
 * ALGEBRAIC[dss] is dss in component ICaL (dimensionless).
 * STATES[d] is d in component ICaL (dimensionless).
 * ALGEBRAIC[fss] is fss in component ICaL (dimensionless).
 * CONSTANTS[Aff] is Aff in component ICaL (dimensionless).
 * CONSTANTS[Afs] is Afs in component ICaL (dimensionless).
 * STATES[ff] is ff in component ICaL (dimensionless).
 * STATES[fs] is fs in component ICaL (dimensionless).
 * ALGEBRAIC[f] is f in component ICaL (dimensionless).
 * ALGEBRAIC[fcass] is fcass in component ICaL (dimensionless).
 * ALGEBRAIC[Afcaf] is Afcaf in component ICaL (dimensionless).
 * ALGEBRAIC[Afcas] is Afcas in component ICaL (dimensionless).
 * STATES[fcaf] is fcaf in component ICaL (dimensionless).
 * STATES[fcas] is fcas in component ICaL (dimensionless).
 * ALGEBRAIC[fca] is fca in component ICaL (dimensionless).
 * STATES[jca] is jca in component ICaL (dimensionless).
 * STATES[ffp] is ffp in component ICaL (dimensionless).
 * ALGEBRAIC[fp] is fp in component ICaL (dimensionless).
 * STATES[fcafp] is fcafp in component ICaL (dimensionless).
 * ALGEBRAIC[fcap] is fcap in component ICaL (dimensionless).
 * ALGEBRAIC[km2n] is km2n in component ICaL (per_millisecond).
 * ALGEBRAIC[anca] is anca in component ICaL (dimensionless).
 * STATES[nca] is nca in component ICaL (dimensionless).
 * ALGEBRAIC[PhiCaL] is PhiCaL in component ICaL (dimensionless).
 * ALGEBRAIC[PhiCaNa] is PhiCaNa in component ICaL (dimensionless).
 * ALGEBRAIC[PhiCaK] is PhiCaK in component ICaL (dimensionless).
 * CONSTANTS[PCa] is PCa in component ICaL (dimensionless).
 * CONSTANTS[PCap] is PCap in component ICaL (dimensionless).
 * CONSTANTS[PCaNa] is PCaNa in component ICaL (dimensionless).
 * CONSTANTS[PCaK] is PCaK in component ICaL (dimensionless).
 * CONSTANTS[PCaNap] is PCaNap in component ICaL (dimensionless).
 * CONSTANTS[PCaKp] is PCaKp in component ICaL (dimensionless).
 * ALGEBRAIC[fICaLp] is fICaLp in component ICaL (dimensionless).
 * ALGEBRAIC[td] is td in component ICaL (millisecond).
 * ALGEBRAIC[tff] is tff in component ICaL (millisecond).
 * ALGEBRAIC[tfs] is tfs in component ICaL (millisecond).
 * ALGEBRAIC[tfcaf] is tfcaf in component ICaL (millisecond).
 * ALGEBRAIC[tfcas] is tfcas in component ICaL (millisecond).
 * CONSTANTS[tjca] is tjca in component ICaL (millisecond).
 * ALGEBRAIC[tffp] is tffp in component ICaL (millisecond).
 * ALGEBRAIC[tfcafp] is tfcafp in component ICaL (millisecond).
 * CONSTANTS[v0_CaL] is v0 in component ICaL (millivolt).
 * ALGEBRAIC[A_1] is A_1 in component ICaL (dimensionless).
 * CONSTANTS[B_1] is B_1 in component ICaL (per_millivolt).
 * ALGEBRAIC[U_1] is U_1 in component ICaL (dimensionless).
 * ALGEBRAIC[A_2] is A_2 in component ICaL (dimensionless).
 * CONSTANTS[B_2] is B_2 in component ICaL (per_millivolt).
 * ALGEBRAIC[U_2] is U_2 in component ICaL (dimensionless).
 * ALGEBRAIC[A_3] is A_3 in component ICaL (dimensionless).
 * CONSTANTS[B_3] is B_3 in component ICaL (per_millivolt).
 * ALGEBRAIC[U_3] is U_3 in component ICaL (dimensionless).
 * CONSTANTS[GKr_b] is GKr_b in component IKr (milliS_per_microF).
 * STATES[IC1] is IC1 in component IKr (dimensionless).
 * STATES[IC2] is IC2 in component IKr (dimensionless).
 * STATES[C1] is C1 in component IKr (dimensionless).
 * STATES[C2] is C2 in component IKr (dimensionless).
 * STATES[O] is O in component IKr (dimensionless).
 * STATES[IO] is IO in component IKr (dimensionless).
 * STATES[IObound] is IObound in component IKr (dimensionless).
 * STATES[Obound] is Obound in component IKr (dimensionless).
 * STATES[Cbound] is Cbound in component IKr (dimensionless).
 * STATES[D] is D in component IKr (dimensionless).
 * CONSTANTS[GKr] is GKr in component IKr (milliS_per_microF).
 * CONSTANTS[A1] is A1 in component IKr (per_millisecond).
 * CONSTANTS[B1] is B1 in component IKr (per_millivolt).
 * CONSTANTS[q1] is q1 in component IKr (dimensionless).
 * CONSTANTS[A2] is A2 in component IKr (per_millisecond).
 * CONSTANTS[B2] is B2 in component IKr (per_millivolt).
 * CONSTANTS[q2] is q2 in component IKr (dimensionless).
 * CONSTANTS[A3] is A3 in component IKr (per_millisecond).
 * CONSTANTS[B3] is B3 in component IKr (per_millivolt).
 * CONSTANTS[q3] is q3 in component IKr (dimensionless).
 * CONSTANTS[A4] is A4 in component IKr (per_millisecond).
 * CONSTANTS[B4] is B4 in component IKr (per_millivolt).
 * CONSTANTS[q4] is q4 in component IKr (dimensionless).
 * CONSTANTS[A11] is A11 in component IKr (per_millisecond).
 * CONSTANTS[B11] is B11 in component IKr (per_millivolt).
 * CONSTANTS[q11] is q11 in component IKr (dimensionless).
 * CONSTANTS[A21] is A21 in component IKr (per_millisecond).
 * CONSTANTS[B21] is B21 in component IKr (per_millivolt).
 * CONSTANTS[q21] is q21 in component IKr (dimensionless).
 * CONSTANTS[A31] is A31 in component IKr (per_millisecond).
 * CONSTANTS[B31] is B31 in component IKr (per_millivolt).
 * CONSTANTS[q31] is q31 in component IKr (dimensionless).
 * CONSTANTS[A41] is A41 in component IKr (per_millisecond).
 * CONSTANTS[B41] is B41 in component IKr (per_millivolt).
 * CONSTANTS[q41] is q41 in component IKr (dimensionless).
 * CONSTANTS[A51] is A51 in component IKr (per_millisecond).
 * CONSTANTS[B51] is B51 in component IKr (per_millivolt).
 * CONSTANTS[q51] is q51 in component IKr (dimensionless).
 * CONSTANTS[A52] is A52 in component IKr (per_millisecond).
 * CONSTANTS[B52] is B52 in component IKr (per_millivolt).
 * CONSTANTS[q52] is q52 in component IKr (dimensionless).
 * CONSTANTS[A53] is A53 in component IKr (per_millisecond).
 * CONSTANTS[B53] is B53 in component IKr (per_millivolt).
 * CONSTANTS[q53] is q53 in component IKr (dimensionless).
 * CONSTANTS[A61] is A61 in component IKr (per_millisecond).
 * CONSTANTS[B61] is B61 in component IKr (per_millivolt).
 * CONSTANTS[q61] is q61 in component IKr (dimensionless).
 * CONSTANTS[A62] is A62 in component IKr (per_millisecond).
 * CONSTANTS[B62] is B62 in component IKr (per_millivolt).
 * CONSTANTS[q62] is q62 in component IKr (dimensionless).
 * CONSTANTS[A63] is A63 in component IKr (per_millisecond).
 * CONSTANTS[B63] is B63 in component IKr (per_millivolt).
 * CONSTANTS[q63] is q63 in component IKr (dimensionless).
 * CONSTANTS[Kmax] is Kmax in component IKr (dimensionless).
 * CONSTANTS[Ku] is Ku in component IKr (per_millisecond).
 * CONSTANTS[n] is n in component IKr (dimensionless).
 * CONSTANTS[halfmax] is halfmax in component IKr (dimensionless).
 * CONSTANTS[Kt] is Kt in component IKr (per_millisecond).
 * CONSTANTS[Vhalf] is Vhalf in component IKr (millivolt).
 * CONSTANTS[Temp] is Temp in component IKr (dimensionless).
 * CONSTANTS[GKs_b] is GKs_b in component IKs (milliS_per_microF).
 * CONSTANTS[GKs] is GKs in component IKs (milliS_per_microF).
 * ALGEBRAIC[xs1ss] is xs1ss in component IKs (dimensionless).
 * ALGEBRAIC[xs2ss] is xs2ss in component IKs (dimensionless).
 * ALGEBRAIC[txs1] is txs1 in component IKs (millisecond).
 * CONSTANTS[txs1_max] is txs1_max in component IKs (millisecond).
 * STATES[xs1] is xs1 in component IKs (dimensionless).
 * STATES[xs2] is xs2 in component IKs (dimensionless).
 * ALGEBRAIC[KsCa] is KsCa in component IKs (dimensionless).
 * ALGEBRAIC[txs2] is txs2 in component IKs (millisecond).
 * CONSTANTS[GK1] is GK1 in component IK1 (milliS_per_microF).
 * CONSTANTS[GK1_b] is GK1_b in component IK1 (milliS_per_microF).
 * ALGEBRAIC[xk1ss] is xk1ss in component IK1 (dimensionless).
 * ALGEBRAIC[txk1] is txk1 in component IK1 (millisecond).
 * STATES[xk1] is xk1 in component IK1 (dimensionless).
 * ALGEBRAIC[rk1] is rk1 in component IK1 (millisecond).
 * CONSTANTS[kna1] is kna1 in component INaCa_i (per_millisecond).
 * CONSTANTS[kna2] is kna2 in component INaCa_i (per_millisecond).
 * CONSTANTS[kna3] is kna3 in component INaCa_i (per_millisecond).
 * CONSTANTS[kasymm] is kasymm in component INaCa_i (dimensionless).
 * CONSTANTS[wna] is wna in component INaCa_i (dimensionless).
 * CONSTANTS[wca] is wca in component INaCa_i (dimensionless).
 * CONSTANTS[wnaca] is wnaca in component INaCa_i (dimensionless).
 * CONSTANTS[kcaon] is kcaon in component INaCa_i (per_millisecond).
 * CONSTANTS[kcaoff] is kcaoff in component INaCa_i (per_millisecond).
 * CONSTANTS[qna] is qna in component INaCa_i (dimensionless).
 * CONSTANTS[qca] is qca in component INaCa_i (dimensionless).
 * ALGEBRAIC[hna] is hna in component INaCa_i (dimensionless).
 * ALGEBRAIC[hca] is hca in component INaCa_i (dimensionless).
 * CONSTANTS[KmCaAct] is KmCaAct in component INaCa_i (millimolar).
 * CONSTANTS[Gncx_b] is Gncx_b in component INaCa_i (milliS_per_microF).
 * CONSTANTS[Gncx] is Gncx in component INaCa_i (milliS_per_microF).
 * ALGEBRAIC[h1_i] is h1_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[h2_i] is h2_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[h3_i] is h3_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[h4_i] is h4_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[h5_i] is h5_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[h6_i] is h6_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[h7_i] is h7_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[h8_i] is h8_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[h9_i] is h9_i in component INaCa_i (dimensionless).
 * CONSTANTS[h10_i] is h10_i in component INaCa_i (dimensionless).
 * CONSTANTS[h11_i] is h11_i in component INaCa_i (dimensionless).
 * CONSTANTS[h12_i] is h12_i in component INaCa_i (dimensionless).
 * CONSTANTS[k1_i] is k1_i in component INaCa_i (dimensionless).
 * CONSTANTS[k2_i] is k2_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[k3p_i] is k3p_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[k3pp_i] is k3pp_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[k3_i] is k3_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[k4_i] is k4_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[k4p_i] is k4p_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[k4pp_i] is k4pp_i in component INaCa_i (dimensionless).
 * CONSTANTS[k5_i] is k5_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[k6_i] is k6_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[k7_i] is k7_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[k8_i] is k8_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[x1_i] is x1_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[x2_i] is x2_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[x3_i] is x3_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[x4_i] is x4_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[E1_i] is E1_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[E2_i] is E2_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[E3_i] is E3_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[E4_i] is E4_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[allo_i] is allo_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[JncxNa_i] is JncxNa_i in component INaCa_i (millimolar_per_millisecond).
 * ALGEBRAIC[JncxCa_i] is JncxCa_i in component INaCa_i (millimolar_per_millisecond).
 * ALGEBRAIC[h1_ss] is h1_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[h2_ss] is h2_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[h3_ss] is h3_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[h4_ss] is h4_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[h5_ss] is h5_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[h6_ss] is h6_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[h7_ss] is h7_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[h8_ss] is h8_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[h9_ss] is h9_ss in component INaCa_i (dimensionless).
 * CONSTANTS[h10_ss] is h10_ss in component INaCa_i (dimensionless).
 * CONSTANTS[h11_ss] is h11_ss in component INaCa_i (dimensionless).
 * CONSTANTS[h12_ss] is h12_ss in component INaCa_i (dimensionless).
 * CONSTANTS[k1_ss] is k1_ss in component INaCa_i (dimensionless).
 * CONSTANTS[k2_ss] is k2_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[k3p_ss] is k3p_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[k3pp_ss] is k3pp_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[k3_ss] is k3_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[k4_ss] is k4_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[k4p_ss] is k4p_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[k4pp_ss] is k4pp_ss in component INaCa_i (dimensionless).
 * CONSTANTS[k5_ss] is k5_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[k6_ss] is k6_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[k7_ss] is k7_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[k8_ss] is k8_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[x1_ss] is x1_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[x2_ss] is x2_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[x3_ss] is x3_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[x4_ss] is x4_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[E1_ss] is E1_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[E2_ss] is E2_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[E3_ss] is E3_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[E4_ss] is E4_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[allo_ss] is allo_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[JncxNa_ss] is JncxNa_ss in component INaCa_i (millimolar_per_millisecond).
 * ALGEBRAIC[JncxCa_ss] is JncxCa_ss in component INaCa_i (millimolar_per_millisecond).
 * CONSTANTS[k1p] is k1p in component INaK (per_millisecond).
 * CONSTANTS[k1m] is k1m in component INaK (per_millisecond).
 * CONSTANTS[k2p] is k2p in component INaK (per_millisecond).
 * CONSTANTS[k2m] is k2m in component INaK (per_millisecond).
 * CONSTANTS[k3p] is k3p in component INaK (per_millisecond).
 * CONSTANTS[k3m] is k3m in component INaK (per_millisecond).
 * CONSTANTS[k4p] is k4p in component INaK (per_millisecond).
 * CONSTANTS[k4m] is k4m in component INaK (per_millisecond).
 * CONSTANTS[Knai0] is Knai0 in component INaK (millimolar).
 * CONSTANTS[Knao0] is Knao0 in component INaK (millimolar).
 * CONSTANTS[delta] is delta in component INaK (millivolt).
 * CONSTANTS[Kki] is Kki in component INaK (per_millisecond).
 * CONSTANTS[Kko] is Kko in component INaK (per_millisecond).
 * CONSTANTS[MgADP] is MgADP in component INaK (millimolar).
 * CONSTANTS[MgATP] is MgATP in component INaK (millimolar).
 * CONSTANTS[Kmgatp] is Kmgatp in component INaK (millimolar).
 * CONSTANTS[H] is H in component INaK (millimolar).
 * CONSTANTS[eP] is eP in component INaK (dimensionless).
 * CONSTANTS[Khp] is Khp in component INaK (millimolar).
 * CONSTANTS[Knap] is Knap in component INaK (millimolar).
 * CONSTANTS[Kxkur] is Kxkur in component INaK (millimolar).
 * CONSTANTS[Pnak_b] is Pnak_b in component INaK (milliS_per_microF).
 * CONSTANTS[Pnak] is Pnak in component INaK (milliS_per_microF).
 * ALGEBRAIC[Knai] is Knai in component INaK (millimolar).
 * ALGEBRAIC[Knao] is Knao in component INaK (millimolar).
 * ALGEBRAIC[P] is P in component INaK (dimensionless).
 * ALGEBRAIC[a1] is a1 in component INaK (dimensionless).
 * CONSTANTS[b1] is b1 in component INaK (dimensionless).
 * CONSTANTS[a2] is a2 in component INaK (dimensionless).
 * ALGEBRAIC[b2] is b2 in component INaK (dimensionless).
 * ALGEBRAIC[a3] is a3 in component INaK (dimensionless).
 * ALGEBRAIC[b3] is b3 in component INaK (dimensionless).
 * CONSTANTS[a4] is a4 in component INaK (dimensionless).
 * ALGEBRAIC[b4] is b4 in component INaK (dimensionless).
 * ALGEBRAIC[x1] is x1 in component INaK (dimensionless).
 * ALGEBRAIC[x2] is x2 in component INaK (dimensionless).
 * ALGEBRAIC[x3] is x3 in component INaK (dimensionless).
 * ALGEBRAIC[x4] is x4 in component INaK (dimensionless).
 * ALGEBRAIC[E1] is E1 in component INaK (dimensionless).
 * ALGEBRAIC[E2] is E2 in component INaK (dimensionless).
 * ALGEBRAIC[E3] is E3 in component INaK (dimensionless).
 * ALGEBRAIC[E4] is E4 in component INaK (dimensionless).
 * ALGEBRAIC[JnakNa] is JnakNa in component INaK (millimolar_per_millisecond).
 * ALGEBRAIC[JnakK] is JnakK in component INaK (millimolar_per_millisecond).
 * ALGEBRAIC[xkb] is xkb in component IKb (dimensionless).
 * CONSTANTS[GKb_b] is GKb_b in component IKb (milliS_per_microF).
 * CONSTANTS[GKb] is GKb in component IKb (milliS_per_microF).
 * CONSTANTS[PNab] is PNab in component INab (milliS_per_microF).
 * ALGEBRAIC[A_Nab] is A in component INab (microA_per_microF).
 * CONSTANTS[B_Nab] is B in component INab (per_millivolt).
 * CONSTANTS[v0_Nab] is v0 in component INab (millivolt).
 * ALGEBRAIC[U] is U in component INab (dimensionless).
 * CONSTANTS[PCab] is PCab in component ICab (milliS_per_microF).
 * ALGEBRAIC[A_Cab] is A in component ICab (microA_per_microF).
 * CONSTANTS[B_Cab] is B in component ICab (per_millivolt).
 * CONSTANTS[v0_Cab] is v0 in component ICab (millivolt).
 * ALGEBRAIC[U] is U in component ICab (dimensionless).
 * CONSTANTS[GpCa] is GpCa in component IpCa (milliS_per_microF).
 * CONSTANTS[KmCap] is KmCap in component IpCa (millimolar).
 * CONSTANTS[bt] is bt in component ryr (millisecond).
 * CONSTANTS[a_rel] is a_rel in component ryr (millisecond).
 * ALGEBRAIC[Jrel_inf] is Jrel_inf in component ryr (dimensionless).
 * ALGEBRAIC[tau_rel] is tau_rel in component ryr (millisecond).
 * ALGEBRAIC[Jrel_infp] is Jrel_infp in component ryr (dimensionless).
 * ALGEBRAIC[Jrel_temp] is Jrel_temp in component ryr (dimensionless).
 * ALGEBRAIC[tau_relp] is tau_relp in component ryr (millisecond).
 * STATES[Jrelnp] is Jrelnp in component ryr (dimensionless).
 * STATES[Jrelp] is Jrelp in component ryr (dimensionless).
 * CONSTANTS[btp] is btp in component ryr (millisecond).
 * CONSTANTS[a_relp] is a_relp in component ryr (millisecond).
 * ALGEBRAIC[Jrel_inf_temp] is Jrel_inf_temp in component ryr (dimensionless).
 * ALGEBRAIC[fJrelp] is fJrelp in component ryr (dimensionless).
 * CONSTANTS[Jrel_scaling_factor] is Jrel_scaling_factor in component ryr (dimensionless).
 * ALGEBRAIC[tau_rel_temp] is tau_rel_temp in component ryr (millisecond).
 * ALGEBRAIC[tau_relp_temp] is tau_relp_temp in component ryr (millisecond).
 * CONSTANTS[upScale] is upScale in component SERCA (dimensionless).
 * ALGEBRAIC[Jupnp] is Jupnp in component SERCA (millimolar_per_millisecond).
 * ALGEBRAIC[Jupp] is Jupp in component SERCA (millimolar_per_millisecond).
 * ALGEBRAIC[fJupp] is fJupp in component SERCA (dimensionless).
 * ALGEBRAIC[Jleak] is Jleak in component SERCA (millimolar_per_millisecond).
 * CONSTANTS[Jup_b] is Jup_b in component SERCA (dimensionless).
 * RATES[V] is d/dt v in component membrane (millivolt).
 * RATES[CaMKt] is d/dt CaMKt in component CaMK (millimolar).
 * RATES[nai] is d/dt nai in component intracellular_ions (millimolar).
 * RATES[nass] is d/dt nass in component intracellular_ions (millimolar).
 * RATES[ki] is d/dt ki in component intracellular_ions (millimolar).
 * RATES[kss] is d/dt kss in component intracellular_ions (millimolar).
 * RATES[cai] is d/dt cai in component intracellular_ions (millimolar).
 * RATES[cass] is d/dt cass in component intracellular_ions (millimolar).
 * RATES[cansr] is d/dt cansr in component intracellular_ions (millimolar).
 * RATES[cajsr] is d/dt cajsr in component intracellular_ions (millimolar).
 * RATES[m] is d/dt m in component INa (dimensionless).
 * RATES[hf] is d/dt hf in component INa (dimensionless).
 * RATES[hs] is d/dt hs in component INa (dimensionless).
 * RATES[j] is d/dt j in component INa (dimensionless).
 * RATES[hsp] is d/dt hsp in component INa (dimensionless).
 * RATES[jp] is d/dt jp in component INa (dimensionless).
 * RATES[mL] is d/dt mL in component INaL (dimensionless).
 * RATES[hL] is d/dt hL in component INaL (dimensionless).
 * RATES[hLp] is d/dt hLp in component INaL (dimensionless).
 * RATES[a] is d/dt a in component Ito (dimensionless).
 * RATES[iF] is d/dt iF in component Ito (dimensionless).
 * RATES[iS] is d/dt iS in component Ito (dimensionless).
 * RATES[ap] is d/dt ap in component Ito (dimensionless).
 * RATES[iFp] is d/dt iFp in component Ito (dimensionless).
 * RATES[iSp] is d/dt iSp in component Ito (dimensionless).
 * RATES[d] is d/dt d in component ICaL (dimensionless).
 * RATES[ff] is d/dt ff in component ICaL (dimensionless).
 * RATES[fs] is d/dt fs in component ICaL (dimensionless).
 * RATES[fcaf] is d/dt fcaf in component ICaL (dimensionless).
 * RATES[fcas] is d/dt fcas in component ICaL (dimensionless).
 * RATES[jca] is d/dt jca in component ICaL (dimensionless).
 * RATES[ffp] is d/dt ffp in component ICaL (dimensionless).
 * RATES[fcafp] is d/dt fcafp in component ICaL (dimensionless).
 * RATES[nca] is d/dt nca in component ICaL (dimensionless).
 * RATES[IC1] is d/dt IC1 in component IKr (dimensionless).
 * RATES[IC2] is d/dt IC2 in component IKr (dimensionless).
 * RATES[C1] is d/dt C1 in component IKr (dimensionless).
 * RATES[C2] is d/dt C2 in component IKr (dimensionless).
 * RATES[O] is d/dt O in component IKr (dimensionless).
 * RATES[IO] is d/dt IO in component IKr (dimensionless).
 * RATES[IObound] is d/dt IObound in component IKr (dimensionless).
 * RATES[Obound] is d/dt Obound in component IKr (dimensionless).
 * RATES[Cbound] is d/dt Cbound in component IKr (dimensionless).
 * RATES[D] is d/dt D in component IKr (dimensionless).
 * RATES[xs1] is d/dt xs1 in component IKs (dimensionless).
 * RATES[xs2] is d/dt xs2 in component IKs (dimensionless).
 * RATES[xk1] is d/dt xk1 in component IK1 (dimensionless).
 * RATES[Jrelnp] is d/dt Jrelnp in component ryr (dimensionless).
 * RATES[Jrelp] is d/dt Jrelp in component ryr (dimensionless).
 */

__device__ void ___initConsts(double *CONSTANTS, double *STATES, double type, double bcl, int offset)
{
algebraic_size = 200;
constants_size = 206;
states_size = 49;
rates_size = 49;
  // consider to put all of the sizes here,  as in 2011 ord
CONSTANTS[(constant_size * offset) + celltype] = type;
CONSTANTS[(constant_size * offset) + nao] = 140;
CONSTANTS[(constant_size * offset) + cao] = 1.8;
CONSTANTS[(constant_size * offset) + ko] = 5.4;
CONSTANTS[(constant_size * offset) + R] = 8314;
CONSTANTS[(constant_size * offset) + T] = 310;
CONSTANTS[(constant_size * offset) + F] = 96485;
CONSTANTS[(constant_size * offset) + zna] = 1;
CONSTANTS[(constant_size * offset) + zca] = 2;
CONSTANTS[(constant_size * offset) + zk] = 1;
CONSTANTS[(constant_size * offset) + L] = 0.01;
CONSTANTS[(constant_size * offset) + rad] = 0.0011;
STATES[(states_size * offset) +  V] = -88.00190465;
CONSTANTS[(constant_size * offset) + stim_start] = 10;
CONSTANTS[(constant_size * offset) + stim_end] = 100000000000000000;
CONSTANTS[(constant_size * offset) + amp] = -80;
CONSTANTS[(constant_size * offset) + BCL] = 1000;
CONSTANTS[(constant_size * offset) + duration] = 0.5;
CONSTANTS[(constant_size * offset) + KmCaMK] = 0.15;
CONSTANTS[(constant_size * offset) + aCaMK] = 0.05;
CONSTANTS[(constant_size * offset) + bCaMK] = 0.00068;
CONSTANTS[(constant_size * offset) + CaMKo] = 0.05;
CONSTANTS[(constant_size * offset) + KmCaM] = 0.0015;
STATES[(states_size * offset) +  CaMKt] = 0.0125840447;
STATES[(states_size * offset) +  cass] = 8.49e-05;
CONSTANTS[(constant_size * offset) + cmdnmax_b] = 0.05;
CONSTANTS[(constant_size * offset) + kmcmdn] = 0.00238;
CONSTANTS[(constant_size * offset) + trpnmax] = 0.07;
CONSTANTS[(constant_size * offset) + kmtrpn] = 0.0005;
CONSTANTS[(constant_size * offset) + BSRmax] = 0.047;
CONSTANTS[(constant_size * offset) + KmBSR] = 0.00087;
CONSTANTS[(constant_size * offset) + BSLmax] = 1.124;
CONSTANTS[(constant_size * offset) + KmBSL] = 0.0087;
CONSTANTS[(constant_size * offset) + csqnmax] = 10;
CONSTANTS[(constant_size * offset) + kmcsqn] = 0.8;
STATES[(states_size * offset) +  nai] = 7.268004498;
STATES[(states_size * offset) +  nass] = 7.268089977;
STATES[(states_size * offset) +  ki] = 144.6555918;
STATES[(states_size * offset) +  kss] = 144.6555651;
STATES[(states_size * offset) +  cansr] = 1.619574538;
STATES[(states_size * offset) +  cajsr] = 1.571234014;
STATES[(states_size * offset) +  cai] = 8.6e-05;
CONSTANTS[(constant_size * offset) + cm] = 1;
CONSTANTS[(constant_size * offset) + PKNa] = 0.01833;
CONSTANTS[(constant_size * offset) + mssV1] = 39.57;
CONSTANTS[(constant_size * offset) + mssV2] = 9.871;
CONSTANTS[(constant_size * offset) + mtV1] = 11.64;
CONSTANTS[(constant_size * offset) + mtV2] = 34.77;
CONSTANTS[(constant_size * offset) + mtD1] = 6.765;
CONSTANTS[(constant_size * offset) + mtD2] = 8.552;
CONSTANTS[(constant_size * offset) + mtV3] = 77.42;
CONSTANTS[(constant_size * offset) + mtV4] = 5.955;
STATES[(states_size * offset) +  m] = 0.007344121102;
CONSTANTS[(constant_size * offset) + hssV1] = 82.9;
CONSTANTS[(constant_size * offset) + hssV2] = 6.086;
CONSTANTS[(constant_size * offset) + Ahf] = 0.99;
STATES[(states_size * offset) +  hf] = 0.6981071913;
STATES[(states_size * offset) +  hs] = 0.6980895801;
CONSTANTS[(constant_size * offset) + GNa] = 75;
CONSTANTS[(constant_size * offset) + shift_INa_inact] = 0;
STATES[(states_size * offset) +  j] = 0.6979908432;
STATES[(states_size * offset) +  hsp] = 0.4549485525;
STATES[(states_size * offset) +  jp] = 0.6979245865;
STATES[(states_size * offset) +  mL] = 0.0001882617273;
CONSTANTS[(constant_size * offset) + thL] = 200;
STATES[(states_size * offset) +  hL] = 0.5008548855;
STATES[(states_size * offset) + hLp] = 0.2693065357;
CONSTANTS[(constant_size * offset) + GNaL_b] = 0.019957499999999975;
CONSTANTS[(constant_size * offset) + Gto_b] = 0.02;
STATES[(states_size * offset) + a] = 0.001001097687;
STATES[(states_size * offset) + iF] = 0.9995541745;
STATES[(states_size * offset) + iS] = 0.5865061736;
STATES[(states_size * offset) + ap] = 0.0005100862934;
STATES[(states_size * offset) + iFp] = 0.9995541823;
STATES[(states_size * offset) + iSp] = 0.6393399482;
CONSTANTS[(constant_size * offset) + Kmn] = 0.002;
CONSTANTS[(constant_size * offset) + k2n] = 1000;
CONSTANTS[(constant_size * offset) + PCa_b] = 0.0001007;
STATES[(states_size * offset) + d] = 2.34e-9;
STATES[(states_size * offset) + ff] = 0.9999999909;
STATES[(states_size * offset) + fs] = 0.9102412777;
STATES[(states_size * offset) + fcaf] = 0.9999999909;
STATES[(states_size * offset) + fcas] = 0.9998046777;
STATES[(states_size * offset) + jca] = 0.9999738312;
STATES[(states_size * offset) + ffp] = 0.9999999909;
STATES[(states_size * offset) + fcafp] = 0.9999999909;
STATES[(states_size * offset) + nca] = 0.002749414044;
CONSTANTS[(constant_size * offset) + GKr_b] = 0.04658545454545456;
STATES[(states_size * offset) + IC1] = 0.999637;
STATES[(states_size * offset) + IC2] = 6.83208e-05;
STATES[(states_size * offset) + C1] = 1.80145e-08;
STATES[(states_size * offset) + C2] = 8.26619e-05;
STATES[(states_size * offset) + O] = 0.00015551;
STATES[(states_size * offset) + IO] = 5.67623e-05;
STATES[(states_size * offset) + IObound] = 0;
STATES[(states_size * offset) + Obound] = 0;
STATES[(states_size * offset) + Cbound] = 0;
STATES[(states_size * offset) + D] = 0;
CONSTANTS[(constant_size * offset) + A1] = 0.0264;
CONSTANTS[(constant_size * offset) + B1] = 4.631E-05;
CONSTANTS[(constant_size * offset) + q1] = 4.843;
CONSTANTS[(constant_size * offset) + A2] = 4.986E-06;
CONSTANTS[(constant_size * offset) + B2] = -0.004226;
CONSTANTS[(constant_size * offset) + q2] = 4.23;
CONSTANTS[(constant_size * offset) + A3] = 0.001214;
CONSTANTS[(constant_size * offset) + B3] = 0.008516;
CONSTANTS[(constant_size * offset) + q3] = 4.962;
CONSTANTS[(constant_size * offset) + A4] = 1.854E-05;
CONSTANTS[(constant_size * offset) + B4] = -0.04641;
CONSTANTS[(constant_size * offset) + q4] = 3.769;
CONSTANTS[(constant_size * offset) + A11] = 0.0007868;
CONSTANTS[(constant_size * offset) + B11] = 1.535E-08;
CONSTANTS[(constant_size * offset) + q11] = 4.942;
CONSTANTS[(constant_size * offset) + A21] = 5.455E-06;
CONSTANTS[(constant_size * offset) + B21] = -0.1688;
CONSTANTS[(constant_size * offset) + q21] = 4.156;
CONSTANTS[(constant_size * offset) + A31] = 0.005509;
CONSTANTS[(constant_size * offset) + B31] = 7.771E-09;
CONSTANTS[(constant_size * offset) + q31] = 4.22;
CONSTANTS[(constant_size * offset) + A41] = 0.001416;
CONSTANTS[(constant_size * offset) + B41] = -0.02877;
CONSTANTS[(constant_size * offset) + q41] = 1.459;
CONSTANTS[(constant_size * offset) + A51] = 0.4492;
CONSTANTS[(constant_size * offset) + B51] = 0.008595;
CONSTANTS[(constant_size * offset) + q51] = 5;
CONSTANTS[(constant_size * offset) + A52] = 0.3181;
CONSTANTS[(constant_size * offset) + B52] = 3.613E-08;
CONSTANTS[(constant_size * offset) + q52] = 4.663;
CONSTANTS[(constant_size * offset) + A53] = 0.149;
CONSTANTS[(constant_size * offset) + B53] = 0.004668;
CONSTANTS[(constant_size * offset) + q53] = 2.412;
CONSTANTS[(constant_size * offset) + A61] = 0.01241;
CONSTANTS[(constant_size * offset) + B61] = 0.1725;
CONSTANTS[(constant_size * offset) + q61] = 5.568;
CONSTANTS[(constant_size * offset) + A62] = 0.3226;
CONSTANTS[(constant_size * offset) + B62] = -0.0006575;
CONSTANTS[(constant_size * offset) + q62] = 5;
CONSTANTS[(constant_size * offset) + A63] = 0.008978;
CONSTANTS[(constant_size * offset) + B63] = -0.02215;
CONSTANTS[(constant_size * offset) + q63] = 5.682;
CONSTANTS[(constant_size * offset) + Kmax] = 0;
CONSTANTS[(constant_size * offset) + Ku] = 0;
CONSTANTS[(constant_size * offset) + n] = 1;
CONSTANTS[(constant_size * offset) + halfmax] = 1;
CONSTANTS[(constant_size * offset) + Kt] = 0;
CONSTANTS[(constant_size * offset) + Vhalf] = 1;
CONSTANTS[(constant_size * offset) + Temp] = 37;
CONSTANTS[(constant_size * offset) + GKs_b] = 0.006358000000000001;
CONSTANTS[(constant_size * offset) + txs1_max] = 817.3;
STATES[(states_size * offset) + xs1] = 0.2707758025;
STATES[(states_size * offset) + xs2] = 0.0001928503426;
CONSTANTS[(constant_size * offset) + GK1_b] = 0.3239783999999998;
STATES[(states_size * offset) + xk1] = 0.9967597594;
CONSTANTS[(constant_size * offset) + kna1] = 15;
CONSTANTS[(constant_size * offset) + kna2] = 5;
CONSTANTS[(constant_size * offset) + kna3] = 88.12;
CONSTANTS[(constant_size * offset) + kasymm] = 12.5;
CONSTANTS[(constant_size * offset) + wna] = 6e4;
CONSTANTS[(constant_size * offset) + wca] = 6e4;
CONSTANTS[(constant_size * offset) + wnaca] = 5e3;
CONSTANTS[(constant_size * offset) + kcaon] = 1.5e6;
CONSTANTS[(constant_size * offset) + kcaoff] = 5e3;
CONSTANTS[(constant_size * offset) + qna] = 0.5224;
CONSTANTS[(constant_size * offset) + qca] = 0.167;
CONSTANTS[(constant_size * offset) + KmCaAct] = 150e-6;
CONSTANTS[(constant_size * offset) + Gncx_b] = 0.0008;
CONSTANTS[(constant_size * offset) + k1p] = 949.5;
CONSTANTS[(constant_size * offset) + k1m] = 182.4;
CONSTANTS[(constant_size * offset) + k2p] = 687.2;
CONSTANTS[(constant_size * offset) + k2m] = 39.4;
CONSTANTS[(constant_size * offset) + k3p] = 1899;
CONSTANTS[(constant_size * offset) + k3m] = 79300;
CONSTANTS[(constant_size * offset) + k4p] = 639;
CONSTANTS[(constant_size * offset) + k4m] = 40;
CONSTANTS[(constant_size * offset) + Knai0] = 9.073;
CONSTANTS[(constant_size * offset) + Knao0] = 27.78;
CONSTANTS[(constant_size * offset) + delta] = -0.155;
CONSTANTS[(constant_size * offset) + Kki] = 0.5;
CONSTANTS[(constant_size * offset) + Kko] = 0.3582;
CONSTANTS[(constant_size * offset) + MgADP] = 0.05;
CONSTANTS[(constant_size * offset) + MgATP] = 9.8;
CONSTANTS[(constant_size * offset) + Kmgatp] = 1.698e-7;
CONSTANTS[(constant_size * offset) + H] = 1e-7;
CONSTANTS[(constant_size * offset) + eP] = 4.2;
CONSTANTS[(constant_size * offset) + Khp] = 1.698e-7;
CONSTANTS[(constant_size * offset) + Knap] = 224;
CONSTANTS[(constant_size * offset) + Kxkur] = 292;
CONSTANTS[(constant_size * offset) + Pnak_b] = 30;
CONSTANTS[(constant_size * offset) + GKb_b] = 0.003;
CONSTANTS[(constant_size * offset) + PNab] = 3.75e-10;
CONSTANTS[(constant_size * offset) + PCab] = 2.5e-8;
CONSTANTS[(constant_size * offset) + GpCa] = 0.0005;
CONSTANTS[(constant_size * offset) + KmCap] = 0.0005;
CONSTANTS[(constant_size * offset) + bt] = 4.75;
STATES[(states_size * offset) + Jrelnp] = 2.5e-7;
STATES[(states_size * offset) + Jrelp] = 3.12e-7;
CONSTANTS[(constant_size * offset) + Jrel_scaling_factor] = 1.0;
CONSTANTS[(constant_size * offset) + Jup_b] = 1.0;
CONSTANTS[(constant_size * offset) + frt] = CONSTANTS[(constant_size * offset) + F]/( CONSTANTS[(constant_size * offset) + R]*CONSTANTS[(constant_size * offset) + T]);
CONSTANTS[(constant_size * offset) + cmdnmax] = (CONSTANTS[(constant_size * offset) + celltype]==1.00000 ?  CONSTANTS[(constant_size * offset) + cmdnmax_b]*1.30000 : CONSTANTS[(constant_size * offset) + cmdnmax_b]);
CONSTANTS[(constant_size * offset) + Ahs] = 1.00000 - CONSTANTS[(constant_size * offset) + Ahf];
CONSTANTS[(constant_size * offset) + thLp] =  3.00000*CONSTANTS[(constant_size * offset) + thL];
CONSTANTS[(constant_size * offset) + GNaL] = (CONSTANTS[(constant_size * offset) + celltype]==1.00000 ?  CONSTANTS[(constant_size * offset) + GNaL_b]*0.600000 : CONSTANTS[(constant_size * offset) + GNaL_b]);
CONSTANTS[(constant_size * offset) + Gto] = (CONSTANTS[(constant_size * offset) + celltype]==1.00000 ?  CONSTANTS[(constant_size * offset) + Gto_b]*4.00000 : CONSTANTS[(constant_size * offset) + celltype]==2.00000 ?  CONSTANTS[(constant_size * offset) + Gto_b]*4.00000 : CONSTANTS[(constant_size * offset) + Gto_b]);
CONSTANTS[(constant_size * offset) + Aff] = 0.600000;
CONSTANTS[(constant_size * offset) + PCa] = (CONSTANTS[(constant_size * offset) + celltype]==1.00000 ?  CONSTANTS[(constant_size * offset) + PCa_b]*1.20000 : CONSTANTS[(constant_size * offset) + celltype]==2.00000 ?  CONSTANTS[(constant_size * offset) + PCa_b]*2.50000 : CONSTANTS[(constant_size * offset) + PCa_b]);
CONSTANTS[(constant_size * offset) + tjca] = 75.0000;
CONSTANTS[(constant_size * offset) + v0_CaL] = 0.000000;
CONSTANTS[(constant_size * offset) + GKr] = (CONSTANTS[(constant_size * offset) + celltype]==1.00000 ?  CONSTANTS[(constant_size * offset) + GKr_b]*1.30000 : CONSTANTS[(constant_size * offset) + celltype]==2.00000 ?  CONSTANTS[(constant_size * offset) + GKr_b]*0.800000 : CONSTANTS[(constant_size * offset) + GKr_b]);
CONSTANTS[(constant_size * offset) + GKs] = (CONSTANTS[(constant_size * offset) + celltype]==1.00000 ?  CONSTANTS[(constant_size * offset) + GKs_b]*1.40000 : CONSTANTS[(constant_size * offset) + GKs_b]);
CONSTANTS[(constant_size * offset) + GK1] = (CONSTANTS[(constant_size * offset) + celltype]==1.00000 ?  CONSTANTS[(constant_size * offset) + GK1_b]*1.20000 : CONSTANTS[(constant_size * offset) + celltype]==2.00000 ?  CONSTANTS[(constant_size * offset) + GK1_b]*1.30000 : CONSTANTS[(constant_size * offset) + GK1_b]);
CONSTANTS[(constant_size * offset) + vcell] =  1000.00*3.14000*CONSTANTS[(constant_size * offset) + rad]*CONSTANTS[(constant_size * offset) + rad]*CONSTANTS[(constant_size * offset) + L];
CONSTANTS[(constant_size * offset) + GKb] = (CONSTANTS[(constant_size * offset) + celltype]==1.00000 ?  CONSTANTS[(constant_size * offset) + GKb_b]*0.600000 : CONSTANTS[(constant_size * offset) + GKb_b]);
CONSTANTS[(constant_size * offset) + v0_Nab] = 0.000000;
CONSTANTS[(constant_size * offset) + v0_Cab] = 0.000000;
CONSTANTS[(constant_size * offset) + a_rel] =  0.500000*CONSTANTS[(constant_size * offset) + bt];
CONSTANTS[(constant_size * offset) + btp] =  1.25000*CONSTANTS[(constant_size * offset) + bt];
CONSTANTS[(constant_size * offset) + upScale] = (CONSTANTS[(constant_size * offset) + celltype]==1.00000 ? 1.30000 : 1.00000);
CONSTANTS[(constant_size * offset) + cnc] = 0.000000;
CONSTANTS[(constant_size * offset) + ffrt] =  CONSTANTS[(constant_size * offset) + F]*CONSTANTS[(constant_size * offset) + frt];
CONSTANTS[(constant_size * offset) + Afs] = 1.00000 - CONSTANTS[(constant_size * offset) + Aff];
CONSTANTS[(constant_size * offset) + PCap] =  1.10000*CONSTANTS[(constant_size * offset) + PCa];
CONSTANTS[(constant_size * offset) + PCaNa] =  0.00125000*CONSTANTS[(constant_size * offset) + PCa];
CONSTANTS[(constant_size * offset) + PCaK] =  0.000357400*CONSTANTS[(constant_size * offset) + PCa];
CONSTANTS[(constant_size * offset) + B_1] =  2.00000*CONSTANTS[(constant_size * offset) + frt];
CONSTANTS[(constant_size * offset) + B_2] = CONSTANTS[(constant_size * offset) + frt];
CONSTANTS[(constant_size * offset) + B_3] = CONSTANTS[(constant_size * offset) + frt];
CONSTANTS[(constant_size * offset) + Ageo] =  2.00000*3.14000*CONSTANTS[(constant_size * offset) + rad]*CONSTANTS[(constant_size * offset) + rad]+ 2.00000*3.14000*CONSTANTS[(constant_size * offset) + rad]*CONSTANTS[(constant_size * offset) + L];
CONSTANTS[(constant_size * offset) + B_Nab] = CONSTANTS[(constant_size * offset) + frt];
CONSTANTS[(constant_size * offset) + B_Cab] =  2.00000*CONSTANTS[(constant_size * offset) + frt];
CONSTANTS[(constant_size * offset) + a_relp] =  0.500000*CONSTANTS[(constant_size * offset) + btp];
CONSTANTS[(constant_size * offset) + PCaNap] =  0.00125000*CONSTANTS[(constant_size * offset) + PCap];
CONSTANTS[(constant_size * offset) + PCaKp] =  0.000357400*CONSTANTS[(constant_size * offset) + PCap];
CONSTANTS[(constant_size * offset) + Acap] =  2.00000*CONSTANTS[(constant_size * offset) + Ageo];
CONSTANTS[(constant_size * offset) + vmyo] =  0.680000*CONSTANTS[(constant_size * offset) + vcell];
CONSTANTS[(constant_size * offset) + vnsr] =  0.0552000*CONSTANTS[(constant_size * offset) + vcell];
CONSTANTS[(constant_size * offset) + vjsr] =  0.00480000*CONSTANTS[(constant_size * offset) + vcell];
CONSTANTS[(constant_size * offset) + vss] =  0.0200000*CONSTANTS[(constant_size * offset) + vcell];
CONSTANTS[(constant_size * offset) + h10_i] = CONSTANTS[(constant_size * offset) + kasymm]+1.00000+ (CONSTANTS[(constant_size * offset) + nao]/CONSTANTS[(constant_size * offset) + kna1])*(1.00000+CONSTANTS[(constant_size * offset) + nao]/CONSTANTS[(constant_size * offset) + kna2]);
CONSTANTS[(constant_size * offset) + h11_i] = ( CONSTANTS[(constant_size * offset) + nao]*CONSTANTS[(constant_size * offset) + nao])/( CONSTANTS[(constant_size * offset) + h10_i]*CONSTANTS[(constant_size * offset) + kna1]*CONSTANTS[(constant_size * offset) + kna2]);
CONSTANTS[(constant_size * offset) + h12_i] = 1.00000/CONSTANTS[(constant_size * offset) + h10_i];
CONSTANTS[(constant_size * offset) + k1_i] =  CONSTANTS[(constant_size * offset) + h12_i]*CONSTANTS[(constant_size * offset) + cao]*CONSTANTS[(constant_size * offset) + kcaon];
CONSTANTS[(constant_size * offset) + k2_i] = CONSTANTS[(constant_size * offset) + kcaoff];
CONSTANTS[(constant_size * offset) + k5_i] = CONSTANTS[(constant_size * offset) + kcaoff];
CONSTANTS[(constant_size * offset) + Gncx] = (CONSTANTS[(constant_size * offset) + celltype]==1.00000 ?  CONSTANTS[(constant_size * offset) + Gncx_b]*1.10000 : CONSTANTS[(constant_size * offset) + celltype]==2.00000 ?  CONSTANTS[(constant_size * offset) + Gncx_b]*1.40000 : CONSTANTS[(constant_size * offset) + Gncx_b]);
CONSTANTS[(constant_size * offset) + h10_ss] = CONSTANTS[(constant_size * offset) + kasymm]+1.00000+ (CONSTANTS[(constant_size * offset) + nao]/CONSTANTS[(constant_size * offset) + kna1])*(1.00000+CONSTANTS[(constant_size * offset) + nao]/CONSTANTS[(constant_size * offset) + kna2]);
CONSTANTS[(constant_size * offset) + h11_ss] = ( CONSTANTS[(constant_size * offset) + nao]*CONSTANTS[(constant_size * offset) + nao])/( CONSTANTS[(constant_size * offset) + h10_ss]*CONSTANTS[(constant_size * offset) + kna1]*CONSTANTS[(constant_size * offset) + kna2]);
CONSTANTS[(constant_size * offset) + h12_ss] = 1.00000/CONSTANTS[(constant_size * offset) + h10_ss];
CONSTANTS[(constant_size * offset) + k1_ss] =  CONSTANTS[(constant_size * offset) + h12_ss]*CONSTANTS[(constant_size * offset) + cao]*CONSTANTS[(constant_size * offset) + kcaon];
CONSTANTS[(constant_size * offset) + k2_ss] = CONSTANTS[(constant_size * offset) + kcaoff];
CONSTANTS[(constant_size * offset) + k5_ss] = CONSTANTS[(constant_size * offset) + kcaoff];
CONSTANTS[(constant_size * offset) + b1] =  CONSTANTS[(constant_size * offset) + k1m]*CONSTANTS[(constant_size * offset) + MgADP];
CONSTANTS[(constant_size * offset) + a2] = CONSTANTS[(constant_size * offset) + k2p];
CONSTANTS[(constant_size * offset) + a4] = (( CONSTANTS[(constant_size * offset) + k4p]*CONSTANTS[(constant_size * offset) + MgATP])/CONSTANTS[(constant_size * offset) + Kmgatp])/(1.00000+CONSTANTS[(constant_size * offset) + MgATP]/CONSTANTS[(constant_size * offset) + Kmgatp]);
CONSTANTS[(constant_size * offset) + Pnak] = (CONSTANTS[(constant_size * offset) + celltype]==1.00000 ?  CONSTANTS[(constant_size * offset) + Pnak_b]*0.900000 : CONSTANTS[(constant_size * offset) + celltype]==2.00000 ?  CONSTANTS[(constant_size * offset) + Pnak_b]*0.700000 : CONSTANTS[(constant_size * offset) + Pnak_b]);
}

__device__ void applyDrugEffect(double *CONSTANTS, double conc, double *ic50, double epsilon, int offset)
{
CONSTANTS[(constant_size * offset) + GK1] = CONSTANTS[(constant_size * offset) + GK1] * ((hill[2] > 10E-14 && hill[3] > 10E-14) ? 1./(1.+pow(conc/hill[2],hill[3])) : 1.);
CONSTANTS[(constant_size * offset) + GKs] = CONSTANTS[(constant_size * offset) + GKs] * ((hill[4] > 10E-14 && hill[5] > 10E-14) ? 1./(1.+pow(conc/hill[4],hill[5])) : 1.);
CONSTANTS[(constant_size * offset) + GNaL] = CONSTANTS[(constant_size * offset) + GNaL] * ((hill[8] > 10E-14 && hill[9] > 10E-14) ? 1./(1.+pow(conc/hill[8],hill[9])) : 1.);
CONSTANTS[(constant_size * offset) + GNa] = CONSTANTS[(constant_size * offset) + GNa] * ((hill[6] > 10E-14 && hill[7] > 10E-14) ? 1./(1.+pow(conc/hill[6],hill[7])) : 1.);
CONSTANTS[(constant_size * offset) + Gto] = CONSTANTS[(constant_size * offset) + Gto] * ((hill[10] > 10E-14 && hill[11] > 10E-14) ? 1./(1.+pow(conc/hill[10],hill[11])) : 1.);
CONSTANTS[(constant_size * offset) + PCa] = CONSTANTS[(constant_size * offset) + PCa] * ( (hill[0] > 10E-14 && hill[1] > 10E-14) ? 1./(1.+pow(conc/hill[0],hill[1])) : 1.);
}

__device__ void ___applyHERGBinding(double conc, const double *herg)
{
if(conc > 10E-14){
CONSTANTS[(constant_size * offset) + Kmax] = herg[0];
CONSTANTS[(constant_size * offset) + Ku] = herg[1];
CONSTANTS[(constant_size * offset) + n] = herg[2];
CONSTANTS[(constant_size * offset) + halfmax] = herg[3];
CONSTANTS[(constant_size * offset) + Vhalf] = herg[4];
CONSTANTS[(constant_size * offset) + cnc] = conc;
STATES[(states_size * offset) + D] = CONSTANTS[(constant_size * offset) + cnc];
}
}

// void ohara_rudy_cipa_v1_2017::initConsts()
// {
// 	___initConsts(0.);
// }

// void ohara_rudy_cipa_v1_2017::initConsts(double type)
// {
// 	___initConsts(type);
// }

__device__ void initConsts(double *CONSTANTS, double *STATES, double type, double conc, double *ic50, double *cvar, bool is_dutta, bool is_cvar, double bcl, int offset)
{
	___initConsts(type);
	// mpi_printf(0,"Celltype: %lf\n", CONSTANTS[celltype]);
	// mpi_printf(0,"Control %lf %lf %lf %lf %lf\n", CONSTANTS[PCa], CONSTANTS[GK1], CONSTANTS[GKs], CONSTANTS[GNaL], CONSTANTS[GKr]);
	___applyDrugEffect(conc, hill);
	// mpi_printf(0,"After drug %lf %lf %lf %lf %lf\n", CONSTANTS[PCa], CONSTANTS[GK1], CONSTANTS[GKs], CONSTANTS[GNaL], CONSTANTS[GKr]);
	// mpi_printf(0,"Control hERG binding %lf %lf %lf %lf %lf %lf\n", CONSTANTS[Kmax], CONSTANTS[Ku], CONSTANTS[n], CONSTANTS[halfmax], CONSTANTS[Vhalf], CONSTANTS[cnc]);
	___applyHERGBinding(conc, herg);
	// mpi_printf(0,"Bootstrapped hERG binding %lf %lf %lf %lf %lf %lf\n", CONSTANTS[Kmax], CONSTANTS[Ku], CONSTANTS[n], CONSTANTS[halfmax], CONSTANTS[Vhalf], CONSTANTS[cnc]);
}

__device__ void computeRates( double TIME, double *CONSTANTS, double *RATES, double *STATES, double *ALGEBRAIC, int offset )
{
ALGEBRAIC[(algebraic_size * offset) + hLss] = 1.00000/(1.00000+exp((STATES[(states_size * offset) + V]+87.6100)/7.48800));
ALGEBRAIC[(algebraic_size * offset) + hLssp] = 1.00000/(1.00000+exp((STATES[(states_size * offset) + V]+93.8100)/7.48800));
ALGEBRAIC[(algebraic_size * offset) + mss] = 1.00000/(1.00000+exp(- (STATES[(states_size * offset) + V]+CONSTANTS[mssV1])/CONSTANTS[(constant_size * offset) + mssV2]));
ALGEBRAIC[(algebraic_size * offset) + tm] = 1.00000/( CONSTANTS[(constant_size * offset) + mtD1]*exp((STATES[(states_size * offset) + V]+CONSTANTS[(constant_size * offset) + mtV1])/CONSTANTS[(constant_size * offset) + mtV2])+ CONSTANTS[(constant_size * offset) + mtD2]*exp(- (STATES[(states_size * offset) + V]+CONSTANTS[(constant_size * offset) + mtV3])/CONSTANTS[(constant_size * offset) + mtV4]));
ALGEBRAIC[(algebraic_size * offset) + hss] = 1.00000/(1.00000+exp(((STATES[(states_size * offset) + V]+CONSTANTS[(constant_size * offset) + hssV1]) - CONSTANTS[(constant_size * offset) + shift_INa_inact])/CONSTANTS[(constant_size * offset) + hssV2]));
ALGEBRAIC[(algebraic_size * offset) + thf] = 1.00000/( 1.43200e-05*exp(- ((STATES[(states_size * offset) + V]+1.19600) - CONSTANTS[(constant_size * offset) + shift_INa_inact])/6.28500)+ 6.14900*exp(((STATES[(states_size * offset) + V]+0.509600) - CONSTANTS[(constant_size * offset) + shift_INa_inact])/20.2700));
ALGEBRAIC[(algebraic_size * offset) + ths] = 1.00000/( 0.00979400*exp(- ((STATES[(states_size * offset) + V]+17.9500) - CONSTANTS[(constant_size * offset) + shift_INa_inact])/28.0500)+ 0.334300*exp(((STATES[(states_size * offset) + V]+5.73000) - CONSTANTS[(constant_size * offset) + shift_INa_inact])/56.6600));
ALGEBRAIC[(algebraic_size * offset) + ass] = 1.00000/(1.00000+exp(- (STATES[(states_size * offset) + V] - 14.3400)/14.8200));
ALGEBRAIC[(algebraic_size * offset) + ta] = 1.05150/(1.00000/( 1.20890*(1.00000+exp(- (STATES[(states_size * offset) + V] - 18.4099)/29.3814)))+3.50000/(1.00000+exp((STATES[(states_size * offset) + V]+100.000)/29.3814)));
ALGEBRAIC[(algebraic_size * offset) + dss] = 1.00000/(1.00000+exp(- (STATES[(states_size * offset) + V]+3.94000)/4.23000));
ALGEBRAIC[(algebraic_size * offset) + td] = 0.600000+1.00000/(exp( - 0.0500000*(STATES[(states_size * offset) + V]+6.00000))+exp( 0.0900000*(STATES[(states_size * offset) + V]+14.0000)));
ALGEBRAIC[(algebraic_size * offset) + fss] = 1.00000/(1.00000+exp((STATES[(states_size * offset) + V]+19.5800)/3.69600));
ALGEBRAIC[(algebraic_size * offset) + tff] = 7.00000+1.00000/( 0.00450000*exp(- (STATES[(states_size * offset) + V]+20.0000)/10.0000)+ 0.00450000*exp((STATES[(states_size * offset) + V]+20.0000)/10.0000));
ALGEBRAIC[(algebraic_size * offset) + tfs] = 1000.00+1.00000/( 3.50000e-05*exp(- (STATES[(states_size * offset) + V]+5.00000)/4.00000)+ 3.50000e-05*exp((STATES[(states_size * offset) + V]+5.00000)/6.00000));
ALGEBRAIC[(algebraic_size * offset) + fcass] = ALGEBRAIC[(algebraic_size * offset) + fss];
ALGEBRAIC[(algebraic_size * offset) + km2n] =  STATES[(states_size * offset) + jca]*1.00000;
ALGEBRAIC[(algebraic_size * offset) + anca] = 1.00000/(CONSTANTS[(constant_size * offset) + k2n]/ALGEBRAIC[(algebraic_size * offset) + km2n]+pow(1.00000+CONSTANTS[(constant_size * offset) + Kmn]/STATES[(states_size * offset) + cass], 4.00000));
ALGEBRAIC[(algebraic_size * offset) + xs1ss] = 1.00000/(1.00000+exp(- (STATES[(states_size * offset) + V]+11.6000)/8.93200));
ALGEBRAIC[(algebraic_size * offset) + txs1] = CONSTANTS[(constant_size * offset) + txs1_max]+1.00000/( 0.000232600*exp((STATES[(states_size * offset) + V]+48.2800)/17.8000)+ 0.00129200*exp(- (STATES[(states_size * offset) + V]+210.000)/230.000));
ALGEBRAIC[(algebraic_size * offset) + xk1ss] = 1.00000/(1.00000+exp(- (STATES[(states_size * offset) + V]+ 2.55380*CONSTANTS[(constant_size * offset) + ko]+144.590)/( 1.56920*CONSTANTS[(constant_size * offset) + ko]+3.81150)));
ALGEBRAIC[(algebraic_size * offset) + txk1] = 122.200/(exp(- (STATES[(states_size * offset) + V]+127.200)/20.3600)+exp((STATES[(states_size * offset) + V]+236.800)/69.3300));
ALGEBRAIC[(algebraic_size * offset) + CaMKb] = ( CONSTANTS[(constant_size * offset) + CaMKo]*(1.00000 - STATES[(states_size * offset) + CaMKt]))/(1.00000+CONSTANTS[(constant_size * offset) + KmCaM]/STATES[(states_size * offset) + cass]);
ALGEBRAIC[(algebraic_size * offset) + jss] = ALGEBRAIC[(algebraic_size * offset) + hss];
ALGEBRAIC[(algebraic_size * offset) + tj] = 2.03800+1.00000/( 0.0213600*exp(- ((STATES[(states_size * offset) + V]+100.600) - CONSTANTS[(constant_size * offset) + shift_INa_inact])/8.28100)+ 0.305200*exp(((STATES[(states_size * offset) + V]+0.994100) - CONSTANTS[(constant_size * offset) + shift_INa_inact])/38.4500));
ALGEBRAIC[(algebraic_size * offset) + assp] = 1.00000/(1.00000+exp(- (STATES[(states_size * offset) + V] - 24.3400)/14.8200));
ALGEBRAIC[(algebraic_size * offset) + tfcaf] = 7.00000+1.00000/( 0.0400000*exp(- (STATES[(states_size * offset) + V] - 4.00000)/7.00000)+ 0.0400000*exp((STATES[(states_size * offset) + V] - 4.00000)/7.00000));
ALGEBRAIC[(algebraic_size * offset) + tfcas] = 100.000+1.00000/( 0.000120000*exp(- STATES[(states_size * offset) + V]/3.00000)+ 0.000120000*exp(STATES[(states_size * offset) + V]/7.00000));
ALGEBRAIC[(algebraic_size * offset) + tffp] =  2.50000*ALGEBRAIC[(algebraic_size * offset) + tff];
ALGEBRAIC[(algebraic_size * offset) + xs2ss] = ALGEBRAIC[(algebraic_size * offset) + xs1ss];
ALGEBRAIC[(algebraic_size * offset) + txs2] = 1.00000/( 0.0100000*exp((STATES[(states_size * offset) + V] - 50.0000)/20.0000)+ 0.0193000*exp(- (STATES[(states_size * offset) + V]+66.5400)/31.0000));
ALGEBRAIC[(algebraic_size * offset) + hssp] = 1.00000/(1.00000+exp(((STATES[(states_size * offset) + V]+89.1000) - CONSTANTS[(constant_size * offset) + shift_INa_inact])/6.08600));
ALGEBRAIC[(algebraic_size * offset) + thsp] =  3.00000*ALGEBRAIC[(algebraic_size * offset) + ths];
ALGEBRAIC[(algebraic_size * offset) + tjp] =  1.46000*ALGEBRAIC[(algebraic_size * offset) + tj];
ALGEBRAIC[(algebraic_size * offset) + mLss] = 1.00000/(1.00000+exp(- (STATES[(states_size * offset) + V]+42.8500)/5.26400));
ALGEBRAIC[(algebraic_size * offset) + tmL] = ALGEBRAIC[(algebraic_size * offset) + tm];
ALGEBRAIC[(algebraic_size * offset) + tfcafp] =  2.50000*ALGEBRAIC[(algebraic_size * offset) + tfcaf];
ALGEBRAIC[(algebraic_size * offset) + iss] = 1.00000/(1.00000+exp((STATES[(states_size * offset) + V]+43.9400)/5.71100));
ALGEBRAIC[(algebraic_size * offset) + delta_epi] = (CONSTANTS[(constant_size * offset) + celltype]==1.00000 ? 1.00000 - 0.950000/(1.00000+exp((STATES[(states_size * offset) + V]+70.0000)/5.00000)) : 1.00000);
ALGEBRAIC[(algebraic_size * offset) + tiF_b] = 4.56200+1.00000/( 0.393300*exp(- (STATES[(states_size * offset) + V]+100.000)/100.000)+ 0.0800400*exp((STATES[(states_size * offset) + V]+50.0000)/16.5900));
ALGEBRAIC[(algebraic_size * offset) + tiF] =  ALGEBRAIC[(algebraic_size * offset) + tiF_b]*ALGEBRAIC[(algebraic_size * offset) + delta_epi];
ALGEBRAIC[(algebraic_size * offset) + tiS_b] = 23.6200+1.00000/( 0.00141600*exp(- (STATES[(states_size * offset) + V]+96.5200)/59.0500)+ 1.78000e-08*exp((STATES[(states_size * offset) + V]+114.100)/8.07900));
ALGEBRAIC[(algebraic_size * offset) + tiS] =  ALGEBRAIC[(algebraic_size * offset) + tiS_b]*ALGEBRAIC[(algebraic_size * offset) + delta_epi];
ALGEBRAIC[(algebraic_size * offset) + dti_develop] = 1.35400+0.000100000/(exp((STATES[(states_size * offset) + V] - 167.400)/15.8900)+exp(- (STATES[(states_size * offset) + V] - 12.2300)/0.215400));
ALGEBRAIC[(algebraic_size * offset) + dti_recover] = 1.00000 - 0.500000/(1.00000+exp((STATES[(states_size * offset) + V]+70.0000)/20.0000));
ALGEBRAIC[(algebraic_size * offset) + tiFp] =  ALGEBRAIC[(algebraic_size * offset) + dti_develop]*ALGEBRAIC[(algebraic_size * offset) + dti_recover]*ALGEBRAIC[(algebraic_size * offset) + tiF];
ALGEBRAIC[(algebraic_size * offset) + tiSp] =  ALGEBRAIC[(algebraic_size * offset) + dti_develop]*ALGEBRAIC[(algebraic_size * offset) + dti_recover]*ALGEBRAIC[(algebraic_size * offset) + tiS];
ALGEBRAIC[(algebraic_size * offset) + f] =  CONSTANTS[(constant_size * offset) + Aff]*STATES[(states_size * offset) + ff]+ CONSTANTS[(constant_size * offset) + Afs]*STATES[(states_size * offset) + fs];
ALGEBRAIC[(algebraic_size * offset) + Afcaf] = 0.300000+0.600000/(1.00000+exp((STATES[(states_size * offset) + V] - 10.0000)/10.0000));
ALGEBRAIC[(algebraic_size * offset) + Afcas] = 1.00000 - ALGEBRAIC[(algebraic_size * offset) + Afcaf];
ALGEBRAIC[(algebraic_size * offset) + fca] =  ALGEBRAIC[(algebraic_size * offset) + Afcaf]*STATES[(states_size * offset) + fcaf]+ ALGEBRAIC[(algebraic_size * offset) + Afcas]*STATES[(states_size * offset) + fcas];
ALGEBRAIC[(algebraic_size * offset) + fp] =  CONSTANTS[(constant_size * offset) + Aff]*STATES[(states_size * offset) + ffp]+ CONSTANTS[(constant_size * offset) + Afs]*STATES[(states_size * offset) + fs];
ALGEBRAIC[(algebraic_size * offset) + fcap] =  ALGEBRAIC[(algebraic_size * offset) + Afcaf]*STATES[(states_size * offset) + fcafp]+ ALGEBRAIC[(algebraic_size * offset) + Afcas]*STATES[(states_size * offset) + fcas];
ALGEBRAIC[(algebraic_size * offset) + vfrt] =  STATES[(states_size * offset) + V]*CONSTANTS[(constant_size * offset) + frt];
ALGEBRAIC[(algebraic_size * offset) + A_1] = ( 4.00000*CONSTANTS[(constant_size * offset) + ffrt]*( STATES[(states_size * offset) + cass]*exp( 2.00000*ALGEBRAIC[(algebraic_size * offset) + vfrt]) -  0.341000*CONSTANTS[(constant_size * offset) + cao]))/CONSTANTS[(constant_size * offset) + B_1];
ALGEBRAIC[(algebraic_size * offset) + U_1] =  CONSTANTS[(constant_size * offset) + B_1]*(STATES[(states_size * offset) + V] - CONSTANTS[(constant_size * offset) + v0_CaL]);
ALGEBRAIC[(algebraic_size * offset) + PhiCaL] = (- 1.00000e-07<=ALGEBRAIC[(algebraic_size * offset) + U_1]&&ALGEBRAIC[(algebraic_size * offset) + U_1]<=1.00000e-07 ?  ALGEBRAIC[(algebraic_size * offset) + A_1]*(1.00000 -  0.500000*ALGEBRAIC[(algebraic_size * offset) + U_1]) : ( ALGEBRAIC[(algebraic_size * offset) + A_1]*ALGEBRAIC[(algebraic_size * offset) + U_1])/(exp(ALGEBRAIC[(algebraic_size * offset) + U_1]) - 1.00000));
ALGEBRAIC[(algebraic_size * offset) + CaMKa] = ALGEBRAIC[(algebraic_size * offset) + CaMKb]+STATES[(states_size * offset) + CaMKt];
ALGEBRAIC[(algebraic_size * offset) + fICaLp] = 1.00000/(1.00000+CONSTANTS[(constant_size * offset) + KmCaMK]/ALGEBRAIC[(algebraic_size * offset) + CaMKa]);
ALGEBRAIC[(algebraic_size * offset) + ICaL] =  (1.00000 - ALGEBRAIC[(algebraic_size * offset) + fICaLp])*CONSTANTS[(constant_size * offset) + PCa]*ALGEBRAIC[(algebraic_size * offset) + PhiCaL]*STATES[(states_size * offset) + d]*( ALGEBRAIC[(algebraic_size * offset) + f]*(1.00000 - STATES[(states_size * offset) + nca])+ STATES[(states_size * offset) + jca]*ALGEBRAIC[(algebraic_size * offset) + fca]*STATES[(states_size * offset) + nca])+ ALGEBRAIC[(algebraic_size * offset) + fICaLp]*CONSTANTS[(constant_size * offset) + PCap]*ALGEBRAIC[(algebraic_size * offset) + PhiCaL]*STATES[(states_size * offset) + d]*( ALGEBRAIC[(algebraic_size * offset) + fp]*(1.00000 - STATES[(states_size * offset) + nca])+ STATES[(states_size * offset) + jca]*ALGEBRAIC[(algebraic_size * offset) + fcap]*STATES[(states_size * offset) + nca]);
ALGEBRAIC[(algebraic_size * offset) + Jrel_inf_temp] = ( CONSTANTS[(constant_size * offset) + a_rel]*- ALGEBRAIC[(algebraic_size * offset) + ICaL])/(1.00000+ 1.00000*pow(1.50000/STATES[(states_size * offset) + cajsr], 8.00000));
ALGEBRAIC[(algebraic_size * offset) + Jrel_inf] = (CONSTANTS[(constant_size * offset) + celltype]==2.00000 ?  ALGEBRAIC[(algebraic_size * offset) + Jrel_inf_temp]*1.70000 : ALGEBRAIC[(algebraic_size * offset) + Jrel_inf_temp]);
ALGEBRAIC[(algebraic_size * offset) + tau_rel_temp] = CONSTANTS[(constant_size * offset) + bt]/(1.00000+0.0123000/STATES[(states_size * offset) + cajsr]);
ALGEBRAIC[(algebraic_size * offset) + tau_rel] = (ALGEBRAIC[(algebraic_size * offset) + tau_rel_temp]<0.00100000 ? 0.00100000 : ALGEBRAIC[(algebraic_size * offset) + tau_rel_temp]);
ALGEBRAIC[(algebraic_size * offset) + Jrel_temp] = ( CONSTANTS[(constant_size * offset) + a_relp]*- ALGEBRAIC[(algebraic_size * offset) + ICaL])/(1.00000+pow(1.50000/STATES[(states_size * offset) + cajsr], 8.00000));
ALGEBRAIC[(algebraic_size * offset) + Jrel_infp] = (CONSTANTS[(constant_size * offset) + celltype]==2.00000 ?  ALGEBRAIC[(algebraic_size * offset) + Jrel_temp]*1.70000 : ALGEBRAIC[(algebraic_size * offset) + Jrel_temp]);
ALGEBRAIC[(algebraic_size * offset) + tau_relp_temp] = CONSTANTS[(constant_size * offset) + btp]/(1.00000+0.0123000/STATES[(states_size * offset) + cajsr]);
ALGEBRAIC[(algebraic_size * offset) + tau_relp] = (ALGEBRAIC[(algebraic_size * offset) + tau_relp_temp]<0.00100000 ? 0.00100000 : ALGEBRAIC[(algebraic_size * offset) + tau_relp_temp]);
ALGEBRAIC[(algebraic_size * offset) + EK] =  (( CONSTANTS[(constant_size * offset) + R]*CONSTANTS[(constant_size * offset) + T])/CONSTANTS[(constant_size * offset) + F])*log(CONSTANTS[(constant_size * offset) + ko]/STATES[(states_size * offset) + ki]);
ALGEBRAIC[(algebraic_size * offset) + AiF] = 1.00000/(1.00000+exp((STATES[(states_size * offset) + V] - 213.600)/151.200));
ALGEBRAIC[(algebraic_size * offset) + AiS] = 1.00000 - ALGEBRAIC[(algebraic_size * offset) + AiF];
ALGEBRAIC[(algebraic_size * offset) + i] =  ALGEBRAIC[(algebraic_size * offset) + AiF]*STATES[(states_size * offset) + iF]+ ALGEBRAIC[(algebraic_size * offset) + AiS]*STATES[(states_size * offset) + iS];
ALGEBRAIC[(algebraic_size * offset) + ip] =  ALGEBRAIC[(algebraic_size * offset) + AiF]*STATES[(states_size * offset) + iFp]+ ALGEBRAIC[(algebraic_size * offset) + AiS]*STATES[(states_size * offset) + iSp];
ALGEBRAIC[(algebraic_size * offset) + fItop] = 1.00000/(1.00000+CONSTANTS[(constant_size * offset) + KmCaMK]/ALGEBRAIC[(algebraic_size * offset) + CaMKa]);
ALGEBRAIC[(algebraic_size * offset) + Ito] =  CONSTANTS[(constant_size * offset) + Gto]*(STATES[(states_size * offset) + V] - ALGEBRAIC[(algebraic_size * offset) + EK])*( (1.00000 - ALGEBRAIC[(algebraic_size * offset) + fItop])*STATES[(states_size * offset) + a]*ALGEBRAIC[(algebraic_size * offset) + i]+ ALGEBRAIC[(algebraic_size * offset) + fItop]*STATES[(states_size * offset) + ap]*ALGEBRAIC[(algebraic_size * offset) + ip]);
ALGEBRAIC[(algebraic_size * offset) + IKr] =  CONSTANTS[(constant_size * offset) + GKr]* pow((CONSTANTS[(constant_size * offset) + ko]/5.40000), 1.0 / 2)*STATES[(states_size * offset) + O]*(STATES[(states_size * offset) + V] - ALGEBRAIC[(algebraic_size * offset) + EK]);
ALGEBRAIC[(algebraic_size * offset) + EKs] =  (( CONSTANTS[(constant_size * offset) + R]*CONSTANTS[(constant_size * offset) + T])/CONSTANTS[(constant_size * offset) + F])*log((CONSTANTS[(constant_size * offset) + ko]+ CONSTANTS[(constant_size * offset) + PKNa]*CONSTANTS[(constant_size * offset) + nao])/(STATES[(states_size * offset) + ki]+ CONSTANTS[(constant_size * offset) + PKNa]*STATES[(states_size * offset) + nai]));
ALGEBRAIC[(algebraic_size * offset) + KsCa] = 1.00000+0.600000/(1.00000+pow(3.80000e-05/STATES[(states_size * offset) + cai], 1.40000));
ALGEBRAIC[(algebraic_size * offset) + IKs] =  CONSTANTS[(constant_size * offset) + GKs]*ALGEBRAIC[(algebraic_size * offset) + KsCa]*STATES[(states_size * offset) + xs1]*STATES[(states_size * offset) + xs2]*(STATES[(states_size * offset) + V] - ALGEBRAIC[(algebraic_size * offset) + EKs]);
ALGEBRAIC[(algebraic_size * offset) + rk1] = 1.00000/(1.00000+exp(((STATES[(states_size * offset) + V]+105.800) -  2.60000*CONSTANTS[(constant_size * offset) + ko])/9.49300));
ALGEBRAIC[(algebraic_size * offset) + IK1] =  CONSTANTS[(constant_size * offset) + GK1]* pow(CONSTANTS[(constant_size * offset) + ko], 1.0 / 2)*ALGEBRAIC[(algebraic_size * offset) + rk1]*STATES[(states_size * offset) + xk1]*(STATES[(states_size * offset) + V] - ALGEBRAIC[(algebraic_size * offset) + EK]);
ALGEBRAIC[(algebraic_size * offset) + Knao] =  CONSTANTS[(constant_size * offset) + Knao0]*exp(( (1.00000 - CONSTANTS[(constant_size * offset) + delta])*STATES[(states_size * offset) + V]*CONSTANTS[(constant_size * offset) + F])/( 3.00000*CONSTANTS[(constant_size * offset) + R]*CONSTANTS[(constant_size * offset) + T]));
ALGEBRAIC[(algebraic_size * offset) + a3] = ( CONSTANTS[(constant_size * offset) + k3p]*pow(CONSTANTS[(constant_size * offset) + ko]/CONSTANTS[(constant_size * offset) + Kko], 2.00000))/((pow(1.00000+CONSTANTS[(constant_size * offset) + nao]/ALGEBRAIC[(algebraic_size * offset) + Knao], 3.00000)+pow(1.00000+CONSTANTS[(constant_size * offset) + ko]/CONSTANTS[(constant_size * offset) + Kko], 2.00000)) - 1.00000);
ALGEBRAIC[(algebraic_size * offset) + P] = CONSTANTS[(constant_size * offset) + eP]/(1.00000+CONSTANTS[(constant_size * offset) + H]/CONSTANTS[(constant_size * offset) + Khp]+STATES[(states_size * offset) + nai]/CONSTANTS[(constant_size * offset) + Knap]+STATES[(states_size * offset) + ki]/CONSTANTS[(constant_size * offset) + Kxkur]);
ALGEBRAIC[(algebraic_size * offset) + b3] = ( CONSTANTS[(constant_size * offset) + k3m]*ALGEBRAIC[(algebraic_size * offset) + P]*CONSTANTS[(constant_size * offset) + H])/(1.00000+CONSTANTS[(constant_size * offset) + MgATP]/CONSTANTS[(constant_size * offset) + Kmgatp]);
ALGEBRAIC[(algebraic_size * offset) + Knai] =  CONSTANTS[(constant_size * offset) + Knai0]*exp(( CONSTANTS[(constant_size * offset) + delta]*STATES[(states_size * offset) + V]*CONSTANTS[(constant_size * offset) + F])/( 3.00000*CONSTANTS[(constant_size * offset) + R]*CONSTANTS[(constant_size * offset) + T]));
ALGEBRAIC[(algebraic_size * offset) + a1] = ( CONSTANTS[(constant_size * offset) + k1p]*pow(STATES[(states_size * offset) + nai]/ALGEBRAIC[(algebraic_size * offset) + Knai], 3.00000))/((pow(1.00000+STATES[(states_size * offset) + nai]/ALGEBRAIC[(algebraic_size * offset) + Knai], 3.00000)+pow(1.00000+STATES[(states_size * offset) + ki]/CONSTANTS[(constant_size * offset) + Kki], 2.00000)) - 1.00000);
ALGEBRAIC[(algebraic_size * offset) + b2] = ( CONSTANTS[(constant_size * offset) + k2m]*pow(CONSTANTS[(constant_size * offset) + nao]/ALGEBRAIC[(algebraic_size * offset) + Knao], 3.00000))/((pow(1.00000+CONSTANTS[(constant_size * offset) + nao]/ALGEBRAIC[(algebraic_size * offset) + Knao], 3.00000)+pow(1.00000+CONSTANTS[(constant_size * offset) + ko]/CONSTANTS[(constant_size * offset) + Kko], 2.00000)) - 1.00000);
ALGEBRAIC[(algebraic_size * offset) + b4] = ( CONSTANTS[(constant_size * offset) + k4m]*pow(STATES[(states_size * offset) + ki]/CONSTANTS[(constant_size * offset) + Kki], 2.00000))/((pow(1.00000+STATES[(states_size * offset) + nai]/ALGEBRAIC[(algebraic_size * offset) + Knai], 3.00000)+pow(1.00000+STATES[(states_size * offset) + ki]/CONSTANTS[(constant_size * offset) + Kki], 2.00000)) - 1.00000);
ALGEBRAIC[(algebraic_size * offset) + x1] =  CONSTANTS[(constant_size * offset) + a4]*ALGEBRAIC[(algebraic_size * offset) + a1]*CONSTANTS[(constant_size * offset) + a2]+ ALGEBRAIC[(algebraic_size * offset) + b2]*ALGEBRAIC[(algebraic_size * offset) + b4]*ALGEBRAIC[(algebraic_size * offset) + b3]+ CONSTANTS[(constant_size * offset) + a2]*ALGEBRAIC[(algebraic_size * offset) + b4]*ALGEBRAIC[(algebraic_size * offset) + b3]+ ALGEBRAIC[(algebraic_size * offset) + b3]*ALGEBRAIC[(algebraic_size * offset) + a1]*CONSTANTS[(constant_size * offset) + a2];
ALGEBRAIC[(algebraic_size * offset) + x2] =  ALGEBRAIC[(algebraic_size * offset) + b2]*CONSTANTS[(constant_size * offset) + b1]*ALGEBRAIC[(algebraic_size * offset) + b4]+ ALGEBRAIC[(algebraic_size * offset) + a1]*CONSTANTS[(constant_size * offset) + a2]*ALGEBRAIC[(algebraic_size * offset) + a3]+ ALGEBRAIC[(algebraic_size * offset) + a3]*CONSTANTS[(constant_size * offset) + b1]*ALGEBRAIC[(algebraic_size * offset) + b4]+ CONSTANTS[(constant_size * offset) + a2]*ALGEBRAIC[(algebraic_size * offset) + a3]*ALGEBRAIC[(algebraic_size * offset) + b4];
ALGEBRAIC[(algebraic_size * offset) + x3] =  CONSTANTS[(constant_size * offset) + a2]*ALGEBRAIC[(algebraic_size * offset) + a3]*CONSTANTS[(constant_size * offset) + a4]+ ALGEBRAIC[(algebraic_size * offset) + b3]*ALGEBRAIC[(algebraic_size * offset) + b2]*CONSTANTS[(constant_size * offset) + b1]+ ALGEBRAIC[(algebraic_size * offset) + b2]*CONSTANTS[(constant_size * offset) + b1]*CONSTANTS[(constant_size * offset) + a4]+ ALGEBRAIC[(algebraic_size * offset) + a3]*CONSTANTS[(constant_size * offset) + a4]*CONSTANTS[(constant_size * offset) + b1];
ALGEBRAIC[(algebraic_size * offset) + x4] =  ALGEBRAIC[(algebraic_size * offset) + b4]*ALGEBRAIC[(algebraic_size * offset) + b3]*ALGEBRAIC[(algebraic_size * offset) + b2]+ ALGEBRAIC[(algebraic_size * offset) + a3]*CONSTANTS[(constant_size * offset) + a4]*ALGEBRAIC[(algebraic_size * offset) + a1]+ ALGEBRAIC[(algebraic_size * offset) + b2]*CONSTANTS[(constant_size * offset) + a4]*ALGEBRAIC[(algebraic_size * offset) + a1]+ ALGEBRAIC[(algebraic_size * offset) + b3]*ALGEBRAIC[(algebraic_size * offset) + b2]*ALGEBRAIC[(algebraic_size * offset) + a1];
ALGEBRAIC[(algebraic_size * offset) + E1] = ALGEBRAIC[(algebraic_size * offset) + x1]/(ALGEBRAIC[(algebraic_size * offset) + x1]+ALGEBRAIC[(algebraic_size * offset) + x2]+ALGEBRAIC[(algebraic_size * offset) + x3]+ALGEBRAIC[(algebraic_size * offset) + x4]);
ALGEBRAIC[(algebraic_size * offset) + E2] = ALGEBRAIC[(algebraic_size * offset) + x2]/(ALGEBRAIC[(algebraic_size * offset) + x1]+ALGEBRAIC[(algebraic_size * offset) + x2]+ALGEBRAIC[(algebraic_size * offset) + x3]+ALGEBRAIC[(algebraic_size * offset) + x4]);
ALGEBRAIC[(algebraic_size * offset) + JnakNa] =  3.00000*( ALGEBRAIC[(algebraic_size * offset) + E1]*ALGEBRAIC[(algebraic_size * offset) + a3] -  ALGEBRAIC[(algebraic_size * offset) + E2]*ALGEBRAIC[(algebraic_size * offset) + b3]);
ALGEBRAIC[(algebraic_size * offset) + E3] = ALGEBRAIC[(algebraic_size * offset) + x3]/(ALGEBRAIC[(algebraic_size * offset) + x1]+ALGEBRAIC[(algebraic_size * offset) + x2]+ALGEBRAIC[(algebraic_size * offset) + x3]+ALGEBRAIC[(algebraic_size * offset) + x4]);
ALGEBRAIC[(algebraic_size * offset) + E4] = ALGEBRAIC[(algebraic_size * offset) + x4]/(ALGEBRAIC[(algebraic_size * offset) + x1]+ALGEBRAIC[(algebraic_size * offset) + x2]+ALGEBRAIC[(algebraic_size * offset) + x3]+ALGEBRAIC[(algebraic_size * offset) + x4]);
ALGEBRAIC[(algebraic_size * offset) + JnakK] =  2.00000*( ALGEBRAIC[(algebraic_size * offset) + E4]*CONSTANTS[(constant_size * offset) + b1] -  ALGEBRAIC[(algebraic_size * offset) + E3]*ALGEBRAIC[(algebraic_size * offset) + a1]);
ALGEBRAIC[(algebraic_size * offset) + INaK] =  CONSTANTS[(constant_size * offset) + Pnak]*( CONSTANTS[(constant_size * offset) + zna]*ALGEBRAIC[(algebraic_size * offset) + JnakNa]+ CONSTANTS[(constant_size * offset) + zk]*ALGEBRAIC[(algebraic_size * offset) + JnakK]);
ALGEBRAIC[(algebraic_size * offset) + xkb] = 1.00000/(1.00000+exp(- (STATES[(states_size * offset) + V] - 14.4800)/18.3400));
ALGEBRAIC[(algebraic_size * offset) + IKb] =  CONSTANTS[(constant_size * offset) + GKb]*ALGEBRAIC[(algebraic_size * offset) + xkb]*(STATES[(states_size * offset) + V] - ALGEBRAIC[(algebraic_size * offset) + EK]);
ALGEBRAIC[(algebraic_size * offset) + Istim] = (TIME>=CONSTANTS[(constant_size * offset) + stim_start]&&TIME<=CONSTANTS[(constant_size * offset) + stim_end]&&(TIME - CONSTANTS[(constant_size * offset) + stim_start]) -  floor((TIME - CONSTANTS[(constant_size * offset) + stim_start])/CONSTANTS[(constant_size * offset) + BCL])*CONSTANTS[(constant_size * offset) + BCL]<=CONSTANTS[(constant_size * offset) + duration] ? CONSTANTS[(constant_size * offset) + amp] : 0.000000);
ALGEBRAIC[(algebraic_size * offset) + JdiffK] = (STATES[(states_size * offset) + kss] - STATES[(states_size * offset) + ki])/2.00000;
ALGEBRAIC[(algebraic_size * offset) + A_3] = ( 0.750000*CONSTANTS[(constant_size * offset) + ffrt]*( STATES[(states_size * offset) + kss]*exp(ALGEBRAIC[(algebraic_size * offset) + vfrt]) - CONSTANTS[(constant_size * offset) + ko]))/CONSTANTS[(constant_size * offset) + B_3];
ALGEBRAIC[(algebraic_size * offset) + U_3] =  CONSTANTS[(constant_size * offset) + B_3]*(STATES[(states_size * offset) + V] - CONSTANTS[(constant_size * offset) + v0_CaL]);
ALGEBRAIC[(algebraic_size * offset) + PhiCaK] = (- 1.00000e-07<=ALGEBRAIC[(algebraic_size * offset) + U_3]&&ALGEBRAIC[(algebraic_size * offset) + U_3]<=1.00000e-07 ?  ALGEBRAIC[(algebraic_size * offset) + A_3]*(1.00000 -  0.500000*ALGEBRAIC[(algebraic_size * offset) + U_3]) : ( ALGEBRAIC[(algebraic_size * offset) + A_3]*ALGEBRAIC[(algebraic_size * offset) + U_3])/(exp(ALGEBRAIC[(algebraic_size * offset) + U_3]) - 1.00000));
ALGEBRAIC[(algebraic_size * offset) + ICaK] =  (1.00000 - ALGEBRAIC[(algebraic_size * offset) + fICaLp])*CONSTANTS[(constant_size * offset) + PCaK]*ALGEBRAIC[(algebraic_size * offset) + PhiCaK]*STATES[(states_size * offset) + d]*( ALGEBRAIC[(algebraic_size * offset) + f]*(1.00000 - STATES[(states_size * offset) + nca])+ STATES[(states_size * offset) + jca]*ALGEBRAIC[(algebraic_size * offset) + fca]*STATES[(states_size * offset) + nca])+ ALGEBRAIC[(algebraic_size * offset) + fICaLp]*CONSTANTS[(constant_size * offset) + PCaKp]*ALGEBRAIC[(algebraic_size * offset) + PhiCaK]*STATES[(states_size * offset) + d]*( ALGEBRAIC[(algebraic_size * offset) + fp]*(1.00000 - STATES[(states_size * offset) + nca])+ STATES[(states_size * offset) + jca]*ALGEBRAIC[(algebraic_size * offset) + fcap]*STATES[(states_size * offset) + nca]);
ALGEBRAIC[(algebraic_size * offset) + ENa] =  (( CONSTANTS[(constant_size * offset) + R]*CONSTANTS[(constant_size * offset) + T])/CONSTANTS[(constant_size * offset) + F])*log(CONSTANTS[(constant_size * offset) + nao]/STATES[(states_size * offset) + nai]);
ALGEBRAIC[(algebraic_size * offset) + h] =  CONSTANTS[(constant_size * offset) + Ahf]*STATES[(states_size * offset) + hf]+ CONSTANTS[(constant_size * offset) + Ahs]*STATES[(states_size * offset) + hs];
ALGEBRAIC[(algebraic_size * offset) + hp] =  CONSTANTS[(constant_size * offset) + Ahf]*STATES[(states_size * offset) + hf]+ CONSTANTS[(constant_size * offset) + Ahs]*STATES[(states_size * offset) + hsp];
ALGEBRAIC[(algebraic_size * offset) + fINap] = 1.00000/(1.00000+CONSTANTS[(constant_size * offset) + KmCaMK]/ALGEBRAIC[(algebraic_size * offset) + CaMKa]);
ALGEBRAIC[(algebraic_size * offset) + INa] =  CONSTANTS[(constant_size * offset) + GNa]*(STATES[(states_size * offset) + V] - ALGEBRAIC[(algebraic_size * offset) + ENa])*pow(STATES[(states_size * offset) + m], 3.00000)*( (1.00000 - ALGEBRAIC[(algebraic_size * offset) + fINap])*ALGEBRAIC[(algebraic_size * offset) + h]*STATES[(states_size * offset) + j]+ ALGEBRAIC[(algebraic_size * offset) + fINap]*ALGEBRAIC[(algebraic_size * offset) + hp]*STATES[(states_size * offset) + jp]);
ALGEBRAIC[(algebraic_size * offset) + fINaLp] = 1.00000/(1.00000+CONSTANTS[(constant_size * offset) + KmCaMK]/ALGEBRAIC[(algebraic_size * offset) + CaMKa]);
ALGEBRAIC[(algebraic_size * offset) + INaL] =  CONSTANTS[(constant_size * offset) + GNaL]*(STATES[(states_size * offset) + V] - ALGEBRAIC[(algebraic_size * offset) + ENa])*STATES[(states_size * offset) + mL]*( (1.00000 - ALGEBRAIC[(algebraic_size * offset) + fINaLp])*STATES[(states_size * offset) + hL]+ ALGEBRAIC[(algebraic_size * offset) + fINaLp]*STATES[(states_size * offset) + hLp]);
ALGEBRAIC[(algebraic_size * offset) + allo_i] = 1.00000/(1.00000+pow(CONSTANTS[(constant_size * offset) + KmCaAct]/STATES[(states_size * offset) + cai], 2.00000));
ALGEBRAIC[(algebraic_size * offset) + hna] = exp(( CONSTANTS[(constant_size * offset) + qna]*STATES[(states_size * offset) + V]*CONSTANTS[(constant_size * offset) + F])/( CONSTANTS[(constant_size * offset) + R]*CONSTANTS[(constant_size * offset) + T]));
ALGEBRAIC[(algebraic_size * offset) + h7_i] = 1.00000+ (CONSTANTS[(constant_size * offset) + nao]/CONSTANTS[(constant_size * offset) + kna3])*(1.00000+1.00000/ALGEBRAIC[(algebraic_size * offset) + hna]);
ALGEBRAIC[(algebraic_size * offset) + h8_i] = CONSTANTS[(constant_size * offset) + nao]/( CONSTANTS[(constant_size * offset) + kna3]*ALGEBRAIC[(algebraic_size * offset) + hna]*ALGEBRAIC[(algebraic_size * offset) + h7_i]);
ALGEBRAIC[(algebraic_size * offset) + k3pp_i] =  ALGEBRAIC[(algebraic_size * offset) + h8_i]*CONSTANTS[(constant_size * offset) + wnaca];
ALGEBRAIC[(algebraic_size * offset) + h1_i] = 1.00000+ (STATES[(states_size * offset) + nai]/CONSTANTS[(constant_size * offset) + kna3])*(1.00000+ALGEBRAIC[(algebraic_size * offset) + hna]);
ALGEBRAIC[(algebraic_size * offset) + h2_i] = ( STATES[(states_size * offset) + nai]*ALGEBRAIC[(algebraic_size * offset) + hna])/( CONSTANTS[(constant_size * offset) + kna3]*ALGEBRAIC[(algebraic_size * offset) + h1_i]);
ALGEBRAIC[(algebraic_size * offset) + k4pp_i] =  ALGEBRAIC[(algebraic_size * offset) + h2_i]*CONSTANTS[(constant_size * offset) + wnaca];
ALGEBRAIC[(algebraic_size * offset) + h4_i] = 1.00000+ (STATES[(states_size * offset) + nai]/CONSTANTS[(constant_size * offset) + kna1])*(1.00000+STATES[(states_size * offset) + nai]/CONSTANTS[(constant_size * offset) + kna2]);
ALGEBRAIC[(algebraic_size * offset) + h5_i] = ( STATES[(states_size * offset) + nai]*STATES[(states_size * offset) + nai])/( ALGEBRAIC[(algebraic_size * offset) + h4_i]*CONSTANTS[(constant_size * offset) + kna1]*CONSTANTS[(constant_size * offset) + kna2]);
ALGEBRAIC[(algebraic_size * offset) + k7_i] =  ALGEBRAIC[(algebraic_size * offset) + h5_i]*ALGEBRAIC[(algebraic_size * offset) + h2_i]*CONSTANTS[(constant_size * offset) + wna];
ALGEBRAIC[(algebraic_size * offset) + k8_i] =  ALGEBRAIC[(algebraic_size * offset) + h8_i]*CONSTANTS[(constant_size * offset) + h11_i]*CONSTANTS[(constant_size * offset) + wna];
ALGEBRAIC[(algebraic_size * offset) + h9_i] = 1.00000/ALGEBRAIC[(algebraic_size * offset) + h7_i];
ALGEBRAIC[(algebraic_size * offset) + k3p_i] =  ALGEBRAIC[(algebraic_size * offset) + h9_i]*CONSTANTS[(constant_size * offset) + wca];
ALGEBRAIC[(algebraic_size * offset) + k3_i] = ALGEBRAIC[(algebraic_size * offset) + k3p_i]+ALGEBRAIC[(algebraic_size * offset) + k3pp_i];
ALGEBRAIC[(algebraic_size * offset) + hca] = exp(( CONSTANTS[(constant_size * offset) + qca]*STATES[(states_size * offset) + V]*CONSTANTS[(constant_size * offset) + F])/( CONSTANTS[(constant_size * offset) + R]*CONSTANTS[(constant_size * offset) + T]));
ALGEBRAIC[(algebraic_size * offset) + h3_i] = 1.00000/ALGEBRAIC[(algebraic_size * offset) + h1_i];
ALGEBRAIC[(algebraic_size * offset) + k4p_i] = ( ALGEBRAIC[(algebraic_size * offset) + h3_i]*CONSTANTS[(constant_size * offset) + wca])/ALGEBRAIC[(algebraic_size * offset) + hca];
ALGEBRAIC[(algebraic_size * offset) + k4_i] = ALGEBRAIC[(algebraic_size * offset) + k4p_i]+ALGEBRAIC[(algebraic_size * offset) + k4pp_i];
ALGEBRAIC[(algebraic_size * offset) + h6_i] = 1.00000/ALGEBRAIC[(algebraic_size * offset) + h4_i];
ALGEBRAIC[(algebraic_size * offset) + k6_i] =  ALGEBRAIC[(algebraic_size * offset) + h6_i]*STATES[(states_size * offset) + cai]*CONSTANTS[(constant_size * offset) + kcaon];
ALGEBRAIC[(algebraic_size * offset) + x1_i] =  CONSTANTS[(constant_size * offset) + k2_i]*ALGEBRAIC[(algebraic_size * offset) + k4_i]*(ALGEBRAIC[(algebraic_size * offset) + k7_i]+ALGEBRAIC[(algebraic_size * offset) + k6_i])+ CONSTANTS[(constant_size * offset) + k5_i]*ALGEBRAIC[(algebraic_size * offset) + k7_i]*(CONSTANTS[(constant_size * offset) + k2_i]+ALGEBRAIC[(algebraic_size * offset) + k3_i]);
ALGEBRAIC[(algebraic_size * offset) + x2_i] =  CONSTANTS[(constant_size * offset) + k1_i]*ALGEBRAIC[(algebraic_size * offset) + k7_i]*(ALGEBRAIC[(algebraic_size * offset) + k4_i]+CONSTANTS[(constant_size * offset) + k5_i])+ ALGEBRAIC[(algebraic_size * offset) + k4_i]*ALGEBRAIC[(algebraic_size * offset) + k6_i]*(CONSTANTS[(constant_size * offset) + k1_i]+ALGEBRAIC[(algebraic_size * offset) + k8_i]);
ALGEBRAIC[(algebraic_size * offset) + x3_i] =  CONSTANTS[(constant_size * offset) + k1_i]*ALGEBRAIC[(algebraic_size * offset) + k3_i]*(ALGEBRAIC[(algebraic_size * offset) + k7_i]+ALGEBRAIC[(algebraic_size * offset) + k6_i])+ ALGEBRAIC[(algebraic_size * offset) + k8_i]*ALGEBRAIC[(algebraic_size * offset) + k6_i]*(CONSTANTS[(constant_size * offset) + k2_i]+ALGEBRAIC[(algebraic_size * offset) + k3_i]);
ALGEBRAIC[(algebraic_size * offset) + x4_i] =  CONSTANTS[(constant_size * offset) + k2_i]*ALGEBRAIC[(algebraic_size * offset) + k8_i]*(ALGEBRAIC[(algebraic_size * offset) + k4_i]+CONSTANTS[(constant_size * offset) + k5_i])+ ALGEBRAIC[(algebraic_size * offset) + k3_i]*CONSTANTS[(constant_size * offset) + k5_i]*(CONSTANTS[(constant_size * offset) + k1_i]+ALGEBRAIC[(algebraic_size * offset) + k8_i]);
ALGEBRAIC[(algebraic_size * offset) + E1_i] = ALGEBRAIC[(algebraic_size * offset) + x1_i]/(ALGEBRAIC[(algebraic_size * offset) + x1_i]+ALGEBRAIC[(algebraic_size * offset) + x2_i]+ALGEBRAIC[(algebraic_size * offset) + x3_i]+ALGEBRAIC[(algebraic_size * offset) + x4_i]);
ALGEBRAIC[(algebraic_size * offset) + E2_i] = ALGEBRAIC[(algebraic_size * offset) + x2_i]/(ALGEBRAIC[(algebraic_size * offset) + x1_i]+ALGEBRAIC[(algebraic_size * offset) + x2_i]+ALGEBRAIC[(algebraic_size * offset) + x3_i]+ALGEBRAIC[(algebraic_size * offset) + x4_i]);
ALGEBRAIC[(algebraic_size * offset) + E3_i] = ALGEBRAIC[(algebraic_size * offset) + x3_i]/(ALGEBRAIC[(algebraic_size * offset) + x1_i]+ALGEBRAIC[(algebraic_size * offset) + x2_i]+ALGEBRAIC[(algebraic_size * offset) + x3_i]+ALGEBRAIC[(algebraic_size * offset) + x4_i]);
ALGEBRAIC[(algebraic_size * offset) + E4_i] = ALGEBRAIC[(algebraic_size * offset) + x4_i]/(ALGEBRAIC[(algebraic_size * offset) + x1_i]+ALGEBRAIC[(algebraic_size * offset) + x2_i]+ALGEBRAIC[(algebraic_size * offset) + x3_i]+ALGEBRAIC[(algebraic_size * offset) + x4_i]);
ALGEBRAIC[(algebraic_size * offset) + JncxNa_i] = ( 3.00000*( ALGEBRAIC[(algebraic_size * offset) + E4_i]*ALGEBRAIC[(algebraic_size * offset) + k7_i] -  ALGEBRAIC[(algebraic_size * offset) + E1_i]*ALGEBRAIC[(algebraic_size * offset) + k8_i])+ ALGEBRAIC[(algebraic_size * offset) + E3_i]*ALGEBRAIC[(algebraic_size * offset) + k4pp_i]) -  ALGEBRAIC[(algebraic_size * offset) + E2_i]*ALGEBRAIC[(algebraic_size * offset) + k3pp_i];
ALGEBRAIC[(algebraic_size * offset) + JncxCa_i] =  ALGEBRAIC[(algebraic_size * offset) + E2_i]*CONSTANTS[(constant_size * offset) + k2_i] -  ALGEBRAIC[(algebraic_size * offset) + E1_i]*CONSTANTS[(constant_size * offset) + k1_i];
ALGEBRAIC[(algebraic_size * offset) + INaCa_i] =  0.800000*CONSTANTS[(constant_size * offset) + Gncx]*ALGEBRAIC[(algebraic_size * offset) + allo_i]*( CONSTANTS[(constant_size * offset) + zna]*ALGEBRAIC[(algebraic_size * offset) + JncxNa_i]+ CONSTANTS[(constant_size * offset) + zca]*ALGEBRAIC[(algebraic_size * offset) + JncxCa_i]);
ALGEBRAIC[(algebraic_size * offset) + A_Nab] = ( CONSTANTS[(constant_size * offset) + PNab]*CONSTANTS[(constant_size * offset) + ffrt]*( STATES[(states_size * offset) + nai]*exp(ALGEBRAIC[(algebraic_size * offset) + vfrt]) - CONSTANTS[(constant_size * offset) + nao]))/CONSTANTS[(constant_size * offset) + B_Nab];
ALGEBRAIC[(algebraic_size * offset) + U_Nab] =  CONSTANTS[(constant_size * offset) + B_Nab]*(STATES[(states_size * offset) + V] - CONSTANTS[(constant_size * offset) + v0_Nab]);
ALGEBRAIC[(algebraic_size * offset) + INab] = (- 1.00000e-07<=ALGEBRAIC[(algebraic_size * offset) + U_Nab]&&ALGEBRAIC[(algebraic_size * offset) + U_Nab]<=1.00000e-07 ?  ALGEBRAIC[(algebraic_size * offset) + A_Nab]*(1.00000 -  0.500000*ALGEBRAIC[(algebraic_size * offset) + U_Nab]) : ( ALGEBRAIC[(algebraic_size * offset) + A_Nab]*ALGEBRAIC[(algebraic_size * offset) + U_Nab])/(exp(ALGEBRAIC[(algebraic_size * offset) + U_Nab]) - 1.00000));
ALGEBRAIC[(algebraic_size * offset) + JdiffNa] = (STATES[(states_size * offset) + nass] - STATES[(states_size * offset) + nai])/2.00000;
ALGEBRAIC[(algebraic_size * offset) + A_2] = ( 0.750000*CONSTANTS[(constant_size * offset) + ffrt]*( STATES[(states_size * offset) + nass]*exp(ALGEBRAIC[(algebraic_size * offset) + vfrt]) - CONSTANTS[(constant_size * offset) + nao]))/CONSTANTS[(constant_size * offset) + B_2];
ALGEBRAIC[(algebraic_size * offset) + U_2] =  CONSTANTS[(constant_size * offset) + B_2]*(STATES[(states_size * offset) + V] - CONSTANTS[(constant_size * offset) + v0_CaL]);
ALGEBRAIC[(algebraic_size * offset) + PhiCaNa] = (- 1.00000e-07<=ALGEBRAIC[(algebraic_size * offset) + U_2]&&ALGEBRAIC[(algebraic_size * offset) + U_2]<=1.00000e-07 ?  ALGEBRAIC[(algebraic_size * offset) + A_2]*(1.00000 -  0.500000*ALGEBRAIC[(algebraic_size * offset) + U_2]) : ( ALGEBRAIC[(algebraic_size * offset) + A_2]*ALGEBRAIC[(algebraic_size * offset) + U_2])/(exp(ALGEBRAIC[(algebraic_size * offset) + U_2]) - 1.00000));
ALGEBRAIC[(algebraic_size * offset) + ICaNa] =  (1.00000 - ALGEBRAIC[(algebraic_size * offset) + fICaLp])*CONSTANTS[(constant_size * offset) + PCaNa]*ALGEBRAIC[(algebraic_size * offset) + PhiCaNa]*STATES[(states_size * offset) + d]*( ALGEBRAIC[(algebraic_size * offset) + f]*(1.00000 - STATES[(states_size * offset) + nca])+ STATES[(states_size * offset) + jca]*ALGEBRAIC[(algebraic_size * offset) + fca]*STATES[(states_size * offset) + nca])+ ALGEBRAIC[(algebraic_size * offset) + fICaLp]*CONSTANTS[(constant_size * offset) + PCaNap]*ALGEBRAIC[(algebraic_size * offset) + PhiCaNa]*STATES[(states_size * offset) + d]*( ALGEBRAIC[(algebraic_size * offset) + fp]*(1.00000 - STATES[(states_size * offset) + nca])+ STATES[(states_size * offset) + jca]*ALGEBRAIC[(algebraic_size * offset) + fcap]*STATES[(states_size * offset) + nca]);
ALGEBRAIC[(algebraic_size * offset) + allo_ss] = 1.00000/(1.00000+pow(CONSTANTS[(constant_size * offset) + KmCaAct]/STATES[(states_size * offset) + cass], 2.00000));
ALGEBRAIC[(algebraic_size * offset) + h7_ss] = 1.00000+ (CONSTANTS[(constant_size * offset) + nao]/CONSTANTS[(constant_size * offset) + kna3])*(1.00000+1.00000/ALGEBRAIC[(algebraic_size * offset) + hna]);
ALGEBRAIC[(algebraic_size * offset) + h8_ss] = CONSTANTS[(constant_size * offset) + nao]/( CONSTANTS[(constant_size * offset) + kna3]*ALGEBRAIC[(algebraic_size * offset) + hna]*ALGEBRAIC[(algebraic_size * offset) + h7_ss]);
ALGEBRAIC[(algebraic_size * offset) + k3pp_ss] =  ALGEBRAIC[(algebraic_size * offset) + h8_ss]*CONSTANTS[(constant_size * offset) + wnaca];
ALGEBRAIC[(algebraic_size * offset) + h1_ss] = 1.00000+ (STATES[(states_size * offset) + nass]/CONSTANTS[(constant_size * offset) + kna3])*(1.00000+ALGEBRAIC[(algebraic_size * offset) + hna]);
ALGEBRAIC[(algebraic_size * offset) + h2_ss] = ( STATES[(states_size * offset) + nass]*ALGEBRAIC[(algebraic_size * offset) + hna])/( CONSTANTS[(constant_size * offset) + kna3]*ALGEBRAIC[(algebraic_size * offset) + h1_ss]);
ALGEBRAIC[(algebraic_size * offset) + k4pp_ss] =  ALGEBRAIC[(algebraic_size * offset) + h2_ss]*CONSTANTS[(constant_size * offset) + wnaca];
ALGEBRAIC[(algebraic_size * offset) + h4_ss] = 1.00000+ (STATES[(states_size * offset) + nass]/CONSTANTS[(constant_size * offset) + kna1])*(1.00000+STATES[(states_size * offset) + nass]/CONSTANTS[(constant_size * offset) + kna2]);
ALGEBRAIC[(algebraic_size * offset) + h5_ss] = ( STATES[(states_size * offset) + nass]*STATES[(states_size * offset) + nass])/( ALGEBRAIC[(algebraic_size * offset) + h4_ss]*CONSTANTS[(constant_size * offset) + kna1]*CONSTANTS[(constant_size * offset) + kna2]);
ALGEBRAIC[(algebraic_size * offset) + k7_ss] =  ALGEBRAIC[(algebraic_size * offset) + h5_ss]*ALGEBRAIC[(algebraic_size * offset) + h2_ss]*CONSTANTS[(constant_size * offset) + wna];
ALGEBRAIC[(algebraic_size * offset) + k8_ss] =  ALGEBRAIC[(algebraic_size * offset) + h8_ss]*CONSTANTS[(constant_size * offset) + h11_ss]*CONSTANTS[(constant_size * offset) + wna];
ALGEBRAIC[(algebraic_size * offset) + h9_ss] = 1.00000/ALGEBRAIC[(algebraic_size * offset) + h7_ss];
ALGEBRAIC[(algebraic_size * offset) + k3p_ss] =  ALGEBRAIC[(algebraic_size * offset) + h9_ss]*CONSTANTS[(constant_size * offset) + wca];
ALGEBRAIC[(algebraic_size * offset) + k3_ss] = ALGEBRAIC[(algebraic_size * offset) + k3p_ss]+ALGEBRAIC[(algebraic_size * offset) + k3pp_ss];
ALGEBRAIC[(algebraic_size * offset) + h3_ss] = 1.00000/ALGEBRAIC[(algebraic_size * offset) + h1_ss];
ALGEBRAIC[(algebraic_size * offset) + k4p_ss] = ( ALGEBRAIC[(algebraic_size * offset) + h3_ss]*CONSTANTS[(constant_size * offset) + wca])/ALGEBRAIC[(algebraic_size * offset) + hca];
ALGEBRAIC[(algebraic_size * offset) + k4_ss] = ALGEBRAIC[(algebraic_size * offset) + k4p_ss]+ALGEBRAIC[(algebraic_size * offset) + k4pp_ss];
ALGEBRAIC[(algebraic_size * offset) + h6_ss] = 1.00000/ALGEBRAIC[(algebraic_size * offset) + h4_ss];
ALGEBRAIC[(algebraic_size * offset) + k6_ss] =  ALGEBRAIC[(algebraic_size * offset) + h6_ss]*STATES[(states_size * offset) + cass]*CONSTANTS[(constant_size * offset) + kcaon];
ALGEBRAIC[(algebraic_size * offset) + x1_ss] =  CONSTANTS[(constant_size * offset) + k2_ss]*ALGEBRAIC[(algebraic_size * offset) + k4_ss]*(ALGEBRAIC[(algebraic_size * offset) + k7_ss]+ALGEBRAIC[(algebraic_size * offset) + k6_ss])+ CONSTANTS[(constant_size * offset) + k5_ss]*ALGEBRAIC[(algebraic_size * offset) + k7_ss]*(CONSTANTS[(constant_size * offset) + k2_ss]+ALGEBRAIC[(algebraic_size * offset) + k3_ss]);
ALGEBRAIC[(algebraic_size * offset) + x2_ss] =  CONSTANTS[(constant_size * offset) + k1_ss]*ALGEBRAIC[(algebraic_size * offset) + k7_ss]*(ALGEBRAIC[(algebraic_size * offset) + k4_ss]+CONSTANTS[(constant_size * offset) + k5_ss])+ ALGEBRAIC[(algebraic_size * offset) + k4_ss]*ALGEBRAIC[(algebraic_size * offset) + k6_ss]*(CONSTANTS[(constant_size * offset) + k1_ss]+ALGEBRAIC[(algebraic_size * offset) + k8_ss]);
ALGEBRAIC[(algebraic_size * offset) + x3_ss] =  CONSTANTS[(constant_size * offset) + k1_ss]*ALGEBRAIC[(algebraic_size * offset) + k3_ss]*(ALGEBRAIC[(algebraic_size * offset) + k7_ss]+ALGEBRAIC[(algebraic_size * offset) + k6_ss])+ ALGEBRAIC[(algebraic_size * offset) + k8_ss]*ALGEBRAIC[(algebraic_size * offset) + k6_ss]*(CONSTANTS[(constant_size * offset) + k2_ss]+ALGEBRAIC[(algebraic_size * offset) + k3_ss]);
ALGEBRAIC[(algebraic_size * offset) + x4_ss] =  CONSTANTS[(constant_size * offset) + k2_ss]*ALGEBRAIC[(algebraic_size * offset) + k8_ss]*(ALGEBRAIC[(algebraic_size * offset) + k4_ss]+CONSTANTS[(constant_size * offset) + k5_ss])+ ALGEBRAIC[(algebraic_size * offset) + k3_ss]*CONSTANTS[(constant_size * offset) + k5_ss]*(CONSTANTS[(constant_size * offset) + k1_ss]+ALGEBRAIC[(algebraic_size * offset) + k8_ss]);
ALGEBRAIC[(algebraic_size * offset) + E1_ss] = ALGEBRAIC[(algebraic_size * offset) + x1_ss]/(ALGEBRAIC[(algebraic_size * offset) + x1_ss]+ALGEBRAIC[(algebraic_size * offset) + x2_ss]+ALGEBRAIC[(algebraic_size * offset) + x3_ss]+ALGEBRAIC[(algebraic_size * offset) + x4_ss]);
ALGEBRAIC[(algebraic_size * offset) + E2_ss] = ALGEBRAIC[(algebraic_size * offset) + x2_ss]/(ALGEBRAIC[(algebraic_size * offset) + x1_ss]+ALGEBRAIC[(algebraic_size * offset) + x2_ss]+ALGEBRAIC[(algebraic_size * offset) + x3_ss]+ALGEBRAIC[(algebraic_size * offset) + x4_ss]);
ALGEBRAIC[(algebraic_size * offset) + E3_ss] = ALGEBRAIC[(algebraic_size * offset) + x3_ss]/(ALGEBRAIC[(algebraic_size * offset) + x1_ss]+ALGEBRAIC[(algebraic_size * offset) + x2_ss]+ALGEBRAIC[(algebraic_size * offset) + x3_ss]+ALGEBRAIC[(algebraic_size * offset) + x4_ss]);
ALGEBRAIC[(algebraic_size * offset) + E4_ss] = ALGEBRAIC[(algebraic_size * offset) + x4_ss]/(ALGEBRAIC[(algebraic_size * offset) + x1_ss]+ALGEBRAIC[(algebraic_size * offset) + x2_ss]+ALGEBRAIC[(algebraic_size * offset) + x3_ss]+ALGEBRAIC[(algebraic_size * offset) + x4_ss]);
ALGEBRAIC[(algebraic_size * offset) + JncxNa_ss] = ( 3.00000*( ALGEBRAIC[(algebraic_size * offset) + E4_ss]*ALGEBRAIC[(algebraic_size * offset) + k7_ss] -  ALGEBRAIC[(algebraic_size * offset) + E1_ss]*ALGEBRAIC[(algebraic_size * offset) + k8_ss])+ ALGEBRAIC[(algebraic_size * offset) + E3_ss]*ALGEBRAIC[(algebraic_size * offset) + k4pp_ss]) -  ALGEBRAIC[(algebraic_size * offset) + E2_ss]*ALGEBRAIC[(algebraic_size * offset) + k3pp_ss];
ALGEBRAIC[(algebraic_size * offset) + JncxCa_ss] =  ALGEBRAIC[(algebraic_size * offset) + E2_ss]*CONSTANTS[(constant_size * offset) + k2_ss] -  ALGEBRAIC[(algebraic_size * offset) + E1_ss]*CONSTANTS[(constant_size * offset) + k1_ss];
ALGEBRAIC[(algebraic_size * offset) + INaCa_ss] =  0.200000*CONSTANTS[(constant_size * offset) + Gncx]*ALGEBRAIC[(algebraic_size * offset) + allo_ss]*( CONSTANTS[(constant_size * offset) + zna]*ALGEBRAIC[(algebraic_size * offset) + JncxNa_ss]+ CONSTANTS[(constant_size * offset) + zca]*ALGEBRAIC[(algebraic_size * offset) + JncxCa_ss]);
ALGEBRAIC[(algebraic_size * offset) + IpCa] = ( CONSTANTS[(constant_size * offset) + GpCa]*STATES[(states_size * offset) + cai])/(CONSTANTS[(constant_size * offset) + KmCap]+STATES[(states_size * offset) + cai]);
ALGEBRAIC[(algebraic_size * offset) + A_Cab] = ( CONSTANTS[(constant_size * offset) + PCab]*4.00000*CONSTANTS[(constant_size * offset) + ffrt]*( STATES[(states_size * offset) + cai]*exp( 2.00000*ALGEBRAIC[(algebraic_size * offset) + vfrt]) -  0.341000*CONSTANTS[(constant_size * offset) + cao]))/CONSTANTS[(constant_size * offset) + B_Cab];
ALGEBRAIC[(algebraic_size * offset) + U_Cab] =  CONSTANTS[(constant_size * offset) + B_Cab]*(STATES[(states_size * offset) + V] - CONSTANTS[(constant_size * offset) + v0_Cab]);
ALGEBRAIC[(algebraic_size * offset) + ICab] = (- 1.00000e-07<=ALGEBRAIC[(algebraic_size * offset) + U_Cab]&&ALGEBRAIC[(algebraic_size * offset) + U_Cab]<=1.00000e-07 ?  ALGEBRAIC[(algebraic_size * offset) + A_Cab]*(1.00000 -  0.500000*ALGEBRAIC[(algebraic_size * offset) + U_Cab]) : ( ALGEBRAIC[(algebraic_size * offset) + A_Cab]*ALGEBRAIC[(algebraic_size * offset) + U_Cab])/(exp(ALGEBRAIC[(algebraic_size * offset) + U_Cab]) - 1.00000));
ALGEBRAIC[(algebraic_size * offset) + Jdiff] = (STATES[(states_size * offset) + cass] - STATES[(states_size * offset) + cai])/0.200000;
ALGEBRAIC[(algebraic_size * offset) + fJrelp] = 1.00000/(1.00000+CONSTANTS[(constant_size * offset) + KmCaMK]/ALGEBRAIC[(algebraic_size * offset) + CaMKa]);
ALGEBRAIC[(algebraic_size * offset) + Jrel] =  CONSTANTS[(constant_size * offset) + Jrel_scaling_factor]*( (1.00000 - ALGEBRAIC[(algebraic_size * offset) + fJrelp])*STATES[(states_size * offset) + Jrelnp]+ ALGEBRAIC[(algebraic_size * offset) + fJrelp]*STATES[(states_size * offset) + Jrelp]);
ALGEBRAIC[(algebraic_size * offset) + Bcass] = 1.00000/(1.00000+( CONSTANTS[(constant_size * offset) + BSRmax]*CONSTANTS[(constant_size * offset) + KmBSR])/pow(CONSTANTS[(constant_size * offset) + KmBSR]+STATES[(states_size * offset) + cass], 2.00000)+( CONSTANTS[(constant_size * offset) + BSLmax]*CONSTANTS[(constant_size * offset) + KmBSL])/pow(CONSTANTS[(constant_size * offset) + KmBSL]+STATES[(states_size * offset) + cass], 2.00000));
ALGEBRAIC[(algebraic_size * offset) + Jupnp] = ( CONSTANTS[(constant_size * offset) + upScale]*0.00437500*STATES[(states_size * offset) + cai])/(STATES[(states_size * offset) + cai]+0.000920000);
ALGEBRAIC[(algebraic_size * offset) + Jupp] = ( CONSTANTS[(constant_size * offset) + upScale]*2.75000*0.00437500*STATES[(states_size * offset) + cai])/((STATES[(states_size * offset) + cai]+0.000920000) - 0.000170000);
ALGEBRAIC[(algebraic_size * offset) + fJupp] = 1.00000/(1.00000+CONSTANTS[(constant_size * offset) + KmCaMK]/ALGEBRAIC[(algebraic_size * offset) + CaMKa]);
ALGEBRAIC[(algebraic_size * offset) + Jleak] = ( 0.00393750*STATES[(states_size * offset) + cansr])/15.0000;
ALGEBRAIC[(algebraic_size * offset) + Jup] =  CONSTANTS[(constant_size * offset) + Jup_b]*(( (1.00000 - ALGEBRAIC[(algebraic_size * offset) + fJupp])*ALGEBRAIC[(algebraic_size * offset) + Jupnp]+ ALGEBRAIC[(algebraic_size * offset) + fJupp]*ALGEBRAIC[(algebraic_size * offset) + Jupp]) - ALGEBRAIC[(algebraic_size * offset) + Jleak]);
ALGEBRAIC[(algebraic_size * offset) + Bcai] = 1.00000/(1.00000+( CONSTANTS[(constant_size * offset) + cmdnmax]*CONSTANTS[(constant_size * offset) + kmcmdn])/pow(CONSTANTS[(constant_size * offset) + kmcmdn]+STATES[(states_size * offset) + cai], 2.00000)+( CONSTANTS[(constant_size * offset) + trpnmax]*CONSTANTS[(constant_size * offset) + kmtrpn])/pow(CONSTANTS[(constant_size * offset) + kmtrpn]+STATES[(states_size * offset) + cai], 2.00000));
ALGEBRAIC[(algebraic_size * offset) + Jtr] = (STATES[(states_size * offset) + cansr] - STATES[(states_size * offset) + cajsr])/100.000;
ALGEBRAIC[(algebraic_size * offset) + Bcajsr] = 1.00000/(1.00000+( CONSTANTS[(constant_size * offset) + csqnmax]*CONSTANTS[(constant_size * offset) + kmcsqn])/pow(CONSTANTS[(constant_size * offset) + kmcsqn]+STATES[(states_size * offset) + cajsr], 2.00000));

//RATES[D] = CONSTANTS[cnc];
RATES[(rates_size * offset) + D] = 0.;
RATES[(rates_size * offset) + IC1] = (- ( CONSTANTS[(constant_size * offset) +  A11]*exp( CONSTANTS[(constant_size * offset) +  B11]*STATES[(states_size * offset) + V])*STATES[(states_size * offset) + IC1]*exp(( (CONSTANTS[(constant_size * offset) +  Temp] - 20.0000)*log(CONSTANTS[(constant_size * offset) +  q11]))/10.0000) -  CONSTANTS[(constant_size * offset) +  A21]*exp( CONSTANTS[(constant_size * offset) +  B21]*STATES[(states_size * offset) + V])*STATES[(states_size * offset) + IC2]*exp(( (CONSTANTS[(constant_size * offset) +  Temp] - 20.0000)*log(CONSTANTS[(constant_size * offset) +  q21]))/10.0000))+ CONSTANTS[(constant_size * offset) +  A51]*exp( CONSTANTS[(constant_size * offset) +  B51]*STATES[(states_size * offset) + V])*STATES[(states_size * offset) + C1]*exp(( (CONSTANTS[(constant_size * offset) +  Temp] - 20.0000)*log(CONSTANTS[(constant_size * offset) +  q51]))/10.0000)) -  CONSTANTS[(constant_size * offset) +  A61]*exp( CONSTANTS[(constant_size * offset) +  B61]*STATES[(states_size * offset) + V])*STATES[(states_size * offset) + IC1]*exp(( (CONSTANTS[(constant_size * offset) +  Temp] - 20.0000)*log(CONSTANTS[(constant_size * offset) +  q61]))/10.0000);
RATES[(rates_size * offset) + IC2] = ((( CONSTANTS[(constant_size * offset) +  A11]*exp( CONSTANTS[(constant_size * offset) +  B11]*STATES[(states_size * offset) + V])*STATES[(states_size * offset) + IC1]*exp(( (CONSTANTS[(constant_size * offset) +  Temp] - 20.0000)*log(CONSTANTS[(constant_size * offset) +  q11]))/10.0000) -  CONSTANTS[(constant_size * offset) +  A21]*exp( CONSTANTS[(constant_size * offset) +  B21]*STATES[(states_size * offset) + V])*STATES[(states_size * offset) + IC2]*exp(( (CONSTANTS[(constant_size * offset) +  Temp] - 20.0000)*log(CONSTANTS[(constant_size * offset) +  q21]))/10.0000)) - ( CONSTANTS[(constant_size * offset) +  A3]*exp( CONSTANTS[(constant_size * offset) +  B3]*STATES[(states_size * offset) + V])*STATES[(states_size * offset) + IC2]*exp(( (CONSTANTS[(constant_size * offset) +  Temp] - 20.0000)*log(CONSTANTS[(constant_size * offset) +  q3]))/10.0000) -  CONSTANTS[(constant_size * offset) +  A4]*exp( CONSTANTS[(constant_size * offset) +  B4]*STATES[(states_size * offset) + V])*STATES[(states_size * offset) + IO]*exp(( (CONSTANTS[(constant_size * offset) +  Temp] - 20.0000)*log(CONSTANTS[(constant_size * offset) +  q4]))/10.0000)))+ CONSTANTS[(constant_size * offset) +  A52]*exp( CONSTANTS[(constant_size * offset) +  B52]*STATES[(states_size * offset) + V])*STATES[(states_size * offset) + C2]*exp(( (CONSTANTS[(constant_size * offset) +  Temp] - 20.0000)*log(CONSTANTS[(constant_size * offset) +  q52]))/10.0000)) -  CONSTANTS[(constant_size * offset) +  A62]*exp( CONSTANTS[(constant_size * offset) +  B62]*STATES[(states_size * offset) + V])*STATES[(states_size * offset) + IC2]*exp(( (CONSTANTS[(constant_size * offset) +  Temp] - 20.0000)*log(CONSTANTS[(constant_size * offset) +  q62]))/10.0000);
RATES[(rates_size * offset) + C1] = - ( CONSTANTS[(constant_size * offset) +  A1]*exp( CONSTANTS[(constant_size * offset) +  B1]*STATES[(states_size * offset) + V])*STATES[(states_size * offset) + C1]*exp(( (CONSTANTS[(constant_size * offset) +  Temp] - 20.0000)*log(CONSTANTS[(constant_size * offset) +  q1]))/10.0000) -  CONSTANTS[(constant_size * offset) +  A2]*exp( CONSTANTS[(constant_size * offset) +  B2]*STATES[(states_size * offset) + V])*STATES[(states_size * offset) + C2]*exp(( (CONSTANTS[(constant_size * offset) +  Temp] - 20.0000)*log(CONSTANTS[(constant_size * offset) +  q2]))/10.0000)) - ( CONSTANTS[(constant_size * offset) +  A51]*exp( CONSTANTS[(constant_size * offset) +  B51]*STATES[(states_size * offset) + V])*STATES[(states_size * offset) + C1]*exp(( (CONSTANTS[(constant_size * offset) +  Temp] - 20.0000)*log(CONSTANTS[(constant_size * offset) +  q51]))/10.0000) -  CONSTANTS[(constant_size * offset) +  A61]*exp( CONSTANTS[(constant_size * offset) +  B61]*STATES[(states_size * offset) + V])*STATES[(states_size * offset) + IC1]*exp(( (CONSTANTS[(constant_size * offset) +  Temp] - 20.0000)*log(CONSTANTS[(constant_size * offset) +  q61]))/10.0000));
RATES[(rates_size * offset) + C2] = (( CONSTANTS[(constant_size * offset) +  A1]*exp( CONSTANTS[(constant_size * offset) +  B1]*STATES[(states_size * offset) + V])*STATES[(states_size * offset) + C1]*exp(( (CONSTANTS[(constant_size * offset) +  Temp] - 20.0000)*log(CONSTANTS[(constant_size * offset) +  q1]))/10.0000) -  CONSTANTS[(constant_size * offset) +  A2]*exp( CONSTANTS[(constant_size * offset) +  B2]*STATES[(states_size * offset) + V])*STATES[(states_size * offset) + C2]*exp(( (CONSTANTS[(constant_size * offset) +  Temp] - 20.0000)*log(CONSTANTS[(constant_size * offset) +  q2]))/10.0000)) - ( CONSTANTS[(constant_size * offset) +  A31]*exp( CONSTANTS[(constant_size * offset) +  B31]*STATES[(states_size * offset) + V])*STATES[(states_size * offset) + C2]*exp(( (CONSTANTS[(constant_size * offset) +  Temp] - 20.0000)*log(CONSTANTS[(constant_size * offset) +  q31]))/10.0000) -  CONSTANTS[(constant_size * offset) +  A41]*exp( CONSTANTS[(constant_size * offset) +  B41]*STATES[(states_size * offset) + V])*STATES[(states_size * offset) + O]*exp(( (CONSTANTS[(constant_size * offset) +  Temp] - 20.0000)*log(CONSTANTS[(constant_size * offset) +  q41]))/10.0000))) - ( CONSTANTS[(constant_size * offset) +  A52]*exp( CONSTANTS[(constant_size * offset) +  B52]*STATES[(states_size * offset) + V])*STATES[(states_size * offset) + C2]*exp(( (CONSTANTS[(constant_size * offset) +  Temp] - 20.0000)*log(CONSTANTS[(constant_size * offset) +  q52]))/10.0000) -  CONSTANTS[(constant_size * offset) +  A62]*exp( CONSTANTS[(constant_size * offset) +  B62]*STATES[(states_size * offset) + V])*STATES[(states_size * offset) + IC2]*exp(( (CONSTANTS[(constant_size * offset) +  Temp] - 20.0000)*log(CONSTANTS[(constant_size * offset) +  q62]))/10.0000));
RATES[(rates_size * offset) + O] = (( CONSTANTS[(constant_size * offset) +  A31]*exp( CONSTANTS[(constant_size * offset) +  B31]*STATES[(states_size * offset) + V])*STATES[(states_size * offset) + C2]*exp(( (CONSTANTS[(constant_size * offset) +  Temp] - 20.0000)*log(CONSTANTS[(constant_size * offset) +  q31]))/10.0000) -  CONSTANTS[(constant_size * offset) +  A41]*exp( CONSTANTS[(constant_size * offset) +  B41]*STATES[(states_size * offset) + V])*STATES[(states_size * offset) + O]*exp(( (CONSTANTS[(constant_size * offset) +  Temp] - 20.0000)*log(CONSTANTS[(constant_size * offset) +  q41]))/10.0000)) - ( CONSTANTS[(constant_size * offset) +  A53]*exp( CONSTANTS[(constant_size * offset) +  B53]*STATES[(states_size * offset) + V])*STATES[(states_size * offset) + O]*exp(( (CONSTANTS[(constant_size * offset) +  Temp] - 20.0000)*log(CONSTANTS[(constant_size * offset) +  q53]))/10.0000) -  CONSTANTS[(constant_size * offset) +  A63]*exp( CONSTANTS[(constant_size * offset) +  B63]*STATES[(states_size * offset) + V])*STATES[(states_size * offset) + IO]*exp(( (CONSTANTS[(constant_size * offset) +  Temp] - 20.0000)*log(CONSTANTS[(constant_size * offset) +  q63]))/10.0000))) - ( (( CONSTANTS[(constant_size * offset) +  Kmax]*CONSTANTS[(constant_size * offset) +  Ku]*pow( STATES[(states_size * offset) + D],CONSTANTS[(constant_size * offset) +  n]))/(pow( STATES[(states_size * offset) + D],CONSTANTS[(constant_size * offset) +  n])+CONSTANTS[(constant_size * offset) +  halfmax]))*STATES[(states_size * offset) + O] -  CONSTANTS[(constant_size * offset) +  Ku]*STATES[(states_size * offset) + Obound]);
RATES[(rates_size * offset) + IO] = ((( CONSTANTS[(constant_size * offset) +  A3]*exp( CONSTANTS[(constant_size * offset) +  B3]*STATES[(states_size * offset) + V])*STATES[(states_size * offset) + IC2]*exp(( (CONSTANTS[(constant_size * offset) +  Temp] - 20.0000)*log(CONSTANTS[(constant_size * offset) +  q3]))/10.0000) -  CONSTANTS[(constant_size * offset) +  A4]*exp( CONSTANTS[(constant_size * offset) +  B4]*STATES[(states_size * offset) + V])*STATES[(states_size * offset) + IO]*exp(( (CONSTANTS[(constant_size * offset) +  Temp] - 20.0000)*log(CONSTANTS[(constant_size * offset) +  q4]))/10.0000))+ CONSTANTS[(constant_size * offset) +  A53]*exp( CONSTANTS[(constant_size * offset) +  B53]*STATES[(states_size * offset) + V])*STATES[(states_size * offset) + O]*exp(( (CONSTANTS[(constant_size * offset) +  Temp] - 20.0000)*log(CONSTANTS[(constant_size * offset) +  q53]))/10.0000)) -  CONSTANTS[(constant_size * offset) +  A63]*exp( CONSTANTS[(constant_size * offset) +  B63]*STATES[(states_size * offset) + V])*STATES[(states_size * offset) + IO]*exp(( (CONSTANTS[(constant_size * offset) +  Temp] - 20.0000)*log(CONSTANTS[(constant_size * offset) +  q63]))/10.0000)) - ( (( CONSTANTS[(constant_size * offset) +  Kmax]*CONSTANTS[(constant_size * offset) +  Ku]*pow( STATES[(states_size * offset) + D],CONSTANTS[(constant_size * offset) +  n]))/(pow( STATES[(states_size * offset) + D],CONSTANTS[(constant_size * offset) +  n])+CONSTANTS[(constant_size * offset) +  halfmax]))*STATES[(states_size * offset) + IO] -  (( CONSTANTS[(constant_size * offset) +  Ku]*CONSTANTS[(constant_size * offset) +  A53]*exp( CONSTANTS[(constant_size * offset) +  B53]*STATES[(states_size * offset) + V])*exp(( (CONSTANTS[(constant_size * offset) +  Temp] - 20.0000)*log(CONSTANTS[(constant_size * offset) +  q53]))/10.0000))/( CONSTANTS[(constant_size * offset) +  A63]*exp( CONSTANTS[(constant_size * offset) +  B63]*STATES[(states_size * offset) + V])*exp(( (CONSTANTS[(constant_size * offset) +  Temp] - 20.0000)*log(CONSTANTS[(constant_size * offset) +  q63]))/10.0000)))*STATES[(states_size * offset) + IObound]);
RATES[(rates_size * offset) + IObound] = (( (( CONSTANTS[(constant_size * offset) +  Kmax]*CONSTANTS[(constant_size * offset) +  Ku]*pow( STATES[(states_size * offset) + D],CONSTANTS[(constant_size * offset) +  n]))/(pow( STATES[(states_size * offset) + D],CONSTANTS[(constant_size * offset) +  n])+CONSTANTS[(constant_size * offset) +  halfmax]))*STATES[(states_size * offset) + IO] -  (( CONSTANTS[(constant_size * offset) +  Ku]*CONSTANTS[(constant_size * offset) +  A53]*exp( CONSTANTS[(constant_size * offset) +  B53]*STATES[(states_size * offset) + V])*exp(( (CONSTANTS[(constant_size * offset) +  Temp] - 20.0000)*log(CONSTANTS[(constant_size * offset) +  q53]))/10.0000))/( CONSTANTS[(constant_size * offset) +  A63]*exp( CONSTANTS[(constant_size * offset) +  B63]*STATES[(states_size * offset) + V])*exp(( (CONSTANTS[(constant_size * offset) +  Temp] - 20.0000)*log(CONSTANTS[(constant_size * offset) +  q63]))/10.0000)))*STATES[(states_size * offset) + IObound])+ (CONSTANTS[(constant_size * offset) +  Kt]/(1.00000+exp(- (STATES[(states_size * offset) + V] - CONSTANTS[(constant_size * offset) +  Vhalf])/6.78900)))*STATES[(states_size * offset) + Cbound]) -  CONSTANTS[(constant_size * offset) +  Kt]*STATES[(states_size * offset) + IObound];
RATES[(rates_size * offset) + Obound] = (( (( CONSTANTS[(constant_size * offset) +  Kmax]*CONSTANTS[(constant_size * offset) +  Ku]*pow( STATES[(states_size * offset) + D],CONSTANTS[(constant_size * offset) +  n]))/(pow( STATES[(states_size * offset) + D],CONSTANTS[(constant_size * offset) +  n])+CONSTANTS[(constant_size * offset) +  halfmax]))*STATES[(states_size * offset) + O] -  CONSTANTS[(constant_size * offset) +  Ku]*STATES[(states_size * offset) + Obound])+ (CONSTANTS[(constant_size * offset) +  Kt]/(1.00000+exp(- (STATES[(states_size * offset) + V] - CONSTANTS[(constant_size * offset) +  Vhalf])/6.78900)))*STATES[(states_size * offset) + Cbound]) -  CONSTANTS[(constant_size * offset) +  Kt]*STATES[(states_size * offset) + Obound];
RATES[(rates_size * offset) + Cbound] = - ( (CONSTANTS[(constant_size * offset) +  Kt]/(1.00000+exp(- (STATES[(states_size * offset) + V] - CONSTANTS[(constant_size * offset) +  Vhalf])/6.78900)))*STATES[(states_size * offset) + Cbound] -  CONSTANTS[(constant_size * offset) +  Kt]*STATES[(states_size * offset) + Obound]) - ( (CONSTANTS[(constant_size * offset) +  Kt]/(1.00000+exp(- (STATES[(states_size * offset) + V] - CONSTANTS[(constant_size * offset) +  Vhalf])/6.78900)))*STATES[(states_size * offset) + Cbound] -  CONSTANTS[(constant_size * offset) +  Kt]*STATES[(states_size * offset) + IObound]);
RATES[(rates_size * offset) + hL] = (ALGEBRAIC[(algebraic_size * offset) + hLss] - STATES[(states_size * offset) + hL])/CONSTANTS[(constant_size * offset) +  thL];
RATES[(rates_size * offset) + hLp] = (ALGEBRAIC[(algebraic_size * offset) + hLssp] - STATES[(states_size * offset) + hLp])/CONSTANTS[(constant_size * offset) +  thLp];
RATES[(rates_size * offset) + m] = (ALGEBRAIC[(algebraic_size * offset) + mss] - STATES[(states_size * offset) + m])/ALGEBRAIC[(algebraic_size * offset) + tm];
RATES[(rates_size * offset) + hf] = (ALGEBRAIC[(algebraic_size * offset) + hss] - STATES[(states_size * offset) + hf])/ALGEBRAIC[(algebraic_size * offset) + thf];
RATES[(rates_size * offset) + hs] = (ALGEBRAIC[(algebraic_size * offset) + hss] - STATES[(states_size * offset) + hs])/ALGEBRAIC[(algebraic_size * offset) + ths];
RATES[(rates_size * offset) + a] = (ALGEBRAIC[(algebraic_size * offset) + ass] - STATES[(states_size * offset) + a])/ALGEBRAIC[(algebraic_size * offset) + ta];
RATES[(rates_size * offset) + d] = (ALGEBRAIC[(algebraic_size * offset) + dss] - STATES[(states_size * offset) + d])/ALGEBRAIC[(algebraic_size * offset) + td];
RATES[(rates_size * offset) + ff] = (ALGEBRAIC[(algebraic_size * offset) + fss] - STATES[(states_size * offset) + ff])/ALGEBRAIC[(algebraic_size * offset) + tff];
RATES[(rates_size * offset) + fs] = (ALGEBRAIC[(algebraic_size * offset) + fss] - STATES[(states_size * offset) + fs])/ALGEBRAIC[(algebraic_size * offset) + tfs];
RATES[(rates_size * offset) + jca] = (ALGEBRAIC[(algebraic_size * offset) + fcass] - STATES[(states_size * offset) + jca])/CONSTANTS[(constant_size * offset) +  tjca];
RATES[(rates_size * offset) + nca] =  ALGEBRAIC[(algebraic_size * offset) + anca]*CONSTANTS[(constant_size * offset) +  k2n] -  STATES[(states_size * offset) + nca]*ALGEBRAIC[(algebraic_size * offset) + km2n];
RATES[(rates_size * offset) + xs1] = (ALGEBRAIC[(algebraic_size * offset) + xs1ss] - STATES[(states_size * offset) + xs1])/ALGEBRAIC[(algebraic_size * offset) + txs1];
RATES[(rates_size * offset) + xk1] = (ALGEBRAIC[(algebraic_size * offset) + xk1ss] - STATES[(states_size * offset) + xk1])/ALGEBRAIC[(algebraic_size * offset) + txk1];
RATES[(rates_size * offset) + CaMKt] =  CONSTANTS[(constant_size * offset) +  aCaMK]*ALGEBRAIC[(algebraic_size * offset) + CaMKb]*(ALGEBRAIC[(algebraic_size * offset) + CaMKb]+STATES[(states_size * offset) + CaMKt]) -  CONSTANTS[(constant_size * offset) +  bCaMK]*STATES[(states_size * offset) + CaMKt];
RATES[(rates_size * offset) + j] = (ALGEBRAIC[(algebraic_size * offset) + jss] - STATES[(states_size * offset) + j])/ALGEBRAIC[(algebraic_size * offset) + tj];
RATES[(rates_size * offset) + ap] = (ALGEBRAIC[(algebraic_size * offset) + assp] - STATES[(states_size * offset) + ap])/ALGEBRAIC[(algebraic_size * offset) + ta];
RATES[(rates_size * offset) + fcaf] = (ALGEBRAIC[(algebraic_size * offset) + fcass] - STATES[(states_size * offset) + fcaf])/ALGEBRAIC[(algebraic_size * offset) + tfcaf];
RATES[(rates_size * offset) + fcas] = (ALGEBRAIC[(algebraic_size * offset) + fcass] - STATES[(states_size * offset) + fcas])/ALGEBRAIC[(algebraic_size * offset) + tfcas];
RATES[(rates_size * offset) + ffp] = (ALGEBRAIC[(algebraic_size * offset) + fss] - STATES[(states_size * offset) + ffp])/ALGEBRAIC[(algebraic_size * offset) + tffp];
RATES[(rates_size * offset) + xs2] = (ALGEBRAIC[(algebraic_size * offset) + xs2ss] - STATES[(states_size * offset) + xs2])/ALGEBRAIC[(algebraic_size * offset) + txs2];
RATES[(rates_size * offset) + hsp] = (ALGEBRAIC[(algebraic_size * offset) + hssp] - STATES[(states_size * offset) + hsp])/ALGEBRAIC[(algebraic_size * offset) + thsp];
RATES[(rates_size * offset) + jp] = (ALGEBRAIC[(algebraic_size * offset) + jss] - STATES[(states_size * offset) + jp])/ALGEBRAIC[(algebraic_size * offset) + tjp];
RATES[(rates_size * offset) + mL] = (ALGEBRAIC[(algebraic_size * offset) + mLss] - STATES[(states_size * offset) + mL])/ALGEBRAIC[(algebraic_size * offset) + tmL];
RATES[(rates_size * offset) + fcafp] = (ALGEBRAIC[(algebraic_size * offset) + fcass] - STATES[(states_size * offset) + fcafp])/ALGEBRAIC[(algebraic_size * offset) + tfcafp];
RATES[(rates_size * offset) + iF] = (ALGEBRAIC[(algebraic_size * offset) + iss] - STATES[(states_size * offset) + iF])/ALGEBRAIC[(algebraic_size * offset) + tiF];
RATES[(rates_size * offset) + iS] = (ALGEBRAIC[(algebraic_size * offset) + iss] - STATES[(states_size * offset) + iS])/ALGEBRAIC[(algebraic_size * offset) + tiS];
RATES[(rates_size * offset) + iFp] = (ALGEBRAIC[(algebraic_size * offset) + iss] - STATES[(states_size * offset) + iFp])/ALGEBRAIC[(algebraic_size * offset) + tiFp];
RATES[(rates_size * offset) + iSp] = (ALGEBRAIC[(algebraic_size * offset) + iss] - STATES[(states_size * offset) + iSp])/ALGEBRAIC[(algebraic_size * offset) + tiSp];
RATES[(rates_size * offset) + Jrelnp] = (ALGEBRAIC[(algebraic_size * offset) + Jrel_inf] - STATES[(states_size * offset) + Jrelnp])/ALGEBRAIC[(algebraic_size * offset) + tau_rel];
RATES[(rates_size * offset) + Jrelp] = (ALGEBRAIC[(algebraic_size * offset) + Jrel_infp] - STATES[(states_size * offset) + Jrelp])/ALGEBRAIC[(algebraic_size * offset) + tau_relp];
RATES[(rates_size * offset) + ki] = ( - ((ALGEBRAIC[(algebraic_size * offset) + Ito]+ALGEBRAIC[(algebraic_size * offset) + IKr]+ALGEBRAIC[(algebraic_size * offset) + IKs]+ALGEBRAIC[(algebraic_size * offset) + IK1]+ALGEBRAIC[(algebraic_size * offset) + IKb]+ALGEBRAIC[(algebraic_size * offset) + Istim]) -  2.00000*ALGEBRAIC[(algebraic_size * offset) + INaK])*CONSTANTS[(constant_size * offset) +  cm]*CONSTANTS[(constant_size * offset) +  Acap])/( CONSTANTS[(constant_size * offset) +  F]*CONSTANTS[(constant_size * offset) +  vmyo])+( ALGEBRAIC[(algebraic_size * offset) + JdiffK]*CONSTANTS[(constant_size * offset) +  vss])/CONSTANTS[(constant_size * offset) +  vmyo];
RATES[(rates_size * offset) + kss] = ( - ALGEBRAIC[(algebraic_size * offset) + ICaK]*CONSTANTS[(constant_size * offset) +  cm]*CONSTANTS[(constant_size * offset) +  Acap])/( CONSTANTS[(constant_size * offset) +  F]*CONSTANTS[(constant_size * offset) +  vss]) - ALGEBRAIC[(algebraic_size * offset) + JdiffK];
RATES[(rates_size * offset) + nai] = ( - (ALGEBRAIC[(algebraic_size * offset) + INa]+ALGEBRAIC[(algebraic_size * offset) + INaL]+ 3.00000*ALGEBRAIC[(algebraic_size * offset) + INaCa_i]+ 3.00000*ALGEBRAIC[(algebraic_size * offset) + INaK]+ALGEBRAIC[(algebraic_size * offset) + INab])*CONSTANTS[(constant_size * offset) +  Acap]*CONSTANTS[(constant_size * offset) +  cm])/( CONSTANTS[(constant_size * offset) +  F]*CONSTANTS[(constant_size * offset) +  vmyo])+( ALGEBRAIC[(algebraic_size * offset) + JdiffNa]*CONSTANTS[(constant_size * offset) +  vss])/CONSTANTS[(constant_size * offset) +  vmyo];
RATES[(rates_size * offset) + nass] = ( - (ALGEBRAIC[(algebraic_size * offset) + ICaNa]+ 3.00000*ALGEBRAIC[(algebraic_size * offset) + INaCa_ss])*CONSTANTS[(constant_size * offset) +  cm]*CONSTANTS[(constant_size * offset) +  Acap])/( CONSTANTS[(constant_size * offset) +  F]*CONSTANTS[(constant_size * offset) +  vss]) - ALGEBRAIC[(algebraic_size * offset) + JdiffNa];
RATES[(rates_size * offset) + V] = - (ALGEBRAIC[(algebraic_size * offset) + INa]+ALGEBRAIC[(algebraic_size * offset) + INaL]+ALGEBRAIC[(algebraic_size * offset) + Ito]+ALGEBRAIC[(algebraic_size * offset) + ICaL]+ALGEBRAIC[(algebraic_size * offset) + ICaNa]+ALGEBRAIC[(algebraic_size * offset) + ICaK]+ALGEBRAIC[(algebraic_size * offset) + IKr]+ALGEBRAIC[(algebraic_size * offset) + IKs]+ALGEBRAIC[(algebraic_size * offset) + IK1]+ALGEBRAIC[(algebraic_size * offset) + INaCa_i]+ALGEBRAIC[(algebraic_size * offset) + INaCa_ss]+ALGEBRAIC[(algebraic_size * offset) + INaK]+ALGEBRAIC[(algebraic_size * offset) + INab]+ALGEBRAIC[(algebraic_size * offset) + IKb]+ALGEBRAIC[(algebraic_size * offset) + IpCa]+ALGEBRAIC[(algebraic_size * offset) + ICab]+ALGEBRAIC[(algebraic_size * offset) + Istim]);
RATES[(rates_size * offset) + cass] =  ALGEBRAIC[(algebraic_size * offset) + Bcass]*((( - (ALGEBRAIC[(algebraic_size * offset) + ICaL] -  2.00000*ALGEBRAIC[(algebraic_size * offset) + INaCa_ss])*CONSTANTS[(constant_size * offset) +  cm]*CONSTANTS[(constant_size * offset) +  Acap])/( 2.00000*CONSTANTS[(constant_size * offset) +  F]*CONSTANTS[(constant_size * offset) +  vss])+( ALGEBRAIC[(algebraic_size * offset) + Jrel]*CONSTANTS[(constant_size * offset) +  vjsr])/CONSTANTS[(constant_size * offset) +  vss]) - ALGEBRAIC[(algebraic_size * offset) + Jdiff]);
RATES[(rates_size * offset) + cai] =  ALGEBRAIC[(algebraic_size * offset) + Bcai]*((( - ((ALGEBRAIC[(algebraic_size * offset) + IpCa]+ALGEBRAIC[(algebraic_size * offset) + ICab]) -  2.00000*ALGEBRAIC[(algebraic_size * offset) + INaCa_i])*CONSTANTS[(constant_size * offset) +  cm]*CONSTANTS[(constant_size * offset) +  Acap])/( 2.00000*CONSTANTS[(constant_size * offset) +  F]*CONSTANTS[(constant_size * offset) +  vmyo]) - ( ALGEBRAIC[(algebraic_size * offset) + Jup]*CONSTANTS[(constant_size * offset) +  vnsr])/CONSTANTS[(constant_size * offset) +  vmyo])+( ALGEBRAIC[(algebraic_size * offset) + Jdiff]*CONSTANTS[(constant_size * offset) +  vss])/CONSTANTS[(constant_size * offset) +  vmyo]);
RATES[(rates_size * offset) + cansr] = ALGEBRAIC[(algebraic_size * offset) + Jup] - ( ALGEBRAIC[(algebraic_size * offset) + Jtr]*CONSTANTS[(constant_size * offset) +  vjsr])/CONSTANTS[(constant_size * offset) +  vnsr];
RATES[(rates_size * offset) + cajsr] =  ALGEBRAIC[(algebraic_size * offset) + Bcajsr]*(ALGEBRAIC[(algebraic_size * offset) + Jtr] - ALGEBRAIC[(algebraic_size * offset) + Jrel]);
}


__device__ void solveAnalytical(double *CONSTANTS, double *STATES, double *ALGEBRAIC, double *RATES, double dt, int offset)
{
////==============
////Exact solution
////==============
////INa
  STATES[(states_size * offset) + m] = ALGEBRAIC[(algebraic_size * offset) + mss] - (ALGEBRAIC[(algebraic_size * offset) + mss] - STATES[(states_size * offset) + m]) * exp(-dt / ALGEBRAIC[(algebraic_size * offset) + tm]);
  STATES[(states_size * offset) + hf] = ALGEBRAIC[(algebraic_size * offset) + hss] - (ALGEBRAIC[(algebraic_size * offset) + hss] - STATES[(states_size * offset) + hf]) * exp(-dt / ALGEBRAIC[(algebraic_size * offset) + thf]);
  STATES[(states_size * offset) + hs] = ALGEBRAIC[(algebraic_size * offset) + hss] - (ALGEBRAIC[(algebraic_size * offset) + hss] - STATES[(states_size * offset) + hs]) * exp(-dt / ALGEBRAIC[(algebraic_size * offset) + ths]);
  STATES[(states_size * offset) + j] = ALGEBRAIC[(algebraic_size * offset) + jss] - (ALGEBRAIC[(algebraic_size * offset) + jss] - STATES[(states_size * offset) + j]) * exp(-dt / ALGEBRAIC[(algebraic_size * offset) + tj]);
  STATES[(states_size * offset) + hsp] = ALGEBRAIC[(algebraic_size * offset) + hssp] - (ALGEBRAIC[(algebraic_size * offset) + hssp] - STATES[(states_size * offset) + hsp]) * exp(-dt / ALGEBRAIC[(algebraic_size * offset) + thsp]);
  STATES[(states_size * offset) + jp] = ALGEBRAIC[(algebraic_size * offset) + jss] - (ALGEBRAIC[(algebraic_size * offset) + jss] - STATES[(states_size * offset) + jp]) * exp(-dt / ALGEBRAIC[(algebraic_size * offset) + tjp]);
  STATES[(states_size * offset) + mL] = ALGEBRAIC[(algebraic_size * offset) + mLss] - (ALGEBRAIC[(algebraic_size * offset) + mLss] - STATES[(states_size * offset) + mL]) * exp(-dt / ALGEBRAIC[(algebraic_size * offset) + tmL]);
  STATES[(states_size * offset) + hL] = ALGEBRAIC[(algebraic_size * offset) + hLss] - (ALGEBRAIC[(algebraic_size * offset) + hLss] - STATES[(states_size * offset) + hL]) * exp(-dt / CONSTANTS[(constant_size * offset) +  thL]);
  STATES[(states_size * offset) + hLp] = ALGEBRAIC[(algebraic_size * offset) + hLssp] - (ALGEBRAIC[(algebraic_size * offset) + hLssp] - STATES[(states_size * offset) + hLp]) * exp(-dt / CONSTANTS[(constant_size * offset) +  thLp]);
////Ito
  STATES[(states_size * offset) + a] = ALGEBRAIC[(algebraic_size * offset) + ass] - (ALGEBRAIC[(algebraic_size * offset) + ass] - STATES[(states_size * offset) + a]) * exp(-dt / ALGEBRAIC[(algebraic_size * offset) + ta]);
  STATES[(states_size * offset) + iF] = ALGEBRAIC[(algebraic_size * offset) + iss] - (ALGEBRAIC[(algebraic_size * offset) + iss] - STATES[(states_size * offset) + iF]) * exp(-dt / ALGEBRAIC[(algebraic_size * offset) + tiF]);
  STATES[(states_size * offset) + iS] = ALGEBRAIC[(algebraic_size * offset) + iss] - (ALGEBRAIC[(algebraic_size * offset) + iss] - STATES[(states_size * offset) + iS]) * exp(-dt / ALGEBRAIC[(algebraic_size * offset) + tiS]);
  STATES[(states_size * offset) + ap] = ALGEBRAIC[(algebraic_size * offset) + assp] - (ALGEBRAIC[(algebraic_size * offset) + assp] - STATES[(states_size * offset) + ap]) * exp(-dt / ALGEBRAIC[(algebraic_size * offset) + ta]);
  STATES[(states_size * offset) + iFp] = ALGEBRAIC[(algebraic_size * offset) + iss] - (ALGEBRAIC[(algebraic_size * offset) + iss] - STATES[(states_size * offset) + iFp]) * exp(-dt / ALGEBRAIC[(algebraic_size * offset) + tiFp]);
  STATES[(states_size * offset) + iSp] = ALGEBRAIC[(algebraic_size * offset) + iss] - (ALGEBRAIC[(algebraic_size * offset) + iss] - STATES[(states_size * offset) + iSp]) * exp(-dt / ALGEBRAIC[(algebraic_size * offset) + tiSp]);
////ICaL
  STATES[(states_size * offset) + d] = ALGEBRAIC[(algebraic_size * offset) + dss] - (ALGEBRAIC[(algebraic_size * offset) + dss] - STATES[(states_size * offset) + d]) * exp(-dt / ALGEBRAIC[(algebraic_size * offset) + td]);
  STATES[(states_size * offset) + ff] = ALGEBRAIC[(algebraic_size * offset) + fss] - (ALGEBRAIC[(algebraic_size * offset) + fss] - STATES[(states_size * offset) + ff]) * exp(-dt / ALGEBRAIC[(algebraic_size * offset) + tff]);
  STATES[(states_size * offset) + fs] = ALGEBRAIC[(algebraic_size * offset) + fss] - (ALGEBRAIC[(algebraic_size * offset) + fss] - STATES[(states_size * offset) + fs]) * exp(-dt / ALGEBRAIC[(algebraic_size * offset) + tfs]);
  STATES[(states_size * offset) + fcaf] = ALGEBRAIC[(algebraic_size * offset) + fcass] - (ALGEBRAIC[(algebraic_size * offset) + fcass] - STATES[(states_size * offset) + fcaf]) * exp(-dt / ALGEBRAIC[(algebraic_size * offset) + tfcaf]);
  STATES[(states_size * offset) + fcas] = ALGEBRAIC[(algebraic_size * offset) + fcass] - (ALGEBRAIC[(algebraic_size * offset) + fcass] - STATES[(states_size * offset) + fcas]) * exp(-dt / ALGEBRAIC[(algebraic_size * offset) + tfcas]);
  STATES[(states_size * offset) + jca] = ALGEBRAIC[(algebraic_size * offset) + fcass] - (ALGEBRAIC[(algebraic_size * offset) + fcass] - STATES[(states_size * offset) + jca]) * exp(- dt / CONSTANTS[(constant_size * offset) +  tjca]);
  STATES[(states_size * offset) + ffp] = ALGEBRAIC[(algebraic_size * offset) + fss] - (ALGEBRAIC[(algebraic_size * offset) + fss] - STATES[(states_size * offset) + ffp]) * exp(-dt / ALGEBRAIC[(algebraic_size * offset) + tffp]);
  STATES[(states_size * offset) + fcafp] = ALGEBRAIC[(algebraic_size * offset) + fcass] - (ALGEBRAIC[(algebraic_size * offset) + fcass] - STATES[(states_size * offset) + fcafp]) * exp(-d / ALGEBRAIC[(algebraic_size * offset) + tfcafp]);
  STATES[(states_size * offset) + nca] = ALGEBRAIC[(algebraic_size * offset) + anca] * CONSTANTS[(constant_size * offset) +  k2n] / ALGEBRAIC[(algebraic_size * offset) + km2n] -
      (ALGEBRAIC[(algebraic_size * offset) + anca] * CONSTANTS[(constant_size * offset) +  k2n] / ALGEBRAIC[(algebraic_size * offset) + km2n] - STATES[(states_size * offset) + nca]) * exp(-ALGEBRAIC[(algebraic_size * offset) + km2n] * dt);
////IKs
  STATES[(states_size * offset) + xs1] = ALGEBRAIC[(algebraic_size * offset) + xs1ss] - (ALGEBRAIC[(algebraic_size * offset) + xs1ss] - STATES[(states_size * offset) + xs1]) * exp(-dt / ALGEBRAIC[(algebraic_size * offset) + txs1]);
  STATES[(states_size * offset) + xs2] = ALGEBRAIC[(algebraic_size * offset) + xs2ss] - (ALGEBRAIC[(algebraic_size * offset) + xs2ss] - STATES[(states_size * offset) + xs2]) * exp(-dt / ALGEBRAIC[(algebraic_size * offset) + txs2]);
////IK1
  STATES[(states_size * offset) + xk1] = ALGEBRAIC[(algebraic_size * offset) + xk1ss] - (ALGEBRAIC[(algebraic_size * offset) + xk1ss] - STATES[(states_size * offset) + xk1]) * exp(-dt / ALGEBRAIC[(algebraic_size * offset) + txk1]);
////RyR receptors
  STATES[(states_size * offset) + Jrelnp] = ALGEBRAIC[(algebraic_size * offset) + Jrel_inf] - (ALGEBRAIC[(algebraic_size * offset) + Jrel_inf] - STATES[(states_size * offset) + Jrelnp]) * exp(-dt / ALGEBRAIC[(algebraic_size * offset) + tau_rel]);
  STATES[(states_size * offset) + Jrelp] = ALGEBRAIC[(algebraic_size * offset) + Jrel_infp] - (ALGEBRAIC[(algebraic_size * offset) + Jrel_infp] - STATES[(states_size * offset) + Jrelp]) * exp(-dt / ALGEBRAIC[(algebraic_size * offset) + tau_relp]);
////=============================
////Approximated solution (Backward Euler)
////=============================
////IKr
  double* coeffs = new double[31];
  coeffs[0] = - CONSTANTS[(constant_size * offset) +  A11]*exp( CONSTANTS[(constant_size * offset) +  B11]*STATES[(states_size * offset) + V])*exp(( (CONSTANTS[(constant_size * offset) +  Temp] - 20.0000)*log(CONSTANTS[(constant_size * offset) +  q11]))/10.0000) - CONSTANTS[(constant_size * offset) +  A61]*exp( CONSTANTS[(constant_size * offset) +  B61]*STATES[(states_size * offset) + V])*exp(( (CONSTANTS[(constant_size * offset) +  Temp] - 20.0000)*log(CONSTANTS[(constant_size * offset) +  q61]))/10.0000);
  coeffs[1] = CONSTANTS[(constant_size * offset) +  A21]*exp( CONSTANTS[(constant_size * offset) +  B21]*STATES[(states_size * offset) + V])*exp(( (CONSTANTS[(constant_size * offset) +  Temp] - 20.0000)*log(CONSTANTS[(constant_size * offset) +  q21]))/10.0000);
  coeffs[2] = CONSTANTS[(constant_size * offset) +  A51]*exp( CONSTANTS[(constant_size * offset) +  B51]*STATES[(states_size * offset) + V])*exp(( (CONSTANTS[(constant_size * offset) +  Temp] - 20.0000)*log(CONSTANTS[(constant_size * offset) +  q51]))/10.0000);

  coeffs[3] = CONSTANTS[(constant_size * offset) +  A11]*exp( CONSTANTS[(constant_size * offset) +  B11]*STATES[(states_size * offset) + V])*exp(( (CONSTANTS[(constant_size * offset) +  Temp] - 20.0000)*log(CONSTANTS[(constant_size * offset) +  q11]))/10.0000);
  coeffs[4] = - CONSTANTS[(constant_size * offset) +  A21]*exp( CONSTANTS[(constant_size * offset) +  B21]*STATES[(states_size * offset) + V])*exp(( (CONSTANTS[(constant_size * offset) +  Temp] - 20.0000)*log(CONSTANTS[(constant_size * offset) +  q21]))/10.0000) - CONSTANTS[(constant_size * offset) +  A3]*exp( CONSTANTS[(constant_size * offset) +  B3]*STATES[(states_size * offset) + V])*exp(( (CONSTANTS[(constant_size * offset) +  Temp] - 20.0000)*log(CONSTANTS[(constant_size * offset) +  q3]))/10.0000) - CONSTANTS[(constant_size * offset) +  A62]*exp( CONSTANTS[(constant_size * offset) +  B62]*STATES[(states_size * offset) + V])*exp(( (CONSTANTS[(constant_size * offset) +  Temp] - 20.0000)*log(CONSTANTS[(constant_size * offset) +  q62]))/10.0000);
  coeffs[5] = CONSTANTS[(constant_size * offset) +  A52]*exp( CONSTANTS[(constant_size * offset) +  B52]*STATES[(states_size * offset) + V])*exp(( (CONSTANTS[(constant_size * offset) +  Temp] - 20.0000)*log(CONSTANTS[(constant_size * offset) +  q52]))/10.0000);
  coeffs[6] = CONSTANTS[(constant_size * offset) +  A4]*exp( CONSTANTS[(constant_size * offset) +  B4]*STATES[(states_size * offset) + V])*exp(( (CONSTANTS[(constant_size * offset) +  Temp] - 20.0000)*log(CONSTANTS[(constant_size * offset) +  q4]))/10.0000);

  coeffs[7] = CONSTANTS[(constant_size * offset) +  A61]*exp( CONSTANTS[(constant_size * offset) +  B61]*STATES[(states_size * offset) + V])*exp(( (CONSTANTS[(constant_size * offset) +  Temp] - 20.0000)*log(CONSTANTS[(constant_size * offset) +  q61]))/10.0000);
  coeffs[8] = - CONSTANTS[(constant_size * offset) +  A1]*exp( CONSTANTS[(constant_size * offset) +  B1]*STATES[(states_size * offset) + V])*exp(( (CONSTANTS[(constant_size * offset) +  Temp] - 20.0000)*log(CONSTANTS[(constant_size * offset) +  q1]))/10.0000) - CONSTANTS[(constant_size * offset) +  A51]*exp( CONSTANTS[(constant_size * offset) +  B51]*STATES[(states_size * offset) + V])*exp(( (CONSTANTS[(constant_size * offset) +  Temp] - 20.0000)*log(CONSTANTS[(constant_size * offset) +  q51]))/10.0000);
  coeffs[9] = CONSTANTS[(constant_size * offset) +  A2]*exp( CONSTANTS[(constant_size * offset) +  B2]*STATES[(states_size * offset) + V])*exp(( (CONSTANTS[(constant_size * offset) +  Temp] - 20.0000)*log(CONSTANTS[(constant_size * offset) +  q2]))/10.0000);

  coeffs[10] = CONSTANTS[(constant_size * offset) +  A62]*exp( CONSTANTS[(constant_size * offset) +  B62]*STATES[(states_size * offset) + V])*exp(( (CONSTANTS[(constant_size * offset) +  Temp] - 20.0000)*log(CONSTANTS[(constant_size * offset) +  q62]))/10.0000);
  coeffs[11] = CONSTANTS[(constant_size * offset) +  A1]*exp( CONSTANTS[(constant_size * offset) +  B1]*STATES[(states_size * offset) + V])*exp(( (CONSTANTS[(constant_size * offset) +  Temp] - 20.0000)*log(CONSTANTS[(constant_size * offset) +  q1]))/10.0000);
  coeffs[12] = - CONSTANTS[(constant_size * offset) +  A2]*exp( CONSTANTS[(constant_size * offset) +  B2]*STATES[(states_size * offset) + V])*exp(( (CONSTANTS[(constant_size * offset) +  Temp] - 20.0000)*log(CONSTANTS[(constant_size * offset) +  q2]))/10.0000) - CONSTANTS[(constant_size * offset) +  A31]*exp( CONSTANTS[(constant_size * offset) +  B31]*STATES[(states_size * offset) + V])*exp(( (CONSTANTS[(constant_size * offset) +  Temp] - 20.0000)*log(CONSTANTS[(constant_size * offset) +  q31]))/10.0000) - CONSTANTS[(constant_size * offset) +  A52]*exp( CONSTANTS[(constant_size * offset) +  B52]*STATES[(states_size * offset) + V])*exp(( (CONSTANTS[(constant_size * offset) +  Temp] - 20.0000)*log(CONSTANTS[(constant_size * offset) +  q52]))/10.0000);
  coeffs[13] = CONSTANTS[(constant_size * offset) +  A41]*exp( CONSTANTS[(constant_size * offset) +  B41]*STATES[(states_size * offset) + V])*exp(( (CONSTANTS[(constant_size * offset) +  Temp] - 20.0000)*log(CONSTANTS[(constant_size * offset) +  q41]))/10.0000);

  coeffs[14] = CONSTANTS[(constant_size * offset) +  A31]*exp( CONSTANTS[(constant_size * offset) +  B31]*STATES[(states_size * offset) + V])*exp(( (CONSTANTS[(constant_size * offset) +  Temp] - 20.0000)*log(CONSTANTS[(constant_size * offset) +  q31]))/10.0000);
  coeffs[15] = - CONSTANTS[(constant_size * offset) +  A41]*exp( CONSTANTS[(constant_size * offset) +  B41]*STATES[(states_size * offset) + V])*exp(( (CONSTANTS[(constant_size * offset) +  Temp] - 20.0000)*log(CONSTANTS[(constant_size * offset) +  q41]))/10.0000) - CONSTANTS[(constant_size * offset) +  A53]*exp( CONSTANTS[(constant_size * offset) +  B53]*STATES[(states_size * offset) + V])*exp(( (CONSTANTS[(constant_size * offset) +  Temp] - 20.0000)*log(CONSTANTS[(constant_size * offset) +  q53]))/10.0000) - (( CONSTANTS[(constant_size * offset) +  Kmax]*CONSTANTS[(constant_size * offset) +  Ku]*pow( STATES[(states_size * offset) + D],CONSTANTS[(constant_size * offset) +  n]))/(pow( STATES[(states_size * offset) + D],CONSTANTS[(constant_size * offset) +  n])+CONSTANTS[(constant_size * offset) +  halfmax]));
  coeffs[16] = CONSTANTS[(constant_size * offset) +  A63]*exp( CONSTANTS[(constant_size * offset) +  B63]*STATES[(states_size * offset) + V])*exp(( (CONSTANTS[(constant_size * offset) +  Temp] - 20.0000)*log(CONSTANTS[(constant_size * offset) +  q63]))/10.0000);
  coeffs[17] = CONSTANTS[(constant_size * offset) +  Kt];

  coeffs[18] = CONSTANTS[(constant_size * offset) +  A3]*exp( CONSTANTS[(constant_size * offset) +  B3]*STATES[(states_size * offset) + V])*exp(( (CONSTANTS[(constant_size * offset) +  Temp] - 20.0000)*log(CONSTANTS[(constant_size * offset) +  q3]))/10.0000);
  coeffs[19] = CONSTANTS[(constant_size * offset) +  A53]*exp( CONSTANTS[(constant_size * offset) +  B53]*STATES[(states_size * offset) + V])*exp(( (CONSTANTS[(constant_size * offset) +  Temp] - 20.0000)*log(CONSTANTS[(constant_size * offset) +  q53]))/10.0000);
  coeffs[20] = - CONSTANTS[(constant_size * offset) +  A4]*exp( CONSTANTS[(constant_size * offset) +  B4]*STATES[(states_size * offset) + V])*exp(( (CONSTANTS[(constant_size * offset) +  Temp] - 20.0000)*log(CONSTANTS[(constant_size * offset) +  q4]))/10.0000) - CONSTANTS[(constant_size * offset) +  A63]*exp( CONSTANTS[(constant_size * offset) +  B63]*STATES[(states_size * offset) + V])*exp(( (CONSTANTS[(constant_size * offset) +  Temp] - 20.0000)*log(CONSTANTS[(constant_size * offset) +  q63]))/10.0000) - (( CONSTANTS[(constant_size * offset) +  Kmax]*CONSTANTS[(constant_size * offset) +  Ku]*pow( STATES[(states_size * offset) + D],CONSTANTS[(constant_size * offset) +  n]))/(pow( STATES[(states_size * offset) + D],CONSTANTS[(constant_size * offset) +  n])+CONSTANTS[(constant_size * offset) +  halfmax]));
  coeffs[21] = (( CONSTANTS[(constant_size * offset) +  Ku]*CONSTANTS[(constant_size * offset) +  A53]*exp( CONSTANTS[(constant_size * offset) +  B53]*STATES[(states_size * offset) + V])*exp(( (CONSTANTS[(constant_size * offset) +  Temp] - 20.0000)*log(CONSTANTS[(constant_size * offset) +  q53]))/10.0000))/( CONSTANTS[(constant_size * offset) +  A63]*exp( CONSTANTS[(constant_size * offset) +  B63]*STATES[(states_size * offset) + V])*exp(( (CONSTANTS[(constant_size * offset) +  Temp] - 20.0000)*log(CONSTANTS[(constant_size * offset) +  q63]))/10.0000)));

  coeffs[22] = (( CONSTANTS[(constant_size * offset) +  Kmax]*CONSTANTS[(constant_size * offset) +  Ku]*pow( STATES[(states_size * offset) + D],CONSTANTS[(constant_size * offset) +  n]))/(pow( STATES[(states_size * offset) + D],CONSTANTS[(constant_size * offset) +  n])+CONSTANTS[(constant_size * offset) +  halfmax]));
  coeffs[23] = -  CONSTANTS[(constant_size * offset) +  Ku] - CONSTANTS[(constant_size * offset) +  Kt];
  coeffs[24] = (CONSTANTS[(constant_size * offset) +  Kt]/(1.00000+exp(- (STATES[(states_size * offset) + V] - CONSTANTS[(constant_size * offset) +  Vhalf])/6.78900)));

  coeffs[25] = (( CONSTANTS[(constant_size * offset) +  Kmax]*CONSTANTS[(constant_size * offset) +  Ku]*pow( STATES[(states_size * offset) + D],CONSTANTS[(constant_size * offset) +  n]))/(pow( STATES[(states_size * offset) + D],CONSTANTS[(constant_size * offset) +  n])+CONSTANTS[(constant_size * offset) +  halfmax]));
  coeffs[26] = - (( CONSTANTS[(constant_size * offset) +  Ku]*CONSTANTS[(constant_size * offset) +  A53]*exp( CONSTANTS[(constant_size * offset) +  B53]*STATES[(states_size * offset) + V])*exp(( (CONSTANTS[(constant_size * offset) +  Temp] - 20.0000)*log(CONSTANTS[(constant_size * offset) +  q53]))/10.0000))/( CONSTANTS[(constant_size * offset) +  A63]*exp( CONSTANTS[(constant_size * offset) +  B63]*STATES[(states_size * offset) + V])*exp(( (CONSTANTS[(constant_size * offset) +  Temp] - 20.0000)*log(CONSTANTS[(constant_size * offset) +  q63]))/10.0000))) - CONSTANTS[(constant_size * offset) +  Kt];
  coeffs[27] = (CONSTANTS[(constant_size * offset) +  Kt]/(1.00000+exp(- (STATES[(states_size * offset) + V] - CONSTANTS[(constant_size * offset) +  Vhalf])/6.78900)));

  coeffs[28] = CONSTANTS[(constant_size * offset) +  Kt];
  coeffs[29] = CONSTANTS[(constant_size * offset) +  Kt];
  coeffs[30] = - (CONSTANTS[(constant_size * offset) +  Kt]/(1.00000+exp(- (STATES[(states_size * offset) + V] - CONSTANTS[(constant_size * offset) +  Vhalf])/6.78900))) - (CONSTANTS[(constant_size * offset) +  Kt]/(1.00000+exp(- (STATES[(states_size * offset) + V] - CONSTANTS[(constant_size * offset) +  Vhalf])/6.78900)));
  int m = 9;
  double* a = new double[m*m]; // Flattened a
  a[0 * m + 0] = 1.0 - dt * coeffs[0];   a[0 * m + 1] = - dt * coeffs[1];     a[0 * m + 2] = - dt * coeffs[2];     a[0 * m + 3] = 0.0;                      a[0 * m + 4] = 0.0;                      a[0 * m + 5] = 0.0;                      a[0 * m + 6] = 0.0;                      a[0 * m + 7] = 0.0;                      a[0 * m + 8] = 0.0;
  a[1 * m + 0] = - dt * coeffs[3];       a[1 * m + 1] = 1.0 - dt * coeffs[4]; a[1 * m + 2] = 0.0;                  a[1 * m + 3] = - dt * coeffs[5];         a[1 * m + 4] = 0.0;                      a[1 * m + 5] = - dt * coeffs[6];         a[1 * m + 6] = 0.0;                      a[1 * m + 7] = 0.0;                      a[1 * m + 8] = 0.0;
  a[2 * m + 0] = - dt * coeffs[7];       a[2 * m + 1] = 0.0;                  a[2 * m + 2] = 1.0 - dt * coeffs[8]; a[2 * m + 3] = - dt * coeffs[9];         a[2 * m + 4] = 0.0;                      a[2 * m + 5] = 0.0;                      a[2 * m + 6] = 0.0;                      a[2 * m + 7] = 0.0;                      a[2 * m + 8] = 0.0;
  a[3 * m + 0] = 0.0;                    a[3 * m + 1] = - dt * coeffs[10];    a[3 * m + 2] = - dt * coeffs[11];    a[3 * m + 3] = 1.0 - dt * coeffs[12];    a[3 * m + 4] = - dt * coeffs[13];        a[3 * m + 5] = 0.0;                      a[3 * m + 6] = 0.0;                      a[3 * m + 7] = 0.0;                      a[3 * m + 8] = 0.0;
  a[4 * m + 0] = 0.0;                    a[4 * m + 1] = 0.0;                  a[4 * m + 2] = 0.0;                  a[4 * m + 3] = - dt * coeffs[14];        a[4 * m + 4] = 1.0 - dt * coeffs[15];    a[4 * m + 5] = - dt * coeffs[16];        a[4 * m + 6] = - dt * coeffs[17];        a[4 * m + 7] = 0.0;                      a[4 * m + 8] = 0.0;
  a[5 * m + 0] = 0.0;                    a[5 * m + 1] = - dt * coeffs[18];    a[5 * m + 2] = 0.0;                  a[5 * m + 3] = 0.0;                      a[5 * m + 4] = - dt * coeffs[19];        a[5 * m + 5] = 1.0 - dt * coeffs[20];    a[5 * m + 6] = - dt * coeffs[21];        a[5 * m + 7] = 0.0;                      a[5 * m + 8] = 0.0;
  a[6 * m + 0] = 0.0;                    a[6 * m + 1] = 0.0;                  a[6 * m + 2] = 0.0;                  a[6 * m + 3] = 0.0;                      a[6 * m + 4] = - dt * coeffs[22];        a[6 * m + 5] = 0.0;                      a[6 * m + 6] = 1.0 - dt * coeffs[23];    a[6 * m + 7] = 0.0;                      a[6 * m + 8] = - dt * coeffs[24];
  a[7 * m + 0] = 0.0;                    a[7 * m + 1] = 0.0;                  a[7 * m + 2] = 0.0;                  a[7 * m + 3] = 0.0;                      a[7 * m + 4] = 0.0;                      a[7 * m + 5] = - dt * coeffs[25];        a[7 * m + 6] = 0.0;                      a[7 * m + 7] = 1.0 - dt * coeffs[26];    a[7 * m + 8] = - dt * coeffs[27];
  a[8 * m + 0] = 0.0;                    a[8 * m + 1] = 0.0;                  a[8 * m + 2] = 0.0;                  a[8 * m + 3] = 0.0;                      a[8 * m + 4] = 0.0;                      a[8 * m + 5] = 0.0;                      a[8 * m + 6] = - dt * coeffs[28];        a[8 * m + 7] = - dt * coeffs[29];        a[8 * m + 8] = 1.0 - dt * coeffs[30];
  double* b = new double[m];
  b[0] = STATES[(states_size * offset) + IC1];
  b[1] = STATES[(states_size * offset) + IC2];
  b[2] = STATES[(states_size * offset) + C1];
  b[3] = STATES[(states_size * offset) + C2];
  b[4] = STATES[(states_size * offset) + O];
  b[5] = STATES[(states_size * offset) + IO];
  b[6] = STATES[(states_size * offset) + Obound];
  b[7] = STATES[(states_size * offset) + IObound];
  b[8] = STATES[(states_size * offset) + Cbound];
  double* x = new double[m];
  for(int i = 0; i < m; i++){
    x[i] = 0.0;
  }
  __device__ double ___gaussElimination(a,b,x,m); // gpu capable?
  STATES[(states_size * offset) + IC1] = x[0];
  STATES[(states_size * offset) + IC2] = x[1];
  STATES[(states_size * offset) + C1] = x[2];
  STATES[(states_size * offset) + C2] = x[3];
  STATES[(states_size * offset) + O] = x[4];
  STATES[(states_size * offset) + IO] = x[5];
  STATES[(states_size * offset) + Obound] = x[6];
  STATES[(states_size * offset) + IObound] = x[7];
  STATES[(states_size * offset) + Cbound] = x[8];
  delete[] coeffs;
  delete[] a;
  delete[] b;
  delete[] x;
//  STATES[IC1] = STATES[IC1] + RATES[IC1] * dt;
//  STATES[IC2] = STATES[IC2] + RATES[IC2] * dt;
//  STATES[C1] = STATES[C1] + RATES[C1] * dt;
//  STATES[C2] = STATES[C2] + RATES[C2] * dt;
//  STATES[O] = STATES[O] + RATES[O] * dt;
//  STATES[IO] = STATES[IO] + RATES[IO] * dt;
//  STATES[D] = STATES[D] + RATES[D] * dt;
//  STATES[IObound] = STATES[IObound] + RATES[IObound] * dt;
//  STATES[Obound] = STATES[Obound] + RATES[Obound] * dt;
//  STATES[Cbound] = STATES[Cbound] + RATES[Cbound] * dt;
////=============================
////Approximated solution (Forward Euler)
////=============================
////CaMK
  STATES[(states_size * offset) + CaMKt] = STATES[(states_size * offset) + CaMKt] + RATES[(rates_size * offset) + CaMKt] * dt;
////Membrane potential
  STATES[(states_size * offset) + V] = STATES[(states_size * offset) + V] + RATES[(rates_size * offset) + V] * dt;
////Ion Concentrations and Buffers
  STATES[(states_size * offset) + nai] = STATES[(states_size * offset) + nai] + RATES[(rates_size * offset) + nai] * dt;
  STATES[(states_size * offset) + nass] = STATES[(states_size * offset) + nass] + RATES[(rates_size * offset) + nass] * dt;
  STATES[(states_size * offset) + ki] = STATES[(states_size * offset) + ki] + RATES[(rates_size * offset) + ki] * dt;
  STATES[(states_size * offset) + kss] = STATES[(states_size * offset) + kss] + RATES[(rates_size * offset) + kss] * dt;
  STATES[(states_size * offset) + cai] = STATES[(states_size * offset) + cai] + RATES[(rates_size * offset) + cai] * dt;
  STATES[(states_size * offset) + cass] = STATES[(states_size * offset) + cass] + RATES[(rates_size * offset) + cass] * dt;
  STATES[(states_size * offset) + cansr] = STATES[(states_size * offset) + cansr] + RATES[(rates_size * offset) + cansr] * dt;
  STATES[(states_size * offset) + cajsr] = STATES[(states_size * offset) + cajsr] + RATES[(rates_size * offset) + cajsr] * dt;
//for(int i=0;i<states_size;i++){
//    STATES[i] = STATES[i] + RATES[i] * dt;
//}
}

__device__ double ___gaussElimination(double *A, double *b, double *x, int N) {
        // Using A as a flat array to represent an N x N matrix
    for (int i = 0; i < N; i++) {
        // Search for maximum in this column
        double maxEl = fabs(A[i*N + i]);
        int maxRow = i;
        for (int k = i + 1; k < N; k++) {
            if (fabs(A[k*N + i]) > maxEl) {
                maxEl = fabs(A[k*N + i]);
                maxRow = k;
            }
        }

        // Swap maximum row with current row (column by column)
        for (int k = i; k < N; k++) {
            double tmp = A[maxRow*N + k];
            A[maxRow*N + k] = A[i*N + k];
            A[i*N + k] = tmp;
        }
        double tmp = b[maxRow];
        b[maxRow] = b[i];
        b[i] = tmp;

        // Make all rows below this one 0 in current column
        for (int k = i + 1; k < N; k++) {
            double c = -A[k*N + i] / A[i*N + i];
            for (int j = i; j < N; j++) {
                if (i == j) {
                    A[k*N + j] = 0;
                } else {
                    A[k*N + j] += c * A[i*N + j];
                }
            }
            b[k] += c * b[i];
        }
    }

    // Solve equation Ax=b for an upper triangular matrix A
    for (int i = N - 1; i >= 0; i--) {
        x[i] = b[i] / A[i*N + i];
        for (int k = i - 1; k >= 0; k--) {
            b[k] -= A[k*N + i] * x[i];
        }
    }
}

// void ohara_rudy_cipa_v1_2017::solveRK4(double TIME, double dt)
// {
// 	unsigned short idx;
// 	double k1[49],k2[49],k3[49],k4[49];
// 	double states_temp[49];
	
// 	computeRates(TIME, CONSTANTS, RATES, STATES, ALGEBRAIC );
// 	for(idx = 0; idx < states_size; idx++){
// 		k1[idx] = dt * RATES[idx];
// 		states_temp[idx] = STATES[idx] + k1[idx]*0.5;
// 	}
// 	computeRates(TIME+(dt*0.5), CONSTANTS, RATES, states_temp, ALGEBRAIC );
// 	for(idx = 0; idx < states_size; idx++){
// 		k2[idx] = dt * RATES[idx];
// 		states_temp[idx] = STATES[idx] + k2[idx]*0.5;
// 	}
// 	computeRates(TIME+(dt*0.5), CONSTANTS, RATES, states_temp, ALGEBRAIC );
// 	for(idx = 0; idx < states_size; idx++){
// 		k3[idx] = dt * RATES[idx];
// 		states_temp[idx] = STATES[idx] + k3[idx];
// 	}
// 	computeRates(TIME+dt, CONSTANTS, RATES, states_temp, ALGEBRAIC );
// 	for(idx = 0; idx < states_size; idx++){
// 		k4[idx] = dt * RATES[idx];
// 		STATES[idx] += (k1[idx]/6) + (k2[idx]/3) + (k3[idx]/3) + (k4[idx]/6) ;
// 	}
// }

__device__ double set_time_step (double TIME,
                                              double time_point,
                                              double min_time_step,
                                              double max_time_step,
                                              double min_dV,
                                              double max_dV) {
 double time_step = min_time_step;
 if (TIME <= time_point || (TIME - floor(TIME / CONSTANTS[(constant_size * offset) +  BCL]) * CONSTANTS[(constant_size * offset) +  BCL]) <= time_point) {
    //printf("TIME <= time_point ms\n");
    return time_step;
    //printf("TIME = %E, dV = %E, time_step = %E\n",TIME, RATES[V] * time_step, time_step);
  }
  else {
    //printf("TIME > time_point ms\n");
    if (std::abs(RATES[(rates_size * offset) + V] * time_step) <= min_dV) {//Slow changes in V
        //printf("dV/dt <= 0.2\n");
        time_step = std::abs(max_dV / RATES[(rates_size * offset) + V]);
        //Make sure time_step is between min time step and max_time_step
        if (time_step < min_time_step) {
            time_step = min_time_step;
        }
        else if (time_step > max_time_step) {
            time_step = max_time_step;
        }
        //printf("TIME = %E, dV = %E, time_step = %E\n",TIME, RATES[V] * time_step, time_step);
    }
    else if (std::abs(RATES[(rates_size * offset) + V] * time_step) >= max_dV) {//Fast changes in V
        //printf("dV/dt >= 0.8\n");
        time_step = std::abs(min_dV / RATES[(rates_size * offset) + V]);
        //Make sure time_step is not less than 0.005
        if (time_step < min_time_step) {
            time_step = min_time_step;
        }
        //printf("TIME = %E, dV = %E, time_step = %E\n",TIME, RATES[V] * time_step, time_step);
    } else {
        time_step = min_time_step;
    }
    return time_step;
  }
}



