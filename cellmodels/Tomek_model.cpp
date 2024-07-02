/*
   There are a total of 223 entries in the algebraic variable array.
   There are a total of 43 entries in each of the rate and state variable arrays.
   There are a total of 163 entries in the constant variable array.
 */

#include "Tomek_model.hpp"
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include "../modules/glob_funct.hpp"
#include <cuda_runtime.h>
#include <cuda.h>

// #include "../../functions/inputoutput.hpp" // disabled for GPU operations

/*
 * TIME is time in component environment (millisecond).
 * CONSTANTS[celltype] is celltype in component environment (dimensionless).
 * CONSTANTS[nao] is nao in component extracellular (millimolar).
 * CONSTANTS[cao] is cao in component extracellular (millimolar).
 * CONSTANTS[ko] is ko in component extracellular (millimolar).
 * CONSTANTS[clo] is clo in component extracellular (millimolar).
 * CONSTANTS[R] is R in component physical_constants (joule_per_kilomole_kelvin).
 * CONSTANTS[T] is T in component physical_constants (kelvin).
 * CONSTANTS[F] is F in component physical_constants (coulomb_per_mole).
 * CONSTANTS[zna] is zna in component physical_constants (dimensionless).
 * CONSTANTS[zca] is zca in component physical_constants (dimensionless).
 * CONSTANTS[zk] is zk in component physical_constants (dimensionless).
 * CONSTANTS[zcl] is zcl in component physical_constants (dimensionless).
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
 * ALGEBRAIC[vffrt] is vffrt in component membrane (coulomb_per_mole).
 * ALGEBRAIC[vfrt] is vfrt in component membrane (dimensionless).
 * ALGEBRAIC[INa] is INa in component INa (microA_per_microF).
 * ALGEBRAIC[INaL] is INaL in component INaL (microA_per_microF).
 * ALGEBRAIC[Ito] is Ito in component Ito (microA_per_microF).
 * ALGEBRAIC[ICaL] is ICaL in component ICaL (microA_per_microF).
 * ALGEBRAIC[ICaNa] is ICaNa in component ICaL (microA_per_microF).
 * ALGEBRAIC[ICaK] is ICaK in component ICaL (microA_per_microF).
 * ALGEBRAIC[IKr] is IKr in component IKr (microA_per_microF).
 * ALGEBRAIC[IKs] is IKs in component IKs (microA_per_microF).
 * ALGEBRAIC[IK1] is IK1 in component IK1 (microA_per_microF).
 * ALGEBRAIC[INaCa_i] is INaCa_i in component INaCa (microA_per_microF).
 * ALGEBRAIC[INaCa_ss] is INaCa_ss in component INaCa (microA_per_microF).
 * ALGEBRAIC[INaK] is INaK in component INaK (microA_per_microF).
 * ALGEBRAIC[INab] is INab in component INab (microA_per_microF).
 * ALGEBRAIC[IKb] is IKb in component IKb (microA_per_microF).
 * ALGEBRAIC[IpCa] is IpCa in component IpCa (microA_per_microF).
 * ALGEBRAIC[ICab] is ICab in component ICab (microA_per_microF).
 * ALGEBRAIC[IClCa] is IClCa in component ICl (microA_per_microF).
 * ALGEBRAIC[IClb] is IClb in component ICl (microA_per_microF).
 * ALGEBRAIC[I_katp] is I_katp in component I_katp (microA_per_microF).
 * ALGEBRAIC[Istim] is Istim in component membrane (microA_per_microF).
 * CONSTANTS[stim_start] is stim_start in component membrane (millisecond).
 * CONSTANTS[i_Stim_End] is i_Stim_End in component membrane (millisecond).
 * CONSTANTS[i_Stim_Amplitude] is i_Stim_Amplitude in component membrane (microA_per_microF).
 * CONSTANTS[BCL] is BCL in component membrane (millisecond).
 * CONSTANTS[i_Stim_PulseDuration] is i_Stim_PulseDuration in component membrane (millisecond).
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
 * CONSTANTS[cli] is cli in component intracellular_ions (millimolar).
 * ALGEBRAIC[ICaL_ss] is ICaL_ss in component ICaL (microA_per_microF).
 * ALGEBRAIC[ICaNa_ss] is ICaNa_ss in component ICaL (microA_per_microF).
 * ALGEBRAIC[ICaK_ss] is ICaK_ss in component ICaL (microA_per_microF).
 * ALGEBRAIC[ICaL_i] is ICaL_i in component ICaL (microA_per_microF).
 * ALGEBRAIC[ICaNa_i] is ICaNa_i in component ICaL (microA_per_microF).
 * ALGEBRAIC[ICaK_i] is ICaK_i in component ICaL (microA_per_microF).
 * ALGEBRAIC[JdiffNa] is JdiffNa in component diff (millimolar_per_millisecond).
 * ALGEBRAIC[Jdiff] is Jdiff in component diff (millimolar_per_millisecond).
 * ALGEBRAIC[Jup] is Jup in component SERCA (millimolar_per_millisecond).
 * ALGEBRAIC[JdiffK] is JdiffK in component diff (millimolar_per_millisecond).
 * ALGEBRAIC[Jrel] is Jrel in component ryr (millimolar_per_millisecond).
 * ALGEBRAIC[Jtr] is Jtr in component trans_flux (millimolar_per_millisecond).
 * ALGEBRAIC[Bcai] is Bcai in component intracellular_ions (dimensionless).
 * ALGEBRAIC[Bcajsr] is Bcajsr in component intracellular_ions (dimensionless).
 * ALGEBRAIC[Bcass] is Bcass in component intracellular_ions (dimensionless).
 * CONSTANTS[PKNa] is PKNa in component reversal_potentials (dimensionless).
 * ALGEBRAIC[ENa] is ENa in component reversal_potentials (millivolt).
 * ALGEBRAIC[EK] is EK in component reversal_potentials (millivolt).
 * ALGEBRAIC[EKs] is EKs in component reversal_potentials (millivolt).
 * CONSTANTS[ECl] is ECl in component reversal_potentials (millivolt).
 * CONSTANTS[gkatp] is gkatp in component I_katp (milliS_per_microF).
 * CONSTANTS[fkatp] is fkatp in component I_katp (dimensionless).
 * CONSTANTS[K_o_n] is K_o_n in component I_katp (millimolar).
 * CONSTANTS[A_atp] is A_atp in component I_katp (millimolar).
 * CONSTANTS[K_atp] is K_atp in component I_katp (millimolar).
 * CONSTANTS[akik] is akik in component I_katp (dimensionless).
 * CONSTANTS[bkik] is bkik in component I_katp (dimensionless).
 * ALGEBRAIC[mss] is mss in component INa (dimensionless).
 * ALGEBRAIC[tm] is tm in component INa (millisecond).
 * STATES[m] is m in component INa (dimensionless).
 * ALGEBRAIC[hss] is hss in component INa (dimensionless).
 * ALGEBRAIC[ah] is ah in component INa (dimensionless).
 * ALGEBRAIC[bh] is bh in component INa (dimensionless).
 * ALGEBRAIC[th] is th in component INa (millisecond).
 * STATES[h] is h in component INa (dimensionless).
 * ALGEBRAIC[jss] is jss in component INa (dimensionless).
 * ALGEBRAIC[aj] is aj in component INa (dimensionless).
 * ALGEBRAIC[bj] is bj in component INa (dimensionless).
 * ALGEBRAIC[tj] is tj in component INa (millisecond).
 * STATES[j] is j in component INa (dimensionless).
 * ALGEBRAIC[hssp] is hssp in component INa (dimensionless).
 * STATES[hp] is hp in component INa (dimensionless).
 * ALGEBRAIC[tjp] is tjp in component INa (millisecond).
 * STATES[jp] is jp in component INa (dimensionless).
 * ALGEBRAIC[fINap] is fINap in component INa (dimensionless).
 * CONSTANTS[GNa] is GNa in component INa (milliS_per_microF).
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
 * CONSTANTS[EKshift] is EKshift in component Ito (millivolt).
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
 * ALGEBRAIC[jcass] is jcass in component ICaL (dimensionless).
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
 * ALGEBRAIC[anca_ss] is anca_ss in component ICaL (dimensionless).
 * STATES[nca_ss] is nca_ss in component ICaL (dimensionless).
 * ALGEBRAIC[anca_i] is anca_i in component ICaL (dimensionless).
 * STATES[nca_i] is nca_i in component ICaL (dimensionless).
 * ALGEBRAIC[PhiCaL_ss] is PhiCaL_ss in component ICaL (dimensionless).
 * ALGEBRAIC[PhiCaNa_ss] is PhiCaNa_ss in component ICaL (dimensionless).
 * ALGEBRAIC[PhiCaK_ss] is PhiCaK_ss in component ICaL (dimensionless).
 * ALGEBRAIC[PhiCaL_i] is PhiCaL_i in component ICaL (dimensionless).
 * ALGEBRAIC[PhiCaNa_i] is PhiCaNa_i in component ICaL (dimensionless).
 * ALGEBRAIC[PhiCaK_i] is PhiCaK_i in component ICaL (dimensionless).
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
 * CONSTANTS[vShift] is vShift in component ICaL (millivolt).
 * CONSTANTS[offset] is offset in component ICaL (millisecond).
 * CONSTANTS[Io] is Io in component ICaL (dimensionless).
 * ALGEBRAIC[Iss] is Iss in component ICaL (dimensionless).
 * ALGEBRAIC[Ii] is Ii in component ICaL (dimensionless).
 * CONSTANTS[dielConstant] is dielConstant in component ICaL (per_kelvin).
 * CONSTANTS[constA] is constA in component ICaL (dimensionless).
 * CONSTANTS[gamma_cao] is gamma_cao in component ICaL (dimensionless).
 * ALGEBRAIC[gamma_cass] is gamma_cass in component ICaL (dimensionless).
 * ALGEBRAIC[gamma_cai] is gamma_cai in component ICaL (dimensionless).
 * CONSTANTS[gamma_nao] is gamma_nao in component ICaL (dimensionless).
 * ALGEBRAIC[gamma_nass] is gamma_nass in component ICaL (dimensionless).
 * ALGEBRAIC[gamma_nai] is gamma_nai in component ICaL (dimensionless).
 * CONSTANTS[gamma_ko] is gamma_ko in component ICaL (dimensionless).
 * ALGEBRAIC[gamma_kss] is gamma_kss in component ICaL (dimensionless).
 * ALGEBRAIC[gamma_ki] is gamma_ki in component ICaL (dimensionless).
 * CONSTANTS[ICaL_fractionSS] is ICaL_fractionSS in component ICaL (dimensionless).
 * CONSTANTS[GKr_b] is GKr_b in component IKr (milliS_per_microF).
 * STATES[C1] is C1 in component IKr (dimensionless).
 * STATES[C2] is C2 in component IKr (dimensionless).
 * STATES[C3] is C3 in component IKr (dimensionless).
 * STATES[I] is I in component IKr (dimensionless).
 * STATES[O] is O in component IKr (dimensionless).
 * ALGEBRAIC[alpha] is alpha in component IKr (per_millisecond).
 * ALGEBRAIC[beta] is beta in component IKr (per_millisecond).
 * CONSTANTS[alpha_1] is alpha_1 in component IKr (per_millisecond).
 * CONSTANTS[beta_1] is beta_1 in component IKr (per_millisecond).
 * ALGEBRAIC[alpha_2] is alpha_2 in component IKr (per_millisecond).
 * ALGEBRAIC[beta_2] is beta_2 in component IKr (per_millisecond).
 * ALGEBRAIC[alpha_i] is alpha_i in component IKr (per_millisecond).
 * ALGEBRAIC[beta_i] is beta_i in component IKr (per_millisecond).
 * ALGEBRAIC[alpha_C2ToI] is alpha_C2ToI in component IKr (per_millisecond).
 * ALGEBRAIC[beta_ItoC2] is beta_ItoC2 in component IKr (per_millisecond).
 * CONSTANTS[GKr] is GKr in component IKr (milliS_per_microF).
 * CONSTANTS[GKs_b] is GKs_b in component IKs (milliS_per_microF).
 * CONSTANTS[GKs] is GKs in component IKs (milliS_per_microF).
 * ALGEBRAIC[xs1ss] is xs1ss in component IKs (dimensionless).
 * ALGEBRAIC[xs2ss] is xs2ss in component IKs (dimensionless).
 * ALGEBRAIC[txs1] is txs1 in component IKs (millisecond).
 * STATES[xs1] is xs1 in component IKs (dimensionless).
 * STATES[xs2] is xs2 in component IKs (dimensionless).
 * ALGEBRAIC[KsCa] is KsCa in component IKs (dimensionless).
 * ALGEBRAIC[txs2] is txs2 in component IKs (millisecond).
 * CONSTANTS[GK1] is GK1 in component IK1 (milliS_per_microF).
 * CONSTANTS[GK1_b] is GK1_b in component IK1 (milliS_per_microF).
 * ALGEBRAIC[aK1] is aK1 in component IK1 (dimensionless).
 * ALGEBRAIC[bK1] is bK1 in component IK1 (dimensionless).
 * ALGEBRAIC[K1ss] is K1ss in component IK1 (dimensionless).
 * CONSTANTS[INaCa_fractionSS] is INaCa_fractionSS in component INaCa (dimensionless).
 * CONSTANTS[kna1] is kna1 in component INaCa (per_millisecond).
 * CONSTANTS[kna2] is kna2 in component INaCa (per_millisecond).
 * CONSTANTS[kna3] is kna3 in component INaCa (per_millisecond).
 * CONSTANTS[kasymm] is kasymm in component INaCa (dimensionless).
 * CONSTANTS[wna] is wna in component INaCa (dimensionless).
 * CONSTANTS[wca] is wca in component INaCa (dimensionless).
 * CONSTANTS[wnaca] is wnaca in component INaCa (dimensionless).
 * CONSTANTS[kcaon] is kcaon in component INaCa (per_millisecond).
 * CONSTANTS[kcaoff] is kcaoff in component INaCa (per_millisecond).
 * CONSTANTS[qna] is qna in component INaCa (dimensionless).
 * CONSTANTS[qca] is qca in component INaCa (dimensionless).
 * ALGEBRAIC[hna] is hna in component INaCa (dimensionless).
 * ALGEBRAIC[hca] is hca in component INaCa (dimensionless).
 * CONSTANTS[KmCaAct] is KmCaAct in component INaCa (millimolar).
 * CONSTANTS[Gncx_b] is Gncx_b in component INaCa (milliS_per_microF).
 * CONSTANTS[Gncx] is Gncx in component INaCa (milliS_per_microF).
 * ALGEBRAIC[h1_i] is h1_i in component INaCa (dimensionless).
 * ALGEBRAIC[h2_i] is h2_i in component INaCa (dimensionless).
 * ALGEBRAIC[h3_i] is h3_i in component INaCa (dimensionless).
 * ALGEBRAIC[h4_i] is h4_i in component INaCa (dimensionless).
 * ALGEBRAIC[h5_i] is h5_i in component INaCa (dimensionless).
 * ALGEBRAIC[h6_i] is h6_i in component INaCa (dimensionless).
 * ALGEBRAIC[h7_i] is h7_i in component INaCa (dimensionless).
 * ALGEBRAIC[h8_i] is h8_i in component INaCa (dimensionless).
 * ALGEBRAIC[h9_i] is h9_i in component INaCa (dimensionless).
 * CONSTANTS[h10_i] is h10_i in component INaCa (dimensionless).
 * CONSTANTS[h11_i] is h11_i in component INaCa (dimensionless).
 * CONSTANTS[h12_i] is h12_i in component INaCa (dimensionless).
 * CONSTANTS[k1_i] is k1_i in component INaCa (dimensionless).
 * CONSTANTS[k2_i] is k2_i in component INaCa (dimensionless).
 * ALGEBRAIC[k3p_i] is k3p_i in component INaCa (dimensionless).
 * ALGEBRAIC[k3pp_i] is k3pp_i in component INaCa (dimensionless).
 * ALGEBRAIC[k3_i] is k3_i in component INaCa (dimensionless).
 * ALGEBRAIC[k4_i] is k4_i in component INaCa (dimensionless).
 * ALGEBRAIC[k4p_i] is k4p_i in component INaCa (dimensionless).
 * ALGEBRAIC[k4pp_i] is k4pp_i in component INaCa (dimensionless).
 * CONSTANTS[k5_i] is k5_i in component INaCa (dimensionless).
 * ALGEBRAIC[k6_i] is k6_i in component INaCa (dimensionless).
 * ALGEBRAIC[k7_i] is k7_i in component INaCa (dimensionless).
 * ALGEBRAIC[k8_i] is k8_i in component INaCa (dimensionless).
 * ALGEBRAIC[x1_i] is x1_i in component INaCa (dimensionless).
 * ALGEBRAIC[x2_i] is x2_i in component INaCa (dimensionless).
 * ALGEBRAIC[x3_i] is x3_i in component INaCa (dimensionless).
 * ALGEBRAIC[x4_i] is x4_i in component INaCa (dimensionless).
 * ALGEBRAIC[E1_i] is E1_i in component INaCa (dimensionless).
 * ALGEBRAIC[E2_i] is E2_i in component INaCa (dimensionless).
 * ALGEBRAIC[E3_i] is E3_i in component INaCa (dimensionless).
 * ALGEBRAIC[E4_i] is E4_i in component INaCa (dimensionless).
 * ALGEBRAIC[allo_i] is allo_i in component INaCa (dimensionless).
 * ALGEBRAIC[JncxNa_i] is JncxNa_i in component INaCa (millimolar_per_millisecond).
 * ALGEBRAIC[JncxCa_i] is JncxCa_i in component INaCa (millimolar_per_millisecond).
 * ALGEBRAIC[h1_ss] is h1_ss in component INaCa (dimensionless).
 * ALGEBRAIC[h2_ss] is h2_ss in component INaCa (dimensionless).
 * ALGEBRAIC[h3_ss] is h3_ss in component INaCa (dimensionless).
 * ALGEBRAIC[h4_ss] is h4_ss in component INaCa (dimensionless).
 * ALGEBRAIC[h5_ss] is h5_ss in component INaCa (dimensionless).
 * ALGEBRAIC[h6_ss] is h6_ss in component INaCa (dimensionless).
 * ALGEBRAIC[h7_ss] is h7_ss in component INaCa (dimensionless).
 * ALGEBRAIC[h8_ss] is h8_ss in component INaCa (dimensionless).
 * ALGEBRAIC[h9_ss] is h9_ss in component INaCa (dimensionless).
 * CONSTANTS[h10_ss] is h10_ss in component INaCa (dimensionless).
 * CONSTANTS[h11_ss] is h11_ss in component INaCa (dimensionless).
 * CONSTANTS[h12_ss] is h12_ss in component INaCa (dimensionless).
 * CONSTANTS[k1_ss] is k1_ss in component INaCa (dimensionless).
 * CONSTANTS[k2_ss] is k2_ss in component INaCa (dimensionless).
 * ALGEBRAIC[k3p_ss] is k3p_ss in component INaCa (dimensionless).
 * ALGEBRAIC[k3pp_ss] is k3pp_ss in component INaCa (dimensionless).
 * ALGEBRAIC[k3_ss] is k3_ss in component INaCa (dimensionless).
 * ALGEBRAIC[k4_ss] is k4_ss in component INaCa (dimensionless).
 * ALGEBRAIC[k4p_ss] is k4p_ss in component INaCa (dimensionless).
 * ALGEBRAIC[k4pp_ss] is k4pp_ss in component INaCa (dimensionless).
 * CONSTANTS[k5_ss] is k5_ss in component INaCa (dimensionless).
 * ALGEBRAIC[k6_ss] is k6_ss in component INaCa (dimensionless).
 * ALGEBRAIC[k7_ss] is k7_ss in component INaCa (dimensionless).
 * ALGEBRAIC[k8_ss] is k8_ss in component INaCa (dimensionless).
 * ALGEBRAIC[x1_ss] is x1_ss in component INaCa (dimensionless).
 * ALGEBRAIC[x2_ss] is x2_ss in component INaCa (dimensionless).
 * ALGEBRAIC[x3_ss] is x3_ss in component INaCa (dimensionless).
 * ALGEBRAIC[x4_ss] is x4_ss in component INaCa (dimensionless).
 * ALGEBRAIC[E1_ss] is E1_ss in component INaCa (dimensionless).
 * ALGEBRAIC[E2_ss] is E2_ss in component INaCa (dimensionless).
 * ALGEBRAIC[E3_ss] is E3_ss in component INaCa (dimensionless).
 * ALGEBRAIC[E4_ss] is E4_ss in component INaCa (dimensionless).
 * ALGEBRAIC[allo_ss] is allo_ss in component INaCa (dimensionless).
 * ALGEBRAIC[JncxNa_ss] is JncxNa_ss in component INaCa (millimolar_per_millisecond).
 * ALGEBRAIC[JncxCa_ss] is JncxCa_ss in component INaCa (millimolar_per_millisecond).
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
 * CONSTANTS[PCab] is PCab in component ICab (milliS_per_microF).
 * CONSTANTS[GpCa] is GpCa in component IpCa (milliS_per_microF).
 * CONSTANTS[KmCap] is KmCap in component IpCa (millimolar).
 * CONSTANTS[GClCa] is GClCa in component ICl (milliS_per_microF).
 * CONSTANTS[GClb] is GClb in component ICl (milliS_per_microF).
 * CONSTANTS[KdClCa] is KdClCa in component ICl (millimolar).
 * CONSTANTS[Fjunc] is Fjunc in component ICl (dimensionless).
 * ALGEBRAIC[IClCa_junc] is IClCa_junc in component ICl (microA_per_microF).
 * ALGEBRAIC[IClCa_sl] is IClCa_sl in component ICl (microA_per_microF).
 * CONSTANTS[tauNa] is tauNa in component diff (millisecond).
 * CONSTANTS[tauK] is tauK in component diff (millisecond).
 * CONSTANTS[tauCa] is tauCa in component diff (millisecond).
 * CONSTANTS[bt] is bt in component ryr (millisecond).
 * CONSTANTS[a_rel] is a_rel in component ryr (millimolar_per_millisecond).
 * ALGEBRAIC[Jrel_inf_b] is Jrel_inf_b in component ryr (millimolar_per_millisecond).
 * ALGEBRAIC[Jrel_inf] is Jrel_inf in component ryr (millimolar_per_millisecond).
 * ALGEBRAIC[tau_rel_b] is tau_rel_b in component ryr (millisecond).
 * ALGEBRAIC[tau_rel] is tau_rel in component ryr (millisecond).
 * STATES[Jrel_np] is Jrel_np in component ryr (millimolar_per_millisecond).
 * CONSTANTS[btp] is btp in component ryr (millisecond).
 * CONSTANTS[a_relp] is a_relp in component ryr (millimolar_per_millisecond).
 * ALGEBRAIC[Jrel_infp_b] is Jrel_infp_b in component ryr (millimolar_per_millisecond).
 * ALGEBRAIC[Jrel_infp] is Jrel_infp in component ryr (millimolar_per_millisecond).
 * ALGEBRAIC[tau_relp_b] is tau_relp_b in component ryr (millisecond).
 * ALGEBRAIC[tau_relp] is tau_relp in component ryr (millisecond).
 * STATES[Jrel_p] is Jrel_p in component ryr (millimolar_per_millisecond).
 * CONSTANTS[cajsr_half] is cajsr_half in component ryr (millimolar).
 * ALGEBRAIC[fJrelp] is fJrelp in component ryr (dimensionless).
 * CONSTANTS[Jrel_b] is Jrel_b in component ryr (dimensionless).
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
 * RATES[h] is d/dt h in component INa (dimensionless).
 * RATES[j] is d/dt j in component INa (dimensionless).
 * RATES[hp] is d/dt hp in component INa (dimensionless).
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
 * RATES[nca_ss] is d/dt nca_ss in component ICaL (dimensionless).
 * RATES[nca_i] is d/dt nca_i in component ICaL (dimensionless).
 * RATES[C3] is d/dt C3 in component IKr (dimensionless).
 * RATES[C2] is d/dt C2 in component IKr (dimensionless).
 * RATES[C1] is d/dt C1 in component IKr (dimensionless).
 * RATES[O] is d/dt O in component IKr (dimensionless).
 * RATES[I] is d/dt I in component IKr (dimensionless).
 * RATES[xs1] is d/dt xs1 in component IKs (dimensionless).
 * RATES[xs2] is d/dt xs2 in component IKs (dimensionless).
 * RATES[Jrel_np] is d/dt Jrel_np in component ryr (millimolar_per_millisecond).
 * RATES[Jrel_p] is d/dt Jrel_p in component ryr (millimolar_per_millisecond).
 */


// Tomek_model::Tomek_model()
// {
// algebraic_size = 223;
// constants_size = 163;
// states_size = 43;
// rates_size = 43;
// }

// Tomek_model::~Tomek_model()
// {

// }

__device__ void ___initConsts(double *CONSTANTS, double *STATES, double type, double bcl, int offset)
{
int constants_size = 163;
int states_size = 43;

CONSTANTS[(constants_size * offset) + celltype] = type;
CONSTANTS[(constants_size * offset) + nao] = 140.0;
CONSTANTS[(constants_size * offset) + cao] = 1.8;
CONSTANTS[(constants_size * offset) + ko] = 5.0;
CONSTANTS[(constants_size * offset) + clo] = 150.0;
CONSTANTS[(constants_size * offset) + R] = 8314;
CONSTANTS[(constants_size * offset) + T] = 310;
CONSTANTS[(constants_size * offset) + F] = 96485;
CONSTANTS[(constants_size * offset) + zna] = 1;
CONSTANTS[(constants_size * offset) + zca] = 2;
CONSTANTS[(constants_size * offset) + zk] = 1;
CONSTANTS[(constants_size * offset) + zcl] = -1;
CONSTANTS[(constants_size * offset) + L] = 0.01;
CONSTANTS[(constants_size * offset) + rad] = 0.0011;
STATES[(states_size * offset) + V] = (CONSTANTS[(constants_size * offset) + celltype]==1.00000 ?  -89.14 : CONSTANTS[(constants_size * offset) + celltype]==2.00000 ?  -89.1704 : -88.7638);
CONSTANTS[(constants_size * offset) + stim_start] = 10;
CONSTANTS[(constants_size * offset) + i_Stim_End] = 100000000000000000;
CONSTANTS[(constants_size * offset) + i_Stim_Amplitude] = -53;
CONSTANTS[(constants_size * offset) + BCL] = bcl;
CONSTANTS[(constants_size * offset) + i_Stim_PulseDuration] = 1.0;
CONSTANTS[(constants_size * offset) + KmCaMK] = 0.15;
CONSTANTS[(constants_size * offset) + aCaMK] = 0.05;
CONSTANTS[(constants_size * offset) + bCaMK] = 0.00068;
CONSTANTS[(constants_size * offset) + CaMKo] = 0.05;
CONSTANTS[(constants_size * offset) + KmCaM] = 0.0015;
STATES[(states_size * offset) + CaMKt] = (CONSTANTS[(constants_size * offset) + celltype]==1.00000 ? 0.0129 : CONSTANTS[(constants_size * offset) + celltype]==2.00000 ? 0.0192 : 0.0111);
STATES[(states_size * offset) + cass] = (CONSTANTS[(constants_size * offset) + celltype]==1.00000 ? 5.77E-05 : CONSTANTS[(constants_size * offset) + celltype]==2.00000 ? 6.58E-05 : 7.0305e-5);
CONSTANTS[(constants_size * offset) + cmdnmax_b] = 0.05;
CONSTANTS[(constants_size * offset) + kmcmdn] = 0.00238;
CONSTANTS[(constants_size * offset) + trpnmax] = 0.07;
CONSTANTS[(constants_size * offset) + kmtrpn] = 0.0005;
CONSTANTS[(constants_size * offset) + BSRmax] = 0.047;
CONSTANTS[(constants_size * offset) + KmBSR] = 0.00087;
CONSTANTS[(constants_size * offset) + BSLmax] = 1.124;
CONSTANTS[(constants_size * offset) + KmBSL] = 0.0087;
CONSTANTS[(constants_size * offset) + csqnmax] = 10;
CONSTANTS[(constants_size * offset) + kmcsqn] = 0.8;
STATES[(states_size * offset) + nai] = (CONSTANTS[(constants_size * offset) + celltype]==1.00000 ? 12.1025 : CONSTANTS[(constants_size * offset) + celltype]==2.00000 ? 15.0038 : 12.1025);
STATES[(states_size * offset) + nass] = (CONSTANTS[(constants_size * offset) + celltype]==1.00000 ? 12.8366 : CONSTANTS[(constants_size * offset) + celltype]==2.00000 ? 15.0043 : 12.1029);
STATES[(states_size * offset) + ki] = (CONSTANTS[(constants_size * offset) + celltype]==1.00000 ? 142.6951 : CONSTANTS[(constants_size * offset) + celltype]==2.00000 ? 143.0403 : 142.3002);
STATES[(states_size * offset) + kss] = (CONSTANTS[(constants_size * offset) + celltype]==1.00000 ? 142.6951 : CONSTANTS[(constants_size * offset) + celltype]==2.00000 ? 143.0402 : 142.3002);
STATES[(states_size * offset) + cansr] = (CONSTANTS[(constants_size * offset) + celltype]==1.00000 ? 1.8119 : CONSTANTS[(constants_size * offset) + celltype]==2.00000 ? 1.9557 : 1.5211);
STATES[(states_size * offset) + cajsr] = (CONSTANTS[(constants_size * offset) + celltype]==1.00000 ? 1.8102 : CONSTANTS[(constants_size * offset) + celltype]==2.00000 ? 1.9593 : 1.5214);
STATES[(states_size * offset) + cai] = (CONSTANTS[(constants_size * offset) + celltype]==1.00000 ? 6.63E-05 : CONSTANTS[(constants_size * offset) + celltype]==2.00000 ? 8.17E-05 : 8.1583e-05);
CONSTANTS[(constants_size * offset) + cli] = 24.0;
CONSTANTS[(constants_size * offset) + PKNa] = 0.01833;
CONSTANTS[(constants_size * offset) + gkatp] = 4.3195;
CONSTANTS[(constants_size * offset) + fkatp] = 0.0;
CONSTANTS[(constants_size * offset) + K_o_n] = 5;
CONSTANTS[(constants_size * offset) + A_atp] = 2;
CONSTANTS[(constants_size * offset) + K_atp] = 0.25;
STATES[(states_size * offset) + m] = (CONSTANTS[(constants_size * offset) + celltype]==1.00000 ? 7.43E-04 : CONSTANTS[(constants_size * offset) + celltype]==2.00000 ? 7.38E-04 : 8.0572e-4);
STATES[(states_size * offset) + h] = (CONSTANTS[(constants_size * offset) + celltype]==1.00000 ? 0.836 : CONSTANTS[(constants_size * offset) + celltype]==2.00000 ? 0.8365 : 0.8286);
STATES[(states_size * offset) + j] = (CONSTANTS[(constants_size * offset) + celltype]==1.00000 ? 0.8359 : CONSTANTS[(constants_size * offset) + celltype]==2.00000 ? 0.8363 : 0.8284);
STATES[(states_size * offset) + hp] = (CONSTANTS[(constants_size * offset) + celltype]==1.00000 ? 0.6828 : CONSTANTS[(constants_size * offset) + celltype]==2.00000 ? 0.6838 : 0.6707);
STATES[(states_size * offset) + jp] = (CONSTANTS[(constants_size * offset) + celltype]==1.00000 ? 0.8357 : CONSTANTS[(constants_size * offset) + celltype]==2.00000 ? 0.8358 : 0.8281);
CONSTANTS[(constants_size * offset) + GNa] = 11.7802;
STATES[(states_size * offset) + mL] = (CONSTANTS[(constants_size * offset) + celltype]==1.00000 ? 1.52E-04 : CONSTANTS[(constants_size * offset) + celltype]==2.00000 ? 1.51E-04 : 1.629e-4);
CONSTANTS[(constants_size * offset) + thL] = 200;
STATES[(states_size * offset) + hL] = (CONSTANTS[(constants_size * offset) + celltype]==1.00000 ? 0.5401 : CONSTANTS[(constants_size * offset) + celltype]==2.00000 ? 0.5327 : 0.5255);
STATES[(states_size * offset) + hLp] = (CONSTANTS[(constants_size * offset) + celltype]==1.00000 ? 0.3034 : CONSTANTS[(constants_size * offset) + celltype]==2.00000 ? 0.2834 : 0.2872);
CONSTANTS[(constants_size * offset) + GNaL_b] = 0.0279;
CONSTANTS[(constants_size * offset) + Gto_b] = 0.16;
STATES[(states_size * offset) + a] = (CONSTANTS[(constants_size * offset) + celltype]==1.00000 ? 9.27E-04 : CONSTANTS[(constants_size * offset) + celltype]==2.00000 ? 9.25E-04 : 9.5098e-4);
CONSTANTS[(constants_size * offset) + EKshift] = 0;
STATES[(states_size * offset) + iF] = 0.9996;
STATES[(states_size * offset) + iS] = (CONSTANTS[(constants_size * offset) + celltype]==1.00000 ? 0.9996 : CONSTANTS[(constants_size * offset) + celltype]==2.00000 ? 0.5671 : 0.5936);
STATES[(states_size * offset) + ap] = (CONSTANTS[(constants_size * offset) + celltype]==1.00000 ? 4.72E-04 : CONSTANTS[(constants_size * offset) + celltype]==2.00000 ? 4.71E-04 : 4.8454e-4);
STATES[(states_size * offset) + iFp] = 0.9996;
STATES[(states_size * offset) + iSp] = (CONSTANTS[(constants_size * offset) + celltype]==1.00000 ? 0.9996 : CONSTANTS[(constants_size * offset) + celltype]==2.00000 ? 0.6261 :0.6538);
CONSTANTS[(constants_size * offset) + Kmn] = 0.002;
CONSTANTS[(constants_size * offset) + k2n] = 500;
CONSTANTS[(constants_size * offset) + PCa_b] = 8.3757e-05;
STATES[(states_size * offset) + d] = (CONSTANTS[(constants_size * offset) + celltype]==1.00000 ? 0.0 : CONSTANTS[(constants_size * offset) + celltype]==2.00000 ? 0.0 : 8.1084e-9);
CONSTANTS[(constants_size * offset) + Aff] = 0.6;
STATES[(states_size * offset) + ff] = 1.0;
STATES[(states_size * offset) + fs] = (CONSTANTS[(constants_size * offset) + celltype]==1.00000 ? 0.9485 : CONSTANTS[(constants_size * offset) + celltype]==2.00000 ? 0.92 : 0.939);
STATES[(states_size * offset) + fcaf] = 1.0;
STATES[(states_size * offset) + fcas] = 0.9999;
STATES[(states_size * offset) + jca] = 1.0;
STATES[(states_size * offset) + ffp] = 1.0;
STATES[(states_size * offset) + fcafp] = 1.0;
STATES[(states_size * offset) + nca_ss] = (CONSTANTS[(constants_size * offset) + celltype]==1.00000 ? 3.09E-04 : CONSTANTS[(constants_size * offset) + celltype]==2.00000 ? 5.14E-04 : 6.6462e-4);
STATES[(states_size * offset) + nca_i] = (CONSTANTS[(constants_size * offset) + celltype]==1.00000 ? 5.30E-04 : CONSTANTS[(constants_size * offset) + celltype]==2.00000 ? 0.0012 : 0.0012);
CONSTANTS[(constants_size * offset) + tjca] = 75;
CONSTANTS[(constants_size * offset) + vShift] = 0;
CONSTANTS[(constants_size * offset) + offset] = 0;
CONSTANTS[(constants_size * offset) + dielConstant] = 74;
CONSTANTS[(constants_size * offset) + ICaL_fractionSS] = 0.8;
CONSTANTS[(constants_size * offset) + GKr_b] = 0.0321;
STATES[(states_size * offset) + C1] = (CONSTANTS[(constants_size * offset) + celltype]==1.00000 ? 6.79E-04 : CONSTANTS[(constants_size * offset) + celltype]==2.00000 ? 6.96E-04 : 7.0344e-4);
STATES[(states_size * offset) + C2] = (CONSTANTS[(constants_size * offset) + celltype]==1.00000 ? 8.29E-04 : CONSTANTS[(constants_size * offset) + celltype]==2.00000 ? 8.27E-04 : 8.5109e-4);
STATES[(states_size * offset) + C3] = (CONSTANTS[(constants_size * offset) + celltype]==1.00000 ? 0.9982 : CONSTANTS[(constants_size * offset) + celltype]==2.00000 ? 0.9979 : 0.9981);
STATES[(states_size * offset) + I] = (CONSTANTS[(constants_size * offset) + celltype]==1.00000 ? 9.54E-06 : CONSTANTS[(constants_size * offset) + celltype]==2.00000 ? 1.88E-05 : 1.3289e-5);
STATES[(states_size * offset) + O] = (CONSTANTS[(constants_size * offset) + celltype]==1.00000 ? 2.76E-04 : CONSTANTS[(constants_size * offset) + celltype]==2.00000 ? 5.42E-04 : 3.7585e-4);
CONSTANTS[(constants_size * offset) + alpha_1] = 0.154375;
CONSTANTS[(constants_size * offset) + beta_1] = 0.1911;
CONSTANTS[(constants_size * offset) + GKs_b] = 0.0011;
STATES[(states_size * offset) + xs1] = (CONSTANTS[(constants_size * offset) + celltype]==1.00000 ? 0.2309 : CONSTANTS[(constants_size * offset) + celltype]==2.00000 ? 0.2653 : 0.248);
STATES[(states_size * offset) + xs2] = (CONSTANTS[(constants_size * offset) + celltype]==1.00000 ? 1.70E-04 : CONSTANTS[(constants_size * offset) + celltype]==2.00000 ? 1.69E-04 : 1.7707e-4);
CONSTANTS[(constants_size * offset) + GK1_b] = 0.6992;
CONSTANTS[(constants_size * offset) + INaCa_fractionSS] = 0.35;
CONSTANTS[(constants_size * offset) + kna1] = 15;
CONSTANTS[(constants_size * offset) + kna2] = 5;
CONSTANTS[(constants_size * offset) + kna3] = 88.12;
CONSTANTS[(constants_size * offset) + kasymm] = 12.5;
CONSTANTS[(constants_size * offset) + wna] = 6e4;
CONSTANTS[(constants_size * offset) + wca] = 6e4;
CONSTANTS[(constants_size * offset) + wnaca] = 5e3;
CONSTANTS[(constants_size * offset) + kcaon] = 1.5e6;
CONSTANTS[(constants_size * offset) + kcaoff] = 5e3;
CONSTANTS[(constants_size * offset) + qna] = 0.5224;
CONSTANTS[(constants_size * offset) + qca] = 0.167;
CONSTANTS[(constants_size * offset) + KmCaAct] = 150e-6;
CONSTANTS[(constants_size * offset) + Gncx_b] = 0.0034;
CONSTANTS[(constants_size * offset) + k1p] = 949.5;
CONSTANTS[(constants_size * offset) + k1m] = 182.4;
CONSTANTS[(constants_size * offset) + k2p] = 687.2;
CONSTANTS[(constants_size * offset) + k2m] = 39.4;
CONSTANTS[(constants_size * offset) + k3p] = 1899;
CONSTANTS[(constants_size * offset) + k3m] = 79300;
CONSTANTS[(constants_size * offset) + k4p] = 639;
CONSTANTS[(constants_size * offset) + k4m] = 40;
CONSTANTS[(constants_size * offset) + Knai0] = 9.073;
CONSTANTS[(constants_size * offset) + Knao0] = 27.78;
CONSTANTS[(constants_size * offset) + delta] = -0.155;
CONSTANTS[(constants_size * offset) + Kki] = 0.5;
CONSTANTS[(constants_size * offset) + Kko] = 0.3582;
CONSTANTS[(constants_size * offset) + MgADP] = 0.05;
CONSTANTS[(constants_size * offset) + MgATP] = 9.8;
CONSTANTS[(constants_size * offset) + Kmgatp] = 1.698e-7;
CONSTANTS[(constants_size * offset) + H] = 1e-7;
CONSTANTS[(constants_size * offset) + eP] = 4.2;
CONSTANTS[(constants_size * offset) + Khp] = 1.698e-7;
CONSTANTS[(constants_size * offset) + Knap] = 224;
CONSTANTS[(constants_size * offset) + Kxkur] = 292;
CONSTANTS[(constants_size * offset) + Pnak_b] = 15.4509;
CONSTANTS[(constants_size * offset) + GKb_b] = 0.0189;
CONSTANTS[(constants_size * offset) + PNab] = 1.9239e-09;
CONSTANTS[(constants_size * offset) + PCab] = 5.9194e-08;
CONSTANTS[(constants_size * offset) + GpCa] = 5e-04;
CONSTANTS[(constants_size * offset) + KmCap] = 0.0005;
CONSTANTS[(constants_size * offset) + GClCa] = 0.2843;
CONSTANTS[(constants_size * offset) + GClb] = 1.98e-3;
CONSTANTS[(constants_size * offset) + KdClCa] = 0.1;
CONSTANTS[(constants_size * offset) + Fjunc] = 1;
CONSTANTS[(constants_size * offset) + tauNa] = 2.0;
CONSTANTS[(constants_size * offset) + tauK] = 2.0;
CONSTANTS[(constants_size * offset) + tauCa] = 0.2;
CONSTANTS[(constants_size * offset) + bt] = 4.75;
STATES[(states_size * offset) + Jrel_np] = (CONSTANTS[(constants_size * offset) + celltype]==1.00000 ? 2.82E-24 : CONSTANTS[(constants_size * offset) + celltype]==2.00000 ? 0. : 1.6129e-22);
STATES[(states_size * offset) + Jrel_p] = (CONSTANTS[(constants_size * offset) + celltype]==1.00000 ? 0. : CONSTANTS[(constants_size * offset) + celltype]==2.00000 ? 0. : 1.2475e-20);
CONSTANTS[(constants_size * offset) + cajsr_half] = 1.7;
CONSTANTS[(constants_size * offset) + Jrel_b] = 1.5378;
CONSTANTS[(constants_size * offset) + Jup_b] = 1.0;
CONSTANTS[(constants_size * offset) + vcell] =  1000.00*3.14000*CONSTANTS[(constants_size * offset) + rad]*CONSTANTS[(constants_size * offset) + rad]*CONSTANTS[(constants_size * offset) + L];
CONSTANTS[(constants_size * offset) + cmdnmax] = (CONSTANTS[(constants_size * offset) + celltype]==1.00000 ?  CONSTANTS[(constants_size * offset) + cmdnmax_b]*1.30000 : CONSTANTS[(constants_size * offset) + cmdnmax_b]);
CONSTANTS[(constants_size * offset) + ECl] =  (( CONSTANTS[(constants_size * offset) + R]*CONSTANTS[(constants_size * offset) + T])/( CONSTANTS[(constants_size * offset) + zcl]*CONSTANTS[(constants_size * offset) + F]))*log(CONSTANTS[(constants_size * offset) + clo]/CONSTANTS[(constants_size * offset) + cli]);
CONSTANTS[(constants_size * offset) + akik] = pow(CONSTANTS[(constants_size * offset) + ko]/CONSTANTS[(constants_size * offset) + K_o_n], 0.240000);
CONSTANTS[(constants_size * offset) + bkik] = 1.00000/(1.00000+pow(CONSTANTS[(constants_size * offset) + A_atp]/CONSTANTS[(constants_size * offset) + K_atp], 2.00000));
CONSTANTS[(constants_size * offset) + thLp] =  3.00000*CONSTANTS[(constants_size * offset) + thL];
CONSTANTS[(constants_size * offset) + GNaL] = (CONSTANTS[(constants_size * offset) + celltype]==1.00000 ?  CONSTANTS[(constants_size * offset) + GNaL_b]*0.600000 : CONSTANTS[(constants_size * offset) + GNaL_b]);
CONSTANTS[(constants_size * offset) + Gto] = (CONSTANTS[(constants_size * offset) + celltype]==1.00000 ?  CONSTANTS[(constants_size * offset) + Gto_b]*2.00000 : CONSTANTS[(constants_size * offset) + celltype]==2.00000 ?  CONSTANTS[(constants_size * offset) + Gto_b]*2.00000 : CONSTANTS[(constants_size * offset) + Gto_b]);
CONSTANTS[(constants_size * offset) + Afs] = 1.00000 - CONSTANTS[(constants_size * offset) + Aff];
CONSTANTS[(constants_size * offset) + PCa] = (CONSTANTS[(constants_size * offset) + celltype]==1.00000 ?  CONSTANTS[(constants_size * offset) + PCa_b]*1.20000 : CONSTANTS[(constants_size * offset) + celltype]==2.00000 ?  CONSTANTS[(constants_size * offset) + PCa_b]*2.00000 : CONSTANTS[(constants_size * offset) + PCa_b]);
CONSTANTS[(constants_size * offset) + Io] = ( 0.500000*(CONSTANTS[(constants_size * offset) + nao]+CONSTANTS[(constants_size * offset) + ko]+CONSTANTS[(constants_size * offset) + clo]+ 4.00000*CONSTANTS[(constants_size * offset) + cao]))/1000.00;
CONSTANTS[(constants_size * offset) + GKr] = (CONSTANTS[(constants_size * offset) + celltype]==1.00000 ?  CONSTANTS[(constants_size * offset) + GKr_b]*1.30000 : CONSTANTS[(constants_size * offset) + celltype]==2.00000 ?  CONSTANTS[(constants_size * offset) + GKr_b]*0.800000 : CONSTANTS[(constants_size * offset) + GKr_b]);
CONSTANTS[(constants_size * offset) + GKs] = (CONSTANTS[(constants_size * offset) + celltype]==1.00000 ?  CONSTANTS[(constants_size * offset) + GKs_b]*1.40000 : CONSTANTS[(constants_size * offset) + GKs_b]);
CONSTANTS[(constants_size * offset) + GK1] = (CONSTANTS[(constants_size * offset) + celltype]==1.00000 ?  CONSTANTS[(constants_size * offset) + GK1_b]*1.20000 : CONSTANTS[(constants_size * offset) + celltype]==2.00000 ?  CONSTANTS[(constants_size * offset) + GK1_b]*1.30000 : CONSTANTS[(constants_size * offset) + GK1_b]);
CONSTANTS[(constants_size * offset) + GKb] = (CONSTANTS[(constants_size * offset) + celltype]==1.00000 ?  CONSTANTS[(constants_size * offset) + GKb_b]*0.600000 : CONSTANTS[(constants_size * offset) + GKb_b]);
CONSTANTS[(constants_size * offset) + a_rel] = ( 0.500000*CONSTANTS[(constants_size * offset) + bt])/1.00000;
CONSTANTS[(constants_size * offset) + btp] =  1.25000*CONSTANTS[(constants_size * offset) + bt];
CONSTANTS[(constants_size * offset) + upScale] = (CONSTANTS[(constants_size * offset) + celltype]==1.00000 ? 1.30000 : 1.00000);
CONSTANTS[(constants_size * offset) + Ageo] =  2.00000*3.14000*CONSTANTS[(constants_size * offset) + rad]*CONSTANTS[(constants_size * offset) + rad]+ 2.00000*3.14000*CONSTANTS[(constants_size * offset) + rad]*CONSTANTS[(constants_size * offset) + L];
CONSTANTS[(constants_size * offset) + PCap] =  1.10000*CONSTANTS[(constants_size * offset) + PCa];
CONSTANTS[(constants_size * offset) + PCaNa] =  0.00125000*CONSTANTS[(constants_size * offset) + PCa];
CONSTANTS[(constants_size * offset) + PCaK] =  0.000357400*CONSTANTS[(constants_size * offset) + PCa];
CONSTANTS[(constants_size * offset) + constA] =  1.82000e+06*pow( CONSTANTS[(constants_size * offset) + dielConstant]*CONSTANTS[(constants_size * offset) + T], - 1.50000);
CONSTANTS[(constants_size * offset) + a_relp] = ( 0.500000*CONSTANTS[(constants_size * offset) + btp])/1.00000;
CONSTANTS[(constants_size * offset) + Acap] =  2.00000*CONSTANTS[(constants_size * offset) + Ageo];
CONSTANTS[(constants_size * offset) + PCaNap] =  0.00125000*CONSTANTS[(constants_size * offset) + PCap];
CONSTANTS[(constants_size * offset) + PCaKp] =  0.000357400*CONSTANTS[(constants_size * offset) + PCap];
CONSTANTS[(constants_size * offset) + gamma_cao] = exp( - CONSTANTS[(constants_size * offset) + constA]*4.00000*( pow(CONSTANTS[(constants_size * offset) + Io], 1.0 / 2)/(1.00000+ pow(CONSTANTS[(constants_size * offset) + Io], 1.0 / 2)) -  0.300000*CONSTANTS[(constants_size * offset) + Io]));
CONSTANTS[(constants_size * offset) + gamma_nao] = exp( - CONSTANTS[(constants_size * offset) + constA]*1.00000*( pow(CONSTANTS[(constants_size * offset) + Io], 1.0 / 2)/(1.00000+ pow(CONSTANTS[(constants_size * offset) + Io], 1.0 / 2)) -  0.300000*CONSTANTS[(constants_size * offset) + Io]));
CONSTANTS[(constants_size * offset) + gamma_ko] = exp( - CONSTANTS[(constants_size * offset) + constA]*1.00000*( pow(CONSTANTS[(constants_size * offset) + Io], 1.0 / 2)/(1.00000+ pow(CONSTANTS[(constants_size * offset) + Io], 1.0 / 2)) -  0.300000*CONSTANTS[(constants_size * offset) + Io]));
CONSTANTS[(constants_size * offset) + vmyo] =  0.680000*CONSTANTS[(constants_size * offset) + vcell];
CONSTANTS[(constants_size * offset) + vnsr] =  0.0552000*CONSTANTS[(constants_size * offset) + vcell];
CONSTANTS[(constants_size * offset) + vjsr] =  0.00480000*CONSTANTS[(constants_size * offset) + vcell];
CONSTANTS[(constants_size * offset) + vss] =  0.0200000*CONSTANTS[(constants_size * offset) + vcell];
CONSTANTS[(constants_size * offset) + h10_i] = CONSTANTS[(constants_size * offset) + kasymm]+1.00000+ (CONSTANTS[(constants_size * offset) + nao]/CONSTANTS[(constants_size * offset) + kna1])*(1.00000+CONSTANTS[(constants_size * offset) + nao]/CONSTANTS[(constants_size * offset) + kna2]);
CONSTANTS[(constants_size * offset) + h11_i] = ( CONSTANTS[(constants_size * offset) + nao]*CONSTANTS[(constants_size * offset) + nao])/( CONSTANTS[(constants_size * offset) + h10_i]*CONSTANTS[(constants_size * offset) + kna1]*CONSTANTS[(constants_size * offset) + kna2]);
CONSTANTS[(constants_size * offset) + h12_i] = 1.00000/CONSTANTS[(constants_size * offset) + h10_i];
CONSTANTS[(constants_size * offset) + k1_i] =  CONSTANTS[(constants_size * offset) + h12_i]*CONSTANTS[(constants_size * offset) + cao]*CONSTANTS[(constants_size * offset) + kcaon];
CONSTANTS[(constants_size * offset) + k2_i] = CONSTANTS[(constants_size * offset) + kcaoff];
CONSTANTS[(constants_size * offset) + k5_i] = CONSTANTS[(constants_size * offset) + kcaoff];
CONSTANTS[(constants_size * offset) + Gncx] = (CONSTANTS[(constants_size * offset) + celltype]==1.00000 ?  CONSTANTS[(constants_size * offset) + Gncx_b]*1.10000 : CONSTANTS[(constants_size * offset) + celltype]==2.00000 ?  CONSTANTS[(constants_size * offset) + Gncx_b]*1.40000 : CONSTANTS[(constants_size * offset) + Gncx_b]);
CONSTANTS[(constants_size * offset) + h10_ss] = CONSTANTS[(constants_size * offset) + kasymm]+1.00000+ (CONSTANTS[(constants_size * offset) + nao]/CONSTANTS[(constants_size * offset) + kna1])*(1.00000+CONSTANTS[(constants_size * offset) + nao]/CONSTANTS[(constants_size * offset) + kna2]);
CONSTANTS[(constants_size * offset) + h11_ss] = ( CONSTANTS[(constants_size * offset) + nao]*CONSTANTS[(constants_size * offset) + nao])/( CONSTANTS[(constants_size * offset) + h10_ss]*CONSTANTS[(constants_size * offset) + kna1]*CONSTANTS[(constants_size * offset) + kna2]);
CONSTANTS[(constants_size * offset) + h12_ss] = 1.00000/CONSTANTS[(constants_size * offset) + h10_ss];
CONSTANTS[(constants_size * offset) + k1_ss] =  CONSTANTS[(constants_size * offset) + h12_ss]*CONSTANTS[(constants_size * offset) + cao]*CONSTANTS[(constants_size * offset) + kcaon];
CONSTANTS[(constants_size * offset) + k2_ss] = CONSTANTS[(constants_size * offset) + kcaoff];
CONSTANTS[(constants_size * offset) + k5_ss] = CONSTANTS[(constants_size * offset) + kcaoff];
CONSTANTS[(constants_size * offset) + b1] =  CONSTANTS[(constants_size * offset) + k1m]*CONSTANTS[(constants_size * offset) + MgADP];
CONSTANTS[(constants_size * offset) + a2] = CONSTANTS[(constants_size * offset) + k2p];
CONSTANTS[(constants_size * offset) + a4] = (( CONSTANTS[(constants_size * offset) + k4p]*CONSTANTS[(constants_size * offset) + MgATP])/CONSTANTS[(constants_size * offset) + Kmgatp])/(1.00000+CONSTANTS[(constants_size * offset) + MgATP]/CONSTANTS[(constants_size * offset) + Kmgatp]);
CONSTANTS[(constants_size * offset) + Pnak] = (CONSTANTS[(constants_size * offset) + celltype]==1.00000 ?  CONSTANTS[(constants_size * offset) + Pnak_b]*0.900000 : CONSTANTS[(constants_size * offset) + celltype]==2.00000 ?  CONSTANTS[(constants_size * offset) + Pnak_b]*0.700000 : CONSTANTS[(constants_size * offset) + Pnak_b]);
}

__device__ void applyDrugEffect(double *CONSTANTS, double conc, double *hill, double epsilon, int offset)
{

int constants_size = 163;

CONSTANTS[(constants_size * offset) + GK1] = CONSTANTS[(constants_size * offset) + GK1] * ((hill[2] > 10E-14 && hill[3] > 10E-14) ? 1./(1.+pow(conc/hill[2],hill[3])) : 1.);
CONSTANTS[(constants_size * offset) + GKr] = CONSTANTS[(constants_size * offset) + GKr] * ((hill[12] > 10E-14 && hill[13] > 10E-14) ? 1./(1.+pow(conc/hill[12],hill[13])) : 1.);
CONSTANTS[(constants_size * offset) + GKs] = CONSTANTS[(constants_size * offset) + GKs] * ((hill[4] > 10E-14 && hill[5] > 10E-14) ? 1./(1.+pow(conc/hill[4],hill[5])) : 1.);
CONSTANTS[(constants_size * offset) + GNaL] = CONSTANTS[(constants_size * offset) + GNaL] * ((hill[8] > 10E-14 && hill[9] > 10E-14) ? 1./(1.+pow(conc/hill[8],hill[9])) : 1.);
CONSTANTS[(constants_size * offset) + GNa] = CONSTANTS[(constants_size * offset) + GNa] * ((hill[6] > 10E-14 && hill[7] > 10E-14) ? 1./(1.+pow(conc/hill[6],hill[7])) : 1.);
CONSTANTS[(constants_size * offset) + Gto] = CONSTANTS[(constants_size * offset) + Gto] * ((hill[10] > 10E-14 && hill[11] > 10E-14) ? 1./(1.+pow(conc/hill[10],hill[11])) : 1.);
CONSTANTS[(constants_size * offset) + PCa] = CONSTANTS[(constants_size * offset) + PCa] * ( (hill[0] > 10E-14 && hill[1] > 10E-14) ? 1./(1.+pow(conc/hill[0],hill[1])) : 1.);
}

// void Tomek_model::initConsts()
// {
// 	___initConsts(0.);
// }

// void Tomek_model::initConsts(double type)
// {
// 	___initConsts(type);
// }

__device__ void initConsts(double *CONSTANTS, double *STATES, double type, double conc, double *hill, double *cvar, bool is_dutta, bool is_cvar, double bcl, double epsilon, int offset)
{
	___initConsts(CONSTANTS, STATES, type, bcl, offset);
	
	applyDrugEffect(CONSTANTS, conc, hill, epsilon, offset);
	
}

__device__ void computeRates( double TIME, double *CONSTANTS, double *RATES, double *STATES, double *ALGEBRAIC, int offset )
{
int algebraic_size = 223;
int constants_size = 163;
int states_size = 43;

ALGEBRAIC[(algebraic_size * offset) + hLss] = 1.00000/(1.00000+exp((STATES[(states_size * offset) + V]+87.6100)/7.48800));
ALGEBRAIC[(algebraic_size * offset) + hLssp] = 1.00000/(1.00000+exp((STATES[(states_size * offset) + V]+93.8100)/7.48800));
ALGEBRAIC[(algebraic_size * offset) + jcass] = 1.00000/(1.00000+exp((STATES[(states_size * offset) + V]+18.0800)/2.79160));
ALGEBRAIC[(algebraic_size * offset) + mss] = 1.00000/pow(1.00000+exp(- (STATES[(states_size * offset) + V]+56.8600)/9.03000), 2.00000);
ALGEBRAIC[(algebraic_size * offset) + tm] =  0.129200*exp(- pow((STATES[(states_size * offset) + V]+45.7900)/15.5400, 2.00000))+ 0.0648700*exp(- pow((STATES[(states_size * offset) + V] - 4.82300)/51.1200, 2.00000));
ALGEBRAIC[(algebraic_size * offset) + mLss] = 1.00000/(1.00000+exp(- (STATES[(states_size * offset) + V]+42.8500)/5.26400));
ALGEBRAIC[(algebraic_size * offset) + tmL] =  0.129200*exp(- pow((STATES[(states_size * offset) + V]+45.7900)/15.5400, 2.00000))+ 0.0648700*exp(- pow((STATES[(states_size * offset) + V] - 4.82300)/51.1200, 2.00000));
ALGEBRAIC[(algebraic_size * offset) + ass] = 1.00000/(1.00000+exp(- ((STATES[(states_size * offset) + V]+CONSTANTS[(constants_size * offset) + EKshift]) - 14.3400)/14.8200));
ALGEBRAIC[(algebraic_size * offset) + ta] = 1.05150/(1.00000/( 1.20890*(1.00000+exp(- ((STATES[(states_size * offset) + V]+CONSTANTS[(constants_size * offset) + EKshift]) - 18.4099)/29.3814)))+3.50000/(1.00000+exp((STATES[(states_size * offset) + V]+CONSTANTS[(constants_size * offset) + EKshift]+100.000)/29.3814)));
ALGEBRAIC[(algebraic_size * offset) + dss] = (STATES[(states_size * offset) + V]>=31.4978 ? 1.00000 :  1.07630*exp( - 1.00700*exp( - 0.0829000*STATES[(states_size * offset) + V])));
ALGEBRAIC[(algebraic_size * offset) + td] = CONSTANTS[(constants_size * offset) + offset]+0.600000+1.00000/(exp( - 0.0500000*(STATES[(states_size * offset) + V]+CONSTANTS[(constants_size * offset) + vShift]+6.00000))+exp( 0.0900000*(STATES[(states_size * offset) + V]+CONSTANTS[(constants_size * offset) + vShift]+14.0000)));
ALGEBRAIC[(algebraic_size * offset) + fss] = 1.00000/(1.00000+exp((STATES[(states_size * offset) + V]+19.5800)/3.69600));
ALGEBRAIC[(algebraic_size * offset) + tff] = 7.00000+1.00000/( 0.00450000*exp(- (STATES[(states_size * offset) + V]+20.0000)/10.0000)+ 0.00450000*exp((STATES[(states_size * offset) + V]+20.0000)/10.0000));
ALGEBRAIC[(algebraic_size * offset) + tfs] = 1000.00+1.00000/( 3.50000e-05*exp(- (STATES[(states_size * offset) + V]+5.00000)/4.00000)+ 3.50000e-05*exp((STATES[(states_size * offset) + V]+5.00000)/6.00000));
ALGEBRAIC[(algebraic_size * offset) + km2n] =  STATES[(states_size * offset) + jca]*1.00000;
ALGEBRAIC[(algebraic_size * offset) + anca_ss] = 1.00000/(CONSTANTS[(constants_size * offset) + k2n]/ALGEBRAIC[(algebraic_size * offset) + km2n]+pow(1.00000+CONSTANTS[(constants_size * offset) + Kmn]/STATES[(states_size * offset) + cass], 4.00000));
ALGEBRAIC[(algebraic_size * offset) + anca_i] = 1.00000/(CONSTANTS[(constants_size * offset) + k2n]/ALGEBRAIC[(algebraic_size * offset) + km2n]+pow(1.00000+CONSTANTS[(constants_size * offset) + Kmn]/STATES[(states_size * offset) + cai], 4.00000));
ALGEBRAIC[(algebraic_size * offset) + xs1ss] = 1.00000/(1.00000+exp(- (STATES[(states_size * offset) + V]+11.6000)/8.93200));
ALGEBRAIC[(algebraic_size * offset) + txs1] = 817.300+1.00000/( 0.000232600*exp((STATES[(states_size * offset) + V]+48.2800)/17.8000)+ 0.00129200*exp(- (STATES[(states_size * offset) + V]+210.000)/230.000));
ALGEBRAIC[(algebraic_size * offset) + assp] = 1.00000/(1.00000+exp(- ((STATES[(states_size * offset) + V]+CONSTANTS[(constants_size * offset) + EKshift]) - 24.3400)/14.8200));
ALGEBRAIC[(algebraic_size * offset) + fcass] = ALGEBRAIC[(algebraic_size * offset) + fss];
ALGEBRAIC[(algebraic_size * offset) + tfcaf] = 7.00000+1.00000/( 0.0400000*exp(- (STATES[(states_size * offset) + V] - 4.00000)/7.00000)+ 0.0400000*exp((STATES[(states_size * offset) + V] - 4.00000)/7.00000));
ALGEBRAIC[(algebraic_size * offset) + tfcas] = 100.000+1.00000/( 0.000120000*exp(- STATES[(states_size * offset) + V]/3.00000)+ 0.000120000*exp(STATES[(states_size * offset) + V]/7.00000));
ALGEBRAIC[(algebraic_size * offset) + tffp] =  2.50000*ALGEBRAIC[(algebraic_size * offset) + tff];
ALGEBRAIC[(algebraic_size * offset) + xs2ss] = ALGEBRAIC[(algebraic_size * offset) + xs1ss];
ALGEBRAIC[(algebraic_size * offset) + txs2] = 1.00000/( 0.0100000*exp((STATES[(states_size * offset) + V] - 50.0000)/20.0000)+ 0.0193000*exp(- (STATES[(states_size * offset) + V]+66.5400)/31.0000));
ALGEBRAIC[(algebraic_size * offset) + CaMKb] = ( CONSTANTS[(constants_size * offset) + CaMKo]*(1.00000 - STATES[(states_size * offset) + CaMKt]))/(1.00000+CONSTANTS[(constants_size * offset) + KmCaM]/STATES[(states_size * offset) + cass]);
ALGEBRAIC[(algebraic_size * offset) + hss] = 1.00000/pow(1.00000+exp((STATES[(states_size * offset) + V]+71.5500)/7.43000), 2.00000);
ALGEBRAIC[(algebraic_size * offset) + ah] = (STATES[(states_size * offset) + V]>=- 40.0000 ? 0.00000 :  0.0570000*exp(- (STATES[(states_size * offset) + V]+80.0000)/6.80000));
ALGEBRAIC[(algebraic_size * offset) + bh] = (STATES[(states_size * offset) + V]>=- 40.0000 ? 0.770000/( 0.130000*(1.00000+exp(- (STATES[(states_size * offset) + V]+10.6600)/11.1000))) :  2.70000*exp( 0.0790000*STATES[(states_size * offset) + V])+ 310000.*exp( 0.348500*STATES[(states_size * offset) + V]));
ALGEBRAIC[(algebraic_size * offset) + th] = 1.00000/(ALGEBRAIC[(algebraic_size * offset) + ah]+ALGEBRAIC[(algebraic_size * offset) + bh]);
ALGEBRAIC[(algebraic_size * offset) + tfcafp] =  2.50000*ALGEBRAIC[(algebraic_size * offset) + tfcaf];
ALGEBRAIC[(algebraic_size * offset) + jss] = ALGEBRAIC[(algebraic_size * offset) + hss];
ALGEBRAIC[(algebraic_size * offset) + aj] = (STATES[(states_size * offset) + V]>=- 40.0000 ? 0.00000 : ( ( - 25428.0*exp( 0.244400*STATES[(states_size * offset) + V]) -  6.94800e-06*exp( - 0.0439100*STATES[(states_size * offset) + V]))*(STATES[(states_size * offset) + V]+37.7800))/(1.00000+exp( 0.311000*(STATES[(states_size * offset) + V]+79.2300))));
ALGEBRAIC[(algebraic_size * offset) + bj] = (STATES[(states_size * offset) + V]>=- 40.0000 ? ( 0.600000*exp( 0.0570000*STATES[(states_size * offset) + V]))/(1.00000+exp( - 0.100000*(STATES[(states_size * offset) + V]+32.0000))) : ( 0.0242400*exp( - 0.0105200*STATES[(states_size * offset) + V]))/(1.00000+exp( - 0.137800*(STATES[(states_size * offset) + V]+40.1400))));
ALGEBRAIC[(algebraic_size * offset) + tj] = 1.00000/(ALGEBRAIC[(algebraic_size * offset) + aj]+ALGEBRAIC[(algebraic_size * offset) + bj]);
ALGEBRAIC[(algebraic_size * offset) + hssp] = 1.00000/pow(1.00000+exp((STATES[(states_size * offset) + V]+77.5500)/7.43000), 2.00000);
ALGEBRAIC[(algebraic_size * offset) + iss] = 1.00000/(1.00000+exp((STATES[(states_size * offset) + V]+CONSTANTS[(constants_size * offset) + EKshift]+43.9400)/5.71100));
ALGEBRAIC[(algebraic_size * offset) + delta_epi] = (CONSTANTS[(constants_size * offset) + celltype]==1.00000 ? 1.00000 - 0.950000/(1.00000+exp((STATES[(states_size * offset) + V]+CONSTANTS[(constants_size * offset) + EKshift]+70.0000)/5.00000)) : 1.00000);
ALGEBRAIC[(algebraic_size * offset) + tiF_b] = 4.56200+1.00000/( 0.393300*exp(- (STATES[(states_size * offset) + V]+CONSTANTS[(constants_size * offset) + EKshift]+100.000)/100.000)+ 0.0800400*exp((STATES[(states_size * offset) + V]+CONSTANTS[(constants_size * offset) + EKshift]+50.0000)/16.5900));
ALGEBRAIC[(algebraic_size * offset) + tiF] =  ALGEBRAIC[(algebraic_size * offset) + tiF_b]*ALGEBRAIC[(algebraic_size * offset) + delta_epi];
ALGEBRAIC[(algebraic_size * offset) + vfrt] = ( STATES[(states_size * offset) + V]*CONSTANTS[(constants_size * offset) + F])/( CONSTANTS[(constants_size * offset) + R]*CONSTANTS[(constants_size * offset) + T]);
ALGEBRAIC[(algebraic_size * offset) + alpha] =  0.116100*exp( 0.299000*ALGEBRAIC[(algebraic_size * offset) + vfrt]);
ALGEBRAIC[(algebraic_size * offset) + beta] =  0.244200*exp( - 1.60400*ALGEBRAIC[(algebraic_size * offset) + vfrt]);
ALGEBRAIC[(algebraic_size * offset) + tjp] =  1.46000*ALGEBRAIC[(algebraic_size * offset) + tj];
ALGEBRAIC[(algebraic_size * offset) + tiS_b] = 23.6200+1.00000/( 0.00141600*exp(- (STATES[(states_size * offset) + V]+CONSTANTS[(constants_size * offset) + EKshift]+96.5200)/59.0500)+ 1.78000e-08*exp((STATES[(states_size * offset) + V]+CONSTANTS[(constants_size * offset) + EKshift]+114.100)/8.07900));
ALGEBRAIC[(algebraic_size * offset) + tiS] =  ALGEBRAIC[(algebraic_size * offset) + tiS_b]*ALGEBRAIC[(algebraic_size * offset) + delta_epi];
ALGEBRAIC[(algebraic_size * offset) + alpha_2] =  0.0578000*exp( 0.971000*ALGEBRAIC[(algebraic_size * offset) + vfrt]);
ALGEBRAIC[(algebraic_size * offset) + beta_2] =  0.000349000*exp( - 1.06200*ALGEBRAIC[(algebraic_size * offset) + vfrt]);
ALGEBRAIC[(algebraic_size * offset) + alpha_i] =  0.253300*exp( 0.595300*ALGEBRAIC[(algebraic_size * offset) + vfrt]);
ALGEBRAIC[(algebraic_size * offset) + beta_i] =  0.0652500*exp( - 0.820900*ALGEBRAIC[(algebraic_size * offset) + vfrt]);
ALGEBRAIC[(algebraic_size * offset) + dti_develop] = 1.35400+0.000100000/(exp(((STATES[(states_size * offset) + V]+CONSTANTS[(constants_size * offset) + EKshift]) - 167.400)/15.8900)+exp(- ((STATES[(states_size * offset) + V]+CONSTANTS[(constants_size * offset) + EKshift]) - 12.2300)/0.215400));
ALGEBRAIC[(algebraic_size * offset) + dti_recover] = 1.00000 - 0.500000/(1.00000+exp((STATES[(states_size * offset) + V]+CONSTANTS[(constants_size * offset) + EKshift]+70.0000)/20.0000));
ALGEBRAIC[(algebraic_size * offset) + tiFp] =  ALGEBRAIC[(algebraic_size * offset) + dti_develop]*ALGEBRAIC[(algebraic_size * offset) + dti_recover]*ALGEBRAIC[(algebraic_size * offset) + tiF];
ALGEBRAIC[(algebraic_size * offset) + tiSp] =  ALGEBRAIC[(algebraic_size * offset) + dti_develop]*ALGEBRAIC[(algebraic_size * offset) + dti_recover]*ALGEBRAIC[(algebraic_size * offset) + tiS];
ALGEBRAIC[(algebraic_size * offset) + alpha_C2ToI] =  5.20000e-05*exp( 1.52500*ALGEBRAIC[(algebraic_size * offset) + vfrt]);
ALGEBRAIC[(algebraic_size * offset) + beta_ItoC2] = ( ALGEBRAIC[(algebraic_size * offset) + beta_2]*ALGEBRAIC[(algebraic_size * offset) + beta_i]*ALGEBRAIC[(algebraic_size * offset) + alpha_C2ToI])/( ALGEBRAIC[(algebraic_size * offset) + alpha_2]*ALGEBRAIC[(algebraic_size * offset) + alpha_i]);
ALGEBRAIC[(algebraic_size * offset) + f] =  CONSTANTS[(constants_size * offset) + Aff]*STATES[(states_size * offset) + ff]+ CONSTANTS[(constants_size * offset) + Afs]*STATES[(states_size * offset) + fs];
ALGEBRAIC[(algebraic_size * offset) + Afcaf] = 0.300000+0.600000/(1.00000+exp((STATES[(states_size * offset) + V] - 10.0000)/10.0000));
ALGEBRAIC[(algebraic_size * offset) + Afcas] = 1.00000 - ALGEBRAIC[(algebraic_size * offset) + Afcaf];
ALGEBRAIC[(algebraic_size * offset) + fca] =  ALGEBRAIC[(algebraic_size * offset) + Afcaf]*STATES[(states_size * offset) + fcaf]+ ALGEBRAIC[(algebraic_size * offset) + Afcas]*STATES[(states_size * offset) + fcas];
ALGEBRAIC[(algebraic_size * offset) + fp] =  CONSTANTS[(constants_size * offset) + Aff]*STATES[(states_size * offset) + ffp]+ CONSTANTS[(constants_size * offset) + Afs]*STATES[(states_size * offset) + fs];
ALGEBRAIC[(algebraic_size * offset) + fcap] =  ALGEBRAIC[(algebraic_size * offset) + Afcaf]*STATES[(states_size * offset) + fcafp]+ ALGEBRAIC[(algebraic_size * offset) + Afcas]*STATES[(states_size * offset) + fcas];
ALGEBRAIC[(algebraic_size * offset) + vffrt] = ( STATES[(states_size * offset) + V]*CONSTANTS[(constants_size * offset) + F]*CONSTANTS[(constants_size * offset) + F])/( CONSTANTS[(constants_size * offset) + R]*CONSTANTS[(constants_size * offset) + T]);
ALGEBRAIC[(algebraic_size * offset) + Iss] = ( 0.500000*(STATES[(states_size * offset) + nass]+STATES[(states_size * offset) + kss]+CONSTANTS[(constants_size * offset) + cli]+ 4.00000*STATES[(states_size * offset) + cass]))/1000.00;
ALGEBRAIC[(algebraic_size * offset) + gamma_cass] = exp( - CONSTANTS[(constants_size * offset) + constA]*4.00000*( pow(ALGEBRAIC[(algebraic_size * offset) + Iss], 1.0 / 2)/(1.00000+ pow(ALGEBRAIC[(algebraic_size * offset) + Iss], 1.0 / 2)) -  0.300000*ALGEBRAIC[(algebraic_size * offset) + Iss]));
ALGEBRAIC[(algebraic_size * offset) + PhiCaL_ss] = ( 4.00000*ALGEBRAIC[(algebraic_size * offset) + vffrt]*( ALGEBRAIC[(algebraic_size * offset) + gamma_cass]*STATES[(states_size * offset) + cass]*exp( 2.00000*ALGEBRAIC[(algebraic_size * offset) + vfrt]) -  CONSTANTS[(constants_size * offset) + gamma_cao]*CONSTANTS[(constants_size * offset) + cao]))/(exp( 2.00000*ALGEBRAIC[(algebraic_size * offset) + vfrt]) - 1.00000);
ALGEBRAIC[(algebraic_size * offset) + CaMKa] = ALGEBRAIC[(algebraic_size * offset) + CaMKb]+STATES[(states_size * offset) + CaMKt];
ALGEBRAIC[(algebraic_size * offset) + fICaLp] = 1.00000/(1.00000+CONSTANTS[(constants_size * offset) + KmCaMK]/ALGEBRAIC[(algebraic_size * offset) + CaMKa]);
ALGEBRAIC[(algebraic_size * offset) + ICaL_ss] =  CONSTANTS[(constants_size * offset) + ICaL_fractionSS]*( (1.00000 - ALGEBRAIC[(algebraic_size * offset) + fICaLp])*CONSTANTS[(constants_size * offset) + PCa]*ALGEBRAIC[(algebraic_size * offset) + PhiCaL_ss]*STATES[(states_size * offset) + d]*( ALGEBRAIC[(algebraic_size * offset) + f]*(1.00000 - STATES[(states_size * offset) + nca_ss])+ STATES[(states_size * offset) + jca]*ALGEBRAIC[(algebraic_size * offset) + fca]*STATES[(states_size * offset) + nca_ss])+ ALGEBRAIC[(algebraic_size * offset) + fICaLp]*CONSTANTS[(constants_size * offset) + PCap]*ALGEBRAIC[(algebraic_size * offset) + PhiCaL_ss]*STATES[(states_size * offset) + d]*( ALGEBRAIC[(algebraic_size * offset) + fp]*(1.00000 - STATES[(states_size * offset) + nca_ss])+ STATES[(states_size * offset) + jca]*ALGEBRAIC[(algebraic_size * offset) + fcap]*STATES[(states_size * offset) + nca_ss]));
ALGEBRAIC[(algebraic_size * offset) + Jrel_inf_b] = (( - CONSTANTS[(constants_size * offset) + a_rel]*ALGEBRAIC[(algebraic_size * offset) + ICaL_ss])/1.00000)/(1.00000+pow(CONSTANTS[(constants_size * offset) + cajsr_half]/STATES[(states_size * offset) + cajsr], 8.00000));
ALGEBRAIC[(algebraic_size * offset) + Jrel_inf] = (CONSTANTS[(constants_size * offset) + celltype]==2.00000 ?  ALGEBRAIC[(algebraic_size * offset) + Jrel_inf_b]*1.70000 : ALGEBRAIC[(algebraic_size * offset) + Jrel_inf_b]);
ALGEBRAIC[(algebraic_size * offset) + tau_rel_b] = CONSTANTS[(constants_size * offset) + bt]/(1.00000+0.0123000/STATES[(states_size * offset) + cajsr]);
ALGEBRAIC[(algebraic_size * offset) + tau_rel] = (ALGEBRAIC[(algebraic_size * offset) + tau_rel_b]<0.00100000 ? 0.00100000 : ALGEBRAIC[(algebraic_size * offset) + tau_rel_b]);
ALGEBRAIC[(algebraic_size * offset) + Jrel_infp_b] = (( - CONSTANTS[(constants_size * offset) + a_relp]*ALGEBRAIC[(algebraic_size * offset) + ICaL_ss])/1.00000)/(1.00000+pow(CONSTANTS[(constants_size * offset) + cajsr_half]/STATES[(states_size * offset) + cajsr], 8.00000));
ALGEBRAIC[(algebraic_size * offset) + Jrel_infp] = (CONSTANTS[(constants_size * offset) + celltype]==2.00000 ?  ALGEBRAIC[(algebraic_size * offset) + Jrel_infp_b]*1.70000 : ALGEBRAIC[(algebraic_size * offset) + Jrel_infp_b]);
ALGEBRAIC[(algebraic_size * offset) + tau_relp_b] = CONSTANTS[(constants_size * offset) + btp]/(1.00000+0.0123000/STATES[(states_size * offset) + cajsr]);
ALGEBRAIC[(algebraic_size * offset) + tau_relp] = (ALGEBRAIC[(algebraic_size * offset) + tau_relp_b]<0.00100000 ? 0.00100000 : ALGEBRAIC[(algebraic_size * offset) + tau_relp_b]);
ALGEBRAIC[(algebraic_size * offset) + EK] =  (( CONSTANTS[(constants_size * offset) + R]*CONSTANTS[(constants_size * offset) + T])/( CONSTANTS[(constants_size * offset) + zk]*CONSTANTS[(constants_size * offset) + F]))*log(CONSTANTS[(constants_size * offset) + ko]/STATES[(states_size * offset) + ki]);
ALGEBRAIC[(algebraic_size * offset) + AiF] = 1.00000/(1.00000+exp(((STATES[(states_size * offset) + V]+CONSTANTS[(constants_size * offset) + EKshift]) - 213.600)/151.200));
ALGEBRAIC[(algebraic_size * offset) + AiS] = 1.00000 - ALGEBRAIC[(algebraic_size * offset) + AiF];
ALGEBRAIC[(algebraic_size * offset) + i] =  ALGEBRAIC[(algebraic_size * offset) + AiF]*STATES[(states_size * offset) + iF]+ ALGEBRAIC[(algebraic_size * offset) + AiS]*STATES[(states_size * offset) + iS];
ALGEBRAIC[(algebraic_size * offset) + ip] =  ALGEBRAIC[(algebraic_size * offset) + AiF]*STATES[(states_size * offset) + iFp]+ ALGEBRAIC[(algebraic_size * offset) + AiS]*STATES[(states_size * offset) + iSp];
ALGEBRAIC[(algebraic_size * offset) + fItop] = 1.00000/(1.00000+CONSTANTS[(constants_size * offset) + KmCaMK]/ALGEBRAIC[(algebraic_size * offset) + CaMKa]);
ALGEBRAIC[(algebraic_size * offset) + Ito] =  CONSTANTS[(constants_size * offset) + Gto]*(STATES[(states_size * offset) + V] - ALGEBRAIC[(algebraic_size * offset) + EK])*( (1.00000 - ALGEBRAIC[(algebraic_size * offset) + fItop])*STATES[(states_size * offset) + a]*ALGEBRAIC[(algebraic_size * offset) + i]+ ALGEBRAIC[(algebraic_size * offset) + fItop]*STATES[(states_size * offset) + ap]*ALGEBRAIC[(algebraic_size * offset) + ip]);
ALGEBRAIC[(algebraic_size * offset) + IKr] =  CONSTANTS[(constants_size * offset) + GKr]* pow((CONSTANTS[(constants_size * offset) + ko]/5.00000), 1.0 / 2)*STATES[(states_size * offset) + O]*(STATES[(states_size * offset) + V] - ALGEBRAIC[(algebraic_size * offset) + EK]);
ALGEBRAIC[(algebraic_size * offset) + EKs] =  (( CONSTANTS[(constants_size * offset) + R]*CONSTANTS[(constants_size * offset) + T])/( CONSTANTS[(constants_size * offset) + zk]*CONSTANTS[(constants_size * offset) + F]))*log((CONSTANTS[(constants_size * offset) + ko]+ CONSTANTS[(constants_size * offset) + PKNa]*CONSTANTS[(constants_size * offset) + nao])/(STATES[(states_size * offset) + ki]+ CONSTANTS[(constants_size * offset) + PKNa]*STATES[(states_size * offset) + nai]));
ALGEBRAIC[(algebraic_size * offset) + KsCa] = 1.00000+0.600000/(1.00000+pow(3.80000e-05/STATES[(states_size * offset) + cai], 1.40000));
ALGEBRAIC[(algebraic_size * offset) + IKs] =  CONSTANTS[(constants_size * offset) + GKs]*ALGEBRAIC[(algebraic_size * offset) + KsCa]*STATES[(states_size * offset) + xs1]*STATES[(states_size * offset) + xs2]*(STATES[(states_size * offset) + V] - ALGEBRAIC[(algebraic_size * offset) + EKs]);
ALGEBRAIC[(algebraic_size * offset) + aK1] = 4.09400/(1.00000+exp( 0.121700*((STATES[(states_size * offset) + V] - ALGEBRAIC[(algebraic_size * offset) + EK]) - 49.9340)));
ALGEBRAIC[(algebraic_size * offset) + bK1] = ( 15.7200*exp( 0.0674000*((STATES[(states_size * offset) + V] - ALGEBRAIC[(algebraic_size * offset) + EK]) - 3.25700))+exp( 0.0618000*((STATES[(states_size * offset) + V] - ALGEBRAIC[(algebraic_size * offset) + EK]) - 594.310)))/(1.00000+exp( - 0.162900*((STATES[(states_size * offset) + V] - ALGEBRAIC[(algebraic_size * offset) + EK])+14.2070)));
ALGEBRAIC[(algebraic_size * offset) + K1ss] = ALGEBRAIC[(algebraic_size * offset) + aK1]/(ALGEBRAIC[(algebraic_size * offset) + aK1]+ALGEBRAIC[(algebraic_size * offset) + bK1]);
ALGEBRAIC[(algebraic_size * offset) + IK1] =  CONSTANTS[(constants_size * offset) + GK1]* pow((CONSTANTS[(constants_size * offset) + ko]/5.00000), 1.0 / 2)*ALGEBRAIC[(algebraic_size * offset) + K1ss]*(STATES[(states_size * offset) + V] - ALGEBRAIC[(algebraic_size * offset) + EK]);
ALGEBRAIC[(algebraic_size * offset) + Knao] =  CONSTANTS[(constants_size * offset) + Knao0]*exp(( (1.00000 - CONSTANTS[(constants_size * offset) + delta])*ALGEBRAIC[(algebraic_size * offset) + vfrt])/3.00000);
ALGEBRAIC[(algebraic_size * offset) + a3] = ( CONSTANTS[(constants_size * offset) + k3p]*pow(CONSTANTS[(constants_size * offset) + ko]/CONSTANTS[(constants_size * offset) + Kko], 2.00000))/((pow(1.00000+CONSTANTS[(constants_size * offset) + nao]/ALGEBRAIC[(algebraic_size * offset) + Knao], 3.00000)+pow(1.00000+CONSTANTS[(constants_size * offset) + ko]/CONSTANTS[(constants_size * offset) + Kko], 2.00000)) - 1.00000);
ALGEBRAIC[(algebraic_size * offset) + P] = CONSTANTS[(constants_size * offset) + eP]/(1.00000+CONSTANTS[(constants_size * offset) + H]/CONSTANTS[(constants_size * offset) + Khp]+STATES[(states_size * offset) + nai]/CONSTANTS[(constants_size * offset) + Knap]+STATES[(states_size * offset) + ki]/CONSTANTS[(constants_size * offset) + Kxkur]);
ALGEBRAIC[(algebraic_size * offset) + b3] = ( CONSTANTS[(constants_size * offset) + k3m]*ALGEBRAIC[(algebraic_size * offset) + P]*CONSTANTS[(constants_size * offset) + H])/(1.00000+CONSTANTS[(constants_size * offset) + MgATP]/CONSTANTS[(constants_size * offset) + Kmgatp]);
ALGEBRAIC[(algebraic_size * offset) + Knai] =  CONSTANTS[(constants_size * offset) + Knai0]*exp(( CONSTANTS[(constants_size * offset) + delta]*ALGEBRAIC[(algebraic_size * offset) + vfrt])/3.00000);
ALGEBRAIC[(algebraic_size * offset) + a1] = ( CONSTANTS[(constants_size * offset) + k1p]*pow(STATES[(states_size * offset) + nai]/ALGEBRAIC[(algebraic_size * offset) + Knai], 3.00000))/((pow(1.00000+STATES[(states_size * offset) + nai]/ALGEBRAIC[(algebraic_size * offset) + Knai], 3.00000)+pow(1.00000+STATES[(states_size * offset) + ki]/CONSTANTS[(constants_size * offset) + Kki], 2.00000)) - 1.00000);
ALGEBRAIC[(algebraic_size * offset) + b2] = ( CONSTANTS[(constants_size * offset) + k2m]*pow(CONSTANTS[(constants_size * offset) + nao]/ALGEBRAIC[(algebraic_size * offset) + Knao], 3.00000))/((pow(1.00000+CONSTANTS[(constants_size * offset) + nao]/ALGEBRAIC[(algebraic_size * offset) + Knao], 3.00000)+pow(1.00000+CONSTANTS[(constants_size * offset) + ko]/CONSTANTS[(constants_size * offset) + Kko], 2.00000)) - 1.00000);
ALGEBRAIC[(algebraic_size * offset) + b4] = ( CONSTANTS[(constants_size * offset) + k4m]*pow(STATES[(states_size * offset) + ki]/CONSTANTS[(constants_size * offset) + Kki], 2.00000))/((pow(1.00000+STATES[(states_size * offset) + nai]/ALGEBRAIC[(algebraic_size * offset) + Knai], 3.00000)+pow(1.00000+STATES[(states_size * offset) + ki]/CONSTANTS[(constants_size * offset) + Kki], 2.00000)) - 1.00000);
ALGEBRAIC[(algebraic_size * offset) + x1] =  CONSTANTS[(constants_size * offset) + a4]*ALGEBRAIC[(algebraic_size * offset) + a1]*CONSTANTS[(constants_size * offset) + a2]+ ALGEBRAIC[(algebraic_size * offset) + b2]*ALGEBRAIC[(algebraic_size * offset) + b4]*ALGEBRAIC[(algebraic_size * offset) + b3]+ CONSTANTS[(constants_size * offset) + a2]*ALGEBRAIC[(algebraic_size * offset) + b4]*ALGEBRAIC[(algebraic_size * offset) + b3]+ ALGEBRAIC[(algebraic_size * offset) + b3]*ALGEBRAIC[(algebraic_size * offset) + a1]*CONSTANTS[(constants_size * offset) + a2];
ALGEBRAIC[(algebraic_size * offset) + x2] =  ALGEBRAIC[(algebraic_size * offset) + b2]*CONSTANTS[(constants_size * offset) + b1]*ALGEBRAIC[(algebraic_size * offset) + b4]+ ALGEBRAIC[(algebraic_size * offset) + a1]*CONSTANTS[(constants_size * offset) + a2]*ALGEBRAIC[(algebraic_size * offset) + a3]+ ALGEBRAIC[(algebraic_size * offset) + a3]*CONSTANTS[(constants_size * offset) + b1]*ALGEBRAIC[(algebraic_size * offset) + b4]+ CONSTANTS[(constants_size * offset) + a2]*ALGEBRAIC[(algebraic_size * offset) + a3]*ALGEBRAIC[(algebraic_size * offset) + b4];
ALGEBRAIC[(algebraic_size * offset) + x3] =  CONSTANTS[(constants_size * offset) + a2]*ALGEBRAIC[(algebraic_size * offset) + a3]*CONSTANTS[(constants_size * offset) + a4]+ ALGEBRAIC[(algebraic_size * offset) + b3]*ALGEBRAIC[(algebraic_size * offset) + b2]*CONSTANTS[(constants_size * offset) + b1]+ ALGEBRAIC[(algebraic_size * offset) + b2]*CONSTANTS[(constants_size * offset) + b1]*CONSTANTS[(constants_size * offset) + a4]+ ALGEBRAIC[(algebraic_size * offset) + a3]*CONSTANTS[(constants_size * offset) + a4]*CONSTANTS[(constants_size * offset) + b1];
ALGEBRAIC[(algebraic_size * offset) + x4] =  ALGEBRAIC[(algebraic_size * offset) + b4]*ALGEBRAIC[(algebraic_size * offset) + b3]*ALGEBRAIC[(algebraic_size * offset) + b2]+ ALGEBRAIC[(algebraic_size * offset) + a3]*CONSTANTS[(constants_size * offset) + a4]*ALGEBRAIC[(algebraic_size * offset) + a1]+ ALGEBRAIC[(algebraic_size * offset) + b2]*CONSTANTS[(constants_size * offset) + a4]*ALGEBRAIC[(algebraic_size * offset) + a1]+ ALGEBRAIC[(algebraic_size * offset) + b3]*ALGEBRAIC[(algebraic_size * offset) + b2]*ALGEBRAIC[(algebraic_size * offset) + a1];
ALGEBRAIC[(algebraic_size * offset) + E1] = ALGEBRAIC[(algebraic_size * offset) + x1]/(ALGEBRAIC[(algebraic_size * offset) + x1]+ALGEBRAIC[(algebraic_size * offset) + x2]+ALGEBRAIC[(algebraic_size * offset) + x3]+ALGEBRAIC[(algebraic_size * offset) + x4]);
ALGEBRAIC[(algebraic_size * offset) + E2] = ALGEBRAIC[(algebraic_size * offset) + x2]/(ALGEBRAIC[(algebraic_size * offset) + x1]+ALGEBRAIC[(algebraic_size * offset) + x2]+ALGEBRAIC[(algebraic_size * offset) + x3]+ALGEBRAIC[(algebraic_size * offset) + x4]);
ALGEBRAIC[(algebraic_size * offset) + JnakNa] =  3.00000*( ALGEBRAIC[(algebraic_size * offset) + E1]*ALGEBRAIC[(algebraic_size * offset) + a3] -  ALGEBRAIC[(algebraic_size * offset) + E2]*ALGEBRAIC[(algebraic_size * offset) + b3]);
ALGEBRAIC[(algebraic_size * offset) + E3] = ALGEBRAIC[(algebraic_size * offset) + x3]/(ALGEBRAIC[(algebraic_size * offset) + x1]+ALGEBRAIC[(algebraic_size * offset) + x2]+ALGEBRAIC[(algebraic_size * offset) + x3]+ALGEBRAIC[(algebraic_size * offset) + x4]);
ALGEBRAIC[(algebraic_size * offset) + E4] = ALGEBRAIC[(algebraic_size * offset) + x4]/(ALGEBRAIC[(algebraic_size * offset) + x1]+ALGEBRAIC[(algebraic_size * offset) + x2]+ALGEBRAIC[(algebraic_size * offset) + x3]+ALGEBRAIC[(algebraic_size * offset) + x4]);
ALGEBRAIC[(algebraic_size * offset) + JnakK] =  2.00000*( ALGEBRAIC[(algebraic_size * offset) + E4]*CONSTANTS[(constants_size * offset) + b1] -  ALGEBRAIC[(algebraic_size * offset) + E3]*ALGEBRAIC[(algebraic_size * offset) + a1]);
ALGEBRAIC[(algebraic_size * offset) + INaK] =  CONSTANTS[(constants_size * offset) + Pnak]*( CONSTANTS[(constants_size * offset) + zna]*ALGEBRAIC[(algebraic_size * offset) + JnakNa]+ CONSTANTS[(constants_size * offset) + zk]*ALGEBRAIC[(algebraic_size * offset) + JnakK]);
ALGEBRAIC[(algebraic_size * offset) + xkb] = 1.00000/(1.00000+exp(- (STATES[(states_size * offset) + V] - 10.8968)/23.9871));
ALGEBRAIC[(algebraic_size * offset) + IKb] =  CONSTANTS[(constants_size * offset) + GKb]*ALGEBRAIC[(algebraic_size * offset) + xkb]*(STATES[(states_size * offset) + V] - ALGEBRAIC[(algebraic_size * offset) + EK]);
ALGEBRAIC[(algebraic_size * offset) + I_katp] =  CONSTANTS[(constants_size * offset) + fkatp]*CONSTANTS[(constants_size * offset) + gkatp]*CONSTANTS[(constants_size * offset) + akik]*CONSTANTS[(constants_size * offset) + bkik]*(STATES[(states_size * offset) + V] - ALGEBRAIC[(algebraic_size * offset) + EK]);
ALGEBRAIC[(algebraic_size * offset) + Istim] = (TIME>=CONSTANTS[(constants_size * offset) + stim_start]&&TIME<=CONSTANTS[(constants_size * offset) + i_Stim_End]&&(TIME - CONSTANTS[(constants_size * offset) + stim_start]) -  floor((TIME - CONSTANTS[(constants_size * offset) + stim_start])/CONSTANTS[(constants_size * offset) + BCL])*CONSTANTS[(constants_size * offset) + BCL]<=CONSTANTS[(constants_size * offset) + i_Stim_PulseDuration] ? CONSTANTS[(constants_size * offset) + i_Stim_Amplitude] : 0.00000);
ALGEBRAIC[(algebraic_size * offset) + Ii] = ( 0.500000*(STATES[(states_size * offset) + nai]+STATES[(states_size * offset) + ki]+CONSTANTS[(constants_size * offset) + cli]+ 4.00000*STATES[(states_size * offset) + cai]))/1000.00;
ALGEBRAIC[(algebraic_size * offset) + gamma_ki] = exp( - CONSTANTS[(constants_size * offset) + constA]*1.00000*( pow(ALGEBRAIC[(algebraic_size * offset) + Ii], 1.0 / 2)/(1.00000+ pow(ALGEBRAIC[(algebraic_size * offset) + Ii], 1.0 / 2)) -  0.300000*ALGEBRAIC[(algebraic_size * offset) + Ii]));
ALGEBRAIC[(algebraic_size * offset) + PhiCaK_i] = ( 1.00000*ALGEBRAIC[(algebraic_size * offset) + vffrt]*( ALGEBRAIC[(algebraic_size * offset) + gamma_ki]*STATES[(states_size * offset) + ki]*exp( 1.00000*ALGEBRAIC[(algebraic_size * offset) + vfrt]) -  CONSTANTS[(constants_size * offset) + gamma_ko]*CONSTANTS[(constants_size * offset) + ko]))/(exp( 1.00000*ALGEBRAIC[(algebraic_size * offset) + vfrt]) - 1.00000);
ALGEBRAIC[(algebraic_size * offset) + ICaK_i] =  (1.00000 - CONSTANTS[(constants_size * offset) + ICaL_fractionSS])*( (1.00000 - ALGEBRAIC[(algebraic_size * offset) + fICaLp])*CONSTANTS[(constants_size * offset) + PCaK]*ALGEBRAIC[(algebraic_size * offset) + PhiCaK_i]*STATES[(states_size * offset) + d]*( ALGEBRAIC[(algebraic_size * offset) + f]*(1.00000 - STATES[(states_size * offset) + nca_i])+ STATES[(states_size * offset) + jca]*ALGEBRAIC[(algebraic_size * offset) + fca]*STATES[(states_size * offset) + nca_i])+ ALGEBRAIC[(algebraic_size * offset) + fICaLp]*CONSTANTS[(constants_size * offset) + PCaKp]*ALGEBRAIC[(algebraic_size * offset) + PhiCaK_i]*STATES[(states_size * offset) + d]*( ALGEBRAIC[(algebraic_size * offset) + fp]*(1.00000 - STATES[(states_size * offset) + nca_i])+ STATES[(states_size * offset) + jca]*ALGEBRAIC[(algebraic_size * offset) + fcap]*STATES[(states_size * offset) + nca_i]));
ALGEBRAIC[(algebraic_size * offset) + JdiffK] = (STATES[(states_size * offset) + kss] - STATES[(states_size * offset) + ki])/CONSTANTS[(constants_size * offset) + tauK];
ALGEBRAIC[(algebraic_size * offset) + gamma_kss] = exp( - CONSTANTS[(constants_size * offset) + constA]*1.00000*( pow(ALGEBRAIC[(algebraic_size * offset) + Iss], 1.0 / 2)/(1.00000+ pow(ALGEBRAIC[(algebraic_size * offset) + Iss], 1.0 / 2)) -  0.300000*ALGEBRAIC[(algebraic_size * offset) + Iss]));
ALGEBRAIC[(algebraic_size * offset) + PhiCaK_ss] = ( 1.00000*ALGEBRAIC[(algebraic_size * offset) + vffrt]*( ALGEBRAIC[(algebraic_size * offset) + gamma_kss]*STATES[(states_size * offset) + kss]*exp( 1.00000*ALGEBRAIC[(algebraic_size * offset) + vfrt]) -  CONSTANTS[(constants_size * offset) + gamma_ko]*CONSTANTS[(constants_size * offset) + ko]))/(exp( 1.00000*ALGEBRAIC[(algebraic_size * offset) + vfrt]) - 1.00000);
ALGEBRAIC[(algebraic_size * offset) + ICaK_ss] =  CONSTANTS[(constants_size * offset) + ICaL_fractionSS]*( (1.00000 - ALGEBRAIC[(algebraic_size * offset) + fICaLp])*CONSTANTS[(constants_size * offset) + PCaK]*ALGEBRAIC[(algebraic_size * offset) + PhiCaK_ss]*STATES[(states_size * offset) + d]*( ALGEBRAIC[(algebraic_size * offset) + f]*(1.00000 - STATES[(states_size * offset) + nca_ss])+ STATES[(states_size * offset) + jca]*ALGEBRAIC[(algebraic_size * offset) + fca]*STATES[(states_size * offset) + nca_ss])+ ALGEBRAIC[(algebraic_size * offset) + fICaLp]*CONSTANTS[(constants_size * offset) + PCaKp]*ALGEBRAIC[(algebraic_size * offset) + PhiCaK_ss]*STATES[(states_size * offset) + d]*( ALGEBRAIC[(algebraic_size * offset) + fp]*(1.00000 - STATES[(states_size * offset) + nca_ss])+ STATES[(states_size * offset) + jca]*ALGEBRAIC[(algebraic_size * offset) + fcap]*STATES[(states_size * offset) + nca_ss]));
ALGEBRAIC[(algebraic_size * offset) + ENa] =  (( CONSTANTS[(constants_size * offset) + R]*CONSTANTS[(constants_size * offset) + T])/( CONSTANTS[(constants_size * offset) + zna]*CONSTANTS[(constants_size * offset) + F]))*log(CONSTANTS[(constants_size * offset) + nao]/STATES[(states_size * offset) + nai]);
ALGEBRAIC[(algebraic_size * offset) + fINap] = 1.00000/(1.00000+CONSTANTS[(constants_size * offset) + KmCaMK]/ALGEBRAIC[(algebraic_size * offset) + CaMKa]);
ALGEBRAIC[(algebraic_size * offset) + INa] =  CONSTANTS[(constants_size * offset) + GNa]*(STATES[(states_size * offset) + V] - ALGEBRAIC[(algebraic_size * offset) + ENa])*pow(STATES[(states_size * offset) + m], 3.00000)*( (1.00000 - ALGEBRAIC[(algebraic_size * offset) + fINap])*STATES[(states_size * offset) + h]*STATES[(states_size * offset) + j]+ ALGEBRAIC[(algebraic_size * offset) + fINap]*STATES[(states_size * offset) + hp]*STATES[(states_size * offset) + jp]);
ALGEBRAIC[(algebraic_size * offset) + fINaLp] = 1.00000/(1.00000+CONSTANTS[(constants_size * offset) + KmCaMK]/ALGEBRAIC[(algebraic_size * offset) + CaMKa]);
ALGEBRAIC[(algebraic_size * offset) + INaL] =  CONSTANTS[(constants_size * offset) + GNaL]*(STATES[(states_size * offset) + V] - ALGEBRAIC[(algebraic_size * offset) + ENa])*STATES[(states_size * offset) + mL]*( (1.00000 - ALGEBRAIC[(algebraic_size * offset) + fINaLp])*STATES[(states_size * offset) + hL]+ ALGEBRAIC[(algebraic_size * offset) + fINaLp]*STATES[(states_size * offset) + hLp]);
ALGEBRAIC[(algebraic_size * offset) + allo_i] = 1.00000/(1.00000+pow(CONSTANTS[(constants_size * offset) + KmCaAct]/STATES[(states_size * offset) + cai], 2.00000));
ALGEBRAIC[(algebraic_size * offset) + hna] = exp( CONSTANTS[(constants_size * offset) + qna]*ALGEBRAIC[(algebraic_size * offset) + vfrt]);
ALGEBRAIC[(algebraic_size * offset) + h7_i] = 1.00000+ (CONSTANTS[(constants_size * offset) + nao]/CONSTANTS[(constants_size * offset) + kna3])*(1.00000+1.00000/ALGEBRAIC[(algebraic_size * offset) + hna]);
ALGEBRAIC[(algebraic_size * offset) + h8_i] = CONSTANTS[(constants_size * offset) + nao]/( CONSTANTS[(constants_size * offset) + kna3]*ALGEBRAIC[(algebraic_size * offset) + hna]*ALGEBRAIC[(algebraic_size * offset) + h7_i]);
ALGEBRAIC[(algebraic_size * offset) + k3pp_i] =  ALGEBRAIC[(algebraic_size * offset) + h8_i]*CONSTANTS[(constants_size * offset) + wnaca];
ALGEBRAIC[(algebraic_size * offset) + h1_i] = 1.00000+ (STATES[(states_size * offset) + nai]/CONSTANTS[(constants_size * offset) + kna3])*(1.00000+ALGEBRAIC[(algebraic_size * offset) + hna]);
ALGEBRAIC[(algebraic_size * offset) + h2_i] = ( STATES[(states_size * offset) + nai]*ALGEBRAIC[(algebraic_size * offset) + hna])/( CONSTANTS[(constants_size * offset) + kna3]*ALGEBRAIC[(algebraic_size * offset) + h1_i]);
ALGEBRAIC[(algebraic_size * offset) + k4pp_i] =  ALGEBRAIC[(algebraic_size * offset) + h2_i]*CONSTANTS[(constants_size * offset) + wnaca];
ALGEBRAIC[(algebraic_size * offset) + h4_i] = 1.00000+ (STATES[(states_size * offset) + nai]/CONSTANTS[(constants_size * offset) + kna1])*(1.00000+STATES[(states_size * offset) + nai]/CONSTANTS[(constants_size * offset) + kna2]);
ALGEBRAIC[(algebraic_size * offset) + h5_i] = ( STATES[(states_size * offset) + nai]*STATES[(states_size * offset) + nai])/( ALGEBRAIC[(algebraic_size * offset) + h4_i]*CONSTANTS[(constants_size * offset) + kna1]*CONSTANTS[(constants_size * offset) + kna2]);
ALGEBRAIC[(algebraic_size * offset) + k7_i] =  ALGEBRAIC[(algebraic_size * offset) + h5_i]*ALGEBRAIC[(algebraic_size * offset) + h2_i]*CONSTANTS[(constants_size * offset) + wna];
ALGEBRAIC[(algebraic_size * offset) + k8_i] =  ALGEBRAIC[(algebraic_size * offset) + h8_i]*CONSTANTS[(constants_size * offset) + h11_i]*CONSTANTS[(constants_size * offset) + wna];
ALGEBRAIC[(algebraic_size * offset) + h9_i] = 1.00000/ALGEBRAIC[(algebraic_size * offset) + h7_i];
ALGEBRAIC[(algebraic_size * offset) + k3p_i] =  ALGEBRAIC[(algebraic_size * offset) + h9_i]*CONSTANTS[(constants_size * offset) + wca];
ALGEBRAIC[(algebraic_size * offset) + k3_i] = ALGEBRAIC[(algebraic_size * offset) + k3p_i]+ALGEBRAIC[(algebraic_size * offset) + k3pp_i];
ALGEBRAIC[(algebraic_size * offset) + hca] = exp( CONSTANTS[(constants_size * offset) + qca]*ALGEBRAIC[(algebraic_size * offset) + vfrt]);
ALGEBRAIC[(algebraic_size * offset) + h3_i] = 1.00000/ALGEBRAIC[(algebraic_size * offset) + h1_i];
ALGEBRAIC[(algebraic_size * offset) + k4p_i] = ( ALGEBRAIC[(algebraic_size * offset) + h3_i]*CONSTANTS[(constants_size * offset) + wca])/ALGEBRAIC[(algebraic_size * offset) + hca];
ALGEBRAIC[(algebraic_size * offset) + k4_i] = ALGEBRAIC[(algebraic_size * offset) + k4p_i]+ALGEBRAIC[(algebraic_size * offset) + k4pp_i];
ALGEBRAIC[(algebraic_size * offset) + h6_i] = 1.00000/ALGEBRAIC[(algebraic_size * offset) + h4_i];
ALGEBRAIC[(algebraic_size * offset) + k6_i] =  ALGEBRAIC[(algebraic_size * offset) + h6_i]*STATES[(states_size * offset) + cai]*CONSTANTS[(constants_size * offset) + kcaon];
ALGEBRAIC[(algebraic_size * offset) + x1_i] =  CONSTANTS[(constants_size * offset) + k2_i]*ALGEBRAIC[(algebraic_size * offset) + k4_i]*(ALGEBRAIC[(algebraic_size * offset) + k7_i]+ALGEBRAIC[(algebraic_size * offset) + k6_i])+ CONSTANTS[(constants_size * offset) + k5_i]*ALGEBRAIC[(algebraic_size * offset) + k7_i]*(CONSTANTS[(constants_size * offset) + k2_i]+ALGEBRAIC[(algebraic_size * offset) + k3_i]);
ALGEBRAIC[(algebraic_size * offset) + x2_i] =  CONSTANTS[(constants_size * offset) + k1_i]*ALGEBRAIC[(algebraic_size * offset) + k7_i]*(ALGEBRAIC[(algebraic_size * offset) + k4_i]+CONSTANTS[(constants_size * offset) + k5_i])+ ALGEBRAIC[(algebraic_size * offset) + k4_i]*ALGEBRAIC[(algebraic_size * offset) + k6_i]*(CONSTANTS[(constants_size * offset) + k1_i]+ALGEBRAIC[(algebraic_size * offset) + k8_i]);
ALGEBRAIC[(algebraic_size * offset) + x3_i] =  CONSTANTS[(constants_size * offset) + k1_i]*ALGEBRAIC[(algebraic_size * offset) + k3_i]*(ALGEBRAIC[(algebraic_size * offset) + k7_i]+ALGEBRAIC[(algebraic_size * offset) + k6_i])+ ALGEBRAIC[(algebraic_size * offset) + k8_i]*ALGEBRAIC[(algebraic_size * offset) + k6_i]*(CONSTANTS[(constants_size * offset) + k2_i]+ALGEBRAIC[(algebraic_size * offset) + k3_i]);
ALGEBRAIC[(algebraic_size * offset) + x4_i] =  CONSTANTS[(constants_size * offset) + k2_i]*ALGEBRAIC[(algebraic_size * offset) + k8_i]*(ALGEBRAIC[(algebraic_size * offset) + k4_i]+CONSTANTS[(constants_size * offset) + k5_i])+ ALGEBRAIC[(algebraic_size * offset) + k3_i]*CONSTANTS[(constants_size * offset) + k5_i]*(CONSTANTS[(constants_size * offset) + k1_i]+ALGEBRAIC[(algebraic_size * offset) + k8_i]);
ALGEBRAIC[(algebraic_size * offset) + E1_i] = ALGEBRAIC[(algebraic_size * offset) + x1_i]/(ALGEBRAIC[(algebraic_size * offset) + x1_i]+ALGEBRAIC[(algebraic_size * offset) + x2_i]+ALGEBRAIC[(algebraic_size * offset) + x3_i]+ALGEBRAIC[(algebraic_size * offset) + x4_i]);
ALGEBRAIC[(algebraic_size * offset) + E2_i] = ALGEBRAIC[(algebraic_size * offset) + x2_i]/(ALGEBRAIC[(algebraic_size * offset) + x1_i]+ALGEBRAIC[(algebraic_size * offset) + x2_i]+ALGEBRAIC[(algebraic_size * offset) + x3_i]+ALGEBRAIC[(algebraic_size * offset) + x4_i]);
ALGEBRAIC[(algebraic_size * offset) + E3_i] = ALGEBRAIC[(algebraic_size * offset) + x3_i]/(ALGEBRAIC[(algebraic_size * offset) + x1_i]+ALGEBRAIC[(algebraic_size * offset) + x2_i]+ALGEBRAIC[(algebraic_size * offset) + x3_i]+ALGEBRAIC[(algebraic_size * offset) + x4_i]);
ALGEBRAIC[(algebraic_size * offset) + E4_i] = ALGEBRAIC[(algebraic_size * offset) + x4_i]/(ALGEBRAIC[(algebraic_size * offset) + x1_i]+ALGEBRAIC[(algebraic_size * offset) + x2_i]+ALGEBRAIC[(algebraic_size * offset) + x3_i]+ALGEBRAIC[(algebraic_size * offset) + x4_i]);
ALGEBRAIC[(algebraic_size * offset) + JncxNa_i] = ( 3.00000*( ALGEBRAIC[(algebraic_size * offset) + E4_i]*ALGEBRAIC[(algebraic_size * offset) + k7_i] -  ALGEBRAIC[(algebraic_size * offset) + E1_i]*ALGEBRAIC[(algebraic_size * offset) + k8_i])+ ALGEBRAIC[(algebraic_size * offset) + E3_i]*ALGEBRAIC[(algebraic_size * offset) + k4pp_i]) -  ALGEBRAIC[(algebraic_size * offset) + E2_i]*ALGEBRAIC[(algebraic_size * offset) + k3pp_i];
ALGEBRAIC[(algebraic_size * offset) + JncxCa_i] =  ALGEBRAIC[(algebraic_size * offset) + E2_i]*CONSTANTS[(constants_size * offset) + k2_i] -  ALGEBRAIC[(algebraic_size * offset) + E1_i]*CONSTANTS[(constants_size * offset) + k1_i];
ALGEBRAIC[(algebraic_size * offset) + INaCa_i] =  (1.00000 - CONSTANTS[(constants_size * offset) + INaCa_fractionSS])*CONSTANTS[(constants_size * offset) + Gncx]*ALGEBRAIC[(algebraic_size * offset) + allo_i]*( CONSTANTS[(constants_size * offset) + zna]*ALGEBRAIC[(algebraic_size * offset) + JncxNa_i]+ CONSTANTS[(constants_size * offset) + zca]*ALGEBRAIC[(algebraic_size * offset) + JncxCa_i]);
ALGEBRAIC[(algebraic_size * offset) + INab] = ( CONSTANTS[(constants_size * offset) + PNab]*ALGEBRAIC[(algebraic_size * offset) + vffrt]*( STATES[(states_size * offset) + nai]*exp(ALGEBRAIC[(algebraic_size * offset) + vfrt]) - CONSTANTS[(constants_size * offset) + nao]))/(exp(ALGEBRAIC[(algebraic_size * offset) + vfrt]) - 1.00000);
ALGEBRAIC[(algebraic_size * offset) + gamma_nai] = exp( - CONSTANTS[(constants_size * offset) + constA]*1.00000*( pow(ALGEBRAIC[(algebraic_size * offset) + Ii], 1.0 / 2)/(1.00000+ pow(ALGEBRAIC[(algebraic_size * offset) + Ii], 1.0 / 2)) -  0.300000*ALGEBRAIC[(algebraic_size * offset) + Ii]));
ALGEBRAIC[(algebraic_size * offset) + PhiCaNa_i] = ( 1.00000*ALGEBRAIC[(algebraic_size * offset) + vffrt]*( ALGEBRAIC[(algebraic_size * offset) + gamma_nai]*STATES[(states_size * offset) + nai]*exp( 1.00000*ALGEBRAIC[(algebraic_size * offset) + vfrt]) -  CONSTANTS[(constants_size * offset) + gamma_nao]*CONSTANTS[(constants_size * offset) + nao]))/(exp( 1.00000*ALGEBRAIC[(algebraic_size * offset) + vfrt]) - 1.00000);
ALGEBRAIC[(algebraic_size * offset) + ICaNa_i] =  (1.00000 - CONSTANTS[(constants_size * offset) + ICaL_fractionSS])*( (1.00000 - ALGEBRAIC[(algebraic_size * offset) + fICaLp])*CONSTANTS[(constants_size * offset) + PCaNa]*ALGEBRAIC[(algebraic_size * offset) + PhiCaNa_i]*STATES[(states_size * offset) + d]*( ALGEBRAIC[(algebraic_size * offset) + f]*(1.00000 - STATES[(states_size * offset) + nca_i])+ STATES[(states_size * offset) + jca]*ALGEBRAIC[(algebraic_size * offset) + fca]*STATES[(states_size * offset) + nca_i])+ ALGEBRAIC[(algebraic_size * offset) + fICaLp]*CONSTANTS[(constants_size * offset) + PCaNap]*ALGEBRAIC[(algebraic_size * offset) + PhiCaNa_i]*STATES[(states_size * offset) + d]*( ALGEBRAIC[(algebraic_size * offset) + fp]*(1.00000 - STATES[(states_size * offset) + nca_i])+ STATES[(states_size * offset) + jca]*ALGEBRAIC[(algebraic_size * offset) + fcap]*STATES[(states_size * offset) + nca_i]));
ALGEBRAIC[(algebraic_size * offset) + JdiffNa] = (STATES[(states_size * offset) + nass] - STATES[(states_size * offset) + nai])/CONSTANTS[(constants_size * offset) + tauNa];
ALGEBRAIC[(algebraic_size * offset) + allo_ss] = 1.00000/(1.00000+pow(CONSTANTS[(constants_size * offset) + KmCaAct]/STATES[(states_size * offset) + cass], 2.00000));
ALGEBRAIC[(algebraic_size * offset) + h7_ss] = 1.00000+ (CONSTANTS[(constants_size * offset) + nao]/CONSTANTS[(constants_size * offset) + kna3])*(1.00000+1.00000/ALGEBRAIC[(algebraic_size * offset) + hna]);
ALGEBRAIC[(algebraic_size * offset) + h8_ss] = CONSTANTS[(constants_size * offset) + nao]/( CONSTANTS[(constants_size * offset) + kna3]*ALGEBRAIC[(algebraic_size * offset) + hna]*ALGEBRAIC[(algebraic_size * offset) + h7_ss]);
ALGEBRAIC[(algebraic_size * offset) + k3pp_ss] =  ALGEBRAIC[(algebraic_size * offset) + h8_ss]*CONSTANTS[(constants_size * offset) + wnaca];
ALGEBRAIC[(algebraic_size * offset) + h1_ss] = 1.00000+ (STATES[(states_size * offset) + nass]/CONSTANTS[(constants_size * offset) + kna3])*(1.00000+ALGEBRAIC[(algebraic_size * offset) + hna]);
ALGEBRAIC[(algebraic_size * offset) + h2_ss] = ( STATES[(states_size * offset) + nass]*ALGEBRAIC[(algebraic_size * offset) + hna])/( CONSTANTS[(constants_size * offset) + kna3]*ALGEBRAIC[(algebraic_size * offset) + h1_ss]);
ALGEBRAIC[(algebraic_size * offset) + k4pp_ss] =  ALGEBRAIC[(algebraic_size * offset) + h2_ss]*CONSTANTS[(constants_size * offset) + wnaca];
ALGEBRAIC[(algebraic_size * offset) + h4_ss] = 1.00000+ (STATES[(states_size * offset) + nass]/CONSTANTS[(constants_size * offset) + kna1])*(1.00000+STATES[(states_size * offset) + nass]/CONSTANTS[(constants_size * offset) + kna2]);
ALGEBRAIC[(algebraic_size * offset) + h5_ss] = ( STATES[(states_size * offset) + nass]*STATES[(states_size * offset) + nass])/( ALGEBRAIC[(algebraic_size * offset) + h4_ss]*CONSTANTS[(constants_size * offset) + kna1]*CONSTANTS[(constants_size * offset) + kna2]);
ALGEBRAIC[(algebraic_size * offset) + k7_ss] =  ALGEBRAIC[(algebraic_size * offset) + h5_ss]*ALGEBRAIC[(algebraic_size * offset) + h2_ss]*CONSTANTS[(constants_size * offset) + wna];
ALGEBRAIC[(algebraic_size * offset) + k8_ss] =  ALGEBRAIC[(algebraic_size * offset) + h8_ss]*CONSTANTS[(constants_size * offset) + h11_ss]*CONSTANTS[(constants_size * offset) + wna];
ALGEBRAIC[(algebraic_size * offset) + h9_ss] = 1.00000/ALGEBRAIC[(algebraic_size * offset) + h7_ss];
ALGEBRAIC[(algebraic_size * offset) + k3p_ss] =  ALGEBRAIC[(algebraic_size * offset) + h9_ss]*CONSTANTS[(constants_size * offset) + wca];
ALGEBRAIC[(algebraic_size * offset) + k3_ss] = ALGEBRAIC[(algebraic_size * offset) + k3p_ss]+ALGEBRAIC[(algebraic_size * offset) + k3pp_ss];
ALGEBRAIC[(algebraic_size * offset) + h3_ss] = 1.00000/ALGEBRAIC[(algebraic_size * offset) + h1_ss];
ALGEBRAIC[(algebraic_size * offset) + k4p_ss] = ( ALGEBRAIC[(algebraic_size * offset) + h3_ss]*CONSTANTS[(constants_size * offset) + wca])/ALGEBRAIC[(algebraic_size * offset) + hca];
ALGEBRAIC[(algebraic_size * offset) + k4_ss] = ALGEBRAIC[(algebraic_size * offset) + k4p_ss]+ALGEBRAIC[(algebraic_size * offset) + k4pp_ss];
ALGEBRAIC[(algebraic_size * offset) + h6_ss] = 1.00000/ALGEBRAIC[(algebraic_size * offset) + h4_ss];
ALGEBRAIC[(algebraic_size * offset) + k6_ss] =  ALGEBRAIC[(algebraic_size * offset) + h6_ss]*STATES[(states_size * offset) + cass]*CONSTANTS[(constants_size * offset) + kcaon];
ALGEBRAIC[(algebraic_size * offset) + x1_ss] =  CONSTANTS[(constants_size * offset) + k2_ss]*ALGEBRAIC[(algebraic_size * offset) + k4_ss]*(ALGEBRAIC[(algebraic_size * offset) + k7_ss]+ALGEBRAIC[(algebraic_size * offset) + k6_ss])+ CONSTANTS[(constants_size * offset) + k5_ss]*ALGEBRAIC[(algebraic_size * offset) + k7_ss]*(CONSTANTS[(constants_size * offset) + k2_ss]+ALGEBRAIC[(algebraic_size * offset) + k3_ss]);
ALGEBRAIC[(algebraic_size * offset) + x2_ss] =  CONSTANTS[(constants_size * offset) + k1_ss]*ALGEBRAIC[(algebraic_size * offset) + k7_ss]*(ALGEBRAIC[(algebraic_size * offset) + k4_ss]+CONSTANTS[(constants_size * offset) + k5_ss])+ ALGEBRAIC[(algebraic_size * offset) + k4_ss]*ALGEBRAIC[(algebraic_size * offset) + k6_ss]*(CONSTANTS[(constants_size * offset) + k1_ss]+ALGEBRAIC[(algebraic_size * offset) + k8_ss]);
ALGEBRAIC[(algebraic_size * offset) + x3_ss] =  CONSTANTS[(constants_size * offset) + k1_ss]*ALGEBRAIC[(algebraic_size * offset) + k3_ss]*(ALGEBRAIC[(algebraic_size * offset) + k7_ss]+ALGEBRAIC[(algebraic_size * offset) + k6_ss])+ ALGEBRAIC[(algebraic_size * offset) + k8_ss]*ALGEBRAIC[(algebraic_size * offset) + k6_ss]*(CONSTANTS[(constants_size * offset) + k2_ss]+ALGEBRAIC[(algebraic_size * offset) + k3_ss]);
ALGEBRAIC[(algebraic_size * offset) + x4_ss] =  CONSTANTS[(constants_size * offset) + k2_ss]*ALGEBRAIC[(algebraic_size * offset) + k8_ss]*(ALGEBRAIC[(algebraic_size * offset) + k4_ss]+CONSTANTS[(constants_size * offset) + k5_ss])+ ALGEBRAIC[(algebraic_size * offset) + k3_ss]*CONSTANTS[(constants_size * offset) + k5_ss]*(CONSTANTS[(constants_size * offset) + k1_ss]+ALGEBRAIC[(algebraic_size * offset) + k8_ss]);
ALGEBRAIC[(algebraic_size * offset) + E1_ss] = ALGEBRAIC[(algebraic_size * offset) + x1_ss]/(ALGEBRAIC[(algebraic_size * offset) + x1_ss]+ALGEBRAIC[(algebraic_size * offset) + x2_ss]+ALGEBRAIC[(algebraic_size * offset) + x3_ss]+ALGEBRAIC[(algebraic_size * offset) + x4_ss]);
ALGEBRAIC[(algebraic_size * offset) + E2_ss] = ALGEBRAIC[(algebraic_size * offset) + x2_ss]/(ALGEBRAIC[(algebraic_size * offset) + x1_ss]+ALGEBRAIC[(algebraic_size * offset) + x2_ss]+ALGEBRAIC[(algebraic_size * offset) + x3_ss]+ALGEBRAIC[(algebraic_size * offset) + x4_ss]);
ALGEBRAIC[(algebraic_size * offset) + E3_ss] = ALGEBRAIC[(algebraic_size * offset) + x3_ss]/(ALGEBRAIC[(algebraic_size * offset) + x1_ss]+ALGEBRAIC[(algebraic_size * offset) + x2_ss]+ALGEBRAIC[(algebraic_size * offset) + x3_ss]+ALGEBRAIC[(algebraic_size * offset) + x4_ss]);
ALGEBRAIC[(algebraic_size * offset) + E4_ss] = ALGEBRAIC[(algebraic_size * offset) + x4_ss]/(ALGEBRAIC[(algebraic_size * offset) + x1_ss]+ALGEBRAIC[(algebraic_size * offset) + x2_ss]+ALGEBRAIC[(algebraic_size * offset) + x3_ss]+ALGEBRAIC[(algebraic_size * offset) + x4_ss]);
ALGEBRAIC[(algebraic_size * offset) + JncxNa_ss] = ( 3.00000*( ALGEBRAIC[(algebraic_size * offset) + E4_ss]*ALGEBRAIC[(algebraic_size * offset) + k7_ss] -  ALGEBRAIC[(algebraic_size * offset) + E1_ss]*ALGEBRAIC[(algebraic_size * offset) + k8_ss])+ ALGEBRAIC[(algebraic_size * offset) + E3_ss]*ALGEBRAIC[(algebraic_size * offset) + k4pp_ss]) -  ALGEBRAIC[(algebraic_size * offset) + E2_ss]*ALGEBRAIC[(algebraic_size * offset) + k3pp_ss];
ALGEBRAIC[(algebraic_size * offset) + JncxCa_ss] =  ALGEBRAIC[(algebraic_size * offset) + E2_ss]*CONSTANTS[(constants_size * offset) + k2_ss] -  ALGEBRAIC[(algebraic_size * offset) + E1_ss]*CONSTANTS[(constants_size * offset) + k1_ss];
ALGEBRAIC[(algebraic_size * offset) + INaCa_ss] =  CONSTANTS[(constants_size * offset) + INaCa_fractionSS]*CONSTANTS[(constants_size * offset) + Gncx]*ALGEBRAIC[(algebraic_size * offset) + allo_ss]*( CONSTANTS[(constants_size * offset) + zna]*ALGEBRAIC[(algebraic_size * offset) + JncxNa_ss]+ CONSTANTS[(constants_size * offset) + zca]*ALGEBRAIC[(algebraic_size * offset) + JncxCa_ss]);
ALGEBRAIC[(algebraic_size * offset) + gamma_nass] = exp( - CONSTANTS[(constants_size * offset) + constA]*1.00000*( pow(ALGEBRAIC[(algebraic_size * offset) + Iss], 1.0 / 2)/(1.00000+ pow(ALGEBRAIC[(algebraic_size * offset) + Iss], 1.0 / 2)) -  0.300000*ALGEBRAIC[(algebraic_size * offset) + Iss]));
ALGEBRAIC[(algebraic_size * offset) + PhiCaNa_ss] = ( 1.00000*ALGEBRAIC[(algebraic_size * offset) + vffrt]*( ALGEBRAIC[(algebraic_size * offset) + gamma_nass]*STATES[(states_size * offset) + nass]*exp( 1.00000*ALGEBRAIC[(algebraic_size * offset) + vfrt]) -  CONSTANTS[(constants_size * offset) + gamma_nao]*CONSTANTS[(constants_size * offset) + nao]))/(exp( 1.00000*ALGEBRAIC[(algebraic_size * offset) + vfrt]) - 1.00000);
ALGEBRAIC[(algebraic_size * offset) + ICaNa_ss] =  CONSTANTS[(constants_size * offset) + ICaL_fractionSS]*( (1.00000 - ALGEBRAIC[(algebraic_size * offset) + fICaLp])*CONSTANTS[(constants_size * offset) + PCaNa]*ALGEBRAIC[(algebraic_size * offset) + PhiCaNa_ss]*STATES[(states_size * offset) + d]*( ALGEBRAIC[(algebraic_size * offset) + f]*(1.00000 - STATES[(states_size * offset) + nca_ss])+ STATES[(states_size * offset) + jca]*ALGEBRAIC[(algebraic_size * offset) + fca]*STATES[(states_size * offset) + nca_ss])+ ALGEBRAIC[(algebraic_size * offset) + fICaLp]*CONSTANTS[(constants_size * offset) + PCaNap]*ALGEBRAIC[(algebraic_size * offset) + PhiCaNa_ss]*STATES[(states_size * offset) + d]*( ALGEBRAIC[(algebraic_size * offset) + fp]*(1.00000 - STATES[(states_size * offset) + nca_ss])+ STATES[(states_size * offset) + jca]*ALGEBRAIC[(algebraic_size * offset) + fcap]*STATES[(states_size * offset) + nca_ss]));
ALGEBRAIC[(algebraic_size * offset) + Jdiff] = (STATES[(states_size * offset) + cass] - STATES[(states_size * offset) + cai])/CONSTANTS[(constants_size * offset) + tauCa];
ALGEBRAIC[(algebraic_size * offset) + fJrelp] = 1.00000/(1.00000+CONSTANTS[(constants_size * offset) + KmCaMK]/ALGEBRAIC[(algebraic_size * offset) + CaMKa]);
ALGEBRAIC[(algebraic_size * offset) + Jrel] =  CONSTANTS[(constants_size * offset) + Jrel_b]*( (1.00000 - ALGEBRAIC[(algebraic_size * offset) + fJrelp])*STATES[(states_size * offset) + Jrel_np]+ ALGEBRAIC[(algebraic_size * offset) + fJrelp]*STATES[(states_size * offset) + Jrel_p]);
ALGEBRAIC[(algebraic_size * offset) + Bcass] = 1.00000/(1.00000+( CONSTANTS[(constants_size * offset) + BSRmax]*CONSTANTS[(constants_size * offset) + KmBSR])/pow(CONSTANTS[(constants_size * offset) + KmBSR]+STATES[(states_size * offset) + cass], 2.00000)+( CONSTANTS[(constants_size * offset) + BSLmax]*CONSTANTS[(constants_size * offset) + KmBSL])/pow(CONSTANTS[(constants_size * offset) + KmBSL]+STATES[(states_size * offset) + cass], 2.00000));
ALGEBRAIC[(algebraic_size * offset) + gamma_cai] = exp( - CONSTANTS[(constants_size * offset) + constA]*4.00000*( pow(ALGEBRAIC[(algebraic_size * offset) + Ii], 1.0 / 2)/(1.00000+ pow(ALGEBRAIC[(algebraic_size * offset) + Ii], 1.0 / 2)) -  0.300000*ALGEBRAIC[(algebraic_size * offset) + Ii]));
ALGEBRAIC[(algebraic_size * offset) + PhiCaL_i] = ( 4.00000*ALGEBRAIC[(algebraic_size * offset) + vffrt]*( ALGEBRAIC[(algebraic_size * offset) + gamma_cai]*STATES[(states_size * offset) + cai]*exp( 2.00000*ALGEBRAIC[(algebraic_size * offset) + vfrt]) -  CONSTANTS[(constants_size * offset) + gamma_cao]*CONSTANTS[(constants_size * offset) + cao]))/(exp( 2.00000*ALGEBRAIC[(algebraic_size * offset) + vfrt]) - 1.00000);
ALGEBRAIC[(algebraic_size * offset) + ICaL_i] =  (1.00000 - CONSTANTS[(constants_size * offset) + ICaL_fractionSS])*( (1.00000 - ALGEBRAIC[(algebraic_size * offset) + fICaLp])*CONSTANTS[(constants_size * offset) + PCa]*ALGEBRAIC[(algebraic_size * offset) + PhiCaL_i]*STATES[(states_size * offset) + d]*( ALGEBRAIC[(algebraic_size * offset) + f]*(1.00000 - STATES[(states_size * offset) + nca_i])+ STATES[(states_size * offset) + jca]*ALGEBRAIC[(algebraic_size * offset) + fca]*STATES[(states_size * offset) + nca_i])+ ALGEBRAIC[(algebraic_size * offset) + fICaLp]*CONSTANTS[(constants_size * offset) + PCap]*ALGEBRAIC[(algebraic_size * offset) + PhiCaL_i]*STATES[(states_size * offset) + d]*( ALGEBRAIC[(algebraic_size * offset) + fp]*(1.00000 - STATES[(states_size * offset) + nca_i])+ STATES[(states_size * offset) + jca]*ALGEBRAIC[(algebraic_size * offset) + fcap]*STATES[(states_size * offset) + nca_i]));
ALGEBRAIC[(algebraic_size * offset) + ICaL] = ALGEBRAIC[(algebraic_size * offset) + ICaL_ss]+ALGEBRAIC[(algebraic_size * offset) + ICaL_i];
ALGEBRAIC[(algebraic_size * offset) + ICaNa] = ALGEBRAIC[(algebraic_size * offset) + ICaNa_ss]+ALGEBRAIC[(algebraic_size * offset) + ICaNa_i];
ALGEBRAIC[(algebraic_size * offset) + ICaK] = ALGEBRAIC[(algebraic_size * offset) + ICaK_ss]+ALGEBRAIC[(algebraic_size * offset) + ICaK_i];
ALGEBRAIC[(algebraic_size * offset) + IpCa] = ( CONSTANTS[(constants_size * offset) + GpCa]*STATES[(states_size * offset) + cai])/(CONSTANTS[(constants_size * offset) + KmCap]+STATES[(states_size * offset) + cai]);
ALGEBRAIC[(algebraic_size * offset) + ICab] = ( CONSTANTS[(constants_size * offset) + PCab]*4.00000*ALGEBRAIC[(algebraic_size * offset) + vffrt]*( ALGEBRAIC[(algebraic_size * offset) + gamma_cai]*STATES[(states_size * offset) + cai]*exp( 2.00000*ALGEBRAIC[(algebraic_size * offset) + vfrt]) -  CONSTANTS[(constants_size * offset) + gamma_cao]*CONSTANTS[(constants_size * offset) + cao]))/(exp( 2.00000*ALGEBRAIC[(algebraic_size * offset) + vfrt]) - 1.00000);
ALGEBRAIC[(algebraic_size * offset) + IClCa_junc] =  (( CONSTANTS[(constants_size * offset) + Fjunc]*CONSTANTS[(constants_size * offset) + GClCa])/(1.00000+CONSTANTS[(constants_size * offset) + KdClCa]/STATES[(states_size * offset) + cass]))*(STATES[(states_size * offset) + V] - CONSTANTS[(constants_size * offset) + ECl]);
ALGEBRAIC[(algebraic_size * offset) + IClCa_sl] =  (( (1.00000 - CONSTANTS[(constants_size * offset) + Fjunc])*CONSTANTS[(constants_size * offset) + GClCa])/(1.00000+CONSTANTS[(constants_size * offset) + KdClCa]/STATES[(states_size * offset) + cai]))*(STATES[(states_size * offset) + V] - CONSTANTS[(constants_size * offset) + ECl]);
ALGEBRAIC[(algebraic_size * offset) + IClCa] = ALGEBRAIC[(algebraic_size * offset) + IClCa_junc]+ALGEBRAIC[(algebraic_size * offset) + IClCa_sl];
ALGEBRAIC[(algebraic_size * offset) + IClb] =  CONSTANTS[(constants_size * offset) + GClb]*(STATES[(states_size * offset) + V] - CONSTANTS[(constants_size * offset) + ECl]);
ALGEBRAIC[(algebraic_size * offset) + Jupnp] = ( CONSTANTS[(constants_size * offset) + upScale]*0.00542500*STATES[(states_size * offset) + cai])/(STATES[(states_size * offset) + cai]+0.000920000);
ALGEBRAIC[(algebraic_size * offset) + Jupp] = ( CONSTANTS[(constants_size * offset) + upScale]*2.75000*0.00542500*STATES[(states_size * offset) + cai])/((STATES[(states_size * offset) + cai]+0.000920000) - 0.000170000);
ALGEBRAIC[(algebraic_size * offset) + fJupp] = 1.00000/(1.00000+CONSTANTS[(constants_size * offset) + KmCaMK]/ALGEBRAIC[(algebraic_size * offset) + CaMKa]);
ALGEBRAIC[(algebraic_size * offset) + Jleak] = ( 0.00488250*STATES[(states_size * offset) + cansr])/15.0000;
ALGEBRAIC[(algebraic_size * offset) + Jup] =  CONSTANTS[(constants_size * offset) + Jup_b]*(( (1.00000 - ALGEBRAIC[(algebraic_size * offset) + fJupp])*ALGEBRAIC[(algebraic_size * offset) + Jupnp]+ ALGEBRAIC[(algebraic_size * offset) + fJupp]*ALGEBRAIC[(algebraic_size * offset) + Jupp]) - ALGEBRAIC[(algebraic_size * offset) + Jleak]);
ALGEBRAIC[(algebraic_size * offset) + Bcai] = 1.00000/(1.00000+( CONSTANTS[(constants_size * offset) + cmdnmax]*CONSTANTS[(constants_size * offset) + kmcmdn])/pow(CONSTANTS[(constants_size * offset) + kmcmdn]+STATES[(states_size * offset) + cai], 2.00000)+( CONSTANTS[(constants_size * offset) + trpnmax]*CONSTANTS[(constants_size * offset) + kmtrpn])/pow(CONSTANTS[(constants_size * offset) + kmtrpn]+STATES[(states_size * offset) + cai], 2.00000));
ALGEBRAIC[(algebraic_size * offset) + Jtr] = (STATES[(states_size * offset) + cansr] - STATES[(states_size * offset) + cajsr])/60.0000;
ALGEBRAIC[(algebraic_size * offset) + Bcajsr] = 1.00000/(1.00000+( CONSTANTS[(constants_size * offset) + csqnmax]*CONSTANTS[(constants_size * offset) + kmcsqn])/pow(CONSTANTS[(constants_size * offset) + kmcsqn]+STATES[(states_size * offset) + cajsr], 2.00000));

RATES[ (states_size * offset) +hL] = (ALGEBRAIC[(algebraic_size * offset) + hLss] - STATES[(states_size * offset) + hL])/CONSTANTS[(constants_size * offset) + thL];
RATES[ (states_size * offset) +hLp] = (ALGEBRAIC[(algebraic_size * offset) + hLssp] - STATES[(states_size * offset) + hLp])/CONSTANTS[(constants_size * offset) + thLp];
RATES[ (states_size * offset) +jca] = (ALGEBRAIC[(algebraic_size * offset) + jcass] - STATES[(states_size * offset) + jca])/CONSTANTS[(constants_size * offset) + tjca];
RATES[ (states_size * offset) +m] = (ALGEBRAIC[(algebraic_size * offset) + mss] - STATES[(states_size * offset) + m])/ALGEBRAIC[(algebraic_size * offset) + tm];
RATES[ (states_size * offset) +mL] = (ALGEBRAIC[(algebraic_size * offset) + mLss] - STATES[(states_size * offset) + mL])/ALGEBRAIC[(algebraic_size * offset) + tmL];
RATES[ (states_size * offset) +a] = (ALGEBRAIC[(algebraic_size * offset) + ass] - STATES[(states_size * offset) + a])/ALGEBRAIC[(algebraic_size * offset) + ta];
RATES[ (states_size * offset) +d] = (ALGEBRAIC[(algebraic_size * offset) + dss] - STATES[(states_size * offset) + d])/ALGEBRAIC[(algebraic_size * offset) + td];
RATES[ (states_size * offset) +ff] = (ALGEBRAIC[(algebraic_size * offset) + fss] - STATES[(states_size * offset) + ff])/ALGEBRAIC[(algebraic_size * offset) + tff];
RATES[ (states_size * offset) +fs] = (ALGEBRAIC[(algebraic_size * offset) + fss] - STATES[(states_size * offset) + fs])/ALGEBRAIC[(algebraic_size * offset) + tfs];
RATES[ (states_size * offset) +nca_ss] =  ALGEBRAIC[(algebraic_size * offset) + anca_ss]*CONSTANTS[(constants_size * offset) + k2n] -  STATES[(states_size * offset) + nca_ss]*ALGEBRAIC[(algebraic_size * offset) + km2n];
RATES[ (states_size * offset) +nca_i] =  ALGEBRAIC[(algebraic_size * offset) + anca_i]*CONSTANTS[(constants_size * offset) + k2n] -  STATES[(states_size * offset) + nca_i]*ALGEBRAIC[(algebraic_size * offset) + km2n];
RATES[ (states_size * offset) +xs1] = (ALGEBRAIC[(algebraic_size * offset) + xs1ss] - STATES[(states_size * offset) + xs1])/ALGEBRAIC[(algebraic_size * offset) + txs1];
RATES[ (states_size * offset) +ap] = (ALGEBRAIC[(algebraic_size * offset) + assp] - STATES[(states_size * offset) + ap])/ALGEBRAIC[(algebraic_size * offset) + ta];
RATES[ (states_size * offset) +fcaf] = (ALGEBRAIC[(algebraic_size * offset) + fcass] - STATES[(states_size * offset) + fcaf])/ALGEBRAIC[(algebraic_size * offset) + tfcaf];
RATES[ (states_size * offset) +fcas] = (ALGEBRAIC[(algebraic_size * offset) + fcass] - STATES[(states_size * offset) + fcas])/ALGEBRAIC[(algebraic_size * offset) + tfcas];
RATES[ (states_size * offset) +ffp] = (ALGEBRAIC[(algebraic_size * offset) + fss] - STATES[(states_size * offset) + ffp])/ALGEBRAIC[(algebraic_size * offset) + tffp];
RATES[ (states_size * offset) +xs2] = (ALGEBRAIC[(algebraic_size * offset) + xs2ss] - STATES[(states_size * offset) + xs2])/ALGEBRAIC[(algebraic_size * offset) + txs2];
RATES[ (states_size * offset) +CaMKt] =  CONSTANTS[(constants_size * offset) + aCaMK]*ALGEBRAIC[(algebraic_size * offset) + CaMKb]*(ALGEBRAIC[(algebraic_size * offset) + CaMKb]+STATES[(states_size * offset) + CaMKt]) -  CONSTANTS[(constants_size * offset) + bCaMK]*STATES[(states_size * offset) + CaMKt];
RATES[ (states_size * offset) +h] = (ALGEBRAIC[(algebraic_size * offset) + hss] - STATES[(states_size * offset) + h])/ALGEBRAIC[(algebraic_size * offset) + th];
RATES[ (states_size * offset) +fcafp] = (ALGEBRAIC[(algebraic_size * offset) + fcass] - STATES[(states_size * offset) + fcafp])/ALGEBRAIC[(algebraic_size * offset) + tfcafp];
RATES[ (states_size * offset) +j] = (ALGEBRAIC[(algebraic_size * offset) + jss] - STATES[(states_size * offset) + j])/ALGEBRAIC[(algebraic_size * offset) + tj];
RATES[ (states_size * offset) +hp] = (ALGEBRAIC[(algebraic_size * offset) + hssp] - STATES[(states_size * offset) + hp])/ALGEBRAIC[(algebraic_size * offset) + th];
RATES[ (states_size * offset) +iF] = (ALGEBRAIC[(algebraic_size * offset) + iss] - STATES[(states_size * offset) + iF])/ALGEBRAIC[(algebraic_size * offset) + tiF];
RATES[ (states_size * offset) +C3] =  ALGEBRAIC[(algebraic_size * offset) + beta]*STATES[(states_size * offset) + C2] -  ALGEBRAIC[(algebraic_size * offset) + alpha]*STATES[(states_size * offset) + C3];
RATES[ (states_size * offset) +C2] = ( ALGEBRAIC[(algebraic_size * offset) + alpha]*STATES[(states_size * offset) + C3]+ CONSTANTS[(constants_size * offset) + beta_1]*STATES[(states_size * offset) + C1]) -  (ALGEBRAIC[(algebraic_size * offset) + beta]+CONSTANTS[(constants_size * offset) + alpha_1])*STATES[(states_size * offset) + C2];
RATES[ (states_size * offset) +jp] = (ALGEBRAIC[(algebraic_size * offset) + jss] - STATES[(states_size * offset) + jp])/ALGEBRAIC[(algebraic_size * offset) + tjp];
RATES[ (states_size * offset) +iS] = (ALGEBRAIC[(algebraic_size * offset) + iss] - STATES[(states_size * offset) + iS])/ALGEBRAIC[(algebraic_size * offset) + tiS];
RATES[ (states_size * offset) +O] = ( ALGEBRAIC[(algebraic_size * offset) + alpha_2]*STATES[(states_size * offset) + C1]+ ALGEBRAIC[(algebraic_size * offset) + beta_i]*STATES[(states_size * offset) + I]) -  (ALGEBRAIC[(algebraic_size * offset) + beta_2]+ALGEBRAIC[(algebraic_size * offset) + alpha_i])*STATES[(states_size * offset) + O];
RATES[ (states_size * offset) +iFp] = (ALGEBRAIC[(algebraic_size * offset) + iss] - STATES[(states_size * offset) + iFp])/ALGEBRAIC[(algebraic_size * offset) + tiFp];
RATES[ (states_size * offset) +iSp] = (ALGEBRAIC[(algebraic_size * offset) + iss] - STATES[(states_size * offset) + iSp])/ALGEBRAIC[(algebraic_size * offset) + tiSp];
RATES[ (states_size * offset) +C1] = ( CONSTANTS[(constants_size * offset) + alpha_1]*STATES[(states_size * offset) + C2]+ ALGEBRAIC[(algebraic_size * offset) + beta_2]*STATES[(states_size * offset) + O]+ ALGEBRAIC[(algebraic_size * offset) + beta_ItoC2]*STATES[(states_size * offset) + I]) -  (CONSTANTS[(constants_size * offset) + beta_1]+ALGEBRAIC[(algebraic_size * offset) + alpha_2]+ALGEBRAIC[(algebraic_size * offset) + alpha_C2ToI])*STATES[(states_size * offset) + C1];
RATES[ (states_size * offset) +I] = ( ALGEBRAIC[(algebraic_size * offset) + alpha_C2ToI]*STATES[(states_size * offset) + C1]+ ALGEBRAIC[(algebraic_size * offset) + alpha_i]*STATES[(states_size * offset) + O]) -  (ALGEBRAIC[(algebraic_size * offset) + beta_ItoC2]+ALGEBRAIC[(algebraic_size * offset) + beta_i])*STATES[(states_size * offset) + I];
RATES[ (states_size * offset) +Jrel_np] = (ALGEBRAIC[(algebraic_size * offset) + Jrel_inf] - STATES[(states_size * offset) + Jrel_np])/ALGEBRAIC[(algebraic_size * offset) + tau_rel];
RATES[ (states_size * offset) +Jrel_p] = (ALGEBRAIC[(algebraic_size * offset) + Jrel_infp] - STATES[(states_size * offset) + Jrel_p])/ALGEBRAIC[(algebraic_size * offset) + tau_relp];
RATES[ (states_size * offset) +ki] = ( - (((ALGEBRAIC[(algebraic_size * offset) + Ito]+ALGEBRAIC[(algebraic_size * offset) + IKr]+ALGEBRAIC[(algebraic_size * offset) + IKs]+ALGEBRAIC[(algebraic_size * offset) + IK1]+ALGEBRAIC[(algebraic_size * offset) + IKb]+ALGEBRAIC[(algebraic_size * offset) + I_katp]+ALGEBRAIC[(algebraic_size * offset) + Istim]) -  2.00000*ALGEBRAIC[(algebraic_size * offset) + INaK])+ALGEBRAIC[(algebraic_size * offset) + ICaK_i])*CONSTANTS[(constants_size * offset) + Acap])/( CONSTANTS[(constants_size * offset) + F]*CONSTANTS[(constants_size * offset) + vmyo])+( ALGEBRAIC[(algebraic_size * offset) + JdiffK]*CONSTANTS[(constants_size * offset) + vss])/CONSTANTS[(constants_size * offset) + vmyo];
RATES[ (states_size * offset) +kss] = ( - ALGEBRAIC[(algebraic_size * offset) + ICaK_ss]*CONSTANTS[(constants_size * offset) + Acap])/( CONSTANTS[(constants_size * offset) + F]*CONSTANTS[(constants_size * offset) + vss]) - ALGEBRAIC[(algebraic_size * offset) + JdiffK];
RATES[ (states_size * offset) +nai] = ( - (ALGEBRAIC[(algebraic_size * offset) + INa]+ALGEBRAIC[(algebraic_size * offset) + INaL]+ 3.00000*ALGEBRAIC[(algebraic_size * offset) + INaCa_i]+ALGEBRAIC[(algebraic_size * offset) + ICaNa_i]+ 3.00000*ALGEBRAIC[(algebraic_size * offset) + INaK]+ALGEBRAIC[(algebraic_size * offset) + INab])*CONSTANTS[(constants_size * offset) + Acap])/( CONSTANTS[(constants_size * offset) + F]*CONSTANTS[(constants_size * offset) + vmyo])+( ALGEBRAIC[(algebraic_size * offset) + JdiffNa]*CONSTANTS[(constants_size * offset) + vss])/CONSTANTS[(constants_size * offset) + vmyo];
RATES[ (states_size * offset) +nass] = ( - (ALGEBRAIC[(algebraic_size * offset) + ICaNa_ss]+ 3.00000*ALGEBRAIC[(algebraic_size * offset) + INaCa_ss])*CONSTANTS[(constants_size * offset) + Acap])/( CONSTANTS[(constants_size * offset) + F]*CONSTANTS[(constants_size * offset) + vss]) - ALGEBRAIC[(algebraic_size * offset) + JdiffNa];
RATES[ (states_size * offset) +cass] =  ALGEBRAIC[(algebraic_size * offset) + Bcass]*((( - (ALGEBRAIC[(algebraic_size * offset) + ICaL_ss] -  2.00000*ALGEBRAIC[(algebraic_size * offset) + INaCa_ss])*CONSTANTS[(constants_size * offset) + Acap])/( 2.00000*CONSTANTS[(constants_size * offset) + F]*CONSTANTS[(constants_size * offset) + vss])+( ALGEBRAIC[(algebraic_size * offset) + Jrel]*CONSTANTS[(constants_size * offset) + vjsr])/CONSTANTS[(constants_size * offset) + vss]) - ALGEBRAIC[(algebraic_size * offset) + Jdiff]);
RATES[ (states_size * offset) +V] = - (ALGEBRAIC[(algebraic_size * offset) + INa]+ALGEBRAIC[(algebraic_size * offset) + INaL]+ALGEBRAIC[(algebraic_size * offset) + Ito]+ALGEBRAIC[(algebraic_size * offset) + ICaL]+ALGEBRAIC[(algebraic_size * offset) + ICaNa]+ALGEBRAIC[(algebraic_size * offset) + ICaK]+ALGEBRAIC[(algebraic_size * offset) + IKr]+ALGEBRAIC[(algebraic_size * offset) + IKs]+ALGEBRAIC[(algebraic_size * offset) + IK1]+ALGEBRAIC[(algebraic_size * offset) + INaCa_i]+ALGEBRAIC[(algebraic_size * offset) + INaCa_ss]+ALGEBRAIC[(algebraic_size * offset) + INaK]+ALGEBRAIC[(algebraic_size * offset) + INab]+ALGEBRAIC[(algebraic_size * offset) + IKb]+ALGEBRAIC[(algebraic_size * offset) + IpCa]+ALGEBRAIC[(algebraic_size * offset) + ICab]+ALGEBRAIC[(algebraic_size * offset) + IClCa]+ALGEBRAIC[(algebraic_size * offset) + IClb]+ALGEBRAIC[(algebraic_size * offset) + I_katp]+ALGEBRAIC[(algebraic_size * offset) + Istim]);
RATES[ (states_size * offset) +cai] =  ALGEBRAIC[(algebraic_size * offset) + Bcai]*((( - ((ALGEBRAIC[(algebraic_size * offset) + ICaL_i]+ALGEBRAIC[(algebraic_size * offset) + IpCa]+ALGEBRAIC[(algebraic_size * offset) + ICab]) -  2.00000*ALGEBRAIC[(algebraic_size * offset) + INaCa_i])*CONSTANTS[(constants_size * offset) + Acap])/( 2.00000*CONSTANTS[(constants_size * offset) + F]*CONSTANTS[(constants_size * offset) + vmyo]) - ( ALGEBRAIC[(algebraic_size * offset) + Jup]*CONSTANTS[(constants_size * offset) + vnsr])/CONSTANTS[(constants_size * offset) + vmyo])+( ALGEBRAIC[(algebraic_size * offset) + Jdiff]*CONSTANTS[(constants_size * offset) + vss])/CONSTANTS[(constants_size * offset) + vmyo]);
RATES[ (states_size * offset) +cansr] = ALGEBRAIC[(algebraic_size * offset) + Jup] - ( ALGEBRAIC[(algebraic_size * offset) + Jtr]*CONSTANTS[(constants_size * offset) + vjsr])/CONSTANTS[(constants_size * offset) + vnsr];
RATES[ (states_size * offset) +cajsr] =  ALGEBRAIC[(algebraic_size * offset) + Bcajsr]*(ALGEBRAIC[(algebraic_size * offset) + Jtr] - ALGEBRAIC[(algebraic_size * offset) + Jrel]);
}

// void Tomek_model::solveRK4(double TIME, double dt)
// {
// 	double k1[43],k23[43];
// 	double yk123[43];
// 	int idx;


// 	// assuming first computeRates() have been executed
// 	computeRates( TIME, CONSTANTS, RATES, STATES, ALGEBRAIC );
// 	for( idx = 0; idx < states_size; idx++ ) {
// 		k1[idx] = RATES[ (states_size * offset) +idx];
// 		yk123[idx] = STATES[(states_size * offset) + idx] + (k1[idx]*dt*0.5);
// 	}
// 	computeRates( TIME+(dt*0.5), CONSTANTS, RATES, yk123, ALGEBRAIC );
// 	for( idx = 0; idx < states_size; idx++ ) {
// 		k23[idx] = RATES[ (states_size * offset) +idx];
// 		yk123[idx] = STATES[(states_size * offset) + idx] + (k23[idx]*dt*0.5);
// 	}
// 	computeRates( TIME+(dt*0.5), CONSTANTS, RATES, yk123, ALGEBRAIC );
//   for( idx = 0; idx < states_size; idx++ ) {
//     k23[idx] += RATES[ (states_size * offset) +idx];
//     yk123[idx] = STATES[(states_size * offset) + idx] + (k23[idx]*dt);
//   }
//   computeRates( TIME+dt, CONSTANTS, RATES, yk123, ALGEBRAIC );
// 	for( idx = 0; idx < states_size; idx++ ) {
// 		STATES[(states_size * offset) + (states_size * offset) + idx] += (k1[idx]+(2*k23[idx])+RATES[idx])/6. * dt;
//   }


// }

__device__ void solveAnalytical(double *CONSTANTS, double *STATES, double *ALGEBRAIC, double *RATES, double dt, int offset)
{
int algebraic_size = 223;
int constants_size = 163;
int states_size = 43;

#ifdef EULER
  STATES[(states_size * offset) + (states_size * offset) + V] = STATES[(states_size * offset) + V] + RATES[ (states_size * offset) +V] * dt;
  STATES[(states_size * offset) + CaMKt] = STATES[(states_size * offset) + CaMKt] + RATES[ (states_size * offset) +CaMKt] * dt;
  STATES[(states_size * offset) + cass] = STATES[(states_size * offset) + cass] + RATES[ (states_size * offset) +cass] * dt;
  STATES[(states_size * offset) + nai] = STATES[(states_size * offset) + nai] + RATES[ (states_size * offset) +nai] * dt;
  STATES[(states_size * offset) + nass] = STATES[(states_size * offset) + nass] + RATES[ (states_size * offset) +nass] * dt;
  STATES[(states_size * offset) + ki] = STATES[(states_size * offset) + ki] + RATES[ (states_size * offset) +ki] * dt;
  STATES[(states_size * offset) + kss] = STATES[(states_size * offset) + kss] + RATES[ (states_size * offset) +kss] * dt;
  STATES[(states_size * offset) + cansr] = STATES[(states_size * offset) + cansr] + RATES[ (states_size * offset) +cansr] * dt;
  STATES[(states_size * offset) + cajsr] = STATES[(states_size * offset) + cajsr] + RATES[ (states_size * offset) +cajsr] * dt;
  STATES[(states_size * offset) + cai] = STATES[(states_size * offset) + cai] + RATES[ (states_size * offset) +cai] * dt;
  STATES[(states_size * offset) + m] = STATES[(states_size * offset) + m] + RATES[ (states_size * offset) +m] * dt;
  STATES[(states_size * offset) + h] = STATES[(states_size * offset) + h] + RATES[ (states_size * offset) +h] * dt;
  STATES[(states_size * offset) + j] = STATES[(states_size * offset) + j] + RATES[ (states_size * offset) +j] * dt;
  STATES[(states_size * offset) + hp] = STATES[(states_size * offset) + hp] + RATES[ (states_size * offset) +hp] * dt;
  STATES[(states_size * offset) + jp] = STATES[(states_size * offset) + jp] + RATES[ (states_size * offset) +jp] * dt;
  STATES[(states_size * offset) + mL] = STATES[(states_size * offset) + mL] + RATES[ (states_size * offset) +mL] * dt;
  STATES[(states_size * offset) + hL] = STATES[(states_size * offset) + hL] + RATES[ (states_size * offset) +hL] * dt;
  STATES[(states_size * offset) + hLp] = STATES[(states_size * offset) + hLp] + RATES[ (states_size * offset) +hLp] * dt;
  STATES[(states_size * offset) + a] = STATES[(states_size * offset) + a] + RATES[ (states_size * offset) +a] * dt;
  STATES[(states_size * offset) + iF] = STATES[(states_size * offset) + iF] + RATES[ (states_size * offset) +iF] * dt;
  STATES[(states_size * offset) + iS] = STATES[(states_size * offset) + iS] + RATES[ (states_size * offset) +iS] * dt;
  STATES[(states_size * offset) + ap] = STATES[(states_size * offset) + ap] + RATES[ (states_size * offset) +ap] * dt;
  STATES[(states_size * offset) + iFp] = STATES[(states_size * offset) + iFp] + RATES[ (states_size * offset) +iFp] * dt;
  STATES[(states_size * offset) + iSp] = STATES[(states_size * offset) + iSp] + RATES[ (states_size * offset) +iSp] * dt;
  STATES[(states_size * offset) + d] = STATES[(states_size * offset) + d] + RATES[ (states_size * offset) +d] * dt;
  STATES[(states_size * offset) + ff] = STATES[(states_size * offset) + ff] + RATES[ (states_size * offset) +ff] * dt;
  STATES[(states_size * offset) + fs] = STATES[(states_size * offset) + fs] + RATES[ (states_size * offset) +fs] * dt;
  STATES[(states_size * offset) + fcaf] = STATES[(states_size * offset) + fcaf] + RATES[ (states_size * offset) +fcaf] * dt;
  STATES[(states_size * offset) + fcas] = STATES[(states_size * offset) + fcas] + RATES[ (states_size * offset) +fcas] * dt;
  STATES[(states_size * offset) + jca] = STATES[(states_size * offset) + jca] + RATES[ (states_size * offset) +jca] * dt;
  STATES[(states_size * offset) + ffp] = STATES[(states_size * offset) + ffp] + RATES[ (states_size * offset) +ffp] * dt;
  STATES[(states_size * offset) + fcafp] = STATES[(states_size * offset) + fcafp] + RATES[ (states_size * offset) +fcafp] * dt;
  STATES[(states_size * offset) + nca_ss] = STATES[(states_size * offset) + nca_ss] + RATES[ (states_size * offset) +nca_ss] * dt;
  STATES[(states_size * offset) + nca_i] = STATES[(states_size * offset) + nca_i] + RATES[ (states_size * offset) +nca_i] * dt;
  STATES[(states_size * offset) + O] = STATES[(states_size * offset) + O] + RATES[ (states_size * offset) +O] * dt;
  STATES[(states_size * offset) + I] = STATES[(states_size * offset) + I] + RATES[ (states_size * offset) +I] * dt;
	STATES[(states_size * offset) + C3] = STATES[(states_size * offset) + C3] + RATES[ (states_size * offset) +C3] * dt;
	STATES[(states_size * offset) + C2] = STATES[(states_size * offset) + C2] + RATES[ (states_size * offset) +C2] * dt;
	STATES[(states_size * offset) + C1] = STATES[(states_size * offset) + C1] + RATES[ (states_size * offset) +C1] * dt;
  STATES[(states_size * offset) + xs1] = STATES[(states_size * offset) + xs1] + RATES[ (states_size * offset) +xs1] * dt;
  STATES[(states_size * offset) + xs2] = STATES[(states_size * offset) + xs2] + RATES[ (states_size * offset) +xs2] * dt;
  STATES[(states_size * offset) + Jrel_np] = STATES[(states_size * offset) + Jrel_np] + RATES[ (states_size * offset) +Jrel_np] * dt;
  STATES[(states_size * offset) + Jrel_p] = STATES[(states_size * offset) + Jrel_p] + RATES[(states_size * offset) +Jrel_p] * dt;
#else
////==============
////Exact solution
////==============
////INa
  STATES[(states_size * offset) + m] = ALGEBRAIC[(algebraic_size * offset) + mss] - (ALGEBRAIC[(algebraic_size * offset) + mss] - STATES[(states_size * offset) + m]) * exp(-dt / ALGEBRAIC[(algebraic_size * offset) + tm]);
  STATES[(states_size * offset) + h] = ALGEBRAIC[(algebraic_size * offset) + hss] - (ALGEBRAIC[(algebraic_size * offset) + hss] - STATES[(states_size * offset) + h]) * exp(-dt / ALGEBRAIC[(algebraic_size * offset) + th]);
  STATES[(states_size * offset) + j] = ALGEBRAIC[(algebraic_size * offset) + jss] - (ALGEBRAIC[(algebraic_size * offset) + jss] - STATES[(states_size * offset) + j]) * exp(-dt / ALGEBRAIC[(algebraic_size * offset) + tj]);
  STATES[(states_size * offset) + hp] = ALGEBRAIC[(algebraic_size * offset) + hssp] - (ALGEBRAIC[(algebraic_size * offset) + hssp] - STATES[(states_size * offset) + hp]) * exp(-dt / ALGEBRAIC[(algebraic_size * offset) + th]);
  STATES[(states_size * offset) + jp] = ALGEBRAIC[(algebraic_size * offset) + jss] - (ALGEBRAIC[(algebraic_size * offset) + jss] - STATES[(states_size * offset) + jp]) * exp(-dt / ALGEBRAIC[(algebraic_size * offset) + tjp]);
  STATES[(states_size * offset) + mL] = ALGEBRAIC[(algebraic_size * offset) + mLss] - (ALGEBRAIC[(algebraic_size * offset) + mLss] - STATES[(states_size * offset) + mL]) * exp(-dt / ALGEBRAIC[(algebraic_size * offset) + tmL]);
  STATES[(states_size * offset) + hL] = ALGEBRAIC[(algebraic_size * offset) + hLss] - (ALGEBRAIC[(algebraic_size * offset) + hLss] - STATES[(states_size * offset) + hL]) * exp(-dt / CONSTANTS[(constants_size * offset) + thL]);
  STATES[(states_size * offset) + hLp] = ALGEBRAIC[(algebraic_size * offset) + hLssp] - (ALGEBRAIC[(algebraic_size * offset) + hLssp] - STATES[(states_size * offset) + hLp]) * exp(-dt / CONSTANTS[(constants_size * offset) + thLp]);
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
  STATES[(states_size * offset) + jca] = ALGEBRAIC[(algebraic_size * offset) + jcass] - (ALGEBRAIC[(algebraic_size * offset) + jcass] - STATES[(states_size * offset) + jca]) * exp(- dt / CONSTANTS[(constants_size * offset) + tjca]);
  STATES[(states_size * offset) + ffp] = ALGEBRAIC[(algebraic_size * offset) + fss] - (ALGEBRAIC[(algebraic_size * offset) + fss] - STATES[(states_size * offset) + ffp]) * exp(-dt / ALGEBRAIC[(algebraic_size * offset) + tffp]);
  STATES[(states_size * offset) + fcafp] = ALGEBRAIC[(algebraic_size * offset) + fcass] - (ALGEBRAIC[(algebraic_size * offset) + fcass] - STATES[(states_size * offset) + fcafp]) * exp(-d / ALGEBRAIC[(algebraic_size * offset) + tfcafp]);
	STATES[(states_size * offset) + nca_i] = STATES[(states_size * offset) + nca_i] + RATES[(states_size * offset) +nca_i]*dt;
	STATES[(states_size * offset) + nca_ss] = STATES[(states_size * offset) + nca_ss] + RATES[(states_size * offset) +nca_ss]*dt;
//  STATES[nca_i] = ALGEBRAIC[(algebraic_size * offset) + anca_i] * CONSTANTS[(constants_size * offset) + k2n] / ALGEBRAIC[(algebraic_size * offset) + km2n] -
//      (ALGEBRAIC[(algebraic_size * offset) + anca_i] * CONSTANTS[(constants_size * offset) + k2n] / ALGEBRAIC[km2n] - STATES[nca_i]) * exp(-ALGEBRAIC[km2n] * dt);
//  STATES[nca_ss] = ALGEBRAIC[anca_ss] * CONSTANTS[(constants_size * offset) + k2n] / ALGEBRAIC[km2n] -
//      (ALGEBRAIC[anca_ss] * CONSTANTS[(constants_size * offset) + k2n] / ALGEBRAIC[km2n] - STATES[nca_ss]) * exp(-ALGEBRAIC[km2n] * dt);
////IKr
  //STATES[O] = STATES[O] + RATES[O] * dt;
  //STATES[I] = STATES[I] + RATES[I] * dt;
  //STATES[C3] = STATES[C3] + RATES[C3] * dt;
  //STATES[C2] = STATES[C2] + RATES[C2] * dt;
  //STATES[C1] = STATES[C1] + RATES[C1] * dt;
  double* coeffs = new double[15];
  coeffs[0] = -  (ALGEBRAIC[(algebraic_size * offset) + beta_2]+ALGEBRAIC[(algebraic_size * offset) + alpha_i]);
  coeffs[1] = ALGEBRAIC[(algebraic_size * offset) + beta_i];
  coeffs[2] = ALGEBRAIC[(algebraic_size * offset) + alpha_2];
  coeffs[3] = ALGEBRAIC[(algebraic_size * offset) + alpha_i];
  coeffs[4] = -  (ALGEBRAIC[(algebraic_size * offset) + beta_ItoC2]+ALGEBRAIC[(algebraic_size * offset) + beta_i]);
  coeffs[5] = ALGEBRAIC[(algebraic_size * offset) + alpha_C2ToI];
  coeffs[6] = ALGEBRAIC[(algebraic_size * offset) + beta_2];
  coeffs[7] = ALGEBRAIC[(algebraic_size * offset) + beta_ItoC2];
  coeffs[8] = -  (CONSTANTS[(constants_size * offset) + beta_1]+ALGEBRAIC[(algebraic_size * offset) + alpha_2]+ALGEBRAIC[(algebraic_size * offset) + alpha_C2ToI]);
  coeffs[9] = CONSTANTS[(constants_size * offset) + alpha_1];
  coeffs[10] = CONSTANTS[(constants_size * offset) + beta_1];
  coeffs[11] = -  (ALGEBRAIC[(algebraic_size * offset) + beta]+CONSTANTS[(constants_size * offset) + alpha_1]);
  coeffs[12] = ALGEBRAIC[(algebraic_size * offset) + alpha];
  coeffs[13] = ALGEBRAIC[(algebraic_size * offset) + beta];
  coeffs[14] = -  ALGEBRAIC[(algebraic_size * offset) + alpha];
  int m = 5;
  double* a = new double[m*m]; // Flattened a
  a[0 * m + 0] = 1.0 - dt * coeffs[0];   a[0 * m + 1] = - dt * coeffs[1];     a[0 * m + 2] = - dt * coeffs[2];     a[0 * m + 3] = 0.0;                      a[0 * m + 4] = 0.0;
  a[1 * m + 0] = - dt * coeffs[3];       a[1 * m + 1] = 1.0 - dt * coeffs[4]; a[1 * m + 2] = - dt * coeffs[5];     a[1 * m + 3] = 0.0;                      a[1 * m + 4] = 0.0;
  a[2 * m + 0] = - dt * coeffs[6];       a[2 * m + 1] = - dt * coeffs[7];     a[2 * m + 2] = 1.0 - dt * coeffs[8]; a[2 * m + 3] = - dt * coeffs[9];         a[2 * m + 4] = 0.0;
  a[3 * m + 0] = 0.0;                    a[3 * m + 1] = 0.0;                  a[3 * m + 2] = - dt * coeffs[10];    a[3 * m + 3] = 1.0 - dt * coeffs[11];    a[3 * m + 4] = - dt * coeffs[12];
  a[4 * m + 0] = 0.0;                    a[4 * m + 1] = 0.0;                  a[4 * m + 2] = 0.0;                  a[4 * m + 3] = - dt * coeffs[13];;       a[4 * m + 4] = 1.0 - dt * coeffs[14];
  double* b = new double[m];
  b[0] = STATES[(states_size * offset) + O];
  b[1] = STATES[(states_size * offset) + I];
  b[2] = STATES[(states_size * offset) + C1];
  b[3] = STATES[(states_size * offset) + C2];
  b[4] = STATES[(states_size * offset) + C3];
  double* x = new double[m];
  for(int i = 0; i < m; i++){
    x[i] = 0.0;
  }
  ___gaussElimination(a,b,x,m);
  STATES[(states_size * offset) + O] = x[0];
  STATES[(states_size * offset) + I] = x[1];
  STATES[(states_size * offset) + C1] = x[2];
  STATES[(states_size * offset) + C2] = x[3];
  STATES[(states_size * offset) + C3] = x[4];
  delete[] coeffs;
  delete[] a;
  delete[] b;
  delete[] x;
  
////IKs
  STATES[(states_size * offset) + xs1] = ALGEBRAIC[(algebraic_size * offset) + xs1ss] - (ALGEBRAIC[(algebraic_size * offset) + xs1ss] - STATES[(states_size * offset) + xs1]) * exp(-dt / ALGEBRAIC[(algebraic_size * offset) + txs1]);
  STATES[(states_size * offset) + xs2] = ALGEBRAIC[(algebraic_size * offset) + xs2ss] - (ALGEBRAIC[(algebraic_size * offset) + xs2ss] - STATES[(states_size * offset) + xs2]) * exp(-dt / ALGEBRAIC[(algebraic_size * offset) + txs2]);
////IK1
////RyR receptors
  STATES[(states_size * offset) + Jrel_np] = ALGEBRAIC[(algebraic_size * offset) + Jrel_inf] - (ALGEBRAIC[(algebraic_size * offset) + Jrel_inf] - STATES[(states_size * offset) + Jrel_np]) * exp(-dt / ALGEBRAIC[(algebraic_size * offset) + tau_rel]);
  STATES[(states_size * offset) + Jrel_p] = ALGEBRAIC[(algebraic_size * offset) + Jrel_infp] - (ALGEBRAIC[(algebraic_size * offset) + Jrel_infp] - STATES[(states_size * offset) + Jrel_p]) * exp(-dt / ALGEBRAIC[(algebraic_size * offset) + tau_relp]);
////=============================
////Approximated solution (Euler)
////=============================
////CaMK
  STATES[(states_size * offset) + CaMKt] = STATES[(states_size * offset) + CaMKt] + RATES[(states_size * offset) +CaMKt] * dt;
////Membrane potential
  STATES[(states_size * offset) + V] = STATES[(states_size * offset) + V] + RATES[(states_size * offset) +V] * dt;
////Ion Concentrations and Buffers
  STATES[(states_size * offset) + nai] = STATES[(states_size * offset) + nai] + RATES[(states_size * offset) +nai] * dt;
  STATES[(states_size * offset) + nass] = STATES[(states_size * offset) + nass] + RATES[(states_size * offset) +nass] * dt;
  STATES[(states_size * offset) + ki] = STATES[(states_size * offset) + ki] + RATES[(states_size * offset) +ki] * dt;
  STATES[(states_size * offset) + kss] = STATES[(states_size * offset) + kss] + RATES[(states_size * offset) +kss] * dt;
  STATES[(states_size * offset) + cai] = STATES[(states_size * offset) + cai] + RATES[(states_size * offset) +cai] * dt;
  STATES[(states_size * offset) + cass] = STATES[(states_size * offset) + cass] + RATES[(states_size * offset) +cass] * dt;
  STATES[(states_size * offset) + cansr] = STATES[(states_size * offset) + cansr] + RATES[(states_size * offset) +cansr] * dt;
  STATES[(states_size * offset) + cajsr] = STATES[(states_size * offset) + cajsr] + RATES[(states_size * offset) +cajsr] * dt;
#endif

}

__device__ void ___gaussElimination(double *A, double *b, double *x, int N) {
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

// __device__ void set_time_step(double TIME,
//                                               double time_point,
//                                               double min_time_step,
//                                               double max_time_step,
//                                               double min_dV,
//                                               double max_d,
//                                               int offset) {
//  int constants_size = 163;
//  int rates_size = 43;

//  double time_step = min_time_step;
//  if (TIME <= time_point || (TIME - floor(TIME / CONSTANTS[(constants_size * offset) + BCL]) * CONSTANTS[(constants_size * offset) + BCL]) <= time_point) {
//     //printf("TIME <= time_point ms\n");
//     return time_step;
//     //printf("TIME = %E, dV = %E, time_step = %E\n",TIME, RATES[V] * time_step, time_step);
//   }
//   else {
//     //printf("TIME > time_point ms\n");
//     if (std::abs(RATES[(rates_size * offset) +V] * time_step) <= min_dV) {//Slow changes in V
//         //printf("dV/dt <= 0.2\n");
//         time_step = std::abs(max_dV / RATES[(rates_size * offset) +V]);
//         //Make sure time_step is between min time step and max_time_step
//         if (time_step < min_time_step) {
//             time_step = min_time_step;
//         }
//         else if (time_step > max_time_step) {
//             time_step = max_time_step;
//         }
//         //printf("TIME = %E, dV = %E, time_step = %E\n",TIME, RATES[V] * time_step, time_step);
//     }
//     else if (std::abs(RATES[(rates_size * offset) +V] * time_step) >= max_dV) {//Fast changes in V
//         //printf("dV/dt >= 0.8\n");
//         time_step = std::abs(min_dV / RATES[(rates_size * offset) +V]);
//         //Make sure time_step is not less than 0.005
//         if (time_step < min_time_step) {
//             time_step = min_time_step;
//         }
//         //printf("TIME = %E, dV = %E, time_step = %E\n",TIME, RATES[V] * time_step, time_step);
//     } else {
//         time_step = min_time_step;
//     }
//     return time_step;
//   }
// }

//using ord 2011 set time step
__device__ double set_time_step(double TIME, double time_point, double max_time_step, double *CONSTANTS, double *RATES, int offset) {
  double time_step = 0.005;
  int constants_size = 163;
  int rates_size = 43;

  if (TIME <= time_point || (TIME - floor(TIME / CONSTANTS[BCL + (offset * constants_size)]) * CONSTANTS[BCL + (offset * constants_size)]) <= time_point) {
    //printf("TIME <= time_point ms\n");
    return time_step;
    //printf("dV = %lf, time_step = %lf\n",RATES[V] * time_step, time_step);
  }
  else {
    //printf("TIME > time_point ms\n");
    if (std::abs(RATES[V + (offset * rates_size)] * time_step) <= 0.2) {//Slow changes in V
        // printf("dV/dt <= 0.2\n");
        time_step = std::abs(0.8 / RATES[V + (offset * rates_size)]);
        //Make sure time_step is between 0.005 and max_time_step
        if (time_step < 0.005) {
            time_step = 0.005;
        }
        else if (time_step > max_time_step) {
            time_step = max_time_step;
        }
        //printf("dV = %lf, time_step = %lf\n",std::abs(RATES[V] * time_step), time_step);
    }
    else if (std::abs(RATES[V + (offset * rates_size)] * time_step) >= 0.8) {//Fast changes in V
        // printf("dV/dt >= 0.8\n");
        time_step = std::abs(0.2 / RATES[V + (offset * rates_size)]);
        while (std::abs(RATES[V + (offset * rates_size)]  * time_step) >= 0.8 &&
               0.005 < time_step &&
               time_step < max_time_step) {
            time_step = time_step / 10.0;
            // printf("dV = %lf, time_step = %lf\n",std::abs(RATES[V] * time_step), time_step);
        }
    }
    // __syncthreads();
    return time_step;
  }
}
