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

__device__ void ___initConsts(double *CONSTANTS, double *STATES, double type, double bcl, int foset)
{
int constants_size = 163;
int states_size = 43;

CONSTANTS[(constants_size * foset) + celltype] = type;
CONSTANTS[(constants_size * foset) + nao] = 140.0;
CONSTANTS[(constants_size * foset) + cao] = 1.8;
CONSTANTS[(constants_size * foset) + ko] = 5.0;
CONSTANTS[(constants_size * foset) + clo] = 150.0;
CONSTANTS[(constants_size * foset) + R] = 8314;
CONSTANTS[(constants_size * foset) + T] = 310;
CONSTANTS[(constants_size * foset) + F] = 96485;
CONSTANTS[(constants_size * foset) + zna] = 1;
CONSTANTS[(constants_size * foset) + zca] = 2;
CONSTANTS[(constants_size * foset) + zk] = 1;
CONSTANTS[(constants_size * foset) + zcl] = -1;
CONSTANTS[(constants_size * foset) + L] = 0.01;
CONSTANTS[(constants_size * foset) + rad] = 0.0011;
STATES[(states_size * foset) + V] = (CONSTANTS[(constants_size * foset) + celltype]==1.00000 ?  -89.14 : CONSTANTS[(constants_size * foset) + celltype]==2.00000 ?  -89.1704 : -88.7638);
CONSTANTS[(constants_size * foset) + stim_start] = 10;
CONSTANTS[(constants_size * foset) + i_Stim_End] = 100000000000000000;
CONSTANTS[(constants_size * foset) + i_Stim_Amplitude] = -53;
CONSTANTS[(constants_size * foset) + BCL] = bcl;
CONSTANTS[(constants_size * foset) + i_Stim_PulseDuration] = 1.0;
CONSTANTS[(constants_size * foset) + KmCaMK] = 0.15;
CONSTANTS[(constants_size * foset) + aCaMK] = 0.05;
CONSTANTS[(constants_size * foset) + bCaMK] = 0.00068;
CONSTANTS[(constants_size * foset) + CaMKo] = 0.05;
CONSTANTS[(constants_size * foset) + KmCaM] = 0.0015;
STATES[(states_size * foset) + CaMKt] = (CONSTANTS[(constants_size * foset) + celltype]==1.00000 ? 0.0129 : CONSTANTS[(constants_size * foset) + celltype]==2.00000 ? 0.0192 : 0.0111);
STATES[(states_size * foset) + cass] = (CONSTANTS[(constants_size * foset) + celltype]==1.00000 ? 5.77E-05 : CONSTANTS[(constants_size * foset) + celltype]==2.00000 ? 6.58E-05 : 7.0305e-5);
CONSTANTS[(constants_size * foset) + cmdnmax_b] = 0.05;
CONSTANTS[(constants_size * foset) + kmcmdn] = 0.00238;
CONSTANTS[(constants_size * foset) + trpnmax] = 0.07;
CONSTANTS[(constants_size * foset) + kmtrpn] = 0.0005;
CONSTANTS[(constants_size * foset) + BSRmax] = 0.047;
CONSTANTS[(constants_size * foset) + KmBSR] = 0.00087;
CONSTANTS[(constants_size * foset) + BSLmax] = 1.124;
CONSTANTS[(constants_size * foset) + KmBSL] = 0.0087;
CONSTANTS[(constants_size * foset) + csqnmax] = 10;
CONSTANTS[(constants_size * foset) + kmcsqn] = 0.8;
STATES[(states_size * foset) + nai] = (CONSTANTS[(constants_size * foset) + celltype]==1.00000 ? 12.1025 : CONSTANTS[(constants_size * foset) + celltype]==2.00000 ? 15.0038 : 12.1025);
STATES[(states_size * foset) + nass] = (CONSTANTS[(constants_size * foset) + celltype]==1.00000 ? 12.8366 : CONSTANTS[(constants_size * foset) + celltype]==2.00000 ? 15.0043 : 12.1029);
STATES[(states_size * foset) + ki] = (CONSTANTS[(constants_size * foset) + celltype]==1.00000 ? 142.6951 : CONSTANTS[(constants_size * foset) + celltype]==2.00000 ? 143.0403 : 142.3002);
STATES[(states_size * foset) + kss] = (CONSTANTS[(constants_size * foset) + celltype]==1.00000 ? 142.6951 : CONSTANTS[(constants_size * foset) + celltype]==2.00000 ? 143.0402 : 142.3002);
STATES[(states_size * foset) + cansr] = (CONSTANTS[(constants_size * foset) + celltype]==1.00000 ? 1.8119 : CONSTANTS[(constants_size * foset) + celltype]==2.00000 ? 1.9557 : 1.5211);
STATES[(states_size * foset) + cajsr] = (CONSTANTS[(constants_size * foset) + celltype]==1.00000 ? 1.8102 : CONSTANTS[(constants_size * foset) + celltype]==2.00000 ? 1.9593 : 1.5214);
STATES[(states_size * foset) + cai] = (CONSTANTS[(constants_size * foset) + celltype]==1.00000 ? 6.63E-05 : CONSTANTS[(constants_size * foset) + celltype]==2.00000 ? 8.17E-05 : 8.1583e-05);
CONSTANTS[(constants_size * foset) + cli] = 24.0;
CONSTANTS[(constants_size * foset) + PKNa] = 0.01833;
CONSTANTS[(constants_size * foset) + gkatp] = 4.3195;
CONSTANTS[(constants_size * foset) + fkatp] = 0.0;
CONSTANTS[(constants_size * foset) + K_o_n] = 5;
CONSTANTS[(constants_size * foset) + A_atp] = 2;
CONSTANTS[(constants_size * foset) + K_atp] = 0.25;
STATES[(states_size * foset) + m] = (CONSTANTS[(constants_size * foset) + celltype]==1.00000 ? 7.43E-04 : CONSTANTS[(constants_size * foset) + celltype]==2.00000 ? 7.38E-04 : 8.0572e-4);
STATES[(states_size * foset) + h] = (CONSTANTS[(constants_size * foset) + celltype]==1.00000 ? 0.836 : CONSTANTS[(constants_size * foset) + celltype]==2.00000 ? 0.8365 : 0.8286);
STATES[(states_size * foset) + j] = (CONSTANTS[(constants_size * foset) + celltype]==1.00000 ? 0.8359 : CONSTANTS[(constants_size * foset) + celltype]==2.00000 ? 0.8363 : 0.8284);
STATES[(states_size * foset) + hp] = (CONSTANTS[(constants_size * foset) + celltype]==1.00000 ? 0.6828 : CONSTANTS[(constants_size * foset) + celltype]==2.00000 ? 0.6838 : 0.6707);
STATES[(states_size * foset) + jp] = (CONSTANTS[(constants_size * foset) + celltype]==1.00000 ? 0.8357 : CONSTANTS[(constants_size * foset) + celltype]==2.00000 ? 0.8358 : 0.8281);
CONSTANTS[(constants_size * foset) + GNa] = 11.7802;
STATES[(states_size * foset) + mL] = (CONSTANTS[(constants_size * foset) + celltype]==1.00000 ? 1.52E-04 : CONSTANTS[(constants_size * foset) + celltype]==2.00000 ? 1.51E-04 : 1.629e-4);
CONSTANTS[(constants_size * foset) + thL] = 200;
STATES[(states_size * foset) + hL] = (CONSTANTS[(constants_size * foset) + celltype]==1.00000 ? 0.5401 : CONSTANTS[(constants_size * foset) + celltype]==2.00000 ? 0.5327 : 0.5255);
STATES[(states_size * foset) + hLp] = (CONSTANTS[(constants_size * foset) + celltype]==1.00000 ? 0.3034 : CONSTANTS[(constants_size * foset) + celltype]==2.00000 ? 0.2834 : 0.2872);
CONSTANTS[(constants_size * foset) + GNaL_b] = 0.0279;
CONSTANTS[(constants_size * foset) + Gto_b] = 0.16;
STATES[(states_size * foset) + a] = (CONSTANTS[(constants_size * foset) + celltype]==1.00000 ? 9.27E-04 : CONSTANTS[(constants_size * foset) + celltype]==2.00000 ? 9.25E-04 : 9.5098e-4);
CONSTANTS[(constants_size * foset) + EKshift] = 0;
STATES[(states_size * foset) + iF] = 0.9996;
STATES[(states_size * foset) + iS] = (CONSTANTS[(constants_size * foset) + celltype]==1.00000 ? 0.9996 : CONSTANTS[(constants_size * foset) + celltype]==2.00000 ? 0.5671 : 0.5936);
STATES[(states_size * foset) + ap] = (CONSTANTS[(constants_size * foset) + celltype]==1.00000 ? 4.72E-04 : CONSTANTS[(constants_size * foset) + celltype]==2.00000 ? 4.71E-04 : 4.8454e-4);
STATES[(states_size * foset) + iFp] = 0.9996;
STATES[(states_size * foset) + iSp] = (CONSTANTS[(constants_size * foset) + celltype]==1.00000 ? 0.9996 : CONSTANTS[(constants_size * foset) + celltype]==2.00000 ? 0.6261 :0.6538);
CONSTANTS[(constants_size * foset) + Kmn] = 0.002;
CONSTANTS[(constants_size * foset) + k2n] = 500;
CONSTANTS[(constants_size * foset) + PCa_b] = 8.3757e-05;
STATES[(states_size * foset) + d] = (CONSTANTS[(constants_size * foset) + celltype]==1.00000 ? 0.0 : CONSTANTS[(constants_size * foset) + celltype]==2.00000 ? 0.0 : 8.1084e-9);
CONSTANTS[(constants_size * foset) + Aff] = 0.6;
STATES[(states_size * foset) + ff] = 1.0;
STATES[(states_size * foset) + fs] = (CONSTANTS[(constants_size * foset) + celltype]==1.00000 ? 0.9485 : CONSTANTS[(constants_size * foset) + celltype]==2.00000 ? 0.92 : 0.939);
STATES[(states_size * foset) + fcaf] = 1.0;
STATES[(states_size * foset) + fcas] = 0.9999;
STATES[(states_size * foset) + jca] = 1.0;
STATES[(states_size * foset) + ffp] = 1.0;
STATES[(states_size * foset) + fcafp] = 1.0;
STATES[(states_size * foset) + nca_ss] = (CONSTANTS[(constants_size * foset) + celltype]==1.00000 ? 3.09E-04 : CONSTANTS[(constants_size * foset) + celltype]==2.00000 ? 5.14E-04 : 6.6462e-4);
STATES[(states_size * foset) + nca_i] = (CONSTANTS[(constants_size * foset) + celltype]==1.00000 ? 5.30E-04 : CONSTANTS[(constants_size * foset) + celltype]==2.00000 ? 0.0012 : 0.0012);
CONSTANTS[(constants_size * foset) + tjca] = 75;
CONSTANTS[(constants_size * foset) + vShift] = 0;
CONSTANTS[(constants_size * foset) + offset] = 0;
CONSTANTS[(constants_size * foset) + dielConstant] = 74;
CONSTANTS[(constants_size * foset) + ICaL_fractionSS] = 0.8;
CONSTANTS[(constants_size * foset) + GKr_b] = 0.0321;
STATES[(states_size * foset) + C1] = (CONSTANTS[(constants_size * foset) + celltype]==1.00000 ? 6.79E-04 : CONSTANTS[(constants_size * foset) + celltype]==2.00000 ? 6.96E-04 : 7.0344e-4);
STATES[(states_size * foset) + C2] = (CONSTANTS[(constants_size * foset) + celltype]==1.00000 ? 8.29E-04 : CONSTANTS[(constants_size * foset) + celltype]==2.00000 ? 8.27E-04 : 8.5109e-4);
STATES[(states_size * foset) + C3] = (CONSTANTS[(constants_size * foset) + celltype]==1.00000 ? 0.9982 : CONSTANTS[(constants_size * foset) + celltype]==2.00000 ? 0.9979 : 0.9981);
STATES[(states_size * foset) + I] = (CONSTANTS[(constants_size * foset) + celltype]==1.00000 ? 9.54E-06 : CONSTANTS[(constants_size * foset) + celltype]==2.00000 ? 1.88E-05 : 1.3289e-5);
STATES[(states_size * foset) + O] = (CONSTANTS[(constants_size * foset) + celltype]==1.00000 ? 2.76E-04 : CONSTANTS[(constants_size * foset) + celltype]==2.00000 ? 5.42E-04 : 3.7585e-4);
CONSTANTS[(constants_size * foset) + alpha_1] = 0.154375;
CONSTANTS[(constants_size * foset) + beta_1] = 0.1911;
CONSTANTS[(constants_size * foset) + GKs_b] = 0.0011;
STATES[(states_size * foset) + xs1] = (CONSTANTS[(constants_size * foset) + celltype]==1.00000 ? 0.2309 : CONSTANTS[(constants_size * foset) + celltype]==2.00000 ? 0.2653 : 0.248);
STATES[(states_size * foset) + xs2] = (CONSTANTS[(constants_size * foset) + celltype]==1.00000 ? 1.70E-04 : CONSTANTS[(constants_size * foset) + celltype]==2.00000 ? 1.69E-04 : 1.7707e-4);
CONSTANTS[(constants_size * foset) + GK1_b] = 0.6992;
CONSTANTS[(constants_size * foset) + INaCa_fractionSS] = 0.35;
CONSTANTS[(constants_size * foset) + kna1] = 15;
CONSTANTS[(constants_size * foset) + kna2] = 5;
CONSTANTS[(constants_size * foset) + kna3] = 88.12;
CONSTANTS[(constants_size * foset) + kasymm] = 12.5;
CONSTANTS[(constants_size * foset) + wna] = 6e4;
CONSTANTS[(constants_size * foset) + wca] = 6e4;
CONSTANTS[(constants_size * foset) + wnaca] = 5e3;
CONSTANTS[(constants_size * foset) + kcaon] = 1.5e6;
CONSTANTS[(constants_size * foset) + kcaoff] = 5e3;
CONSTANTS[(constants_size * foset) + qna] = 0.5224;
CONSTANTS[(constants_size * foset) + qca] = 0.167;
CONSTANTS[(constants_size * foset) + KmCaAct] = 150e-6;
CONSTANTS[(constants_size * foset) + Gncx_b] = 0.0034;
CONSTANTS[(constants_size * foset) + k1p] = 949.5;
CONSTANTS[(constants_size * foset) + k1m] = 182.4;
CONSTANTS[(constants_size * foset) + k2p] = 687.2;
CONSTANTS[(constants_size * foset) + k2m] = 39.4;
CONSTANTS[(constants_size * foset) + k3p] = 1899;
CONSTANTS[(constants_size * foset) + k3m] = 79300;
CONSTANTS[(constants_size * foset) + k4p] = 639;
CONSTANTS[(constants_size * foset) + k4m] = 40;
CONSTANTS[(constants_size * foset) + Knai0] = 9.073;
CONSTANTS[(constants_size * foset) + Knao0] = 27.78;
CONSTANTS[(constants_size * foset) + delta] = -0.155;
CONSTANTS[(constants_size * foset) + Kki] = 0.5;
CONSTANTS[(constants_size * foset) + Kko] = 0.3582;
CONSTANTS[(constants_size * foset) + MgADP] = 0.05;
CONSTANTS[(constants_size * foset) + MgATP] = 9.8;
CONSTANTS[(constants_size * foset) + Kmgatp] = 1.698e-7;
CONSTANTS[(constants_size * foset) + H] = 1e-7;
CONSTANTS[(constants_size * foset) + eP] = 4.2;
CONSTANTS[(constants_size * foset) + Khp] = 1.698e-7;
CONSTANTS[(constants_size * foset) + Knap] = 224;
CONSTANTS[(constants_size * foset) + Kxkur] = 292;
CONSTANTS[(constants_size * foset) + Pnak_b] = 15.4509;
CONSTANTS[(constants_size * foset) + GKb_b] = 0.0189;
CONSTANTS[(constants_size * foset) + PNab] = 1.9239e-09;
CONSTANTS[(constants_size * foset) + PCab] = 5.9194e-08;
CONSTANTS[(constants_size * foset) + GpCa] = 5e-04;
CONSTANTS[(constants_size * foset) + KmCap] = 0.0005;
CONSTANTS[(constants_size * foset) + GClCa] = 0.2843;
CONSTANTS[(constants_size * foset) + GClb] = 1.98e-3;
CONSTANTS[(constants_size * foset) + KdClCa] = 0.1;
CONSTANTS[(constants_size * foset) + Fjunc] = 1;
CONSTANTS[(constants_size * foset) + tauNa] = 2.0;
CONSTANTS[(constants_size * foset) + tauK] = 2.0;
CONSTANTS[(constants_size * foset) + tauCa] = 0.2;
CONSTANTS[(constants_size * foset) + bt] = 4.75;
STATES[(states_size * foset) + Jrel_np] = (CONSTANTS[(constants_size * foset) + celltype]==1.00000 ? 2.82E-24 : CONSTANTS[(constants_size * foset) + celltype]==2.00000 ? 0. : 1.6129e-22);
STATES[(states_size * foset) + Jrel_p] = (CONSTANTS[(constants_size * foset) + celltype]==1.00000 ? 0. : CONSTANTS[(constants_size * foset) + celltype]==2.00000 ? 0. : 1.2475e-20);
CONSTANTS[(constants_size * foset) + cajsr_half] = 1.7;
CONSTANTS[(constants_size * foset) + Jrel_b] = 1.5378;
CONSTANTS[(constants_size * foset) + Jup_b] = 1.0;
CONSTANTS[(constants_size * foset) + vcell] =  1000.00*3.14000*CONSTANTS[(constants_size * foset) + rad]*CONSTANTS[(constants_size * foset) + rad]*CONSTANTS[(constants_size * foset) + L];
CONSTANTS[(constants_size * foset) + cmdnmax] = (CONSTANTS[(constants_size * foset) + celltype]==1.00000 ?  CONSTANTS[(constants_size * foset) + cmdnmax_b]*1.30000 : CONSTANTS[(constants_size * foset) + cmdnmax_b]);
CONSTANTS[(constants_size * foset) + ECl] =  (( CONSTANTS[(constants_size * foset) + R]*CONSTANTS[(constants_size * foset) + T])/( CONSTANTS[(constants_size * foset) + zcl]*CONSTANTS[(constants_size * foset) + F]))*log(CONSTANTS[(constants_size * foset) + clo]/CONSTANTS[(constants_size * foset) + cli]);
CONSTANTS[(constants_size * foset) + akik] = pow(CONSTANTS[(constants_size * foset) + ko]/CONSTANTS[(constants_size * foset) + K_o_n], 0.240000);
CONSTANTS[(constants_size * foset) + bkik] = 1.00000/(1.00000+pow(CONSTANTS[(constants_size * foset) + A_atp]/CONSTANTS[(constants_size * foset) + K_atp], 2.00000));
CONSTANTS[(constants_size * foset) + thLp] =  3.00000*CONSTANTS[(constants_size * foset) + thL];
CONSTANTS[(constants_size * foset) + GNaL] = (CONSTANTS[(constants_size * foset) + celltype]==1.00000 ?  CONSTANTS[(constants_size * foset) + GNaL_b]*0.600000 : CONSTANTS[(constants_size * foset) + GNaL_b]);
CONSTANTS[(constants_size * foset) + Gto] = (CONSTANTS[(constants_size * foset) + celltype]==1.00000 ?  CONSTANTS[(constants_size * foset) + Gto_b]*2.00000 : CONSTANTS[(constants_size * foset) + celltype]==2.00000 ?  CONSTANTS[(constants_size * foset) + Gto_b]*2.00000 : CONSTANTS[(constants_size * foset) + Gto_b]);
CONSTANTS[(constants_size * foset) + Afs] = 1.00000 - CONSTANTS[(constants_size * foset) + Aff];
CONSTANTS[(constants_size * foset) + PCa] = (CONSTANTS[(constants_size * foset) + celltype]==1.00000 ?  CONSTANTS[(constants_size * foset) + PCa_b]*1.20000 : CONSTANTS[(constants_size * foset) + celltype]==2.00000 ?  CONSTANTS[(constants_size * foset) + PCa_b]*2.00000 : CONSTANTS[(constants_size * foset) + PCa_b]);
CONSTANTS[(constants_size * foset) + Io] = ( 0.500000*(CONSTANTS[(constants_size * foset) + nao]+CONSTANTS[(constants_size * foset) + ko]+CONSTANTS[(constants_size * foset) + clo]+ 4.00000*CONSTANTS[(constants_size * foset) + cao]))/1000.00;
CONSTANTS[(constants_size * foset) + GKr] = (CONSTANTS[(constants_size * foset) + celltype]==1.00000 ?  CONSTANTS[(constants_size * foset) + GKr_b]*1.30000 : CONSTANTS[(constants_size * foset) + celltype]==2.00000 ?  CONSTANTS[(constants_size * foset) + GKr_b]*0.800000 : CONSTANTS[(constants_size * foset) + GKr_b]);
CONSTANTS[(constants_size * foset) + GKs] = (CONSTANTS[(constants_size * foset) + celltype]==1.00000 ?  CONSTANTS[(constants_size * foset) + GKs_b]*1.40000 : CONSTANTS[(constants_size * foset) + GKs_b]);
CONSTANTS[(constants_size * foset) + GK1] = (CONSTANTS[(constants_size * foset) + celltype]==1.00000 ?  CONSTANTS[(constants_size * foset) + GK1_b]*1.20000 : CONSTANTS[(constants_size * foset) + celltype]==2.00000 ?  CONSTANTS[(constants_size * foset) + GK1_b]*1.30000 : CONSTANTS[(constants_size * foset) + GK1_b]);
CONSTANTS[(constants_size * foset) + GKb] = (CONSTANTS[(constants_size * foset) + celltype]==1.00000 ?  CONSTANTS[(constants_size * foset) + GKb_b]*0.600000 : CONSTANTS[(constants_size * foset) + GKb_b]);
CONSTANTS[(constants_size * foset) + a_rel] = ( 0.500000*CONSTANTS[(constants_size * foset) + bt])/1.00000;
CONSTANTS[(constants_size * foset) + btp] =  1.25000*CONSTANTS[(constants_size * foset) + bt];
CONSTANTS[(constants_size * foset) + upScale] = (CONSTANTS[(constants_size * foset) + celltype]==1.00000 ? 1.30000 : 1.00000);
CONSTANTS[(constants_size * foset) + Ageo] =  2.00000*3.14000*CONSTANTS[(constants_size * foset) + rad]*CONSTANTS[(constants_size * foset) + rad]+ 2.00000*3.14000*CONSTANTS[(constants_size * foset) + rad]*CONSTANTS[(constants_size * foset) + L];
CONSTANTS[(constants_size * foset) + PCap] =  1.10000*CONSTANTS[(constants_size * foset) + PCa];
CONSTANTS[(constants_size * foset) + PCaNa] =  0.00125000*CONSTANTS[(constants_size * foset) + PCa];
CONSTANTS[(constants_size * foset) + PCaK] =  0.000357400*CONSTANTS[(constants_size * foset) + PCa];
CONSTANTS[(constants_size * foset) + constA] =  1.82000e+06*pow( CONSTANTS[(constants_size * foset) + dielConstant]*CONSTANTS[(constants_size * foset) + T], - 1.50000);
CONSTANTS[(constants_size * foset) + a_relp] = ( 0.500000*CONSTANTS[(constants_size * foset) + btp])/1.00000;
CONSTANTS[(constants_size * foset) + Acap] =  2.00000*CONSTANTS[(constants_size * foset) + Ageo];
CONSTANTS[(constants_size * foset) + PCaNap] =  0.00125000*CONSTANTS[(constants_size * foset) + PCap];
CONSTANTS[(constants_size * foset) + PCaKp] =  0.000357400*CONSTANTS[(constants_size * foset) + PCap];
CONSTANTS[(constants_size * foset) + gamma_cao] = exp( - CONSTANTS[(constants_size * foset) + constA]*4.00000*( pow(CONSTANTS[(constants_size * foset) + Io], 1.0 / 2)/(1.00000+ pow(CONSTANTS[(constants_size * foset) + Io], 1.0 / 2)) -  0.300000*CONSTANTS[(constants_size * foset) + Io]));
CONSTANTS[(constants_size * foset) + gamma_nao] = exp( - CONSTANTS[(constants_size * foset) + constA]*1.00000*( pow(CONSTANTS[(constants_size * foset) + Io], 1.0 / 2)/(1.00000+ pow(CONSTANTS[(constants_size * foset) + Io], 1.0 / 2)) -  0.300000*CONSTANTS[(constants_size * foset) + Io]));
CONSTANTS[(constants_size * foset) + gamma_ko] = exp( - CONSTANTS[(constants_size * foset) + constA]*1.00000*( pow(CONSTANTS[(constants_size * foset) + Io], 1.0 / 2)/(1.00000+ pow(CONSTANTS[(constants_size * foset) + Io], 1.0 / 2)) -  0.300000*CONSTANTS[(constants_size * foset) + Io]));
CONSTANTS[(constants_size * foset) + vmyo] =  0.680000*CONSTANTS[(constants_size * foset) + vcell];
CONSTANTS[(constants_size * foset) + vnsr] =  0.0552000*CONSTANTS[(constants_size * foset) + vcell];
CONSTANTS[(constants_size * foset) + vjsr] =  0.00480000*CONSTANTS[(constants_size * foset) + vcell];
CONSTANTS[(constants_size * foset) + vss] =  0.0200000*CONSTANTS[(constants_size * foset) + vcell];
CONSTANTS[(constants_size * foset) + h10_i] = CONSTANTS[(constants_size * foset) + kasymm]+1.00000+ (CONSTANTS[(constants_size * foset) + nao]/CONSTANTS[(constants_size * foset) + kna1])*(1.00000+CONSTANTS[(constants_size * foset) + nao]/CONSTANTS[(constants_size * foset) + kna2]);
CONSTANTS[(constants_size * foset) + h11_i] = ( CONSTANTS[(constants_size * foset) + nao]*CONSTANTS[(constants_size * foset) + nao])/( CONSTANTS[(constants_size * foset) + h10_i]*CONSTANTS[(constants_size * foset) + kna1]*CONSTANTS[(constants_size * foset) + kna2]);
CONSTANTS[(constants_size * foset) + h12_i] = 1.00000/CONSTANTS[(constants_size * foset) + h10_i];
CONSTANTS[(constants_size * foset) + k1_i] =  CONSTANTS[(constants_size * foset) + h12_i]*CONSTANTS[(constants_size * foset) + cao]*CONSTANTS[(constants_size * foset) + kcaon];
CONSTANTS[(constants_size * foset) + k2_i] = CONSTANTS[(constants_size * foset) + kcaoff];
CONSTANTS[(constants_size * foset) + k5_i] = CONSTANTS[(constants_size * foset) + kcaoff];
CONSTANTS[(constants_size * foset) + Gncx] = (CONSTANTS[(constants_size * foset) + celltype]==1.00000 ?  CONSTANTS[(constants_size * foset) + Gncx_b]*1.10000 : CONSTANTS[(constants_size * foset) + celltype]==2.00000 ?  CONSTANTS[(constants_size * foset) + Gncx_b]*1.40000 : CONSTANTS[(constants_size * foset) + Gncx_b]);
CONSTANTS[(constants_size * foset) + h10_ss] = CONSTANTS[(constants_size * foset) + kasymm]+1.00000+ (CONSTANTS[(constants_size * foset) + nao]/CONSTANTS[(constants_size * foset) + kna1])*(1.00000+CONSTANTS[(constants_size * foset) + nao]/CONSTANTS[(constants_size * foset) + kna2]);
CONSTANTS[(constants_size * foset) + h11_ss] = ( CONSTANTS[(constants_size * foset) + nao]*CONSTANTS[(constants_size * foset) + nao])/( CONSTANTS[(constants_size * foset) + h10_ss]*CONSTANTS[(constants_size * foset) + kna1]*CONSTANTS[(constants_size * foset) + kna2]);
CONSTANTS[(constants_size * foset) + h12_ss] = 1.00000/CONSTANTS[(constants_size * foset) + h10_ss];
CONSTANTS[(constants_size * foset) + k1_ss] =  CONSTANTS[(constants_size * foset) + h12_ss]*CONSTANTS[(constants_size * foset) + cao]*CONSTANTS[(constants_size * foset) + kcaon];
CONSTANTS[(constants_size * foset) + k2_ss] = CONSTANTS[(constants_size * foset) + kcaoff];
CONSTANTS[(constants_size * foset) + k5_ss] = CONSTANTS[(constants_size * foset) + kcaoff];
CONSTANTS[(constants_size * foset) + b1] =  CONSTANTS[(constants_size * foset) + k1m]*CONSTANTS[(constants_size * foset) + MgADP];
CONSTANTS[(constants_size * foset) + a2] = CONSTANTS[(constants_size * foset) + k2p];
CONSTANTS[(constants_size * foset) + a4] = (( CONSTANTS[(constants_size * foset) + k4p]*CONSTANTS[(constants_size * foset) + MgATP])/CONSTANTS[(constants_size * foset) + Kmgatp])/(1.00000+CONSTANTS[(constants_size * foset) + MgATP]/CONSTANTS[(constants_size * foset) + Kmgatp]);
CONSTANTS[(constants_size * foset) + Pnak] = (CONSTANTS[(constants_size * foset) + celltype]==1.00000 ?  CONSTANTS[(constants_size * foset) + Pnak_b]*0.900000 : CONSTANTS[(constants_size * foset) + celltype]==2.00000 ?  CONSTANTS[(constants_size * foset) + Pnak_b]*0.700000 : CONSTANTS[(constants_size * foset) + Pnak_b]);
}

__device__ void applyDrugEffect(double *CONSTANTS, double conc, double *hill, double epsilon, int foset)
{

int constant_size = 163;

CONSTANTS[(constant_size * foset) + PCa_b] = CONSTANTS[(constant_size * foset) + PCa_b] * ((hill[(14 * foset) + 0] > epsilon && hill[(14 * foset) + 1] > epsilon) ? 1./(1.+pow(conc/hill[(14 * foset) + 0],hill[(14 * foset) + 1])) : 1.);
CONSTANTS[(constant_size * foset) + GK1_b] = CONSTANTS[(constant_size * foset) + GK1_b] * ((hill[(14 * foset) + 2] > epsilon && hill[(14 * foset) + 3] > epsilon) ? 1./(1.+pow(conc/hill[(14 * foset) + 2],hill[(14 * foset) + 3])) : 1.);
CONSTANTS[(constant_size * foset) + GKs_b] = CONSTANTS[(constant_size * foset) + GKs_b] * ((hill[(14 * foset) + 4] > epsilon && hill[(14 * foset) + 5] > epsilon) ? 1./(1.+pow(conc/hill[(14 * foset) + 4],hill[(14 * foset) + 5])) : 1.);
CONSTANTS[(constant_size * foset) + GNa] = CONSTANTS[(constant_size * foset) + GNa] * ((hill[(14 * foset) + 6] > epsilon && hill[(14 * foset) + 7] > epsilon) ? 1./(1.+pow(conc/hill[(14 * foset) + 6],hill[(14 * foset) + 7])) : 1.);
CONSTANTS[(constant_size * foset) + GNaL_b] = CONSTANTS[(constant_size * foset) + GNaL_b] * ((hill[(14 * foset) + 8] > epsilon && hill[(14 * foset) + 9] > epsilon) ? 1./(1.+pow(conc/hill[(14 * foset) + 8],hill[(14 * foset) + 9])) : 1.);
CONSTANTS[(constant_size * foset) + Gto_b] = CONSTANTS[(constant_size * foset) + Gto_b] * ((hill[(14 * foset) + 10] > epsilon && hill[(14 * foset) + 11] > epsilon) ? 1./(1.+pow(conc/hill[(14 * foset) + 10],hill[(14 * foset) + 11])) : 1.);
CONSTANTS[(constant_size * foset) + GKr_b] = CONSTANTS[(constant_size * foset) + GKr_b] * ((hill[(14 * foset) + 12] > epsilon && hill[(14 * foset) + 13] > epsilon) ? 1./(1.+pow(conc/hill[(14 * foset) + 12],hill[(14 * foset) + 13])) : 1.);


}


__device__ void initConsts(double *CONSTANTS, double *STATES, double type, double conc, double *hill, double *cvar, bool is_cvar, double bcl, double epsilon, int foset)
{
	___initConsts(CONSTANTS, STATES, type, bcl, foset); // clean up later
	
}

__device__ void computeRates(double TIME, double *CONSTANTS, double *RATES, double *STATES, double *ALGEBRAIC, int foset)
{
int algebraic_size = 223;
int constants_size = 163;
int states_size = 43;

//addition from libcml
CONSTANTS[(constants_size * foset) + cmdnmax] = (CONSTANTS[(constants_size * foset) + celltype]==1.00000 ?  CONSTANTS[(constants_size * foset) + cmdnmax_b]*1.30000 : CONSTANTS[(constants_size * foset) + cmdnmax_b]);
CONSTANTS[(constants_size * foset) + GNaL] = (CONSTANTS[(constants_size * foset) + celltype]==1.00000 ?  CONSTANTS[(constants_size * foset) + GNaL_b]*0.600000 : CONSTANTS[(constants_size * foset) + GNaL_b]);
CONSTANTS[(constants_size * foset) + Gto] = (CONSTANTS[(constants_size * foset) + celltype]==1.00000 ?  CONSTANTS[(constants_size * foset) + Gto_b]*2.00000 : CONSTANTS[(constants_size * foset) + celltype]==2.00000 ?  CONSTANTS[(constants_size * foset) + Gto_b]*2.00000 : CONSTANTS[(constants_size * foset) + Gto_b]);
CONSTANTS[(constants_size * foset) + PCa] = (CONSTANTS[(constants_size * foset) + celltype]==1.00000 ?  CONSTANTS[(constants_size * foset) + PCa_b]*1.20000 : CONSTANTS[(constants_size * foset) + celltype]==2.00000 ?  CONSTANTS[(constants_size * foset) + PCa_b]*2.00000 : CONSTANTS[(constants_size * foset) + PCa_b]);
CONSTANTS[(constants_size * foset) + GKr] = (CONSTANTS[(constants_size * foset) + celltype]==1.00000 ?  CONSTANTS[(constants_size * foset) + GKr_b]*1.30000 : CONSTANTS[(constants_size * foset) + celltype]==2.00000 ?  CONSTANTS[(constants_size * foset) + GKr_b]*0.800000 : CONSTANTS[(constants_size * foset) + GKr_b]);
CONSTANTS[(constants_size * foset) + GKs] = (CONSTANTS[(constants_size * foset) + celltype]==1.00000 ?  CONSTANTS[(constants_size * foset) + GKs_b]*1.40000 : CONSTANTS[(constants_size * foset) + GKs_b]);
CONSTANTS[(constants_size * foset) + GK1] = (CONSTANTS[(constants_size * foset) + celltype]==1.00000 ?  CONSTANTS[(constants_size * foset) + GK1_b]*1.20000 : CONSTANTS[(constants_size * foset) + celltype]==2.00000 ?  CONSTANTS[(constants_size * foset) + GK1_b]*1.30000 : CONSTANTS[(constants_size * foset) + GK1_b]);
CONSTANTS[(constants_size * foset) + GKb] = (CONSTANTS[(constants_size * foset) + celltype]==1.00000 ?  CONSTANTS[(constants_size * foset) + GKb_b]*0.600000 : CONSTANTS[(constants_size * foset) + GKb_b]);
CONSTANTS[(constants_size * foset) + upScale] = (CONSTANTS[(constants_size * foset) + celltype]==1.00000 ? 1.30000 : 1.00000);
CONSTANTS[(constants_size * foset) + Gncx] = (CONSTANTS[(constants_size * foset) + celltype]==1.00000 ?  CONSTANTS[(constants_size * foset) + Gncx_b]*1.10000 : CONSTANTS[(constants_size * foset) + celltype]==2.00000 ?  CONSTANTS[(constants_size * foset) + Gncx_b]*1.40000 : CONSTANTS[(constants_size * foset) + Gncx_b]);


ALGEBRAIC[(algebraic_size * foset) + hLss] = 1.00000/(1.00000+exp((STATES[(states_size * foset) + V]+87.6100)/7.48800));
ALGEBRAIC[(algebraic_size * foset) + hLssp] = 1.00000/(1.00000+exp((STATES[(states_size * foset) + V]+93.8100)/7.48800));
ALGEBRAIC[(algebraic_size * foset) + jcass] = 1.00000/(1.00000+exp((STATES[(states_size * foset) + V]+18.0800)/2.79160));
ALGEBRAIC[(algebraic_size * foset) + mss] = 1.00000/pow(1.00000+exp(- (STATES[(states_size * foset) + V]+56.8600)/9.03000), 2.00000);
ALGEBRAIC[(algebraic_size * foset) + tm] =  0.129200*exp(- pow((STATES[(states_size * foset) + V]+45.7900)/15.5400, 2.00000))+ 0.0648700*exp(- pow((STATES[(states_size * foset) + V] - 4.82300)/51.1200, 2.00000));
ALGEBRAIC[(algebraic_size * foset) + mLss] = 1.00000/(1.00000+exp(- (STATES[(states_size * foset) + V]+42.8500)/5.26400));
ALGEBRAIC[(algebraic_size * foset) + tmL] =  0.129200*exp(- pow((STATES[(states_size * foset) + V]+45.7900)/15.5400, 2.00000))+ 0.0648700*exp(- pow((STATES[(states_size * foset) + V] - 4.82300)/51.1200, 2.00000));
ALGEBRAIC[(algebraic_size * foset) + ass] = 1.00000/(1.00000+exp(- ((STATES[(states_size * foset) + V]+CONSTANTS[(constants_size * foset) + EKshift]) - 14.3400)/14.8200));
ALGEBRAIC[(algebraic_size * foset) + ta] = 1.05150/(1.00000/( 1.20890*(1.00000+exp(- ((STATES[(states_size * foset) + V]+CONSTANTS[(constants_size * foset) + EKshift]) - 18.4099)/29.3814)))+3.50000/(1.00000+exp((STATES[(states_size * foset) + V]+CONSTANTS[(constants_size * foset) + EKshift]+100.000)/29.3814)));
ALGEBRAIC[(algebraic_size * foset) + dss] = (STATES[(states_size * foset) + V]>=31.4978 ? 1.00000 :  1.07630*exp( - 1.00700*exp( - 0.0829000*STATES[(states_size * foset) + V])));
ALGEBRAIC[(algebraic_size * foset) + td] = CONSTANTS[(constants_size * foset) + offset]+0.600000+1.00000/(exp( - 0.0500000*(STATES[(states_size * foset) + V]+CONSTANTS[(constants_size * foset) + vShift]+6.00000))+exp( 0.0900000*(STATES[(states_size * foset) + V]+CONSTANTS[(constants_size * foset) + vShift]+14.0000)));
ALGEBRAIC[(algebraic_size * foset) + fss] = 1.00000/(1.00000+exp((STATES[(states_size * foset) + V]+19.5800)/3.69600));
ALGEBRAIC[(algebraic_size * foset) + tff] = 7.00000+1.00000/( 0.00450000*exp(- (STATES[(states_size * foset) + V]+20.0000)/10.0000)+ 0.00450000*exp((STATES[(states_size * foset) + V]+20.0000)/10.0000));
ALGEBRAIC[(algebraic_size * foset) + tfs] = 1000.00+1.00000/( 3.50000e-05*exp(- (STATES[(states_size * foset) + V]+5.00000)/4.00000)+ 3.50000e-05*exp((STATES[(states_size * foset) + V]+5.00000)/6.00000));
ALGEBRAIC[(algebraic_size * foset) + km2n] =  STATES[(states_size * foset) + jca]*1.00000;
ALGEBRAIC[(algebraic_size * foset) + anca_ss] = 1.00000/(CONSTANTS[(constants_size * foset) + k2n]/ALGEBRAIC[(algebraic_size * foset) + km2n]+pow(1.00000+CONSTANTS[(constants_size * foset) + Kmn]/STATES[(states_size * foset) + cass], 4.00000));
ALGEBRAIC[(algebraic_size * foset) + anca_i] = 1.00000/(CONSTANTS[(constants_size * foset) + k2n]/ALGEBRAIC[(algebraic_size * foset) + km2n]+pow(1.00000+CONSTANTS[(constants_size * foset) + Kmn]/STATES[(states_size * foset) + cai], 4.00000));
ALGEBRAIC[(algebraic_size * foset) + xs1ss] = 1.00000/(1.00000+exp(- (STATES[(states_size * foset) + V]+11.6000)/8.93200));
ALGEBRAIC[(algebraic_size * foset) + txs1] = 817.300+1.00000/( 0.000232600*exp((STATES[(states_size * foset) + V]+48.2800)/17.8000)+ 0.00129200*exp(- (STATES[(states_size * foset) + V]+210.000)/230.000));
ALGEBRAIC[(algebraic_size * foset) + assp] = 1.00000/(1.00000+exp(- ((STATES[(states_size * foset) + V]+CONSTANTS[(constants_size * foset) + EKshift]) - 24.3400)/14.8200));
ALGEBRAIC[(algebraic_size * foset) + fcass] = ALGEBRAIC[(algebraic_size * foset) + fss];
ALGEBRAIC[(algebraic_size * foset) + tfcaf] = 7.00000+1.00000/( 0.0400000*exp(- (STATES[(states_size * foset) + V] - 4.00000)/7.00000)+ 0.0400000*exp((STATES[(states_size * foset) + V] - 4.00000)/7.00000));
ALGEBRAIC[(algebraic_size * foset) + tfcas] = 100.000+1.00000/( 0.000120000*exp(- STATES[(states_size * foset) + V]/3.00000)+ 0.000120000*exp(STATES[(states_size * foset) + V]/7.00000));
ALGEBRAIC[(algebraic_size * foset) + tffp] =  2.50000*ALGEBRAIC[(algebraic_size * foset) + tff];
ALGEBRAIC[(algebraic_size * foset) + xs2ss] = ALGEBRAIC[(algebraic_size * foset) + xs1ss];
ALGEBRAIC[(algebraic_size * foset) + txs2] = 1.00000/( 0.0100000*exp((STATES[(states_size * foset) + V] - 50.0000)/20.0000)+ 0.0193000*exp(- (STATES[(states_size * foset) + V]+66.5400)/31.0000));
ALGEBRAIC[(algebraic_size * foset) + CaMKb] = ( CONSTANTS[(constants_size * foset) + CaMKo]*(1.00000 - STATES[(states_size * foset) + CaMKt]))/(1.00000+CONSTANTS[(constants_size * foset) + KmCaM]/STATES[(states_size * foset) + cass]);
ALGEBRAIC[(algebraic_size * foset) + hss] = 1.00000/pow(1.00000+exp((STATES[(states_size * foset) + V]+71.5500)/7.43000), 2.00000);
ALGEBRAIC[(algebraic_size * foset) + ah] = (STATES[(states_size * foset) + V]>=- 40.0000 ? 0.00000 :  0.0570000*exp(- (STATES[(states_size * foset) + V]+80.0000)/6.80000));
ALGEBRAIC[(algebraic_size * foset) + bh] = (STATES[(states_size * foset) + V]>=- 40.0000 ? 0.770000/( 0.130000*(1.00000+exp(- (STATES[(states_size * foset) + V]+10.6600)/11.1000))) :  2.70000*exp( 0.0790000*STATES[(states_size * foset) + V])+ 310000.*exp( 0.348500*STATES[(states_size * foset) + V]));
ALGEBRAIC[(algebraic_size * foset) + th] = 1.00000/(ALGEBRAIC[(algebraic_size * foset) + ah]+ALGEBRAIC[(algebraic_size * foset) + bh]);
ALGEBRAIC[(algebraic_size * foset) + tfcafp] =  2.50000*ALGEBRAIC[(algebraic_size * foset) + tfcaf];
ALGEBRAIC[(algebraic_size * foset) + jss] = ALGEBRAIC[(algebraic_size * foset) + hss];
ALGEBRAIC[(algebraic_size * foset) + aj] = (STATES[(states_size * foset) + V]>=- 40.0000 ? 0.00000 : ( ( - 25428.0*exp( 0.244400*STATES[(states_size * foset) + V]) -  6.94800e-06*exp( - 0.0439100*STATES[(states_size * foset) + V]))*(STATES[(states_size * foset) + V]+37.7800))/(1.00000+exp( 0.311000*(STATES[(states_size * foset) + V]+79.2300))));
ALGEBRAIC[(algebraic_size * foset) + bj] = (STATES[(states_size * foset) + V]>=- 40.0000 ? ( 0.600000*exp( 0.0570000*STATES[(states_size * foset) + V]))/(1.00000+exp( - 0.100000*(STATES[(states_size * foset) + V]+32.0000))) : ( 0.0242400*exp( - 0.0105200*STATES[(states_size * foset) + V]))/(1.00000+exp( - 0.137800*(STATES[(states_size * foset) + V]+40.1400))));
ALGEBRAIC[(algebraic_size * foset) + tj] = 1.00000/(ALGEBRAIC[(algebraic_size * foset) + aj]+ALGEBRAIC[(algebraic_size * foset) + bj]);
ALGEBRAIC[(algebraic_size * foset) + hssp] = 1.00000/pow(1.00000+exp((STATES[(states_size * foset) + V]+77.5500)/7.43000), 2.00000);
ALGEBRAIC[(algebraic_size * foset) + iss] = 1.00000/(1.00000+exp((STATES[(states_size * foset) + V]+CONSTANTS[(constants_size * foset) + EKshift]+43.9400)/5.71100));
ALGEBRAIC[(algebraic_size * foset) + delta_epi] = (CONSTANTS[(constants_size * foset) + celltype]==1.00000 ? 1.00000 - 0.950000/(1.00000+exp((STATES[(states_size * foset) + V]+CONSTANTS[(constants_size * foset) + EKshift]+70.0000)/5.00000)) : 1.00000);
ALGEBRAIC[(algebraic_size * foset) + tiF_b] = 4.56200+1.00000/( 0.393300*exp(- (STATES[(states_size * foset) + V]+CONSTANTS[(constants_size * foset) + EKshift]+100.000)/100.000)+ 0.0800400*exp((STATES[(states_size * foset) + V]+CONSTANTS[(constants_size * foset) + EKshift]+50.0000)/16.5900));
ALGEBRAIC[(algebraic_size * foset) + tiF] =  ALGEBRAIC[(algebraic_size * foset) + tiF_b]*ALGEBRAIC[(algebraic_size * foset) + delta_epi];
ALGEBRAIC[(algebraic_size * foset) + vfrt] = ( STATES[(states_size * foset) + V]*CONSTANTS[(constants_size * foset) + F])/( CONSTANTS[(constants_size * foset) + R]*CONSTANTS[(constants_size * foset) + T]);
ALGEBRAIC[(algebraic_size * foset) + alpha] =  0.116100*exp( 0.299000*ALGEBRAIC[(algebraic_size * foset) + vfrt]);
ALGEBRAIC[(algebraic_size * foset) + beta] =  0.244200*exp( - 1.60400*ALGEBRAIC[(algebraic_size * foset) + vfrt]);
ALGEBRAIC[(algebraic_size * foset) + tjp] =  1.46000*ALGEBRAIC[(algebraic_size * foset) + tj];
ALGEBRAIC[(algebraic_size * foset) + tiS_b] = 23.6200+1.00000/( 0.00141600*exp(- (STATES[(states_size * foset) + V]+CONSTANTS[(constants_size * foset) + EKshift]+96.5200)/59.0500)+ 1.78000e-08*exp((STATES[(states_size * foset) + V]+CONSTANTS[(constants_size * foset) + EKshift]+114.100)/8.07900));
ALGEBRAIC[(algebraic_size * foset) + tiS] =  ALGEBRAIC[(algebraic_size * foset) + tiS_b]*ALGEBRAIC[(algebraic_size * foset) + delta_epi];
ALGEBRAIC[(algebraic_size * foset) + alpha_2] =  0.0578000*exp( 0.971000*ALGEBRAIC[(algebraic_size * foset) + vfrt]);
ALGEBRAIC[(algebraic_size * foset) + beta_2] =  0.000349000*exp( - 1.06200*ALGEBRAIC[(algebraic_size * foset) + vfrt]);
ALGEBRAIC[(algebraic_size * foset) + alpha_i] =  0.253300*exp( 0.595300*ALGEBRAIC[(algebraic_size * foset) + vfrt]);
ALGEBRAIC[(algebraic_size * foset) + beta_i] =  0.0652500*exp( - 0.820900*ALGEBRAIC[(algebraic_size * foset) + vfrt]);
ALGEBRAIC[(algebraic_size * foset) + dti_develop] = 1.35400+0.000100000/(exp(((STATES[(states_size * foset) + V]+CONSTANTS[(constants_size * foset) + EKshift]) - 167.400)/15.8900)+exp(- ((STATES[(states_size * foset) + V]+CONSTANTS[(constants_size * foset) + EKshift]) - 12.2300)/0.215400));
ALGEBRAIC[(algebraic_size * foset) + dti_recover] = 1.00000 - 0.500000/(1.00000+exp((STATES[(states_size * foset) + V]+CONSTANTS[(constants_size * foset) + EKshift]+70.0000)/20.0000));
ALGEBRAIC[(algebraic_size * foset) + tiFp] =  ALGEBRAIC[(algebraic_size * foset) + dti_develop]*ALGEBRAIC[(algebraic_size * foset) + dti_recover]*ALGEBRAIC[(algebraic_size * foset) + tiF];
ALGEBRAIC[(algebraic_size * foset) + tiSp] =  ALGEBRAIC[(algebraic_size * foset) + dti_develop]*ALGEBRAIC[(algebraic_size * foset) + dti_recover]*ALGEBRAIC[(algebraic_size * foset) + tiS];
ALGEBRAIC[(algebraic_size * foset) + alpha_C2ToI] =  5.20000e-05*exp( 1.52500*ALGEBRAIC[(algebraic_size * foset) + vfrt]);
ALGEBRAIC[(algebraic_size * foset) + beta_ItoC2] = ( ALGEBRAIC[(algebraic_size * foset) + beta_2]*ALGEBRAIC[(algebraic_size * foset) + beta_i]*ALGEBRAIC[(algebraic_size * foset) + alpha_C2ToI])/( ALGEBRAIC[(algebraic_size * foset) + alpha_2]*ALGEBRAIC[(algebraic_size * foset) + alpha_i]);
ALGEBRAIC[(algebraic_size * foset) + f] =  CONSTANTS[(constants_size * foset) + Aff]*STATES[(states_size * foset) + ff]+ CONSTANTS[(constants_size * foset) + Afs]*STATES[(states_size * foset) + fs];
ALGEBRAIC[(algebraic_size * foset) + Afcaf] = 0.300000+0.600000/(1.00000+exp((STATES[(states_size * foset) + V] - 10.0000)/10.0000));
ALGEBRAIC[(algebraic_size * foset) + Afcas] = 1.00000 - ALGEBRAIC[(algebraic_size * foset) + Afcaf];
ALGEBRAIC[(algebraic_size * foset) + fca] =  ALGEBRAIC[(algebraic_size * foset) + Afcaf]*STATES[(states_size * foset) + fcaf]+ ALGEBRAIC[(algebraic_size * foset) + Afcas]*STATES[(states_size * foset) + fcas];
ALGEBRAIC[(algebraic_size * foset) + fp] =  CONSTANTS[(constants_size * foset) + Aff]*STATES[(states_size * foset) + ffp]+ CONSTANTS[(constants_size * foset) + Afs]*STATES[(states_size * foset) + fs];
ALGEBRAIC[(algebraic_size * foset) + fcap] =  ALGEBRAIC[(algebraic_size * foset) + Afcaf]*STATES[(states_size * foset) + fcafp]+ ALGEBRAIC[(algebraic_size * foset) + Afcas]*STATES[(states_size * foset) + fcas];
ALGEBRAIC[(algebraic_size * foset) + vffrt] = ( STATES[(states_size * foset) + V]*CONSTANTS[(constants_size * foset) + F]*CONSTANTS[(constants_size * foset) + F])/( CONSTANTS[(constants_size * foset) + R]*CONSTANTS[(constants_size * foset) + T]);
ALGEBRAIC[(algebraic_size * foset) + Iss] = ( 0.500000*(STATES[(states_size * foset) + nass]+STATES[(states_size * foset) + kss]+CONSTANTS[(constants_size * foset) + cli]+ 4.00000*STATES[(states_size * foset) + cass]))/1000.00;
ALGEBRAIC[(algebraic_size * foset) + gamma_cass] = exp( - CONSTANTS[(constants_size * foset) + constA]*4.00000*( pow(ALGEBRAIC[(algebraic_size * foset) + Iss], 1.0 / 2)/(1.00000+ pow(ALGEBRAIC[(algebraic_size * foset) + Iss], 1.0 / 2)) -  0.300000*ALGEBRAIC[(algebraic_size * foset) + Iss]));
ALGEBRAIC[(algebraic_size * foset) + PhiCaL_ss] = ( 4.00000*ALGEBRAIC[(algebraic_size * foset) + vffrt]*( ALGEBRAIC[(algebraic_size * foset) + gamma_cass]*STATES[(states_size * foset) + cass]*exp( 2.00000*ALGEBRAIC[(algebraic_size * foset) + vfrt]) -  CONSTANTS[(constants_size * foset) + gamma_cao]*CONSTANTS[(constants_size * foset) + cao]))/(exp( 2.00000*ALGEBRAIC[(algebraic_size * foset) + vfrt]) - 1.00000);
ALGEBRAIC[(algebraic_size * foset) + CaMKa] = ALGEBRAIC[(algebraic_size * foset) + CaMKb]+STATES[(states_size * foset) + CaMKt];
ALGEBRAIC[(algebraic_size * foset) + fICaLp] = 1.00000/(1.00000+CONSTANTS[(constants_size * foset) + KmCaMK]/ALGEBRAIC[(algebraic_size * foset) + CaMKa]);
ALGEBRAIC[(algebraic_size * foset) + ICaL_ss] =  CONSTANTS[(constants_size * foset) + ICaL_fractionSS]*( (1.00000 - ALGEBRAIC[(algebraic_size * foset) + fICaLp])*CONSTANTS[(constants_size * foset) + PCa]*ALGEBRAIC[(algebraic_size * foset) + PhiCaL_ss]*STATES[(states_size * foset) + d]*( ALGEBRAIC[(algebraic_size * foset) + f]*(1.00000 - STATES[(states_size * foset) + nca_ss])+ STATES[(states_size * foset) + jca]*ALGEBRAIC[(algebraic_size * foset) + fca]*STATES[(states_size * foset) + nca_ss])+ ALGEBRAIC[(algebraic_size * foset) + fICaLp]*CONSTANTS[(constants_size * foset) + PCap]*ALGEBRAIC[(algebraic_size * foset) + PhiCaL_ss]*STATES[(states_size * foset) + d]*( ALGEBRAIC[(algebraic_size * foset) + fp]*(1.00000 - STATES[(states_size * foset) + nca_ss])+ STATES[(states_size * foset) + jca]*ALGEBRAIC[(algebraic_size * foset) + fcap]*STATES[(states_size * foset) + nca_ss]));
ALGEBRAIC[(algebraic_size * foset) + Jrel_inf_b] = (( - CONSTANTS[(constants_size * foset) + a_rel]*ALGEBRAIC[(algebraic_size * foset) + ICaL_ss])/1.00000)/(1.00000+pow(CONSTANTS[(constants_size * foset) + cajsr_half]/STATES[(states_size * foset) + cajsr], 8.00000));
ALGEBRAIC[(algebraic_size * foset) + Jrel_inf] = (CONSTANTS[(constants_size * foset) + celltype]==2.00000 ?  ALGEBRAIC[(algebraic_size * foset) + Jrel_inf_b]*1.70000 : ALGEBRAIC[(algebraic_size * foset) + Jrel_inf_b]);
ALGEBRAIC[(algebraic_size * foset) + tau_rel_b] = CONSTANTS[(constants_size * foset) + bt]/(1.00000+0.0123000/STATES[(states_size * foset) + cajsr]);
ALGEBRAIC[(algebraic_size * foset) + tau_rel] = (ALGEBRAIC[(algebraic_size * foset) + tau_rel_b]<0.00100000 ? 0.00100000 : ALGEBRAIC[(algebraic_size * foset) + tau_rel_b]);
ALGEBRAIC[(algebraic_size * foset) + Jrel_infp_b] = (( - CONSTANTS[(constants_size * foset) + a_relp]*ALGEBRAIC[(algebraic_size * foset) + ICaL_ss])/1.00000)/(1.00000+pow(CONSTANTS[(constants_size * foset) + cajsr_half]/STATES[(states_size * foset) + cajsr], 8.00000));
ALGEBRAIC[(algebraic_size * foset) + Jrel_infp] = (CONSTANTS[(constants_size * foset) + celltype]==2.00000 ?  ALGEBRAIC[(algebraic_size * foset) + Jrel_infp_b]*1.70000 : ALGEBRAIC[(algebraic_size * foset) + Jrel_infp_b]);
ALGEBRAIC[(algebraic_size * foset) + tau_relp_b] = CONSTANTS[(constants_size * foset) + btp]/(1.00000+0.0123000/STATES[(states_size * foset) + cajsr]);
ALGEBRAIC[(algebraic_size * foset) + tau_relp] = (ALGEBRAIC[(algebraic_size * foset) + tau_relp_b]<0.00100000 ? 0.00100000 : ALGEBRAIC[(algebraic_size * foset) + tau_relp_b]);
ALGEBRAIC[(algebraic_size * foset) + EK] =  (( CONSTANTS[(constants_size * foset) + R]*CONSTANTS[(constants_size * foset) + T])/( CONSTANTS[(constants_size * foset) + zk]*CONSTANTS[(constants_size * foset) + F]))*log(CONSTANTS[(constants_size * foset) + ko]/STATES[(states_size * foset) + ki]);
ALGEBRAIC[(algebraic_size * foset) + AiF] = 1.00000/(1.00000+exp(((STATES[(states_size * foset) + V]+CONSTANTS[(constants_size * foset) + EKshift]) - 213.600)/151.200));
ALGEBRAIC[(algebraic_size * foset) + AiS] = 1.00000 - ALGEBRAIC[(algebraic_size * foset) + AiF];
ALGEBRAIC[(algebraic_size * foset) + i] =  ALGEBRAIC[(algebraic_size * foset) + AiF]*STATES[(states_size * foset) + iF]+ ALGEBRAIC[(algebraic_size * foset) + AiS]*STATES[(states_size * foset) + iS];
ALGEBRAIC[(algebraic_size * foset) + ip] =  ALGEBRAIC[(algebraic_size * foset) + AiF]*STATES[(states_size * foset) + iFp]+ ALGEBRAIC[(algebraic_size * foset) + AiS]*STATES[(states_size * foset) + iSp];
ALGEBRAIC[(algebraic_size * foset) + fItop] = 1.00000/(1.00000+CONSTANTS[(constants_size * foset) + KmCaMK]/ALGEBRAIC[(algebraic_size * foset) + CaMKa]);
ALGEBRAIC[(algebraic_size * foset) + Ito] =  CONSTANTS[(constants_size * foset) + Gto]*(STATES[(states_size * foset) + V] - ALGEBRAIC[(algebraic_size * foset) + EK])*( (1.00000 - ALGEBRAIC[(algebraic_size * foset) + fItop])*STATES[(states_size * foset) + a]*ALGEBRAIC[(algebraic_size * foset) + i]+ ALGEBRAIC[(algebraic_size * foset) + fItop]*STATES[(states_size * foset) + ap]*ALGEBRAIC[(algebraic_size * foset) + ip]);
ALGEBRAIC[(algebraic_size * foset) + IKr] =  CONSTANTS[(constants_size * foset) + GKr]* pow((CONSTANTS[(constants_size * foset) + ko]/5.00000), 1.0 / 2)*STATES[(states_size * foset) + O]*(STATES[(states_size * foset) + V] - ALGEBRAIC[(algebraic_size * foset) + EK]);
ALGEBRAIC[(algebraic_size * foset) + EKs] =  (( CONSTANTS[(constants_size * foset) + R]*CONSTANTS[(constants_size * foset) + T])/( CONSTANTS[(constants_size * foset) + zk]*CONSTANTS[(constants_size * foset) + F]))*log((CONSTANTS[(constants_size * foset) + ko]+ CONSTANTS[(constants_size * foset) + PKNa]*CONSTANTS[(constants_size * foset) + nao])/(STATES[(states_size * foset) + ki]+ CONSTANTS[(constants_size * foset) + PKNa]*STATES[(states_size * foset) + nai]));
ALGEBRAIC[(algebraic_size * foset) + KsCa] = 1.00000+0.600000/(1.00000+pow(3.80000e-05/STATES[(states_size * foset) + cai], 1.40000));
ALGEBRAIC[(algebraic_size * foset) + IKs] =  CONSTANTS[(constants_size * foset) + GKs]*ALGEBRAIC[(algebraic_size * foset) + KsCa]*STATES[(states_size * foset) + xs1]*STATES[(states_size * foset) + xs2]*(STATES[(states_size * foset) + V] - ALGEBRAIC[(algebraic_size * foset) + EKs]);
ALGEBRAIC[(algebraic_size * foset) + aK1] = 4.09400/(1.00000+exp( 0.121700*((STATES[(states_size * foset) + V] - ALGEBRAIC[(algebraic_size * foset) + EK]) - 49.9340)));
ALGEBRAIC[(algebraic_size * foset) + bK1] = ( 15.7200*exp( 0.0674000*((STATES[(states_size * foset) + V] - ALGEBRAIC[(algebraic_size * foset) + EK]) - 3.25700))+exp( 0.0618000*((STATES[(states_size * foset) + V] - ALGEBRAIC[(algebraic_size * foset) + EK]) - 594.310)))/(1.00000+exp( - 0.162900*((STATES[(states_size * foset) + V] - ALGEBRAIC[(algebraic_size * foset) + EK])+14.2070)));
ALGEBRAIC[(algebraic_size * foset) + K1ss] = ALGEBRAIC[(algebraic_size * foset) + aK1]/(ALGEBRAIC[(algebraic_size * foset) + aK1]+ALGEBRAIC[(algebraic_size * foset) + bK1]);
ALGEBRAIC[(algebraic_size * foset) + IK1] =  CONSTANTS[(constants_size * foset) + GK1]* pow((CONSTANTS[(constants_size * foset) + ko]/5.00000), 1.0 / 2)*ALGEBRAIC[(algebraic_size * foset) + K1ss]*(STATES[(states_size * foset) + V] - ALGEBRAIC[(algebraic_size * foset) + EK]);
ALGEBRAIC[(algebraic_size * foset) + Knao] =  CONSTANTS[(constants_size * foset) + Knao0]*exp(( (1.00000 - CONSTANTS[(constants_size * foset) + delta])*ALGEBRAIC[(algebraic_size * foset) + vfrt])/3.00000);
ALGEBRAIC[(algebraic_size * foset) + a3] = ( CONSTANTS[(constants_size * foset) + k3p]*pow(CONSTANTS[(constants_size * foset) + ko]/CONSTANTS[(constants_size * foset) + Kko], 2.00000))/((pow(1.00000+CONSTANTS[(constants_size * foset) + nao]/ALGEBRAIC[(algebraic_size * foset) + Knao], 3.00000)+pow(1.00000+CONSTANTS[(constants_size * foset) + ko]/CONSTANTS[(constants_size * foset) + Kko], 2.00000)) - 1.00000);
ALGEBRAIC[(algebraic_size * foset) + P] = CONSTANTS[(constants_size * foset) + eP]/(1.00000+CONSTANTS[(constants_size * foset) + H]/CONSTANTS[(constants_size * foset) + Khp]+STATES[(states_size * foset) + nai]/CONSTANTS[(constants_size * foset) + Knap]+STATES[(states_size * foset) + ki]/CONSTANTS[(constants_size * foset) + Kxkur]);
ALGEBRAIC[(algebraic_size * foset) + b3] = ( CONSTANTS[(constants_size * foset) + k3m]*ALGEBRAIC[(algebraic_size * foset) + P]*CONSTANTS[(constants_size * foset) + H])/(1.00000+CONSTANTS[(constants_size * foset) + MgATP]/CONSTANTS[(constants_size * foset) + Kmgatp]);
ALGEBRAIC[(algebraic_size * foset) + Knai] =  CONSTANTS[(constants_size * foset) + Knai0]*exp(( CONSTANTS[(constants_size * foset) + delta]*ALGEBRAIC[(algebraic_size * foset) + vfrt])/3.00000);
ALGEBRAIC[(algebraic_size * foset) + a1] = ( CONSTANTS[(constants_size * foset) + k1p]*pow(STATES[(states_size * foset) + nai]/ALGEBRAIC[(algebraic_size * foset) + Knai], 3.00000))/((pow(1.00000+STATES[(states_size * foset) + nai]/ALGEBRAIC[(algebraic_size * foset) + Knai], 3.00000)+pow(1.00000+STATES[(states_size * foset) + ki]/CONSTANTS[(constants_size * foset) + Kki], 2.00000)) - 1.00000);
ALGEBRAIC[(algebraic_size * foset) + b2] = ( CONSTANTS[(constants_size * foset) + k2m]*pow(CONSTANTS[(constants_size * foset) + nao]/ALGEBRAIC[(algebraic_size * foset) + Knao], 3.00000))/((pow(1.00000+CONSTANTS[(constants_size * foset) + nao]/ALGEBRAIC[(algebraic_size * foset) + Knao], 3.00000)+pow(1.00000+CONSTANTS[(constants_size * foset) + ko]/CONSTANTS[(constants_size * foset) + Kko], 2.00000)) - 1.00000);
ALGEBRAIC[(algebraic_size * foset) + b4] = ( CONSTANTS[(constants_size * foset) + k4m]*pow(STATES[(states_size * foset) + ki]/CONSTANTS[(constants_size * foset) + Kki], 2.00000))/((pow(1.00000+STATES[(states_size * foset) + nai]/ALGEBRAIC[(algebraic_size * foset) + Knai], 3.00000)+pow(1.00000+STATES[(states_size * foset) + ki]/CONSTANTS[(constants_size * foset) + Kki], 2.00000)) - 1.00000);
ALGEBRAIC[(algebraic_size * foset) + x1] =  CONSTANTS[(constants_size * foset) + a4]*ALGEBRAIC[(algebraic_size * foset) + a1]*CONSTANTS[(constants_size * foset) + a2]+ ALGEBRAIC[(algebraic_size * foset) + b2]*ALGEBRAIC[(algebraic_size * foset) + b4]*ALGEBRAIC[(algebraic_size * foset) + b3]+ CONSTANTS[(constants_size * foset) + a2]*ALGEBRAIC[(algebraic_size * foset) + b4]*ALGEBRAIC[(algebraic_size * foset) + b3]+ ALGEBRAIC[(algebraic_size * foset) + b3]*ALGEBRAIC[(algebraic_size * foset) + a1]*CONSTANTS[(constants_size * foset) + a2];
ALGEBRAIC[(algebraic_size * foset) + x2] =  ALGEBRAIC[(algebraic_size * foset) + b2]*CONSTANTS[(constants_size * foset) + b1]*ALGEBRAIC[(algebraic_size * foset) + b4]+ ALGEBRAIC[(algebraic_size * foset) + a1]*CONSTANTS[(constants_size * foset) + a2]*ALGEBRAIC[(algebraic_size * foset) + a3]+ ALGEBRAIC[(algebraic_size * foset) + a3]*CONSTANTS[(constants_size * foset) + b1]*ALGEBRAIC[(algebraic_size * foset) + b4]+ CONSTANTS[(constants_size * foset) + a2]*ALGEBRAIC[(algebraic_size * foset) + a3]*ALGEBRAIC[(algebraic_size * foset) + b4];
ALGEBRAIC[(algebraic_size * foset) + x3] =  CONSTANTS[(constants_size * foset) + a2]*ALGEBRAIC[(algebraic_size * foset) + a3]*CONSTANTS[(constants_size * foset) + a4]+ ALGEBRAIC[(algebraic_size * foset) + b3]*ALGEBRAIC[(algebraic_size * foset) + b2]*CONSTANTS[(constants_size * foset) + b1]+ ALGEBRAIC[(algebraic_size * foset) + b2]*CONSTANTS[(constants_size * foset) + b1]*CONSTANTS[(constants_size * foset) + a4]+ ALGEBRAIC[(algebraic_size * foset) + a3]*CONSTANTS[(constants_size * foset) + a4]*CONSTANTS[(constants_size * foset) + b1];
ALGEBRAIC[(algebraic_size * foset) + x4] =  ALGEBRAIC[(algebraic_size * foset) + b4]*ALGEBRAIC[(algebraic_size * foset) + b3]*ALGEBRAIC[(algebraic_size * foset) + b2]+ ALGEBRAIC[(algebraic_size * foset) + a3]*CONSTANTS[(constants_size * foset) + a4]*ALGEBRAIC[(algebraic_size * foset) + a1]+ ALGEBRAIC[(algebraic_size * foset) + b2]*CONSTANTS[(constants_size * foset) + a4]*ALGEBRAIC[(algebraic_size * foset) + a1]+ ALGEBRAIC[(algebraic_size * foset) + b3]*ALGEBRAIC[(algebraic_size * foset) + b2]*ALGEBRAIC[(algebraic_size * foset) + a1];
ALGEBRAIC[(algebraic_size * foset) + E1] = ALGEBRAIC[(algebraic_size * foset) + x1]/(ALGEBRAIC[(algebraic_size * foset) + x1]+ALGEBRAIC[(algebraic_size * foset) + x2]+ALGEBRAIC[(algebraic_size * foset) + x3]+ALGEBRAIC[(algebraic_size * foset) + x4]);
ALGEBRAIC[(algebraic_size * foset) + E2] = ALGEBRAIC[(algebraic_size * foset) + x2]/(ALGEBRAIC[(algebraic_size * foset) + x1]+ALGEBRAIC[(algebraic_size * foset) + x2]+ALGEBRAIC[(algebraic_size * foset) + x3]+ALGEBRAIC[(algebraic_size * foset) + x4]);
ALGEBRAIC[(algebraic_size * foset) + JnakNa] =  3.00000*( ALGEBRAIC[(algebraic_size * foset) + E1]*ALGEBRAIC[(algebraic_size * foset) + a3] -  ALGEBRAIC[(algebraic_size * foset) + E2]*ALGEBRAIC[(algebraic_size * foset) + b3]);
ALGEBRAIC[(algebraic_size * foset) + E3] = ALGEBRAIC[(algebraic_size * foset) + x3]/(ALGEBRAIC[(algebraic_size * foset) + x1]+ALGEBRAIC[(algebraic_size * foset) + x2]+ALGEBRAIC[(algebraic_size * foset) + x3]+ALGEBRAIC[(algebraic_size * foset) + x4]);
ALGEBRAIC[(algebraic_size * foset) + E4] = ALGEBRAIC[(algebraic_size * foset) + x4]/(ALGEBRAIC[(algebraic_size * foset) + x1]+ALGEBRAIC[(algebraic_size * foset) + x2]+ALGEBRAIC[(algebraic_size * foset) + x3]+ALGEBRAIC[(algebraic_size * foset) + x4]);
ALGEBRAIC[(algebraic_size * foset) + JnakK] =  2.00000*( ALGEBRAIC[(algebraic_size * foset) + E4]*CONSTANTS[(constants_size * foset) + b1] -  ALGEBRAIC[(algebraic_size * foset) + E3]*ALGEBRAIC[(algebraic_size * foset) + a1]);
ALGEBRAIC[(algebraic_size * foset) + INaK] =  CONSTANTS[(constants_size * foset) + Pnak]*( CONSTANTS[(constants_size * foset) + zna]*ALGEBRAIC[(algebraic_size * foset) + JnakNa]+ CONSTANTS[(constants_size * foset) + zk]*ALGEBRAIC[(algebraic_size * foset) + JnakK]);
ALGEBRAIC[(algebraic_size * foset) + xkb] = 1.00000/(1.00000+exp(- (STATES[(states_size * foset) + V] - 10.8968)/23.9871));
ALGEBRAIC[(algebraic_size * foset) + IKb] =  CONSTANTS[(constants_size * foset) + GKb]*ALGEBRAIC[(algebraic_size * foset) + xkb]*(STATES[(states_size * foset) + V] - ALGEBRAIC[(algebraic_size * foset) + EK]);
ALGEBRAIC[(algebraic_size * foset) + I_katp] =  CONSTANTS[(constants_size * foset) + fkatp]*CONSTANTS[(constants_size * foset) + gkatp]*CONSTANTS[(constants_size * foset) + akik]*CONSTANTS[(constants_size * foset) + bkik]*(STATES[(states_size * foset) + V] - ALGEBRAIC[(algebraic_size * foset) + EK]);
ALGEBRAIC[(algebraic_size * foset) + Istim] = (TIME>=CONSTANTS[(constants_size * foset) + stim_start]&&TIME<=CONSTANTS[(constants_size * foset) + i_Stim_End]&&(TIME - CONSTANTS[(constants_size * foset) + stim_start]) -  floor((TIME - CONSTANTS[(constants_size * foset) + stim_start])/CONSTANTS[(constants_size * foset) + BCL])*CONSTANTS[(constants_size * foset) + BCL]<=CONSTANTS[(constants_size * foset) + i_Stim_PulseDuration] ? CONSTANTS[(constants_size * foset) + i_Stim_Amplitude] : 0.00000);
ALGEBRAIC[(algebraic_size * foset) + Ii] = ( 0.500000*(STATES[(states_size * foset) + nai]+STATES[(states_size * foset) + ki]+CONSTANTS[(constants_size * foset) + cli]+ 4.00000*STATES[(states_size * foset) + cai]))/1000.00;
ALGEBRAIC[(algebraic_size * foset) + gamma_ki] = exp( - CONSTANTS[(constants_size * foset) + constA]*1.00000*( pow(ALGEBRAIC[(algebraic_size * foset) + Ii], 1.0 / 2)/(1.00000+ pow(ALGEBRAIC[(algebraic_size * foset) + Ii], 1.0 / 2)) -  0.300000*ALGEBRAIC[(algebraic_size * foset) + Ii]));
ALGEBRAIC[(algebraic_size * foset) + PhiCaK_i] = ( 1.00000*ALGEBRAIC[(algebraic_size * foset) + vffrt]*( ALGEBRAIC[(algebraic_size * foset) + gamma_ki]*STATES[(states_size * foset) + ki]*exp( 1.00000*ALGEBRAIC[(algebraic_size * foset) + vfrt]) -  CONSTANTS[(constants_size * foset) + gamma_ko]*CONSTANTS[(constants_size * foset) + ko]))/(exp( 1.00000*ALGEBRAIC[(algebraic_size * foset) + vfrt]) - 1.00000);
ALGEBRAIC[(algebraic_size * foset) + ICaK_i] =  (1.00000 - CONSTANTS[(constants_size * foset) + ICaL_fractionSS])*( (1.00000 - ALGEBRAIC[(algebraic_size * foset) + fICaLp])*CONSTANTS[(constants_size * foset) + PCaK]*ALGEBRAIC[(algebraic_size * foset) + PhiCaK_i]*STATES[(states_size * foset) + d]*( ALGEBRAIC[(algebraic_size * foset) + f]*(1.00000 - STATES[(states_size * foset) + nca_i])+ STATES[(states_size * foset) + jca]*ALGEBRAIC[(algebraic_size * foset) + fca]*STATES[(states_size * foset) + nca_i])+ ALGEBRAIC[(algebraic_size * foset) + fICaLp]*CONSTANTS[(constants_size * foset) + PCaKp]*ALGEBRAIC[(algebraic_size * foset) + PhiCaK_i]*STATES[(states_size * foset) + d]*( ALGEBRAIC[(algebraic_size * foset) + fp]*(1.00000 - STATES[(states_size * foset) + nca_i])+ STATES[(states_size * foset) + jca]*ALGEBRAIC[(algebraic_size * foset) + fcap]*STATES[(states_size * foset) + nca_i]));
ALGEBRAIC[(algebraic_size * foset) + JdiffK] = (STATES[(states_size * foset) + kss] - STATES[(states_size * foset) + ki])/CONSTANTS[(constants_size * foset) + tauK];
ALGEBRAIC[(algebraic_size * foset) + gamma_kss] = exp( - CONSTANTS[(constants_size * foset) + constA]*1.00000*( pow(ALGEBRAIC[(algebraic_size * foset) + Iss], 1.0 / 2)/(1.00000+ pow(ALGEBRAIC[(algebraic_size * foset) + Iss], 1.0 / 2)) -  0.300000*ALGEBRAIC[(algebraic_size * foset) + Iss]));
ALGEBRAIC[(algebraic_size * foset) + PhiCaK_ss] = ( 1.00000*ALGEBRAIC[(algebraic_size * foset) + vffrt]*( ALGEBRAIC[(algebraic_size * foset) + gamma_kss]*STATES[(states_size * foset) + kss]*exp( 1.00000*ALGEBRAIC[(algebraic_size * foset) + vfrt]) -  CONSTANTS[(constants_size * foset) + gamma_ko]*CONSTANTS[(constants_size * foset) + ko]))/(exp( 1.00000*ALGEBRAIC[(algebraic_size * foset) + vfrt]) - 1.00000);
ALGEBRAIC[(algebraic_size * foset) + ICaK_ss] =  CONSTANTS[(constants_size * foset) + ICaL_fractionSS]*( (1.00000 - ALGEBRAIC[(algebraic_size * foset) + fICaLp])*CONSTANTS[(constants_size * foset) + PCaK]*ALGEBRAIC[(algebraic_size * foset) + PhiCaK_ss]*STATES[(states_size * foset) + d]*( ALGEBRAIC[(algebraic_size * foset) + f]*(1.00000 - STATES[(states_size * foset) + nca_ss])+ STATES[(states_size * foset) + jca]*ALGEBRAIC[(algebraic_size * foset) + fca]*STATES[(states_size * foset) + nca_ss])+ ALGEBRAIC[(algebraic_size * foset) + fICaLp]*CONSTANTS[(constants_size * foset) + PCaKp]*ALGEBRAIC[(algebraic_size * foset) + PhiCaK_ss]*STATES[(states_size * foset) + d]*( ALGEBRAIC[(algebraic_size * foset) + fp]*(1.00000 - STATES[(states_size * foset) + nca_ss])+ STATES[(states_size * foset) + jca]*ALGEBRAIC[(algebraic_size * foset) + fcap]*STATES[(states_size * foset) + nca_ss]));
ALGEBRAIC[(algebraic_size * foset) + ENa] =  (( CONSTANTS[(constants_size * foset) + R]*CONSTANTS[(constants_size * foset) + T])/( CONSTANTS[(constants_size * foset) + zna]*CONSTANTS[(constants_size * foset) + F]))*log(CONSTANTS[(constants_size * foset) + nao]/STATES[(states_size * foset) + nai]);
ALGEBRAIC[(algebraic_size * foset) + fINap] = 1.00000/(1.00000+CONSTANTS[(constants_size * foset) + KmCaMK]/ALGEBRAIC[(algebraic_size * foset) + CaMKa]);

ALGEBRAIC[(algebraic_size * foset) + INa] = CONSTANTS[(constants_size * foset) + GNa]*(STATES[(states_size * foset) + V] - ALGEBRAIC[(algebraic_size * foset) + ENa])*pow(STATES[(states_size * foset) + m], 3.00000)*( (1.00000 - ALGEBRAIC[(algebraic_size * foset) + fINap])*STATES[(states_size * foset) + h]*STATES[(states_size * foset) + j]+ ALGEBRAIC[(algebraic_size * foset) + fINap]*STATES[(states_size * foset) + hp]*STATES[(states_size * foset) + jp]);

ALGEBRAIC[(algebraic_size * foset) + fINaLp] = 1.00000/(1.00000+CONSTANTS[(constants_size * foset) + KmCaMK]/ALGEBRAIC[(algebraic_size * foset) + CaMKa]);
ALGEBRAIC[(algebraic_size * foset) + INaL] =  CONSTANTS[(constants_size * foset) + GNaL]*(STATES[(states_size * foset) + V] - ALGEBRAIC[(algebraic_size * foset) + ENa])*STATES[(states_size * foset) + mL]*( (1.00000 - ALGEBRAIC[(algebraic_size * foset) + fINaLp])*STATES[(states_size * foset) + hL]+ ALGEBRAIC[(algebraic_size * foset) + fINaLp]*STATES[(states_size * foset) + hLp]);
ALGEBRAIC[(algebraic_size * foset) + allo_i] = 1.00000/(1.00000+pow(CONSTANTS[(constants_size * foset) + KmCaAct]/STATES[(states_size * foset) + cai], 2.00000));
ALGEBRAIC[(algebraic_size * foset) + hna] = exp( CONSTANTS[(constants_size * foset) + qna]*ALGEBRAIC[(algebraic_size * foset) + vfrt]);
ALGEBRAIC[(algebraic_size * foset) + h7_i] = 1.00000+ (CONSTANTS[(constants_size * foset) + nao]/CONSTANTS[(constants_size * foset) + kna3])*(1.00000+1.00000/ALGEBRAIC[(algebraic_size * foset) + hna]);
ALGEBRAIC[(algebraic_size * foset) + h8_i] = CONSTANTS[(constants_size * foset) + nao]/( CONSTANTS[(constants_size * foset) + kna3]*ALGEBRAIC[(algebraic_size * foset) + hna]*ALGEBRAIC[(algebraic_size * foset) + h7_i]);
ALGEBRAIC[(algebraic_size * foset) + k3pp_i] =  ALGEBRAIC[(algebraic_size * foset) + h8_i]*CONSTANTS[(constants_size * foset) + wnaca];
ALGEBRAIC[(algebraic_size * foset) + h1_i] = 1.00000+ (STATES[(states_size * foset) + nai]/CONSTANTS[(constants_size * foset) + kna3])*(1.00000+ALGEBRAIC[(algebraic_size * foset) + hna]);
ALGEBRAIC[(algebraic_size * foset) + h2_i] = ( STATES[(states_size * foset) + nai]*ALGEBRAIC[(algebraic_size * foset) + hna])/( CONSTANTS[(constants_size * foset) + kna3]*ALGEBRAIC[(algebraic_size * foset) + h1_i]);
ALGEBRAIC[(algebraic_size * foset) + k4pp_i] =  ALGEBRAIC[(algebraic_size * foset) + h2_i]*CONSTANTS[(constants_size * foset) + wnaca];
ALGEBRAIC[(algebraic_size * foset) + h4_i] = 1.00000+ (STATES[(states_size * foset) + nai]/CONSTANTS[(constants_size * foset) + kna1])*(1.00000+STATES[(states_size * foset) + nai]/CONSTANTS[(constants_size * foset) + kna2]);
ALGEBRAIC[(algebraic_size * foset) + h5_i] = ( STATES[(states_size * foset) + nai]*STATES[(states_size * foset) + nai])/( ALGEBRAIC[(algebraic_size * foset) + h4_i]*CONSTANTS[(constants_size * foset) + kna1]*CONSTANTS[(constants_size * foset) + kna2]);
ALGEBRAIC[(algebraic_size * foset) + k7_i] =  ALGEBRAIC[(algebraic_size * foset) + h5_i]*ALGEBRAIC[(algebraic_size * foset) + h2_i]*CONSTANTS[(constants_size * foset) + wna];
ALGEBRAIC[(algebraic_size * foset) + k8_i] =  ALGEBRAIC[(algebraic_size * foset) + h8_i]*CONSTANTS[(constants_size * foset) + h11_i]*CONSTANTS[(constants_size * foset) + wna];
ALGEBRAIC[(algebraic_size * foset) + h9_i] = 1.00000/ALGEBRAIC[(algebraic_size * foset) + h7_i];
ALGEBRAIC[(algebraic_size * foset) + k3p_i] =  ALGEBRAIC[(algebraic_size * foset) + h9_i]*CONSTANTS[(constants_size * foset) + wca];
ALGEBRAIC[(algebraic_size * foset) + k3_i] = ALGEBRAIC[(algebraic_size * foset) + k3p_i]+ALGEBRAIC[(algebraic_size * foset) + k3pp_i];
ALGEBRAIC[(algebraic_size * foset) + hca] = exp( CONSTANTS[(constants_size * foset) + qca]*ALGEBRAIC[(algebraic_size * foset) + vfrt]);
ALGEBRAIC[(algebraic_size * foset) + h3_i] = 1.00000/ALGEBRAIC[(algebraic_size * foset) + h1_i];
ALGEBRAIC[(algebraic_size * foset) + k4p_i] = ( ALGEBRAIC[(algebraic_size * foset) + h3_i]*CONSTANTS[(constants_size * foset) + wca])/ALGEBRAIC[(algebraic_size * foset) + hca];
ALGEBRAIC[(algebraic_size * foset) + k4_i] = ALGEBRAIC[(algebraic_size * foset) + k4p_i]+ALGEBRAIC[(algebraic_size * foset) + k4pp_i];
ALGEBRAIC[(algebraic_size * foset) + h6_i] = 1.00000/ALGEBRAIC[(algebraic_size * foset) + h4_i];
ALGEBRAIC[(algebraic_size * foset) + k6_i] =  ALGEBRAIC[(algebraic_size * foset) + h6_i]*STATES[(states_size * foset) + cai]*CONSTANTS[(constants_size * foset) + kcaon];
ALGEBRAIC[(algebraic_size * foset) + x1_i] =  CONSTANTS[(constants_size * foset) + k2_i]*ALGEBRAIC[(algebraic_size * foset) + k4_i]*(ALGEBRAIC[(algebraic_size * foset) + k7_i]+ALGEBRAIC[(algebraic_size * foset) + k6_i])+ CONSTANTS[(constants_size * foset) + k5_i]*ALGEBRAIC[(algebraic_size * foset) + k7_i]*(CONSTANTS[(constants_size * foset) + k2_i]+ALGEBRAIC[(algebraic_size * foset) + k3_i]);
ALGEBRAIC[(algebraic_size * foset) + x2_i] =  CONSTANTS[(constants_size * foset) + k1_i]*ALGEBRAIC[(algebraic_size * foset) + k7_i]*(ALGEBRAIC[(algebraic_size * foset) + k4_i]+CONSTANTS[(constants_size * foset) + k5_i])+ ALGEBRAIC[(algebraic_size * foset) + k4_i]*ALGEBRAIC[(algebraic_size * foset) + k6_i]*(CONSTANTS[(constants_size * foset) + k1_i]+ALGEBRAIC[(algebraic_size * foset) + k8_i]);
ALGEBRAIC[(algebraic_size * foset) + x3_i] =  CONSTANTS[(constants_size * foset) + k1_i]*ALGEBRAIC[(algebraic_size * foset) + k3_i]*(ALGEBRAIC[(algebraic_size * foset) + k7_i]+ALGEBRAIC[(algebraic_size * foset) + k6_i])+ ALGEBRAIC[(algebraic_size * foset) + k8_i]*ALGEBRAIC[(algebraic_size * foset) + k6_i]*(CONSTANTS[(constants_size * foset) + k2_i]+ALGEBRAIC[(algebraic_size * foset) + k3_i]);
ALGEBRAIC[(algebraic_size * foset) + x4_i] =  CONSTANTS[(constants_size * foset) + k2_i]*ALGEBRAIC[(algebraic_size * foset) + k8_i]*(ALGEBRAIC[(algebraic_size * foset) + k4_i]+CONSTANTS[(constants_size * foset) + k5_i])+ ALGEBRAIC[(algebraic_size * foset) + k3_i]*CONSTANTS[(constants_size * foset) + k5_i]*(CONSTANTS[(constants_size * foset) + k1_i]+ALGEBRAIC[(algebraic_size * foset) + k8_i]);

ALGEBRAIC[(algebraic_size * foset) + E1_i] = ALGEBRAIC[(algebraic_size * foset) + x1_i] / (ALGEBRAIC[(algebraic_size * foset) + x1_i] + ALGEBRAIC[(algebraic_size * foset) + x2_i] + ALGEBRAIC[(algebraic_size * foset) + x3_i] + ALGEBRAIC[(algebraic_size * foset) + x4_i]);

ALGEBRAIC[(algebraic_size * foset) + E2_i] = ALGEBRAIC[(algebraic_size * foset) + x2_i]/(ALGEBRAIC[(algebraic_size * foset) + x1_i]+ALGEBRAIC[(algebraic_size * foset) + x2_i]+ALGEBRAIC[(algebraic_size * foset) + x3_i]+ALGEBRAIC[(algebraic_size * foset) + x4_i]);

ALGEBRAIC[(algebraic_size * foset) + E3_i] = ALGEBRAIC[(algebraic_size * foset) + x3_i]/(ALGEBRAIC[(algebraic_size * foset) + x1_i]+ALGEBRAIC[(algebraic_size * foset) + x2_i]+ALGEBRAIC[(algebraic_size * foset) + x3_i]+ALGEBRAIC[(algebraic_size * foset) + x4_i]);
ALGEBRAIC[(algebraic_size * foset) + E4_i] = ALGEBRAIC[(algebraic_size * foset) + x4_i]/(ALGEBRAIC[(algebraic_size * foset) + x1_i]+ALGEBRAIC[(algebraic_size * foset) + x2_i]+ALGEBRAIC[(algebraic_size * foset) + x3_i]+ALGEBRAIC[(algebraic_size * foset) + x4_i]);
ALGEBRAIC[(algebraic_size * foset) + JncxNa_i] = ( 3.00000*( ALGEBRAIC[(algebraic_size * foset) + E4_i]*ALGEBRAIC[(algebraic_size * foset) + k7_i] - ALGEBRAIC[(algebraic_size * foset) + E1_i]*ALGEBRAIC[(algebraic_size * foset) + k8_i])+ ALGEBRAIC[(algebraic_size * foset) + E3_i]*ALGEBRAIC[(algebraic_size * foset) + k4pp_i]) -  ALGEBRAIC[(algebraic_size * foset) + E2_i]*ALGEBRAIC[(algebraic_size * foset) + k3pp_i];

ALGEBRAIC[(algebraic_size * foset) + JncxCa_i] = ALGEBRAIC[(algebraic_size * foset) + E2_i] * CONSTANTS[(constants_size * foset) + k2_i] - ALGEBRAIC[(algebraic_size * foset) + E1_i] * CONSTANTS[(constants_size * foset) + k1_i];

ALGEBRAIC[(algebraic_size * foset) + INaCa_i] = (1.00000 - CONSTANTS[(constants_size * foset) + INaCa_fractionSS]) * CONSTANTS[(constants_size * foset) + Gncx] * ALGEBRAIC[(algebraic_size * foset) + allo_i] * (CONSTANTS[(constants_size * foset) + zna] * ALGEBRAIC[(algebraic_size * foset) + JncxNa_i] + CONSTANTS[(constants_size * foset) + zca] * ALGEBRAIC[(algebraic_size * foset) + JncxCa_i]);

ALGEBRAIC[(algebraic_size * foset) + INab] = ( CONSTANTS[(constants_size * foset) + PNab]*ALGEBRAIC[(algebraic_size * foset) + vffrt]*( STATES[(states_size * foset) + nai]*exp(ALGEBRAIC[(algebraic_size * foset) + vfrt]) - CONSTANTS[(constants_size * foset) + nao]))/(exp(ALGEBRAIC[(algebraic_size * foset) + vfrt]) - 1.00000);
ALGEBRAIC[(algebraic_size * foset) + gamma_nai] = exp( - CONSTANTS[(constants_size * foset) + constA]*1.00000*( pow(ALGEBRAIC[(algebraic_size * foset) + Ii], 1.0 / 2)/(1.00000+ pow(ALGEBRAIC[(algebraic_size * foset) + Ii], 1.0 / 2)) -  0.300000*ALGEBRAIC[(algebraic_size * foset) + Ii]));
ALGEBRAIC[(algebraic_size * foset) + PhiCaNa_i] = ( 1.00000*ALGEBRAIC[(algebraic_size * foset) + vffrt]*( ALGEBRAIC[(algebraic_size * foset) + gamma_nai]*STATES[(states_size * foset) + nai]*exp( 1.00000*ALGEBRAIC[(algebraic_size * foset) + vfrt]) -  CONSTANTS[(constants_size * foset) + gamma_nao]*CONSTANTS[(constants_size * foset) + nao]))/(exp( 1.00000*ALGEBRAIC[(algebraic_size * foset) + vfrt]) - 1.00000);
ALGEBRAIC[(algebraic_size * foset) + ICaNa_i] =  (1.00000 - CONSTANTS[(constants_size * foset) + ICaL_fractionSS])*( (1.00000 - ALGEBRAIC[(algebraic_size * foset) + fICaLp])*CONSTANTS[(constants_size * foset) + PCaNa]*ALGEBRAIC[(algebraic_size * foset) + PhiCaNa_i]*STATES[(states_size * foset) + d]*( ALGEBRAIC[(algebraic_size * foset) + f]*(1.00000 - STATES[(states_size * foset) + nca_i])+ STATES[(states_size * foset) + jca]*ALGEBRAIC[(algebraic_size * foset) + fca]*STATES[(states_size * foset) + nca_i])+ ALGEBRAIC[(algebraic_size * foset) + fICaLp]*CONSTANTS[(constants_size * foset) + PCaNap]*ALGEBRAIC[(algebraic_size * foset) + PhiCaNa_i]*STATES[(states_size * foset) + d]*( ALGEBRAIC[(algebraic_size * foset) + fp]*(1.00000 - STATES[(states_size * foset) + nca_i])+ STATES[(states_size * foset) + jca]*ALGEBRAIC[(algebraic_size * foset) + fcap]*STATES[(states_size * foset) + nca_i]));
ALGEBRAIC[(algebraic_size * foset) + JdiffNa] = (STATES[(states_size * foset) + nass] - STATES[(states_size * foset) + nai])/CONSTANTS[(constants_size * foset) + tauNa];
ALGEBRAIC[(algebraic_size * foset) + allo_ss] = 1.00000/(1.00000+pow(CONSTANTS[(constants_size * foset) + KmCaAct]/STATES[(states_size * foset) + cass], 2.00000));
ALGEBRAIC[(algebraic_size * foset) + h7_ss] = 1.00000+ (CONSTANTS[(constants_size * foset) + nao]/CONSTANTS[(constants_size * foset) + kna3])*(1.00000+1.00000/ALGEBRAIC[(algebraic_size * foset) + hna]);
ALGEBRAIC[(algebraic_size * foset) + h8_ss] = CONSTANTS[(constants_size * foset) + nao]/( CONSTANTS[(constants_size * foset) + kna3]*ALGEBRAIC[(algebraic_size * foset) + hna]*ALGEBRAIC[(algebraic_size * foset) + h7_ss]);
ALGEBRAIC[(algebraic_size * foset) + k3pp_ss] =  ALGEBRAIC[(algebraic_size * foset) + h8_ss]*CONSTANTS[(constants_size * foset) + wnaca];
ALGEBRAIC[(algebraic_size * foset) + h1_ss] = 1.00000+ (STATES[(states_size * foset) + nass]/CONSTANTS[(constants_size * foset) + kna3])*(1.00000+ALGEBRAIC[(algebraic_size * foset) + hna]);
ALGEBRAIC[(algebraic_size * foset) + h2_ss] = ( STATES[(states_size * foset) + nass]*ALGEBRAIC[(algebraic_size * foset) + hna])/( CONSTANTS[(constants_size * foset) + kna3]*ALGEBRAIC[(algebraic_size * foset) + h1_ss]);
ALGEBRAIC[(algebraic_size * foset) + k4pp_ss] =  ALGEBRAIC[(algebraic_size * foset) + h2_ss]*CONSTANTS[(constants_size * foset) + wnaca];
ALGEBRAIC[(algebraic_size * foset) + h4_ss] = 1.00000+ (STATES[(states_size * foset) + nass]/CONSTANTS[(constants_size * foset) + kna1])*(1.00000+STATES[(states_size * foset) + nass]/CONSTANTS[(constants_size * foset) + kna2]);
ALGEBRAIC[(algebraic_size * foset) + h5_ss] = ( STATES[(states_size * foset) + nass]*STATES[(states_size * foset) + nass])/( ALGEBRAIC[(algebraic_size * foset) + h4_ss]*CONSTANTS[(constants_size * foset) + kna1]*CONSTANTS[(constants_size * foset) + kna2]);
ALGEBRAIC[(algebraic_size * foset) + k7_ss] =  ALGEBRAIC[(algebraic_size * foset) + h5_ss]*ALGEBRAIC[(algebraic_size * foset) + h2_ss]*CONSTANTS[(constants_size * foset) + wna];
ALGEBRAIC[(algebraic_size * foset) + k8_ss] =  ALGEBRAIC[(algebraic_size * foset) + h8_ss]*CONSTANTS[(constants_size * foset) + h11_ss]*CONSTANTS[(constants_size * foset) + wna];
ALGEBRAIC[(algebraic_size * foset) + h9_ss] = 1.00000/ALGEBRAIC[(algebraic_size * foset) + h7_ss];
ALGEBRAIC[(algebraic_size * foset) + k3p_ss] =  ALGEBRAIC[(algebraic_size * foset) + h9_ss]*CONSTANTS[(constants_size * foset) + wca];
ALGEBRAIC[(algebraic_size * foset) + k3_ss] = ALGEBRAIC[(algebraic_size * foset) + k3p_ss]+ALGEBRAIC[(algebraic_size * foset) + k3pp_ss];
ALGEBRAIC[(algebraic_size * foset) + h3_ss] = 1.00000/ALGEBRAIC[(algebraic_size * foset) + h1_ss];
ALGEBRAIC[(algebraic_size * foset) + k4p_ss] = ( ALGEBRAIC[(algebraic_size * foset) + h3_ss]*CONSTANTS[(constants_size * foset) + wca])/ALGEBRAIC[(algebraic_size * foset) + hca];
ALGEBRAIC[(algebraic_size * foset) + k4_ss] = ALGEBRAIC[(algebraic_size * foset) + k4p_ss]+ALGEBRAIC[(algebraic_size * foset) + k4pp_ss];
ALGEBRAIC[(algebraic_size * foset) + h6_ss] = 1.00000/ALGEBRAIC[(algebraic_size * foset) + h4_ss];
ALGEBRAIC[(algebraic_size * foset) + k6_ss] =  ALGEBRAIC[(algebraic_size * foset) + h6_ss]*STATES[(states_size * foset) + cass]*CONSTANTS[(constants_size * foset) + kcaon];
ALGEBRAIC[(algebraic_size * foset) + x1_ss] =  CONSTANTS[(constants_size * foset) + k2_ss]*ALGEBRAIC[(algebraic_size * foset) + k4_ss]*(ALGEBRAIC[(algebraic_size * foset) + k7_ss]+ALGEBRAIC[(algebraic_size * foset) + k6_ss])+ CONSTANTS[(constants_size * foset) + k5_ss]*ALGEBRAIC[(algebraic_size * foset) + k7_ss]*(CONSTANTS[(constants_size * foset) + k2_ss]+ALGEBRAIC[(algebraic_size * foset) + k3_ss]);
ALGEBRAIC[(algebraic_size * foset) + x2_ss] =  CONSTANTS[(constants_size * foset) + k1_ss]*ALGEBRAIC[(algebraic_size * foset) + k7_ss]*(ALGEBRAIC[(algebraic_size * foset) + k4_ss]+CONSTANTS[(constants_size * foset) + k5_ss])+ ALGEBRAIC[(algebraic_size * foset) + k4_ss]*ALGEBRAIC[(algebraic_size * foset) + k6_ss]*(CONSTANTS[(constants_size * foset) + k1_ss]+ALGEBRAIC[(algebraic_size * foset) + k8_ss]);
ALGEBRAIC[(algebraic_size * foset) + x3_ss] =  CONSTANTS[(constants_size * foset) + k1_ss]*ALGEBRAIC[(algebraic_size * foset) + k3_ss]*(ALGEBRAIC[(algebraic_size * foset) + k7_ss]+ALGEBRAIC[(algebraic_size * foset) + k6_ss])+ ALGEBRAIC[(algebraic_size * foset) + k8_ss]*ALGEBRAIC[(algebraic_size * foset) + k6_ss]*(CONSTANTS[(constants_size * foset) + k2_ss]+ALGEBRAIC[(algebraic_size * foset) + k3_ss]);
ALGEBRAIC[(algebraic_size * foset) + x4_ss] =  CONSTANTS[(constants_size * foset) + k2_ss]*ALGEBRAIC[(algebraic_size * foset) + k8_ss]*(ALGEBRAIC[(algebraic_size * foset) + k4_ss]+CONSTANTS[(constants_size * foset) + k5_ss])+ ALGEBRAIC[(algebraic_size * foset) + k3_ss]*CONSTANTS[(constants_size * foset) + k5_ss]*(CONSTANTS[(constants_size * foset) + k1_ss]+ALGEBRAIC[(algebraic_size * foset) + k8_ss]);
ALGEBRAIC[(algebraic_size * foset) + E1_ss] = ALGEBRAIC[(algebraic_size * foset) + x1_ss]/(ALGEBRAIC[(algebraic_size * foset) + x1_ss]+ALGEBRAIC[(algebraic_size * foset) + x2_ss]+ALGEBRAIC[(algebraic_size * foset) + x3_ss]+ALGEBRAIC[(algebraic_size * foset) + x4_ss]);
ALGEBRAIC[(algebraic_size * foset) + E2_ss] = ALGEBRAIC[(algebraic_size * foset) + x2_ss]/(ALGEBRAIC[(algebraic_size * foset) + x1_ss]+ALGEBRAIC[(algebraic_size * foset) + x2_ss]+ALGEBRAIC[(algebraic_size * foset) + x3_ss]+ALGEBRAIC[(algebraic_size * foset) + x4_ss]);
ALGEBRAIC[(algebraic_size * foset) + E3_ss] = ALGEBRAIC[(algebraic_size * foset) + x3_ss]/(ALGEBRAIC[(algebraic_size * foset) + x1_ss]+ALGEBRAIC[(algebraic_size * foset) + x2_ss]+ALGEBRAIC[(algebraic_size * foset) + x3_ss]+ALGEBRAIC[(algebraic_size * foset) + x4_ss]);
ALGEBRAIC[(algebraic_size * foset) + E4_ss] = ALGEBRAIC[(algebraic_size * foset) + x4_ss]/(ALGEBRAIC[(algebraic_size * foset) + x1_ss]+ALGEBRAIC[(algebraic_size * foset) + x2_ss]+ALGEBRAIC[(algebraic_size * foset) + x3_ss]+ALGEBRAIC[(algebraic_size * foset) + x4_ss]);
ALGEBRAIC[(algebraic_size * foset) + JncxNa_ss] = ( 3.00000*( ALGEBRAIC[(algebraic_size * foset) + E4_ss]*ALGEBRAIC[(algebraic_size * foset) + k7_ss] -  ALGEBRAIC[(algebraic_size * foset) + E1_ss]*ALGEBRAIC[(algebraic_size * foset) + k8_ss])+ ALGEBRAIC[(algebraic_size * foset) + E3_ss]*ALGEBRAIC[(algebraic_size * foset) + k4pp_ss]) -  ALGEBRAIC[(algebraic_size * foset) + E2_ss]*ALGEBRAIC[(algebraic_size * foset) + k3pp_ss];
ALGEBRAIC[(algebraic_size * foset) + JncxCa_ss] =  ALGEBRAIC[(algebraic_size * foset) + E2_ss]*CONSTANTS[(constants_size * foset) + k2_ss] -  ALGEBRAIC[(algebraic_size * foset) + E1_ss]*CONSTANTS[(constants_size * foset) + k1_ss];
ALGEBRAIC[(algebraic_size * foset) + INaCa_ss] =  CONSTANTS[(constants_size * foset) + INaCa_fractionSS]*CONSTANTS[(constants_size * foset) + Gncx]*ALGEBRAIC[(algebraic_size * foset) + allo_ss]*( CONSTANTS[(constants_size * foset) + zna]*ALGEBRAIC[(algebraic_size * foset) + JncxNa_ss]+ CONSTANTS[(constants_size * foset) + zca]*ALGEBRAIC[(algebraic_size * foset) + JncxCa_ss]);
ALGEBRAIC[(algebraic_size * foset) + gamma_nass] = exp( - CONSTANTS[(constants_size * foset) + constA]*1.00000*( pow(ALGEBRAIC[(algebraic_size * foset) + Iss], 1.0 / 2)/(1.00000+ pow(ALGEBRAIC[(algebraic_size * foset) + Iss], 1.0 / 2)) -  0.300000*ALGEBRAIC[(algebraic_size * foset) + Iss]));
ALGEBRAIC[(algebraic_size * foset) + PhiCaNa_ss] = ( 1.00000*ALGEBRAIC[(algebraic_size * foset) + vffrt]*( ALGEBRAIC[(algebraic_size * foset) + gamma_nass]*STATES[(states_size * foset) + nass]*exp( 1.00000*ALGEBRAIC[(algebraic_size * foset) + vfrt]) -  CONSTANTS[(constants_size * foset) + gamma_nao]*CONSTANTS[(constants_size * foset) + nao]))/(exp( 1.00000*ALGEBRAIC[(algebraic_size * foset) + vfrt]) - 1.00000);
ALGEBRAIC[(algebraic_size * foset) + ICaNa_ss] =  CONSTANTS[(constants_size * foset) + ICaL_fractionSS]*( (1.00000 - ALGEBRAIC[(algebraic_size * foset) + fICaLp])*CONSTANTS[(constants_size * foset) + PCaNa]*ALGEBRAIC[(algebraic_size * foset) + PhiCaNa_ss]*STATES[(states_size * foset) + d]*( ALGEBRAIC[(algebraic_size * foset) + f]*(1.00000 - STATES[(states_size * foset) + nca_ss])+ STATES[(states_size * foset) + jca]*ALGEBRAIC[(algebraic_size * foset) + fca]*STATES[(states_size * foset) + nca_ss])+ ALGEBRAIC[(algebraic_size * foset) + fICaLp]*CONSTANTS[(constants_size * foset) + PCaNap]*ALGEBRAIC[(algebraic_size * foset) + PhiCaNa_ss]*STATES[(states_size * foset) + d]*( ALGEBRAIC[(algebraic_size * foset) + fp]*(1.00000 - STATES[(states_size * foset) + nca_ss])+ STATES[(states_size * foset) + jca]*ALGEBRAIC[(algebraic_size * foset) + fcap]*STATES[(states_size * foset) + nca_ss]));
ALGEBRAIC[(algebraic_size * foset) + Jdiff] = (STATES[(states_size * foset) + cass] - STATES[(states_size * foset) + cai])/CONSTANTS[(constants_size * foset) + tauCa];
ALGEBRAIC[(algebraic_size * foset) + fJrelp] = 1.00000/(1.00000+CONSTANTS[(constants_size * foset) + KmCaMK]/ALGEBRAIC[(algebraic_size * foset) + CaMKa]);
ALGEBRAIC[(algebraic_size * foset) + Jrel] =  CONSTANTS[(constants_size * foset) + Jrel_b]*( (1.00000 - ALGEBRAIC[(algebraic_size * foset) + fJrelp])*STATES[(states_size * foset) + Jrel_np]+ ALGEBRAIC[(algebraic_size * foset) + fJrelp]*STATES[(states_size * foset) + Jrel_p]);
ALGEBRAIC[(algebraic_size * foset) + Bcass] = 1.00000/(1.00000+( CONSTANTS[(constants_size * foset) + BSRmax]*CONSTANTS[(constants_size * foset) + KmBSR])/pow(CONSTANTS[(constants_size * foset) + KmBSR]+STATES[(states_size * foset) + cass], 2.00000)+( CONSTANTS[(constants_size * foset) + BSLmax]*CONSTANTS[(constants_size * foset) + KmBSL])/pow(CONSTANTS[(constants_size * foset) + KmBSL]+STATES[(states_size * foset) + cass], 2.00000));
ALGEBRAIC[(algebraic_size * foset) + gamma_cai] = exp( - CONSTANTS[(constants_size * foset) + constA]*4.00000*( pow(ALGEBRAIC[(algebraic_size * foset) + Ii], 1.0 / 2)/(1.00000+ pow(ALGEBRAIC[(algebraic_size * foset) + Ii], 1.0 / 2)) -  0.300000*ALGEBRAIC[(algebraic_size * foset) + Ii]));
ALGEBRAIC[(algebraic_size * foset) + PhiCaL_i] = ( 4.00000*ALGEBRAIC[(algebraic_size * foset) + vffrt]*( ALGEBRAIC[(algebraic_size * foset) + gamma_cai]*STATES[(states_size * foset) + cai]*exp( 2.00000*ALGEBRAIC[(algebraic_size * foset) + vfrt]) -  CONSTANTS[(constants_size * foset) + gamma_cao]*CONSTANTS[(constants_size * foset) + cao]))/(exp( 2.00000*ALGEBRAIC[(algebraic_size * foset) + vfrt]) - 1.00000);
ALGEBRAIC[(algebraic_size * foset) + ICaL_i] =  (1.00000 - CONSTANTS[(constants_size * foset) + ICaL_fractionSS])*( (1.00000 - ALGEBRAIC[(algebraic_size * foset) + fICaLp])*CONSTANTS[(constants_size * foset) + PCa]*ALGEBRAIC[(algebraic_size * foset) + PhiCaL_i]*STATES[(states_size * foset) + d]*( ALGEBRAIC[(algebraic_size * foset) + f]*(1.00000 - STATES[(states_size * foset) + nca_i])+ STATES[(states_size * foset) + jca]*ALGEBRAIC[(algebraic_size * foset) + fca]*STATES[(states_size * foset) + nca_i])+ ALGEBRAIC[(algebraic_size * foset) + fICaLp]*CONSTANTS[(constants_size * foset) + PCap]*ALGEBRAIC[(algebraic_size * foset) + PhiCaL_i]*STATES[(states_size * foset) + d]*( ALGEBRAIC[(algebraic_size * foset) + fp]*(1.00000 - STATES[(states_size * foset) + nca_i])+ STATES[(states_size * foset) + jca]*ALGEBRAIC[(algebraic_size * foset) + fcap]*STATES[(states_size * foset) + nca_i]));
ALGEBRAIC[(algebraic_size * foset) + ICaL] = ALGEBRAIC[(algebraic_size * foset) + ICaL_ss]+ALGEBRAIC[(algebraic_size * foset) + ICaL_i];
ALGEBRAIC[(algebraic_size * foset) + ICaNa] = ALGEBRAIC[(algebraic_size * foset) + ICaNa_ss]+ALGEBRAIC[(algebraic_size * foset) + ICaNa_i];
ALGEBRAIC[(algebraic_size * foset) + ICaK] = ALGEBRAIC[(algebraic_size * foset) + ICaK_ss]+ALGEBRAIC[(algebraic_size * foset) + ICaK_i];
ALGEBRAIC[(algebraic_size * foset) + IpCa] = ( CONSTANTS[(constants_size * foset) + GpCa]*STATES[(states_size * foset) + cai])/(CONSTANTS[(constants_size * foset) + KmCap]+STATES[(states_size * foset) + cai]);
ALGEBRAIC[(algebraic_size * foset) + ICab] = ( CONSTANTS[(constants_size * foset) + PCab]*4.00000*ALGEBRAIC[(algebraic_size * foset) + vffrt]*( ALGEBRAIC[(algebraic_size * foset) + gamma_cai]*STATES[(states_size * foset) + cai]*exp( 2.00000*ALGEBRAIC[(algebraic_size * foset) + vfrt]) -  CONSTANTS[(constants_size * foset) + gamma_cao]*CONSTANTS[(constants_size * foset) + cao]))/(exp( 2.00000*ALGEBRAIC[(algebraic_size * foset) + vfrt]) - 1.00000);
ALGEBRAIC[(algebraic_size * foset) + IClCa_junc] =  (( CONSTANTS[(constants_size * foset) + Fjunc]*CONSTANTS[(constants_size * foset) + GClCa])/(1.00000+CONSTANTS[(constants_size * foset) + KdClCa]/STATES[(states_size * foset) + cass]))*(STATES[(states_size * foset) + V] - CONSTANTS[(constants_size * foset) + ECl]);
ALGEBRAIC[(algebraic_size * foset) + IClCa_sl] =  (( (1.00000 - CONSTANTS[(constants_size * foset) + Fjunc])*CONSTANTS[(constants_size * foset) + GClCa])/(1.00000+CONSTANTS[(constants_size * foset) + KdClCa]/STATES[(states_size * foset) + cai]))*(STATES[(states_size * foset) + V] - CONSTANTS[(constants_size * foset) + ECl]);
ALGEBRAIC[(algebraic_size * foset) + IClCa] = ALGEBRAIC[(algebraic_size * foset) + IClCa_junc]+ALGEBRAIC[(algebraic_size * foset) + IClCa_sl];
ALGEBRAIC[(algebraic_size * foset) + IClb] =  CONSTANTS[(constants_size * foset) + GClb]*(STATES[(states_size * foset) + V] - CONSTANTS[(constants_size * foset) + ECl]);
ALGEBRAIC[(algebraic_size * foset) + Jupnp] = ( CONSTANTS[(constants_size * foset) + upScale]*0.00542500*STATES[(states_size * foset) + cai])/(STATES[(states_size * foset) + cai]+0.000920000);
ALGEBRAIC[(algebraic_size * foset) + Jupp] = ( CONSTANTS[(constants_size * foset) + upScale]*2.75000*0.00542500*STATES[(states_size * foset) + cai])/((STATES[(states_size * foset) + cai]+0.000920000) - 0.000170000);
ALGEBRAIC[(algebraic_size * foset) + fJupp] = 1.00000/(1.00000+CONSTANTS[(constants_size * foset) + KmCaMK]/ALGEBRAIC[(algebraic_size * foset) + CaMKa]);
ALGEBRAIC[(algebraic_size * foset) + Jleak] = ( 0.00488250*STATES[(states_size * foset) + cansr])/15.0000;
ALGEBRAIC[(algebraic_size * foset) + Jup] =  CONSTANTS[(constants_size * foset) + Jup_b]*(( (1.00000 - ALGEBRAIC[(algebraic_size * foset) + fJupp])*ALGEBRAIC[(algebraic_size * foset) + Jupnp]+ ALGEBRAIC[(algebraic_size * foset) + fJupp]*ALGEBRAIC[(algebraic_size * foset) + Jupp]) - ALGEBRAIC[(algebraic_size * foset) + Jleak]);
ALGEBRAIC[(algebraic_size * foset) + Bcai] = 1.00000/(1.00000+( CONSTANTS[(constants_size * foset) + cmdnmax]*CONSTANTS[(constants_size * foset) + kmcmdn])/pow(CONSTANTS[(constants_size * foset) + kmcmdn]+STATES[(states_size * foset) + cai], 2.00000)+( CONSTANTS[(constants_size * foset) + trpnmax]*CONSTANTS[(constants_size * foset) + kmtrpn])/pow(CONSTANTS[(constants_size * foset) + kmtrpn]+STATES[(states_size * foset) + cai], 2.00000));
ALGEBRAIC[(algebraic_size * foset) + Jtr] = (STATES[(states_size * foset) + cansr] - STATES[(states_size * foset) + cajsr])/60.0000;
ALGEBRAIC[(algebraic_size * foset) + Bcajsr] = 1.00000/(1.00000+( CONSTANTS[(constants_size * foset) + csqnmax]*CONSTANTS[(constants_size * foset) + kmcsqn])/pow(CONSTANTS[(constants_size * foset) + kmcsqn]+STATES[(states_size * foset) + cajsr], 2.00000));

RATES[ (states_size * foset) +hL] = (ALGEBRAIC[(algebraic_size * foset) + hLss] - STATES[(states_size * foset) + hL])/CONSTANTS[(constants_size * foset) + thL];
RATES[ (states_size * foset) +hLp] = (ALGEBRAIC[(algebraic_size * foset) + hLssp] - STATES[(states_size * foset) + hLp])/CONSTANTS[(constants_size * foset) + thLp];
RATES[ (states_size * foset) +jca] = (ALGEBRAIC[(algebraic_size * foset) + jcass] - STATES[(states_size * foset) + jca])/CONSTANTS[(constants_size * foset) + tjca];
RATES[ (states_size * foset) +m] = (ALGEBRAIC[(algebraic_size * foset) + mss] - STATES[(states_size * foset) + m])/ALGEBRAIC[(algebraic_size * foset) + tm];
RATES[ (states_size * foset) +mL] = (ALGEBRAIC[(algebraic_size * foset) + mLss] - STATES[(states_size * foset) + mL])/ALGEBRAIC[(algebraic_size * foset) + tmL];
RATES[ (states_size * foset) +a] = (ALGEBRAIC[(algebraic_size * foset) + ass] - STATES[(states_size * foset) + a])/ALGEBRAIC[(algebraic_size * foset) + ta];
RATES[ (states_size * foset) +d] = (ALGEBRAIC[(algebraic_size * foset) + dss] - STATES[(states_size * foset) + d])/ALGEBRAIC[(algebraic_size * foset) + td];
RATES[ (states_size * foset) +ff] = (ALGEBRAIC[(algebraic_size * foset) + fss] - STATES[(states_size * foset) + ff])/ALGEBRAIC[(algebraic_size * foset) + tff];
RATES[ (states_size * foset) +fs] = (ALGEBRAIC[(algebraic_size * foset) + fss] - STATES[(states_size * foset) + fs])/ALGEBRAIC[(algebraic_size * foset) + tfs];
RATES[ (states_size * foset) +nca_ss] =  ALGEBRAIC[(algebraic_size * foset) + anca_ss]*CONSTANTS[(constants_size * foset) + k2n] -  STATES[(states_size * foset) + nca_ss]*ALGEBRAIC[(algebraic_size * foset) + km2n];
RATES[ (states_size * foset) +nca_i] =  ALGEBRAIC[(algebraic_size * foset) + anca_i]*CONSTANTS[(constants_size * foset) + k2n] -  STATES[(states_size * foset) + nca_i]*ALGEBRAIC[(algebraic_size * foset) + km2n];
RATES[ (states_size * foset) +xs1] = (ALGEBRAIC[(algebraic_size * foset) + xs1ss] - STATES[(states_size * foset) + xs1])/ALGEBRAIC[(algebraic_size * foset) + txs1];
RATES[ (states_size * foset) +ap] = (ALGEBRAIC[(algebraic_size * foset) + assp] - STATES[(states_size * foset) + ap])/ALGEBRAIC[(algebraic_size * foset) + ta];
RATES[ (states_size * foset) +fcaf] = (ALGEBRAIC[(algebraic_size * foset) + fcass] - STATES[(states_size * foset) + fcaf])/ALGEBRAIC[(algebraic_size * foset) + tfcaf];
RATES[ (states_size * foset) +fcas] = (ALGEBRAIC[(algebraic_size * foset) + fcass] - STATES[(states_size * foset) + fcas])/ALGEBRAIC[(algebraic_size * foset) + tfcas];
RATES[ (states_size * foset) +ffp] = (ALGEBRAIC[(algebraic_size * foset) + fss] - STATES[(states_size * foset) + ffp])/ALGEBRAIC[(algebraic_size * foset) + tffp];
RATES[ (states_size * foset) +xs2] = (ALGEBRAIC[(algebraic_size * foset) + xs2ss] - STATES[(states_size * foset) + xs2])/ALGEBRAIC[(algebraic_size * foset) + txs2];
RATES[ (states_size * foset) +CaMKt] =  CONSTANTS[(constants_size * foset) + aCaMK]*ALGEBRAIC[(algebraic_size * foset) + CaMKb]*(ALGEBRAIC[(algebraic_size * foset) + CaMKb]+STATES[(states_size * foset) + CaMKt]) -  CONSTANTS[(constants_size * foset) + bCaMK]*STATES[(states_size * foset) + CaMKt];
RATES[ (states_size * foset) +h] = (ALGEBRAIC[(algebraic_size * foset) + hss] - STATES[(states_size * foset) + h])/ALGEBRAIC[(algebraic_size * foset) + th];
RATES[ (states_size * foset) +fcafp] = (ALGEBRAIC[(algebraic_size * foset) + fcass] - STATES[(states_size * foset) + fcafp])/ALGEBRAIC[(algebraic_size * foset) + tfcafp];
RATES[ (states_size * foset) +j] = (ALGEBRAIC[(algebraic_size * foset) + jss] - STATES[(states_size * foset) + j])/ALGEBRAIC[(algebraic_size * foset) + tj];
RATES[ (states_size * foset) +hp] = (ALGEBRAIC[(algebraic_size * foset) + hssp] - STATES[(states_size * foset) + hp])/ALGEBRAIC[(algebraic_size * foset) + th];
RATES[ (states_size * foset) +iF] = (ALGEBRAIC[(algebraic_size * foset) + iss] - STATES[(states_size * foset) + iF])/ALGEBRAIC[(algebraic_size * foset) + tiF];
RATES[ (states_size * foset) +C3] =  ALGEBRAIC[(algebraic_size * foset) + beta]*STATES[(states_size * foset) + C2] -  ALGEBRAIC[(algebraic_size * foset) + alpha]*STATES[(states_size * foset) + C3];
RATES[ (states_size * foset) +C2] = ( ALGEBRAIC[(algebraic_size * foset) + alpha]*STATES[(states_size * foset) + C3]+ CONSTANTS[(constants_size * foset) + beta_1]*STATES[(states_size * foset) + C1]) -  (ALGEBRAIC[(algebraic_size * foset) + beta]+CONSTANTS[(constants_size * foset) + alpha_1])*STATES[(states_size * foset) + C2];
RATES[ (states_size * foset) +jp] = (ALGEBRAIC[(algebraic_size * foset) + jss] - STATES[(states_size * foset) + jp])/ALGEBRAIC[(algebraic_size * foset) + tjp];
RATES[ (states_size * foset) +iS] = (ALGEBRAIC[(algebraic_size * foset) + iss] - STATES[(states_size * foset) + iS])/ALGEBRAIC[(algebraic_size * foset) + tiS];
RATES[ (states_size * foset) +O] = ( ALGEBRAIC[(algebraic_size * foset) + alpha_2]*STATES[(states_size * foset) + C1]+ ALGEBRAIC[(algebraic_size * foset) + beta_i]*STATES[(states_size * foset) + I]) -  (ALGEBRAIC[(algebraic_size * foset) + beta_2]+ALGEBRAIC[(algebraic_size * foset) + alpha_i])*STATES[(states_size * foset) + O];
RATES[ (states_size * foset) +iFp] = (ALGEBRAIC[(algebraic_size * foset) + iss] - STATES[(states_size * foset) + iFp])/ALGEBRAIC[(algebraic_size * foset) + tiFp];
RATES[ (states_size * foset) +iSp] = (ALGEBRAIC[(algebraic_size * foset) + iss] - STATES[(states_size * foset) + iSp])/ALGEBRAIC[(algebraic_size * foset) + tiSp];
RATES[ (states_size * foset) +C1] = ( CONSTANTS[(constants_size * foset) + alpha_1]*STATES[(states_size * foset) + C2]+ ALGEBRAIC[(algebraic_size * foset) + beta_2]*STATES[(states_size * foset) + O]+ ALGEBRAIC[(algebraic_size * foset) + beta_ItoC2]*STATES[(states_size * foset) + I]) -  (CONSTANTS[(constants_size * foset) + beta_1]+ALGEBRAIC[(algebraic_size * foset) + alpha_2]+ALGEBRAIC[(algebraic_size * foset) + alpha_C2ToI])*STATES[(states_size * foset) + C1];
RATES[ (states_size * foset) +I] = ( ALGEBRAIC[(algebraic_size * foset) + alpha_C2ToI]*STATES[(states_size * foset) + C1]+ ALGEBRAIC[(algebraic_size * foset) + alpha_i]*STATES[(states_size * foset) + O]) -  (ALGEBRAIC[(algebraic_size * foset) + beta_ItoC2]+ALGEBRAIC[(algebraic_size * foset) + beta_i])*STATES[(states_size * foset) + I];
RATES[ (states_size * foset) +Jrel_np] = (ALGEBRAIC[(algebraic_size * foset) + Jrel_inf] - STATES[(states_size * foset) + Jrel_np])/ALGEBRAIC[(algebraic_size * foset) + tau_rel];
RATES[ (states_size * foset) +Jrel_p] = (ALGEBRAIC[(algebraic_size * foset) + Jrel_infp] - STATES[(states_size * foset) + Jrel_p])/ALGEBRAIC[(algebraic_size * foset) + tau_relp];
RATES[ (states_size * foset) +ki] = ( - (((ALGEBRAIC[(algebraic_size * foset) + Ito]+ALGEBRAIC[(algebraic_size * foset) + IKr]+ALGEBRAIC[(algebraic_size * foset) + IKs]+ALGEBRAIC[(algebraic_size * foset) + IK1]+ALGEBRAIC[(algebraic_size * foset) + IKb]+ALGEBRAIC[(algebraic_size * foset) + I_katp]+ALGEBRAIC[(algebraic_size * foset) + Istim]) -  2.00000*ALGEBRAIC[(algebraic_size * foset) + INaK])+ALGEBRAIC[(algebraic_size * foset) + ICaK_i])*CONSTANTS[(constants_size * foset) + Acap])/( CONSTANTS[(constants_size * foset) + F]*CONSTANTS[(constants_size * foset) + vmyo])+( ALGEBRAIC[(algebraic_size * foset) + JdiffK]*CONSTANTS[(constants_size * foset) + vss])/CONSTANTS[(constants_size * foset) + vmyo];
RATES[ (states_size * foset) +kss] = ( - ALGEBRAIC[(algebraic_size * foset) + ICaK_ss]*CONSTANTS[(constants_size * foset) + Acap])/( CONSTANTS[(constants_size * foset) + F]*CONSTANTS[(constants_size * foset) + vss]) - ALGEBRAIC[(algebraic_size * foset) + JdiffK];
RATES[ (states_size * foset) +nai] = ( - (ALGEBRAIC[(algebraic_size * foset) + INa]+ALGEBRAIC[(algebraic_size * foset) + INaL]+ 3.00000*ALGEBRAIC[(algebraic_size * foset) + INaCa_i]+ALGEBRAIC[(algebraic_size * foset) + ICaNa_i]+ 3.00000*ALGEBRAIC[(algebraic_size * foset) + INaK]+ALGEBRAIC[(algebraic_size * foset) + INab])*CONSTANTS[(constants_size * foset) + Acap])/( CONSTANTS[(constants_size * foset) + F]*CONSTANTS[(constants_size * foset) + vmyo])+( ALGEBRAIC[(algebraic_size * foset) + JdiffNa]*CONSTANTS[(constants_size * foset) + vss])/CONSTANTS[(constants_size * foset) + vmyo];
RATES[ (states_size * foset) +nass] = ( - (ALGEBRAIC[(algebraic_size * foset) + ICaNa_ss]+ 3.00000*ALGEBRAIC[(algebraic_size * foset) + INaCa_ss])*CONSTANTS[(constants_size * foset) + Acap])/( CONSTANTS[(constants_size * foset) + F]*CONSTANTS[(constants_size * foset) + vss]) - ALGEBRAIC[(algebraic_size * foset) + JdiffNa];
RATES[ (states_size * foset) +cass] =  ALGEBRAIC[(algebraic_size * foset) + Bcass]*((( - (ALGEBRAIC[(algebraic_size * foset) + ICaL_ss] -  2.00000*ALGEBRAIC[(algebraic_size * foset) + INaCa_ss])*CONSTANTS[(constants_size * foset) + Acap])/( 2.00000*CONSTANTS[(constants_size * foset) + F]*CONSTANTS[(constants_size * foset) + vss])+( ALGEBRAIC[(algebraic_size * foset) + Jrel]*CONSTANTS[(constants_size * foset) + vjsr])/CONSTANTS[(constants_size * foset) + vss]) - ALGEBRAIC[(algebraic_size * foset) + Jdiff]);

RATES[ (states_size * foset) +V] = - (ALGEBRAIC[(algebraic_size * foset) + INa]+ALGEBRAIC[(algebraic_size * foset) + INaL]+ALGEBRAIC[(algebraic_size * foset) + Ito]+ALGEBRAIC[(algebraic_size * foset) + ICaL]+ALGEBRAIC[(algebraic_size * foset) + ICaNa]+ALGEBRAIC[(algebraic_size * foset) + ICaK]+ALGEBRAIC[(algebraic_size * foset) + IKr]+ALGEBRAIC[(algebraic_size * foset) + IKs]+ALGEBRAIC[(algebraic_size * foset) + IK1]+ALGEBRAIC[(algebraic_size * foset) + INaCa_i]+ALGEBRAIC[(algebraic_size * foset) + INaCa_ss]+ALGEBRAIC[(algebraic_size * foset) + INaK]+ALGEBRAIC[(algebraic_size * foset) + INab]+ALGEBRAIC[(algebraic_size * foset) + IKb]+ALGEBRAIC[(algebraic_size * foset) + IpCa]+ALGEBRAIC[(algebraic_size * foset) + ICab]+ALGEBRAIC[(algebraic_size * foset) + IClCa]+ALGEBRAIC[(algebraic_size * foset) + IClb]+ALGEBRAIC[(algebraic_size * foset) + I_katp]+ALGEBRAIC[(algebraic_size * foset) + Istim]);

RATES[ (states_size * foset) +cai] =  ALGEBRAIC[(algebraic_size * foset) + Bcai]*((( - ((ALGEBRAIC[(algebraic_size * foset) + ICaL_i]+ALGEBRAIC[(algebraic_size * foset) + IpCa]+ALGEBRAIC[(algebraic_size * foset) + ICab]) -  2.00000*ALGEBRAIC[(algebraic_size * foset) + INaCa_i])*CONSTANTS[(constants_size * foset) + Acap])/( 2.00000*CONSTANTS[(constants_size * foset) + F]*CONSTANTS[(constants_size * foset) + vmyo]) - ( ALGEBRAIC[(algebraic_size * foset) + Jup]*CONSTANTS[(constants_size * foset) + vnsr])/CONSTANTS[(constants_size * foset) + vmyo])+( ALGEBRAIC[(algebraic_size * foset) + Jdiff]*CONSTANTS[(constants_size * foset) + vss])/CONSTANTS[(constants_size * foset) + vmyo]);
RATES[ (states_size * foset) +cansr] = ALGEBRAIC[(algebraic_size * foset) + Jup] - ( ALGEBRAIC[(algebraic_size * foset) + Jtr]*CONSTANTS[(constants_size * foset) + vjsr])/CONSTANTS[(constants_size * foset) + vnsr];
RATES[ (states_size * foset) +cajsr] =  ALGEBRAIC[(algebraic_size * foset) + Bcajsr]*(ALGEBRAIC[(algebraic_size * foset) + Jtr] - ALGEBRAIC[(algebraic_size * foset) + Jrel]);
}

// void Tomek_model::solveRK4(double TIME, double dt)
// {
// 	double k1[43],k23[43];
// 	double yk123[43];
// 	int idx;


// 	// assuming first computeRates() have been executed
// 	computeRates( TIME, CONSTANTS, RATES, STATES, ALGEBRAIC );
// 	for( idx = 0; idx < states_size; idx++ ) {
// 		k1[idx] = RATES[ (states_size * foset) +idx];
// 		yk123[idx] = STATES[(states_size * foset) + idx] + (k1[idx]*dt*0.5);
// 	}
// 	computeRates( TIME+(dt*0.5), CONSTANTS, RATES, yk123, ALGEBRAIC );
// 	for( idx = 0; idx < states_size; idx++ ) {
// 		k23[idx] = RATES[ (states_size * foset) +idx];
// 		yk123[idx] = STATES[(states_size * foset) + idx] + (k23[idx]*dt*0.5);
// 	}
// 	computeRates( TIME+(dt*0.5), CONSTANTS, RATES, yk123, ALGEBRAIC );
//   for( idx = 0; idx < states_size; idx++ ) {
//     k23[idx] += RATES[ (states_size * foset) +idx];
//     yk123[idx] = STATES[(states_size * foset) + idx] + (k23[idx]*dt);
//   }
//   computeRates( TIME+dt, CONSTANTS, RATES, yk123, ALGEBRAIC );
// 	for( idx = 0; idx < states_size; idx++ ) {
// 		STATES[(states_size * foset) + idx] += (k1[idx]+(2*k23[idx])+RATES[idx])/6. * dt;
//   }


// }

__device__ void solveAnalytical(double *CONSTANTS, double *STATES, double *ALGEBRAIC, double *RATES, double dt, int foset)
{
int algebraic_size = 223;
int constants_size = 163;
int states_size = 43;
////==============
////Exact solution
////==============
////INa
  STATES[(states_size * foset) + m] = ALGEBRAIC[(algebraic_size * foset) + mss] - (ALGEBRAIC[(algebraic_size * foset) + mss] - STATES[(states_size * foset) + m]) * exp(-dt / ALGEBRAIC[(algebraic_size * foset) + tm]);
  STATES[(states_size * foset) + h] = ALGEBRAIC[(algebraic_size * foset) + hss] - (ALGEBRAIC[(algebraic_size * foset) + hss] - STATES[(states_size * foset) + h]) * exp(-dt / ALGEBRAIC[(algebraic_size * foset) + th]);
  STATES[(states_size * foset) + j] = ALGEBRAIC[(algebraic_size * foset) + jss] - (ALGEBRAIC[(algebraic_size * foset) + jss] - STATES[(states_size * foset) + j]) * exp(-dt / ALGEBRAIC[(algebraic_size * foset) + tj]);
  STATES[(states_size * foset) + hp] = ALGEBRAIC[(algebraic_size * foset) + hssp] - (ALGEBRAIC[(algebraic_size * foset) + hssp] - STATES[(states_size * foset) + hp]) * exp(-dt / ALGEBRAIC[(algebraic_size * foset) + th]);
  STATES[(states_size * foset) + jp] = ALGEBRAIC[(algebraic_size * foset) + jss] - (ALGEBRAIC[(algebraic_size * foset) + jss] - STATES[(states_size * foset) + jp]) * exp(-dt / ALGEBRAIC[(algebraic_size * foset) + tjp]);
  STATES[(states_size * foset) + mL] = ALGEBRAIC[(algebraic_size * foset) + mLss] - (ALGEBRAIC[(algebraic_size * foset) + mLss] - STATES[(states_size * foset) + mL]) * exp(-dt / ALGEBRAIC[(algebraic_size * foset) + tmL]);
  STATES[(states_size * foset) + hL] = ALGEBRAIC[(algebraic_size * foset) + hLss] - (ALGEBRAIC[(algebraic_size * foset) + hLss] - STATES[(states_size * foset) + hL]) * exp(-dt / CONSTANTS[(constants_size * foset) + thL]);
  STATES[(states_size * foset) + hLp] = ALGEBRAIC[(algebraic_size * foset) + hLssp] - (ALGEBRAIC[(algebraic_size * foset) + hLssp] - STATES[(states_size * foset) + hLp]) * exp(-dt / CONSTANTS[(constants_size * foset) + thLp]);
////Ito
  STATES[(states_size * foset) + a] = ALGEBRAIC[(algebraic_size * foset) + ass] - (ALGEBRAIC[(algebraic_size * foset) + ass] - STATES[(states_size * foset) + a]) * exp(-dt / ALGEBRAIC[(algebraic_size * foset) + ta]);
  STATES[(states_size * foset) + iF] = ALGEBRAIC[(algebraic_size * foset) + iss] - (ALGEBRAIC[(algebraic_size * foset) + iss] - STATES[(states_size * foset) + iF]) * exp(-dt / ALGEBRAIC[(algebraic_size * foset) + tiF]);
  STATES[(states_size * foset) + iS] = ALGEBRAIC[(algebraic_size * foset) + iss] - (ALGEBRAIC[(algebraic_size * foset) + iss] - STATES[(states_size * foset) + iS]) * exp(-dt / ALGEBRAIC[(algebraic_size * foset) + tiS]);
  STATES[(states_size * foset) + ap] = ALGEBRAIC[(algebraic_size * foset) + assp] - (ALGEBRAIC[(algebraic_size * foset) + assp] - STATES[(states_size * foset) + ap]) * exp(-dt / ALGEBRAIC[(algebraic_size * foset) + ta]);
  STATES[(states_size * foset) + iFp] = ALGEBRAIC[(algebraic_size * foset) + iss] - (ALGEBRAIC[(algebraic_size * foset) + iss] - STATES[(states_size * foset) + iFp]) * exp(-dt / ALGEBRAIC[(algebraic_size * foset) + tiFp]);
  STATES[(states_size * foset) + iSp] = ALGEBRAIC[(algebraic_size * foset) + iss] - (ALGEBRAIC[(algebraic_size * foset) + iss] - STATES[(states_size * foset) + iSp]) * exp(-dt / ALGEBRAIC[(algebraic_size * foset) + tiSp]);
////ICaL
  STATES[(states_size * foset) + d] = ALGEBRAIC[(algebraic_size * foset) + dss] - (ALGEBRAIC[(algebraic_size * foset) + dss] - STATES[(states_size * foset) + d]) * exp(-dt / ALGEBRAIC[(algebraic_size * foset) + td]);
  STATES[(states_size * foset) + ff] = ALGEBRAIC[(algebraic_size * foset) + fss] - (ALGEBRAIC[(algebraic_size * foset) + fss] - STATES[(states_size * foset) + ff]) * exp(-dt / ALGEBRAIC[(algebraic_size * foset) + tff]);
  STATES[(states_size * foset) + fs] = ALGEBRAIC[(algebraic_size * foset) + fss] - (ALGEBRAIC[(algebraic_size * foset) + fss] - STATES[(states_size * foset) + fs]) * exp(-dt / ALGEBRAIC[(algebraic_size * foset) + tfs]);
  STATES[(states_size * foset) + fcaf] = ALGEBRAIC[(algebraic_size * foset) + fcass] - (ALGEBRAIC[(algebraic_size * foset) + fcass] - STATES[(states_size * foset) + fcaf]) * exp(-dt / ALGEBRAIC[(algebraic_size * foset) + tfcaf]);
  STATES[(states_size * foset) + fcas] = ALGEBRAIC[(algebraic_size * foset) + fcass] - (ALGEBRAIC[(algebraic_size * foset) + fcass] - STATES[(states_size * foset) + fcas]) * exp(-dt / ALGEBRAIC[(algebraic_size * foset) + tfcas]);
  STATES[(states_size * foset) + jca] = ALGEBRAIC[(algebraic_size * foset) + jcass] - (ALGEBRAIC[(algebraic_size * foset) + jcass] - STATES[(states_size * foset) + jca]) * exp(- dt / CONSTANTS[(constants_size * foset) + tjca]);
  STATES[(states_size * foset) + ffp] = ALGEBRAIC[(algebraic_size * foset) + fss] - (ALGEBRAIC[(algebraic_size * foset) + fss] - STATES[(states_size * foset) + ffp]) * exp(-dt / ALGEBRAIC[(algebraic_size * foset) + tffp]);
  STATES[(states_size * foset) + fcafp] = ALGEBRAIC[(algebraic_size * foset) + fcass] - (ALGEBRAIC[(algebraic_size * foset) + fcass] - STATES[(states_size * foset) + fcafp]) * exp(-d / ALGEBRAIC[(algebraic_size * foset) + tfcafp]);
	STATES[(states_size * foset) + nca_i] = STATES[(states_size * foset) + nca_i] + RATES[(states_size * foset) +nca_i]*dt;
	STATES[(states_size * foset) + nca_ss] = STATES[(states_size * foset) + nca_ss] + RATES[(states_size * foset) +nca_ss]*dt;
//  STATES[nca_i] = ALGEBRAIC[(algebraic_size * foset) + anca_i] * CONSTANTS[(constants_size * foset) + k2n] / ALGEBRAIC[(algebraic_size * foset) + km2n] -
//      (ALGEBRAIC[(algebraic_size * foset) + anca_i] * CONSTANTS[(constants_size * foset) + k2n] / ALGEBRAIC[km2n] - STATES[nca_i]) * exp(-ALGEBRAIC[km2n] * dt);
//  STATES[nca_ss] = ALGEBRAIC[anca_ss] * CONSTANTS[(constants_size * foset) + k2n] / ALGEBRAIC[km2n] -
//      (ALGEBRAIC[anca_ss] * CONSTANTS[(constants_size * foset) + k2n] / ALGEBRAIC[km2n] - STATES[nca_ss]) * exp(-ALGEBRAIC[km2n] * dt);
////IKr
  //STATES[O] = STATES[O] + RATES[O] * dt;
  //STATES[I] = STATES[I] + RATES[I] * dt;
  //STATES[C3] = STATES[C3] + RATES[C3] * dt;
  //STATES[C2] = STATES[C2] + RATES[C2] * dt;
  //STATES[C1] = STATES[C1] + RATES[C1] * dt;
  double* coeffs = new double[15];
  coeffs[0] = -  (ALGEBRAIC[(algebraic_size * foset) + beta_2]+ALGEBRAIC[(algebraic_size * foset) + alpha_i]);
  coeffs[1] = ALGEBRAIC[(algebraic_size * foset) + beta_i];
  coeffs[2] = ALGEBRAIC[(algebraic_size * foset) + alpha_2];
  coeffs[3] = ALGEBRAIC[(algebraic_size * foset) + alpha_i];
  coeffs[4] = -  (ALGEBRAIC[(algebraic_size * foset) + beta_ItoC2]+ALGEBRAIC[(algebraic_size * foset) + beta_i]);
  coeffs[5] = ALGEBRAIC[(algebraic_size * foset) + alpha_C2ToI];
  coeffs[6] = ALGEBRAIC[(algebraic_size * foset) + beta_2];
  coeffs[7] = ALGEBRAIC[(algebraic_size * foset) + beta_ItoC2];
  coeffs[8] = -  (CONSTANTS[(constants_size * foset) + beta_1]+ALGEBRAIC[(algebraic_size * foset) + alpha_2]+ALGEBRAIC[(algebraic_size * foset) + alpha_C2ToI]);
  coeffs[9] = CONSTANTS[(constants_size * foset) + alpha_1];
  coeffs[10] = CONSTANTS[(constants_size * foset) + beta_1];
  coeffs[11] = -  (ALGEBRAIC[(algebraic_size * foset) + beta]+CONSTANTS[(constants_size * foset) + alpha_1]);
  coeffs[12] = ALGEBRAIC[(algebraic_size * foset) + alpha];
  coeffs[13] = ALGEBRAIC[(algebraic_size * foset) + beta];
  coeffs[14] = -  ALGEBRAIC[(algebraic_size * foset) + alpha];
  int m = 5;
  double* a = new double[m*m]; // Flattened a
  a[0 * m + 0] = 1.0 - dt * coeffs[0];   a[0 * m + 1] = - dt * coeffs[1];     a[0 * m + 2] = - dt * coeffs[2];     a[0 * m + 3] = 0.0;                      a[0 * m + 4] = 0.0;
  a[1 * m + 0] = - dt * coeffs[3];       a[1 * m + 1] = 1.0 - dt * coeffs[4]; a[1 * m + 2] = - dt * coeffs[5];     a[1 * m + 3] = 0.0;                      a[1 * m + 4] = 0.0;
  a[2 * m + 0] = - dt * coeffs[6];       a[2 * m + 1] = - dt * coeffs[7];     a[2 * m + 2] = 1.0 - dt * coeffs[8]; a[2 * m + 3] = - dt * coeffs[9];         a[2 * m + 4] = 0.0;
  a[3 * m + 0] = 0.0;                    a[3 * m + 1] = 0.0;                  a[3 * m + 2] = - dt * coeffs[10];    a[3 * m + 3] = 1.0 - dt * coeffs[11];    a[3 * m + 4] = - dt * coeffs[12];
  a[4 * m + 0] = 0.0;                    a[4 * m + 1] = 0.0;                  a[4 * m + 2] = 0.0;                  a[4 * m + 3] = - dt * coeffs[13];;       a[4 * m + 4] = 1.0 - dt * coeffs[14];
  double* b = new double[m];
  b[0] = STATES[(states_size * foset) + O];
  b[1] = STATES[(states_size * foset) + I];
  b[2] = STATES[(states_size * foset) + C1];
  b[3] = STATES[(states_size * foset) + C2];
  b[4] = STATES[(states_size * foset) + C3];
  double* x = new double[m];
  for(int i = 0; i < m; i++){
    x[i] = 0.0;
  }
  ___gaussElimination(a,b,x,m);
  STATES[(states_size * foset) + O] = x[0];
  STATES[(states_size * foset) + I] = x[1];
  STATES[(states_size * foset) + C1] = x[2];
  STATES[(states_size * foset) + C2] = x[3];
  STATES[(states_size * foset) + C3] = x[4];
  delete[] coeffs;
  delete[] a;
  delete[] b;
  delete[] x;
  
////IKs
  STATES[(states_size * foset) + xs1] = ALGEBRAIC[(algebraic_size * foset) + xs1ss] - (ALGEBRAIC[(algebraic_size * foset) + xs1ss] - STATES[(states_size * foset) + xs1]) * exp(-dt / ALGEBRAIC[(algebraic_size * foset) + txs1]);
  STATES[(states_size * foset) + xs2] = ALGEBRAIC[(algebraic_size * foset) + xs2ss] - (ALGEBRAIC[(algebraic_size * foset) + xs2ss] - STATES[(states_size * foset) + xs2]) * exp(-dt / ALGEBRAIC[(algebraic_size * foset) + txs2]);
////IK1
////RyR receptors
  STATES[(states_size * foset) + Jrel_np] = ALGEBRAIC[(algebraic_size * foset) + Jrel_inf] - (ALGEBRAIC[(algebraic_size * foset) + Jrel_inf] - STATES[(states_size * foset) + Jrel_np]) * exp(-dt / ALGEBRAIC[(algebraic_size * foset) + tau_rel]);
  STATES[(states_size * foset) + Jrel_p] = ALGEBRAIC[(algebraic_size * foset) + Jrel_infp] - (ALGEBRAIC[(algebraic_size * foset) + Jrel_infp] - STATES[(states_size * foset) + Jrel_p]) * exp(-dt / ALGEBRAIC[(algebraic_size * foset) + tau_relp]);
////=============================
////Approximated solution (Euler)
////=============================
////CaMK
  STATES[(states_size * foset) + CaMKt] = STATES[(states_size * foset) + CaMKt] + RATES[(states_size * foset) +CaMKt] * dt;
////Membrane potential
  STATES[(states_size * foset) + V] = STATES[(states_size * foset) + V] + RATES[(states_size * foset) +V] * dt;
////Ion Concentrations and Buffers
  STATES[(states_size * foset) + nai] = STATES[(states_size * foset) + nai] + RATES[(states_size * foset) +nai] * dt;
  STATES[(states_size * foset) + nass] = STATES[(states_size * foset) + nass] + RATES[(states_size * foset) +nass] * dt;
  STATES[(states_size * foset) + ki] = STATES[(states_size * foset) + ki] + RATES[(states_size * foset) +ki] * dt;
  STATES[(states_size * foset) + kss] = STATES[(states_size * foset) + kss] + RATES[(states_size * foset) +kss] * dt;
  STATES[(states_size * foset) + cai] = STATES[(states_size * foset) + cai] + RATES[(states_size * foset) +cai] * dt;
  STATES[(states_size * foset) + cass] = STATES[(states_size * foset) + cass] + RATES[(states_size * foset) +cass] * dt;
  STATES[(states_size * foset) + cansr] = STATES[(states_size * foset) + cansr] + RATES[(states_size * foset) +cansr] * dt;
  STATES[(states_size * foset) + cajsr] = STATES[(states_size * foset) + cajsr] + RATES[(states_size * foset) +cajsr] * dt;
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

__device__ double set_time_step(double TIME, double time_point, double max_time_step, double *CONSTANTS, double *RATES, int foset) {
 int constants_size = 163;
 int rates_size = 43;

 double min_time_step = 0.005;
 double min_dV = 0.2;
 double max_dV = 0.8;
 double time_step = min_time_step;

 if (TIME <= time_point || (TIME - floor(TIME / CONSTANTS[(constants_size * foset) + BCL]) * CONSTANTS[(constants_size * foset) + BCL]) <= time_point) {
    //printf("TIME <= time_point ms\n");
    return time_step;
    //printf("TIME = %E, dV = %E, time_step = %E\n",TIME, RATES[V] * time_step, time_step);
  }
  else {
    //printf("TIME > time_point ms\n");
    if (std::abs(RATES[(rates_size * foset) +V] * time_step) <= min_dV) {//Slow changes in V
        //printf("dV/dt <= 0.2\n");
        time_step = std::abs(max_dV / RATES[(rates_size * foset) +V]);
        //Make sure time_step is between min time step and max_time_step
        if (time_step < min_time_step) {
            time_step = min_time_step;
        }
        else if (time_step > max_time_step) {
            time_step = max_time_step;
        }
        //printf("TIME = %E, dV = %E, time_step = %E\n",TIME, RATES[V] * time_step, time_step);
    }
    else if (std::abs(RATES[(rates_size * foset) +V] * time_step) >= max_dV) {//Fast changes in V
        //printf("dV/dt >= 0.8\n");
        time_step = std::abs(min_dV / RATES[(rates_size * foset) +V]);
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
__device__ void solveEuler(double *STATES, double *RATES, double dt, int foset)
{
  //int rates_size = 43;
  int states_size = 43;
  STATES[(states_size * foset) + V] = STATES[(states_size * foset) + V] + RATES[ (states_size * foset) +V] * dt;
  STATES[(states_size * foset) + CaMKt] = STATES[(states_size * foset) + CaMKt] + RATES[ (states_size * foset) +CaMKt] * dt;
  STATES[(states_size * foset) + cass] = STATES[(states_size * foset) + cass] + RATES[ (states_size * foset) +cass] * dt;
  STATES[(states_size * foset) + nai] = STATES[(states_size * foset) + nai] + RATES[ (states_size * foset) +nai] * dt;
  STATES[(states_size * foset) + nass] = STATES[(states_size * foset) + nass] + RATES[ (states_size * foset) +nass] * dt;
  STATES[(states_size * foset) + ki] = STATES[(states_size * foset) + ki] + RATES[ (states_size * foset) +ki] * dt;
  STATES[(states_size * foset) + kss] = STATES[(states_size * foset) + kss] + RATES[ (states_size * foset) +kss] * dt;
  STATES[(states_size * foset) + cansr] = STATES[(states_size * foset) + cansr] + RATES[ (states_size * foset) +cansr] * dt;
  STATES[(states_size * foset) + cajsr] = STATES[(states_size * foset) + cajsr] + RATES[ (states_size * foset) +cajsr] * dt;
  STATES[(states_size * foset) + cai] = STATES[(states_size * foset) + cai] + RATES[ (states_size * foset) +cai] * dt;
  STATES[(states_size * foset) + m] = STATES[(states_size * foset) + m] + RATES[ (states_size * foset) +m] * dt;
  STATES[(states_size * foset) + h] = STATES[(states_size * foset) + h] + RATES[ (states_size * foset) +h] * dt;
  STATES[(states_size * foset) + j] = STATES[(states_size * foset) + j] + RATES[ (states_size * foset) +j] * dt;
  STATES[(states_size * foset) + hp] = STATES[(states_size * foset) + hp] + RATES[ (states_size * foset) +hp] * dt;
  STATES[(states_size * foset) + jp] = STATES[(states_size * foset) + jp] + RATES[ (states_size * foset) +jp] * dt;
  STATES[(states_size * foset) + mL] = STATES[(states_size * foset) + mL] + RATES[ (states_size * foset) +mL] * dt;
  STATES[(states_size * foset) + hL] = STATES[(states_size * foset) + hL] + RATES[ (states_size * foset) +hL] * dt;
  STATES[(states_size * foset) + hLp] = STATES[(states_size * foset) + hLp] + RATES[ (states_size * foset) +hLp] * dt;
  STATES[(states_size * foset) + a] = STATES[(states_size * foset) + a] + RATES[ (states_size * foset) +a] * dt;
  STATES[(states_size * foset) + iF] = STATES[(states_size * foset) + iF] + RATES[ (states_size * foset) +iF] * dt;
  STATES[(states_size * foset) + iS] = STATES[(states_size * foset) + iS] + RATES[ (states_size * foset) +iS] * dt;
  STATES[(states_size * foset) + ap] = STATES[(states_size * foset) + ap] + RATES[ (states_size * foset) +ap] * dt;
  STATES[(states_size * foset) + iFp] = STATES[(states_size * foset) + iFp] + RATES[ (states_size * foset) +iFp] * dt;
  STATES[(states_size * foset) + iSp] = STATES[(states_size * foset) + iSp] + RATES[ (states_size * foset) +iSp] * dt;
  STATES[(states_size * foset) + d] = STATES[(states_size * foset) + d] + RATES[ (states_size * foset) +d] * dt;
  STATES[(states_size * foset) + ff] = STATES[(states_size * foset) + ff] + RATES[ (states_size * foset) +ff] * dt;
  STATES[(states_size * foset) + fs] = STATES[(states_size * foset) + fs] + RATES[ (states_size * foset) +fs] * dt;
  STATES[(states_size * foset) + fcaf] = STATES[(states_size * foset) + fcaf] + RATES[ (states_size * foset) +fcaf] * dt;
  STATES[(states_size * foset) + fcas] = STATES[(states_size * foset) + fcas] + RATES[ (states_size * foset) +fcas] * dt;
  STATES[(states_size * foset) + jca] = STATES[(states_size * foset) + jca] + RATES[ (states_size * foset) +jca] * dt;
  STATES[(states_size * foset) + ffp] = STATES[(states_size * foset) + ffp] + RATES[ (states_size * foset) +ffp] * dt;
  STATES[(states_size * foset) + fcafp] = STATES[(states_size * foset) + fcafp] + RATES[ (states_size * foset) +fcafp] * dt;
  STATES[(states_size * foset) + nca_ss] = STATES[(states_size * foset) + nca_ss] + RATES[ (states_size * foset) +nca_ss] * dt;
  STATES[(states_size * foset) + nca_i] = STATES[(states_size * foset) + nca_i] + RATES[ (states_size * foset) +nca_i] * dt;
  STATES[(states_size * foset) + O] = STATES[(states_size * foset) + O] + RATES[ (states_size * foset) +O] * dt;
  STATES[(states_size * foset) + I] = STATES[(states_size * foset) + I] + RATES[ (states_size * foset) +I] * dt;
	STATES[(states_size * foset) + C3] = STATES[(states_size * foset) + C3] + RATES[ (states_size * foset) +C3] * dt;
	STATES[(states_size * foset) + C2] = STATES[(states_size * foset) + C2] + RATES[ (states_size * foset) +C2] * dt;
	STATES[(states_size * foset) + C1] = STATES[(states_size * foset) + C1] + RATES[ (states_size * foset) +C1] * dt;
  STATES[(states_size * foset) + xs1] = STATES[(states_size * foset) + xs1] + RATES[ (states_size * foset) +xs1] * dt;
  STATES[(states_size * foset) + xs2] = STATES[(states_size * foset) + xs2] + RATES[ (states_size * foset) +xs2] * dt;
  STATES[(states_size * foset) + Jrel_np] = STATES[(states_size * foset) + Jrel_np] + RATES[ (states_size * foset) +Jrel_np] * dt;
  STATES[(states_size * foset) + Jrel_p] = STATES[(states_size * foset) + Jrel_p] + RATES[(states_size * foset) +Jrel_p] * dt;
}

// ord 2011 set time step
// __device__ double set_time_step(double TIME, double time_point, double max_time_step, double *CONSTANTS, double *RATES, int foset) {
//   double time_step = 0.005;
//   int constants_size = 163;
//   int rates_size = 43;

//   if (TIME <= time_point || (TIME - floor(TIME / CONSTANTS[BCL + (offset * constants_size)]) * CONSTANTS[BCL + (offset * constants_size)]) <= time_point) {
//     // printf("TIME <= time_point ms\n");
//     // printf("dV = %lf, time_step = %lf\n",RATES[V + (offset * rates_size)] * time_step, time_step);
//     return time_step;
//     // printf("dV = %lf, time_step = %lf\n",RATES[V] * time_step, time_step);
//   }
//   else {
//     printf("TIME > time_point ms\n");
//     if (std::abs(RATES[V + (offset * rates_size)] * time_step) <= 0.2) {//Slow changes in V
//         // printf("dV/dt <= 0.2\n");
//         time_step = std::abs(0.8 / RATES[V + (offset * rates_size)]);
//         //Make sure time_step is between 0.005 and max_time_step
//         if (time_step < 0.005) {
//             time_step = 0.005;
//         }
//         else if (time_step > max_time_step) {
//             time_step = max_time_step;
//         }
//         //printf("dV = %lf, time_step = %lf\n",std::abs(RATES[V] * time_step), time_step);
//     }
//     else if (std::abs(RATES[V + (offset * rates_size)] * time_step) >= 0.8) {//Fast changes in V
//         // printf("dV/dt >= 0.8\n");
//         time_step = std::abs(0.2 / RATES[V + (offset * rates_size)]);
//         while (std::abs(RATES[V + (offset * rates_size)]  * time_step) >= 0.8 &&
//                0.005 < time_step &&
//                time_step < max_time_step) {
//             time_step = time_step / 10.0;
//             // printf("dV = %lf, time_step = %lf\n",std::abs(RATES[V] * time_step), time_step);
//         }
//     }
//     // __syncthreads();
//     return time_step;
//   }
// }
