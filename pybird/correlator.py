import os
import sys
import numpy as np
from copy import deepcopy
from scipy.interpolate import interp1d
from scipy.fftpack import dst

from pybird.common import Common, co
from pybird.bird import Bird
from pybird.nonlinear import NonLinear
from pybird.nnlo import NNLO_higher_derivative, NNLO_counterterm
from pybird.resum import Resum
from pybird.projection import Projection
from pybird.greenfunction import GreenFunction
from pybird.fourier import FourierTransform
from pybird.matching import Matching
## MP ##
# from bispectrum_nl import BispectrumNl
#from bispectrum_nl_quad import BispectrumNl
#from bispectrum import Bispectrum


class Correlator(object):

    def __init__(self, config_dict=None, load_engines=True):

        self.cosmo_catalog = {
            "pk_lin": Option("pk_lin", (list, np.ndarray),
                description="Linear matter power spectrum in [Mpc/h]^3",
                default=None) ,
            "kk": Option("kk", (list, np.ndarray),
                description="k-array in [h/Mpc] on which pk_lin is evaluated",
                default=None) ,
            "D": Option("D", (float, list, np.ndarray),
                description="Scale independent growth function. To specify if \'with_time\' is False, e.g., \'with_nonequal_time\' or \'with_redshift_bin\' is True.",
                default=None) ,
            "f": Option("f", (float, list, np.ndarray),
                description="Scale independent growth rate (for RSD). Automatically set to 0 for \'output\': \'m__\'.",
                default=None) ,
            "bias": Option("bias", dict,
                description="EFT parameters in dictionary to specify as \
                    (\'eft_basis\': \'eftoflss\') \{ \'b1\'(a), \'b2\'(a), \'b3\'(a), \'b4\'(a), \'cct\', \'cr1\'(b), \'cr2\'(b), \'ce0\'(d), \'ce1\'(d), \'ce2\'(d)] \} \
                    (\'eft_basis\': \'westcoast\') \{ \'b1\'(a), \'c2\'(a), \'c4\'(a), \'b3\'(a), \'cct\', \'cr1\'(b), \'cr2\'(b), \'ce0\'(d), \'ce1\'(d), \'ce2\'(d)] \} \
                    (\'eft_basis\': \'eastcoast\') \{ \'b1\'(a), \'b2\'(a), \'bG2\'(a), \'bgamma3\'(a), \'c0\', \'c2\'(b), \'c4\'(c), \'ce0\'(d), \'ce1\'(d), \'ce2\'(d)] \} \
                    if (a): \'b\' in \'output\'; (b): \'multipole\'>=2; (d): \'with_stoch\' is True ",
                default=None) ,
            "Omega0_m": Option("Omega0_m", float,
                description="Fractional matter abundance at present time. To specify if \'with_exact_time\' is True.",
                default=None) ,
            "H": Option("H", float,
                description="Hubble parameter by H_0. To specify if \'with_ap\' is True.",
                default=None) ,
            "DA": Option("DA", float,
                description="Angular distance times H_0. To specify if \'with_ap\' is True.",
                default=None) ,
            "z": Option("z", float,
                description="Effective redshift(s). To specify if \'with_time\' is False or \'with_exact_time\' is True.",
                default=None) ,
            "A": Option("A", float,
                description="Amplitude rescaling, i.e, A = A_s / A_s^\{fid\}. Default: A=1. If \'with_time\' is False, can in some ways be used as a fast parameter.",
                default=None) ,
            "w0_fld": Option("w0_fld", float,
                description="Dark energy equation of state parameter. To specify in presence of dark energy if \'with_exact_time\' is True (otherwise w0 = -1).",
                default=None) ,
            "wa_fld": Option("wa_fld", float,
                description="Dark energy equation of state parameterfor evolving model. To specify for exact time dependence if varied (otherwise wa = 0).", 
                default=None) ,
            "alpha_T0": Option("alpha_T0", float,
                description="EFTofDE parameter, today's amplitude of the alpha_T parameter",
                default=None) ,
            "alpha_B0": Option("alpha_B0", float,
                description="EFTofDE parameter, today's amplitude of the alpha_B parameter",
                default=None) ,
            "alpha_M0": Option("alpha_M0", float,
                description="EFTofDE parameter, today's amplitude of the alpha_M parameter, parametrizes the variation fo the Planck mass",
                default=None) ,
            "eta": Option("eta", float,
                description="EFTofDE parameter, exponent for the alpha evolution: alpha(a) = alpha_0 a^eta",
                default=None) ,
            "A_s": Option("A_s", float,
                description="Amplitude of power spectrum", 
                default=None) ,
            "n_s": Option("n_s", float,
                description="Spectral index", 
                default=None) ,
            "h": Option("", float,
                description="Hubble", 
                default=None) ,
            "Dz": Option("Dz", (list, np.ndarray),
                description="Scale independent growth function over redshift bin. To specify if \'with_redshift_bin\' is True.",
                default=None) ,
            "fz": Option("fz", (list, np.ndarray),
                description="Scale independent growth rate over redshift bin. To specify if \'with_redshift_bin\' is True.",
                default=None) ,
            "rz": Option("rz", (list, np.ndarray),
                description="Comoving distance in [Mpc/h] over redshift bin. To specify if \'with_redshift_bin\' or if \'output\':\'w\'.",
                default=None) ,
            "D1": Option("D1", float,
                description="Scale independent growth function at redshift z1. To specify if \'with_nonequal_time\' is True.",
                default=None) ,
            "D2": Option("D2", float,
                description="Scale independent growth function at redshift z2. To specify if \'with_nonequal_time\' is True.",
                default=None) ,
            "f1": Option("f1", float,
                description="Scale independent growth rate at redshift z1. To specify if \'with_nonequal_time\' is True.",
                default=None) ,
            "f2": Option("f2", float,
                description="Scale independent growth rate at redshift z2. To specify if \'with_nonequal_time\' is True.",
                default=None) ,
            "Psmooth": Option("Psmooth", (list, np.ndarray),
                description="Smooth power spectrum. To specify if \'with_nnlo_counterterm\' is True.",
                default=None) ,
            "pk_lin_2": Option("pk_lin_2", (list, np.ndarray),
                description="Alternative linear matter power spectrum in [Mpc/h]^3 replacing \'pk_lin\' in the internal loop integrals (and resummation)",
                default=None) ,
            #GDA_Bisp
            "Pnw": Option("P11", (list, np.ndarray),
                description="No-wiggle linear matter power spectrum in [Mpc/h]^3",
                default=None) ,
            "Pw": Option("P11", (list, np.ndarray),
                description="Wiggle linear matter power spectrum in [Mpc/h]^3",
                default=None) ,
            "Sigma2": Option("Sigma2", float,
                description="Sigma^2 value",
                default=None) ,
            "deltaSigma2": Option("deltaSigma2", float,
                description="delta Sigma^2 value",
                default=None) ,
            "alphain": Option("alphain", dict,
                description="transfer function for fnl",
                default=None) ,
            #"mg_parameters": Option("mg_parameters", dict,
            #    description="parameters for modified gravity model given in config catalog (actually only for BOOTSTRAP)",
            #    default=None) ,
        }

        self.c_catalog = {
            # MP #
            "output": Option("output", str, ["bPk", "bCf", "mPk", "mCf", "bmPk", "bmCf", "w", "B", "PB"], 
                description="Correlator: biased tracers / matter / biased tracers-matter -- power spectrum / correlation function ; \'w\': angular correlation function/ bispectrum / power spectrum and bispectrum. ", 
                default="bPk") ,
            # MP #
            # "triangles": Option("triangles", (np.ndarray),
            #     description="(3,Nk)-array of Fourier modes in [h/Mpc] on which B is evaluated",
            #     default=None) ,
            ######
            ######
            "multipole": Option("multipole", int, [0, 2, 3],
                description="Number of multipoles. 0: real space. 2: monopole + quadrupole. 3: monopole + quadrupole + hexadecapole.",
                default=2) ,
            "z": Option("z", float,
                description="Effective redshift.",
                default=None) ,
            "km": Option("km", float,
                description="Inverse tracer spatial extension scale in [h/Mpc].",
                default=0.7) ,
            "kr": Option("kr", float,
                description="Inverse velocity product renormalization scale in [h/Mpc].",
                default=0.25) ,
            "nd": Option("nd", float,
                description="Mean galaxy density",
                default=3e-4) ,
            "kmax": Option("kmax", float,
                description="kmax in [h/Mpc] for \'output\': \'_Pk\'",
                default=0.25) ,
            "with_bias": Option("with_bias", bool,
                description="Bias (in)dependent evalution. Automatically set to False for \'with_time\': False.",
                   default=False) ,
            "eft_basis": Option("eft_basis", str,
                description="Basis of EFT parameters: \'eftoflss\' (default), \'westcoast\', or \'eastcoast\'. See cosmology command \'bias\' for more details.",
                default="eftoflss") ,
            "with_stoch": Option("with_stoch", bool,
                description="With stochastic terms.",
                   default=False) ,
            "with_nnlo_counterterm": Option("with_nnlo_counterterm", bool,
                description="With next-to-next-to-leading counterterm k^4 pk_lin.",
                default=False) ,
            "with_tidal_alignments": Option("with_tidal_alignments", bool,
                description="With tidal alignements: bq * (\mu^2 - 1/3) \delta_m ",
                default=False) ,
            "with_time": Option("with_time", bool,
                description="Time (in)dependent evaluation. For \'with_redshift_bin\': True, automatically set to False.",
                default=True) ,
            "with_exact_time": Option("with_exact_time", bool,
                description="Exact time dependence or EdS approximation.",
                default=False) ,
            "with_quintessence": Option("with_quintessence", bool,
                description="Clustering quintessence.",
                default=False) ,
            "with_nonequal_time": Option("with_nonequal_time", bool,
                description="Non equal time correlator. Automatically set \'with_time\' to False ",
                default=False) ,
            "Omega_rc": Option("Omega_rc", float,
                description="nDGP parameter",
                default=None) ,
            "fR0": Option("fR0", float,
                description="f(R) modified gravity with LCDM background.",
                default=None) ,
            "expansion_model": Option("expansion_model", str,
                description="Selects the time dependece of the background evolution, at the moment we have lcdm and w0wa.",
                default='lcdm',),
            "mg_model": Option("mg_model", str,
                description="Selects the model for non-linearities, at the moment we have lcdm, quintessence, nDGP, EFTofDE, bootstrap, fR",
                default='lcdm',),
            "gravity_model": Option("gravity_model", str,
                description="Selects the time dependence fo the EFTofDE parameters, at the moment we have propto_omega, propto_scale, eft_alphas_power_law, constant_alphas",
                default='propto_omega',),
            "z1": Option("z1", (float, list, np.ndarray),
                description="Redshift z_1 for non equal time correlator.",
                default=None) ,
            "z2": Option("z2", (float, list, np.ndarray),
                description="Redshift z_2 for non equal time correlator.",
                default=None) ,
            "with_resum": Option("with_resum", bool,
                description="Apply IR-resummation.",
                default=True) ,
            "optiresum": Option("optiresum", bool,
                description="[depreciated: keep on default False] True: Resumming only with the BAO peak. False: Resummation on the full correlation function.",
                default=False) ,
            "xdata": Option("xdata", (list, np.ndarray),
                description="Array of k [h/Mpc] (or s [Mpc/h]) on which to output the correlator. If \'with_binning\' is True, please provide the central k (or s). If not, it can be bin-weighted k (or s). If no \'xdata\' provided, output is on internal default array. ",
                default=None) ,
            "with_binning": Option("with_binning", bool,
                description="Apply binning for linear-spaced bins.",
                default=False) ,
            "binsize": Option("binsize", float,
                description="size of the bin.",
                default=None) ,
            "with_ap": Option("wity_AP", bool,
                description="Apply Alcock Paczynski effect. ",
                default=False) ,
            "H_fid": Option("H_fid", float,
                description="Hubble parameter by H_0. To specify if \'with_ap\' is True.",
                default=None) ,
            "D_fid": Option("D_fid", float,
                description="Angular distance times H_0. To specify if \'with_ap\' is True.",
                default=None) ,
            "with_survey_mask": Option("with_survey_mask", bool,
                description="Apply survey mask. Automatically set to False for \'output\': \'_Cf\'.",
                default=False) ,
            "survey_mask_arr_p": Option("survey_mask_arr_p", (list, np.ndarray),
                description="Mask convolution array for \'output\': \'_Pk\'.",
                default=None) ,
            "survey_mask_mat_kp": Option("survey_mask_mat_kp", (list, np.ndarray),
                description="Mask convolution matrix for \'output\': \'_Pk\'.",
                default=None) ,
            "with_fibercol": Option("with_fibercol", bool,
                description="Apply fiber collision effective window corrections.",
                default=False) ,
            "with_wedge": Option("with_wedge", bool,
                description="Rotate multipoles to wedges",
                default=False) ,
            "wedge_mat_wl": Option("wedge_mat_wl", (list, np.ndarray),
                description="multipole-to-wedge rotation matrix",
                default=None) ,
            "with_redshift_bin": Option("with_redshift_bin", bool,
                description="Account for the galaxy count distribution over a redshift bin.",
                default=False) ,
            "redshift_bin_zz": Option("redshift_bin_zz", (list, np.ndarray),
                description="Array of redshift points inside a redshift bin.",
                default=None) ,
            "redshift_bin_nz": Option("redshift_bin_nz", (list, np.ndarray),
                description="Galaxy counts distribution over a redshift bin.",
                default=None) ,
            "accboost": Option("accboost", int, [1, 2, 3],
                description="Sampling accuracy boost factor. Default k sampling: dk ~ 0.005 (k<0.3), dk ~ 0.01 (k>0.3). ",
                default=1) ,
            "fftaccboost": Option("fftaccboost", int, [1, 2, 3],
                description="FFTLog accuracy boost factor. Default FFTLog sampling : NFFT ~ 256. ",
                default=1) ,
            "fftbias": Option("fftbias", float,
                description="real power bias for fftlog decomposition of pk_lin (usually to keep to default value)",
                default=-1.6) ,
            "with_uvmatch_2": Option("with_uvmatch_2", bool,
                description="In case two linear power spectra \`pk_lin\` and \`pk_lin_2\` are provided (see description in cosmo_catalog), match the UV as in the case if only \`pk_lin\` would be provided. Implemented only for output=\`Pk\`. ",
                default=False) ,
            "keep_loop_pieces_independent": Option("keep_loop_pieces_independent", bool,
                description="keep the loop pieces 13 and 22 independent (mainly for debugging)",
                default=False) ,
        
            #GDA_Bisp
            "with_bisp": Option("with_bisp", bool, 
                description="Calculate or not the bispectrum", 
                default=False) ,
            "tree_level": Option("tree_level", bool, 
                description="Calculate bispectrum only at tree level", 
                default=False) ,
            "triangle_data": Option("triangle_data", dict,
                description="dict of data triangles.",
                default=None) ,
            "matrix_path": Option("matrix_path", str,
                description="Path of loop matrices for the bispectrum monopole.",
                default='./') ,
            "matrix_path_quad": Option("matrix_path_quad", str,
                description="Path of loop matrices for the bispectrum quadrupole.",
                default=None) ,
            "with_bin_bisp": Option("with_bin_bisp", bool,
                description="Apply binning for bispectrum triangles.",
                default=False) ,
            "Pk_fid": Option("Pk_fid", str,
                description="File with linear matter power spectrum in [Mpc/h]^3 on fiducial cosmology for binning",
                default=None) ,
            "bisp_binweights": Option("bisp_binweights", str,
                description="Weights to multiply the tree-level 211 bispectrum evaluated at keff",
                default=None) ,
            "with_bisp_window": Option("with_bisp_window", bool,
                description="Apply mask to bispectrum with 1D approximation.",
                default=False) ,
            "windowPk_bisp": Option("windowPk_bisp", (str, list),
                description="Path to Fourier convolution window file for the bispectrum for \'output\': \'_Pk\'. If not provided, read \'windowCf\', precompute the Fourier one and save it here.",
                default=None) ,
            "windowCf_bisp": Option("windowCf_bisp", (str, list),
                description="Path to configuration space window file for the bispectrum with columns: s [Mpc/h], Q0, Q2, Q4. A list can be provided for multi skycuts. Put \'None\' for each skycut without window.",
                default=None) ,
            "bisp_resum": Option("bisp_resum", bool, 
                description="Apply or not resummation of the bispectrum", 
                default=False) ,
            "with_bisp_nnlo": Option("with_bisp_nnlo", bool, 
                description="Compute or not nnlo terms for the bispectrum", 
                default=False) ,
            "bisp_quad": Option("bisp_quad", int,
                description="Number of quadrupole terms analyzed: 0 (monopole), 1 (002), 3 (002, 200, 020)",
                default=0) ,
            "with_bisp_rebin": Option("with_bisp_rebin", bool,
                description="rebin the bispectrum bins by a factor 2",
                default=False) ,
            "with_bisp_irsub": Option("with_bisp_irsub", bool,
                description="subtract IR pieces from individual bispectrum diagrams",
                default=False) ,
            "with_common_shot_noise_parameter_in_pk_and_bk": Option("with_common_shot_noise_parameter_in_pk_and_bk", bool,
                description="set True to enforce B_shot = P_shot^2",
                default=False) ,
            "with_bisp_normalization": Option("with_bisp_normalization", bool,
                description="Normalize the bispectrum to the FKP normalization",
                default=False) ,
            "bisp_norm": Option("bisp_norm", float,
                description="FKP normalization",
                default=1.) ,
            # MP #
            "with_bisp_stoch": Option("with_bisp_stoch", bool, 
                description="With stochastic terms (bispectrum).",
                   default=True)
            ######
        }

        if config_dict is not None: self.set(config_dict, load_engines=load_engines)


    def info(self, description=True):

        for on in ['config', 'cosmo']:

            print ("\n")
            if on == 'config':
                print ("Configuration commands [.set(config_dict)]")
                print ("----------------------")
                catalog = self.c_catalog
            elif on == 'cosmo':
                print ("Cosmology commands [.compute(cosmo_dict)]")
                print ("------------------")
                catalog = self.cosmo_catalog

            for (name, config) in zip(catalog, catalog.values()):
                if config.list is None: print("\'%s\': %s" % (name, typename(config.type)))
                else: print("\'%s\': %s ; options: %s" % (name, typename(config.type), config.list))
                if description:
                    print ('    - %s' % config.description)
                    print ('    * default: %s' % config.default)

    def set(self, config_dict, load_engines=True):

        # Reading config provided by user
        self.__read_config(config_dict)

        # Setting no-optional config
        self.c["smin"] = 1.
        self.c["smax"] = 1000.
        self.c["kmin"] = 0.001

        # Checking for config conflict
        self.__is_config_conflict()

        if self.c["with_bisp"]:
            if self.c["triangle_data"] is not None:
                if self.c["with_bisp_rebin"]: self.Ntriangles = self.c["triangle_data"]["ltn"].shape[1]
                else: self.Ntriangles = self.c["triangle_data"]["tn"].shape[1]
                if self.c["bisp_quad"] == 3:
                    self.Ntriangles_200 = len(self.c["triangle_data"]["idx_200"])
                    self.Ntriangles_020 = len(self.c["triangle_data"]["idx_020"])
            else:
                print("You must provide triangle data!")

        # Setting list of EFT parameters required by the user to provide later
        self.__set_eft_parameters_list()

        # Loading PyBird engines
        self.__load_engines(load_engines=load_engines)

    def compute(self, cosmo_dict=None, cosmo_module=None, cosmo_engine=None, correlator_engine=None, do_core=True, do_survey_specific=True, bias = None):
        #---------------------------IMPORTANT!!!-------------------------------
        #MM: I am adding "bias" inside compute because with the bootstrap I will 
        # define the epsilon parameters as biases, but since they are
        # "cosmological biases" I need them when I compute things inside Bird.
        #----------------------------------------------------------------------
        if cosmo_dict: cosmo_dict_local = cosmo_dict.copy()
        elif cosmo_module and cosmo_engine: cosmo_dict_local = {}
        else: raise Exception('provide cosmo_dict or CLASSy engine with cosmo_module=\'class\' ')
        if cosmo_module: # works only with classy now
            cosmo_dict_class = self.set_cosmo(cosmo_dict, module=cosmo_module, engine=cosmo_engine)
            cosmo_dict_local.update(cosmo_dict_class)

        self.__read_cosmo(cosmo_dict_local)
        self.__is_cosmo_conflict()

        if do_core:
            if self.c["mg_model"] == 'bootstrap':
                self.__is_bias_conflict(bias)
                self.bird = Bird(self.cosmo, with_bias=self.c["with_bias"], eft_basis=self.c["eft_basis"], with_stoch=self.c["with_stoch"], with_nnlo_counterterm=self.c["with_nnlo_counterterm"], co=self.co, bias = self.bias)
            else:
                self.bird = Bird(self.cosmo, with_bias=self.c["with_bias"], eft_basis=self.c["eft_basis"], with_stoch=self.c["with_stoch"], with_nnlo_counterterm=self.c["with_nnlo_counterterm"], co=self.co)
            if self.c["with_nnlo_counterterm"]: # we use smooth power spectrum since we don't want spurious BAO signals
                ilogPsmooth = interp1d(np.log(self.bird.kin), np.log(self.cosmo["Psmooth"]), fill_value='extrapolate')
                if self.c["with_cf"]: self.nnlo_counterterm.Cf(self.bird, ilogPsmooth)
                else: self.nnlo_counterterm.Ps(self.bird, ilogPsmooth)
            if not correlator_engine: self.nonlinear.PsCf(self.bird)
            elif correlator_engine: correlator_engine.nonlinear.PsCf(self.bird, c_alpha) # emu
            if self.c["with_uvmatch_2"]: self.matching.Ps(self.bird) 
            if self.c["with_bias"]: self.bird.setPsCf(self.bias)
            else: self.bird.setPsCfl()
            if self.c["with_resum"]:
                if not correlator_engine: self.resum.PsCf(self.bird, makeIR=True, makeQ=False, setIR=False, setPs=False, setCf=False) # compute IR-correction pieces
                elif correlator_engine: correlator_engine.resum.PsCf(self.bird, c_alpha) # emu

        if do_survey_specific:
            if not self.c["with_time"]: self.bird.settime(self.cosmo, co=self.co)
            if self.c["with_resum"]: self.resum.PsCf(self.bird, makeIR=False, makeQ=True, setIR=True, setPs=True, setCf=self.c["with_cf"])
            if self.c["with_redshift_bin"]: self.projection.redshift(self.bird, self.cosmo["rz"], self.cosmo["Dz"], self.cosmo["fz"], pk=self.c["output"])
            if self.c["with_ap"]: self.projection.AP(self.bird)
            if self.c["with_fibercol"]: self.projection.fibcolWindow(self.bird)
            if self.c["with_survey_mask"]: self.projection.Window(self.bird)
            elif self.c["with_binning"]: self.projection.xbinning(self.bird) # no binning if 'with_survey_mask' since the mask should account for it.
            elif self.c["xdata"] is not None: self.projection.xdata(self.bird)
            if self.c["with_wedge"]: self.projection.Wedges(self.bird)

    def get(self, bias=None, pk=True, bk = False, concatenate=False, what="full", diagram='all'):

        if not self.c["with_bias"]:
            self.__is_bias_conflict(bias)
            if pk:
                if "Pk" in self.c["output"]: self.bird.setreducePslb(self.bias, what=what)
                elif "Cf" in self.c["output"]: self.bird.setreduceCflb(self.bias, what=what)
            if bk and self.c["with_bisp"]: self.bispectrum.setBisp(self.bias, diagram=diagram)
        correlator = []
        if pk: 
            if "Pk" in self.c["output"]: pk_ell = self.bird.fullPs
            elif "Cf" in self.c["output"]: pk_ell   = self.bird.fullCf
            correlator.append(pk_ell) 
        if bk and self.c["with_bisp"]: 
            bisp = []
            # MP
            self.bispectrum.setBisp(self.bias, diagram=diagram)
            ###
            if self.c["bisp_quad"] >= 0: bisp.append(self.bispectrum.fullBisp)
            if self.c["bisp_quad"] >= 1: bisp.append(self.bispectrum.fullBisp_002)
            if self.c["bisp_quad"] == 3: bisp.extend([self.bispectrum.fullBisp_200, self.bispectrum.fullBisp_020])
            bisp = np.array(bisp)
            correlator.append(bisp)

        if concatenate: return np.concatenate([c.reshape(-1) for c in correlator])
        else: return correlator[0]

    def getmarg(self, bias, marg_gauss_eft_parameters_list):

        for p in marg_gauss_eft_parameters_list:
            if p not in self.gauss_eft_parameters_list:
                raise Exception("The parameter %s specified in getmarg() is not an available Gaussian EFT parameter to marginalize. Check your options. " % p)

        def marg(loopl, ctl, b1, f, stl=None, nnlol=None, bq=0):

            # concatenating multipoles: loopl.shape = (Nl, Nloop, Nk) -> loop.shape = (Nloop, Nl * Nk)
            loop = np.swapaxes(loopl, axis1=0, axis2=1).reshape(loopl.shape[1],-1)
            ct = np.swapaxes(ctl, axis1=0, axis2=1).reshape(ctl.shape[1],-1)
            if stl is not None: st = np.swapaxes(stl, axis1=0, axis2=1).reshape(stl.shape[1],-1)
            if nnlol is not None: nnlo = np.swapaxes(nnlol, axis1=0, axis2=1).reshape(nnlol.shape[1],-1)

            pg = np.empty(shape=(len(marg_gauss_eft_parameters_list), loop.shape[1]))
            for i, p in enumerate(marg_gauss_eft_parameters_list):
                if p in ['b3', 'bGamma3']:
                    if self.co.Nloop == 12: pg[i] = loop[3] + b1 * loop[7]                          # config["with_time"] = True
                    elif self.co.Nloop == 18: pg[i] = loop[3] + b1 * loop[7] + bq * loop[16]        # config["with_time"] = True, config["with_tidal_alignments"] = True
                    elif self.co.Nloop == 22: pg[i] = f * loop[8] + b1 * loop[16]                   # config["with_time"] = False, config["with_exact_time"] = False
                    elif self.co.Nloop == 35: pg[i] = f * loop[18] + b1 * loop[29]                  # config["with_time"] = False, config["with_exact_time"] = True
                    if p == 'bGamma3': pg[i] *= 6. # b3 = b1 + 15. * bG2 + 6. * bGamma3 : config["eft_basis"] = 'eastcoast'
                # counterterm : config["eft_basis"] = 'eftoflss' or 'westcoast'
                elif p == 'cct': pg[i] = 2 * (f * ct[0+3] + b1 * ct[0]) / self.c["km"]**2 # ~ 2 (b1 + f * mu^2) k^2/km^2 pk_lin
                elif p == 'cr1': pg[i] = 2 * (f * ct[1+3] + b1 * ct[1]) / self.c["kr"]**2 # ~ 2 (b1 mu^2 + f * mu^4) k^2/kr^2 pk_lin
                elif p == 'cr2': pg[i] = 2 * (f * ct[2+3] + b1 * ct[2]) / self.c["kr"]**2 # ~ 2 (b1 mu^4 + f * mu^6) k^2/kr^2 pk_lin
                # counterterm : config["eft_basis"] = 'eastcoast'                       # (2.15) and (2.23) of 2004.10607
                elif p in ['c0', 'c2', 'c4']:
                    ct0, ct2, ct4 = - 2 * ct[0], - 2 * f * ct[1], - 2 * f**2 * ct[2]    # - 2 ct0 k^2 pk_lin , - 2 ct2 f mu^2 k^2 pk_lin , - 2 ct4 f^2 mu^4 k^2 pk_lin
                    if p == 'c0':   pg[i] = ct0
                    elif p == 'c2': pg[i] = - f/3. * ct0 + ct2
                    elif p == 'c4': pg[i] = 3/35. * f**2 * ct0 - 6/7. * f * ct2 + ct4
                # stochastic term
                elif p == 'ce0': pg[i] = st[0] / self.c["nd"] # k^0 / nd mono
                elif p == 'ce1': pg[i] = st[1] / self.c["km"]**2 / self.c["nd"] # k^2 / km^2 / nd mono
                elif p == 'ce2': pg[i] = st[2] / self.c["km"]**2 / self.c["nd"] # k^2 / km^2 / nd quad
                # MP #
                #if stl is not None:
                #    if p == 'Be1': # ce0 = Be1
                #        pg[i] = st[0] # k^0 / nd mono
                #    elif p == 'Be2': # ce1 = Be2 + ce2 / 2 , ce1  # k^2 / km^2 / nd mono
                #        pg[i] = st[1]
                #    elif p == 'ce2': # ce1 = Be2 + ce2 / 2 , ce2 k^2 / km^2 / nd quad
                #        pg[i] = st[2] + 0.5 * st[1] 
                ######
                # elif p == 'Be1': # ce0 = Be1
                #     pg[i] = st[0] # k^0 / nd mono
                # elif p == 'Be2': # ce1 = Be2 + ce2 / 2 , ce1  # k^2 / km^2 / nd mono
                #     pg[i] = st[1]
                # elif p == 'ce2': # ce1 = Be2 + ce2 / 2 , ce2 k^2 / km^2 / nd quad
                #     pg[i] = st[2] + 0.5 * st[1] 
                # nnlo term
                elif p == 'cr4': pg[i] = 0.25 * b1**2 * nnlo[0] / self.c["kr"]**4 # ~ 1/4 b1^2 k^4/kr^4 mu^4 pk_lin
                elif p == 'cr6': pg[i] = 0.25 * b1 * nnlo[1] / self.c["kr"]**4    # ~ 1/4 b1 k^4/kr^4 mu^6 pk_lin
                # nnlo term: config["eft_basis"] = 'eastcoast'
                elif p == 'ct': pg[i] = - f**4 * (b1**2 * nnlo[0] + 2. * b1 * f * nnlo[1] + f**2 * nnlo[2]) # ~ k^4 mu^4 pk_lin

            return pg

        

        def marg_bisp(ext_ctr1, ext_ctr2, ext_st1, ext_st2, b1, b2, b5, f1, nnlo=None, real_space=False):
            
            # passing by value instead of reference: 
            ctr1, ctr2, st1, st2, f = 1.*ext_ctr1, 1.*ext_ctr2, 1.*ext_st1, 1.*ext_st2, 1.*f1

            if real_space: f = 0.

            bispg = np.zeros(shape=(len(marg_gauss_eft_parameters_list), loop321_I.shape[1]))

            for i, p in enumerate(marg_gauss_eft_parameters_list):
                # counterterms
                if 'Bc' in p:
                    bngvec = np.array([b1**2, b1 * b2, b1 * b5, b1**2 * f, b1 * f, b2 * f, b5 * f, b1 * f**2, f**2, f**3]) # Because of the ordering, we can express the following as bngvec.kernels, taking care of the f's
                    bvec1 = np.array([b1**2, b1**2 * f, b1 * f, b1 * f**2, f**2, f**3]) # Because of the ordering, we can express the following as bvec.kernels
                    bvec2 = np.array([b1**2, b1 * f, f**2])
                    if p == 'Bc1':
                        bispg[i] = np.dot(bngvec, ctr1[0:10])
                        bispg[i] += np.dot(bvec1, ctr2[0:6])
                    elif p == 'Bc2':
                        bispg[i] = np.dot(bngvec * f, ctr1[10:20])
                        bispg[i] += np.dot(bvec1 * f, ctr2[6:12])
                        bispg[i] *= self.c['km']**2 / self.c['kr']**2
                    elif p == 'Bc3':
                        bispg[i] = np.dot(bngvec * f**2, ctr1[20:30])
                        bispg[i] += np.dot(bvec1 * f**2, ctr2[12:18])
                        bispg[i] *= self.c['km']**2 / self.c['kr']**2
                    elif p == 'Bc4':
                        bispg[i] = np.dot(bngvec * f**2, ctr1[30:40])
                        bispg[i] += np.dot(bvec1 * f**2, ctr2[18:24])
                        bispg[i] *= self.c['km']**2 / self.c['kr']**2
                    elif p == 'Bc5':
                        bispg[i] = np.dot(bvec2, ctr2[24:27])
                    elif p == 'Bc6':
                        bispg[i] = np.dot(bvec2, ctr2[27:30])
                    elif p == 'Bc7':
                        bispg[i] = np.dot(bvec2, ctr2[30:33]) 
                    elif p == 'Bc8':
                        bispg[i] = np.dot(bvec2, ctr2[33:36]) 
                    elif p == 'Bc9':
                        bispg[i] = np.dot(bvec2 * f, ctr2[36:39])
                        bispg[i] *= self.c['km']**2 / self.c['kr']**2
                    elif p == 'Bc10':
                        bispg[i] = np.dot(bvec2 * f**2, ctr2[39:42])
                        bispg[i] *= self.c['km']**2 / self.c['kr']**2
                    elif p == 'Bc11':
                        bispg[i] = np.dot(bvec2 * f**2, ctr2[42:45])
                        bispg[i] *= self.c['km']**2 / self.c['kr']**2
                    elif p == 'Bc12':
                        bispg[i] = np.dot(bvec2 * f**2, ctr2[45:48])
                        bispg[i] *= self.c['km']**2 / self.c['kr']**2
                    elif p == 'Bc13':
                        bispg[i] = np.dot(bvec2 * f**2, ctr2[48:51])
                        bispg[i] *= self.c['km']**2 / self.c['kr']**2
                    elif p == 'Bc14':
                        bispg[i] = np.dot(bvec2 * f**2, ctr2[51:54])
                        bispg[i] *= self.c['km']**2 / self.c['kr']**2
                # stochastic terms
                elif p == 'Bd1': 
                    bispg[i] = st1[0]
                elif p == 'Bd2': 
                    bispg[i] = st1[1]
                elif p == 'Bd3': 
                    bispg[i] = st1[2]
                # semi-stochastic terms
                elif 'Be' in p:
                    bvec1 = np.array([b1, f, b1 * f, f**2])
                    bvec2 = np.array([b1, f])
                    if p == 'Be1':
                        bispg[i] = np.dot(bvec1, st2[0:4])
                    elif p == 'Be2':
                        bispg[i] = np.dot(bvec1, st2[4:8])
                    elif p == 'Be3':
                        bispg[i] = np.dot(bvec1 * f**2, st2[8:12])
                    elif p == 'Be4':
                        bispg[i] = np.dot(bvec1 * f**2, st2[12:16])
                    elif p == 'Be5':
                        bispg[i] = np.dot(bvec2, st2[16:18])
                    elif p == 'Be6':
                        bispg[i] = np.dot(bvec2, st2[18:20])
                    elif p == 'Be7':
                        bispg[i] = np.dot(bvec2, st2[20:22])
                    elif p == 'Be8':
                        bispg[i] = np.dot(bvec2, st2[22:24])
                    elif p == 'Be9':
                        bispg[i] = np.dot(bvec2 * f, st2[24:26])
                    elif p == 'Be10':
                        bispg[i] = np.dot(bvec2 * f, st2[26:28])
                    elif p == 'Be11':
                        bispg[i] = np.dot(bvec2 * f, st2[28:30])
                    elif p == 'Be12':
                        bispg[i] = np.dot(bvec2 * f, st2[30:32])
                # nnlo terms
                elif p == 'Bnnlo1':
                    bvecnnlo1 = np.array([b1**2 * f, b1 * b2 * f, b1 * b5 * f, b1**2 * f**2, b1 * f**2, b2 * f**2, b5 * f**2, b1 * f**3, f**3, f**4])
                    bispg[i] = b1 * np.dot(bvecnnlo1, nnlo[0])
                elif p == 'Bnnlo2': 
                    bvecnnlo2 = np.array([b1**2 * f, b1**2 * f**2, b1 * f**2, b1 * f**3, f**3, f**4])
                    bispg[i] = b1 * np.dot(bvecnnlo2, nnlo[1])

            return bispg

        def marg_from_bird(bird, bias_local):
            self.__is_bias_conflict(bias_local)
            if self.c["with_tidal_alignments"]: bq = self.bias["bq"]
            else: bq = 0.
            if "Pk" in self.c["output"]: return marg(bird.Ploopl, bird.Pctl, self.bias["b1"], bird.f, stl=bird.Pstl, nnlol=bird.Pnnlol, bq=bq)
            elif "Cf" in self.c["output"]: return marg(bird.Cloopl, bird.Cctl, self.bias["b1"], bird.f, stl=bird.Cstl, nnlol=bird.Cnnlol, bq=bq)

        return marg_from_bird(self.bird, bias)
        def marg_from_bisp(bisp_quad=0):
            if bisp_quad == 0: return marg_bisp(self.bispectrum.Bctr1, self.bispectrum.Bctr2, self.bispectrum.Bstoch1, self.bispectrum.Bstoch2, b1=self.bias["Bb1"], b2=self.bias["Bb2"], b5=self.bias["Bb5"], f1=self.bird.f, nnlo=[self.bispectrum.Bnnlo1, self.bispectrum.Bnnlo2], real_space=real_space)
            elif bisp_quad == 1: return marg_bisp(self.bispectrum.Bctr1_002, self.bispectrum.Bctr2_002, self.bispectrum.Bstoch1_002, self.bispectrum.Bstoch2_002, b1=self.bias["Bb1"], b2=self.bias["Bb2"], b5=self.bias["Bb5"], f1=self.bird.f, nnlo=[self.bispectrum.Bnnlo1_002, self.bispectrum.Bnnlo2_002], real_space=real_space)
        
        Tg_ik = marg_from_bird(self.bird)
        
        if self.c["with_bisp"]: 
            if self.c["bisp_quad"] == 0:
                return np.concatenate((Tg_ik, marg_from_bisp(bisp_quad=0, real_space=real_space)), axis=-1)
            elif self.c["bisp_quad"] == 1:
                return np.concatenate((Tg_ik, marg_from_bisp(bisp_quad=0, real_space=real_space), marg_from_bisp(bisp_quad=1)), axis=-1)
        else: 
            return Tg_ik
        
        # if self.c["skycut"] == 1: return marg_from_bird(self.bird, bias)
        # elif self.c["skycut"] > 1: return [ marg_from_bird(bird_i, bias_i) for (bird_i, bias_i) in zip(self.birds, bias) ]

    def __load_engines(self, load_engines=True):

        self.co = Common(
            Nl=self.c["multipole"],
            kmax=self.c["kmax"],
            km=self.c["km"],
            kr=self.c["kr"],
            nd=self.c["nd"],
            eft_basis=self.c["eft_basis"],
            halohalo=self.c["halohalo"],
            with_cf=self.c["with_cf"],
            with_time=self.c["with_time"],
            optiresum=self.c["optiresum"],
            exact_time=self.c["with_exact_time"],
            Omega_rc=self.c["Omega_rc"],
            #fR0=self.c["fR0"],
            background = self.c["expansion_model"],
            model = self.c["mg_model"],
            timedep = self.c["gravity_model"],
            quintessence=self.c["with_quintessence"],
            accboost=self.c["accboost"],
            with_uvmatch=self.c["with_uvmatch_2"],
            with_tidal_alignments=self.c["with_tidal_alignments"],
            nonequaltime=self.c["with_common_nonequal_time"],
            keep_loop_pieces_independent=self.c["keep_loop_pieces_independent"])
        if load_engines:
            self.nonlinear = NonLinear(load=True, save=True, NFFT=256*self.c["fftaccboost"], fftbias=self.c["fftbias"], co=self.co)
            if self.c["with_uvmatch_2"]: self.matching = Matching(self.nonlinear, co=self.co)
            self.resum = Resum(co=self.co)
            self.projection = Projection(self.c["xdata"],
                with_ap=self.c["with_ap"], H_fid=self.c["H_fid"], D_fid=self.c["D_fid"],
                with_survey_mask=self.c["with_survey_mask"], survey_mask_arr_p=self.c["survey_mask_arr_p"], survey_mask_mat_kp=self.c["survey_mask_mat_kp"],
                with_binning=self.c["with_binning"], binsize=self.c["binsize"],
                fibcol=self.c["with_fibercol"],
                with_wedge=self.c["with_wedge"], wedge_mat_wl=self.c["wedge_mat_wl"],
                with_redshift_bin=self.c["with_redshift_bin"], redshift_bin_zz=self.c["redshift_bin_zz"], redshift_bin_nz=self.c["redshift_bin_nz"],
                co=self.co)
            if self.c["with_nnlo_counterterm"]: self.nnlo_counterterm = NNLO_counterterm(co=self.co)
            
            # GDA_Bisp
            if self.c["with_bisp"]:
                if self.c["with_bin_bisp"]:
                    #MM need to correct these commands to account the right AP effect
                    self.bispectrumnl = BispectrumNl(matrix_path=self.c["matrix_path"], matrix_path_quad=self.c["matrix_path_quad"], co=self.co, triangles=self.c["triangle_data"], with_AP=self.c["with_AP"], Om_AP=self.c["Omega_m_AP"], z_AP=self.c["z_AP"], tree_level=self.c["tree_level"], norm=self.c["bisp_norm"])
                else:
                    #MM need to correct these commands to account the right AP effect
                    self.bispectrumnl = BispectrumNl(matrix_path=self.c["matrix_path"], matrix_path_quad=self.c["matrix_path_quad"], co=self.co, triangles=self.c["triangle_data"], with_AP=self.c["with_AP"], Om_AP=self.c["Omega_m_AP"], z_AP=self.c["z_AP"], tree_level=self.c["tree_level"], norm=self.c["bisp_norm"])
                if self.c["with_bisp_window"]:
                    self.co_highkmax = Common(Nl=2, kmax=0.7)  # I put back Nl=2 instead of Nl=0 to try to put some window in the quadrupole
                    self.projection_bisp = Projection(self.c["xdata"], window_fourier_name=self.c["windowPk_bisp"], path_to_window='', window_configspace_file=self.c["windowCf_bisp"], co=self.co_highkmax) # for the bispectrum we ask the window function W_00(k, k') up to k ~ 0.7
                
            

    def __read_cosmo(self, cosmo_dict):

        # Checking if the inputs are consistent with the options
        for (name, cosmo) in zip(self.cosmo_catalog, self.cosmo_catalog.values()):
                for cosmo_key in cosmo_dict:
                    if cosmo_key is name:
                        cosmo.check(cosmo_key, cosmo_dict[cosmo_key])

        # Setting unspecified configs to default value
        for (name, cosmo) in zip(self.cosmo_catalog, self.cosmo_catalog.values()):
            if cosmo.value is None: cosmo.value = cosmo.default

        # Translating the catalog to a dict
        self.cosmo = translate_catalog_to_dict(self.cosmo_catalog)

    def __is_cosmo_conflict(self):

        if self.c["with_bias"]: self.__is_bias_conflict()

        if self.cosmo["kk"] is None or self.cosmo["pk_lin"] is None:
            raise Exception("Please provide a linear matter power spectrum \'pk_lin\' and the corresponding \'kk\'. ")

        if len(self.cosmo["kk"]) != len(self.cosmo["pk_lin"]):
            raise Exception("Please provide a linear matter power spectrum \'pk_lin\' and the corresponding \'kk\' of same length.")

        if self.cosmo["kk"][0] > 1e-4 or self.cosmo["kk"][-1] < 1.:
            raise Exception("Please provide a linear matter spectrum \'pk_lin\' and the corresponding \'kk\' with min(kk) < 1e-4 and max(kk) > 1.")

        if self.c["multipole"] == 0: self.cosmo["f"] = 0.
        elif not self.c["with_redshift_bin"] and self.cosmo["f"] is None:
            raise Exception("Please specify the growth rate \'f\'.")
        elif self.c["with_redshift_bin"] and (self.cosmo["Dz"] is None or self.cosmo["fz"] is None):
            raise Exception("You asked to account the galaxy counts distribution. Please specify \'Dz\' and \'fz\'. ")

        #MM: STILL NEED TO UNDERSTAND MULTIPLE REDSHIFTS!!!
        if not self.c["with_time"] and self.cosmo["D"] is None:
            raise Exception("Please specify the growth factor \'D\'.")

        if self.c["with_nonequal_time"] and (self.cosmo["D1"] is None or self.cosmo["D2"] is None or self.cosmo["f1"] is None or self.cosmo["f2"] is None):
            raise Exception("You asked nonequal time correlator. Pleas specify: \'D1\', \'D2\', \'f1\', \'f2\'.  ")

        # MP #
        #if self.c["mg_model"]=="bootstrap":
        #    pars = ["epsD", "epsf", "epsdg", "epsag", "epsdga"]
        #    for par in pars:
        #        if par not in self.cosmo["mg_parameters"]:
        #            raise Exception(f"{par} not found in 'mg_parameters'. Please specify the following parameters for bootstrap cosmology: {pars}")
        ######


        if self.c["with_ap"] and (self.cosmo["H"] is None or self.cosmo["DA"] is None):
            raise Exception("You asked to apply the AP effect. Please specify \'H\' and \'DA\'. ")

        if not self.c["with_time"] and self.cosmo["A"]: self.cosmo["D"] *= self.cosmo["A"]**.5

    def __is_bias_conflict(self, bias=None):
        if bias is not None: self.cosmo["bias"] = bias
        if self.cosmo["bias"] is None: raise Exception("Please specify \'bias\'. ")
        if isinstance(self.cosmo["bias"], (list, np.ndarray)): self.cosmo["bias"] = self.cosmo["bias"][0]
        if not isinstance(self.cosmo["bias"], dict): raise Exception("Please specify bias in a dict. ")

        for p in self.eft_parameters_list:
            if p not in self.cosmo["bias"]:
                raise Exception ("%s not found, please provide (given command \'eft_basis\': \'%s\') %s" % (p, self.c["eft_basis"], self.eft_parameters_list))

        # PZ: here I should auto-fill the EFT parameters for all output options!!!

        self.bias = self.cosmo["bias"]

        if "b" in self.c["output"]:
            if "westcoast" in self.c["eft_basis"]:
                self.bias["b2"] = 2.**-.5 * (self.bias["c2"] + self.bias["c4"])
                self.bias["b4"] = 2.**-.5 * (self.bias["c2"] - self.bias["c4"])
            elif "eastcoast" in self.c["eft_basis"]:
                self.bias["b2"] = self.bias["b1"] + 7/2. * self.bias["bG2"]
                self.bias["b3"] = self.bias["b1"] + 15. * self.bias["bG2"] + 6. * self.bias["bGamma3"]
                self.bias["b4"] = 1/2. * self.bias["bt2"] - 7/2. * self.bias["bG2"]
        elif "m" in self.c["output"]: self.bias.update({"b1": 1., "b2": 1., "b3": 1., "b4": 0.})
        #MM: this is still work in progres....
        # filling power spectrum EFT parameters from bispectrum EFT parameters
        #self.bias.update({'b1': self.bias['Bb1'], 'b2': self.bias['Bb2'], 'b3': self.bias['Bb3'] + 15 * self.bias['Bb8'], 'b4': self.bias['Bb5'], 
        #    'cct': - self.bias['Bc1'], 'cr1': self.cosmo['f'] * self.bias['Bc2'] - 0.5 * self.cosmo['f']**2 * self.bias['Bc4'], 'cr2': - 0.5 * self.cosmo['f']**2 * self.bias['Bc3'],
        #    'ce0': self.bias['Be1'], 'ce1': self.bias['Be2'] + 0.5 * self.bias['ce2'] })

        #if self.c["with_common_shot_noise_parameter_in_pk_and_bk"]: self.bias["Bd1"] = self.bias["Be1"]**2

        # MP #
        # to link the bias parameters to mg_parameters for bootstrap (temporary solution)
        #if self.c["mg_model"]=="bootstrap":
        #    self.cosmo["mg_parameters"].update({
        #        "b1": self.bias["Bb1"],
        #        "b2": self.bias["Bb2"],
        #        "b4": self.bias["Bb5"],
        #        "Bd1": self.bias["Bd1"],
        #        "Bd2": self.bias["Bd2"],
        #        "Bd3": self.bias["Bd3"]
        #    })
        ######
        # self.bias.update({'b1': 1., 'b2': 1., 'b3': 1., 'b4': 0., 'cct': 0., 'cr1': 0., 'cr2': 0., 'ce0': 0., 'ce1': 0., 'ce2': 0.})

        if self.c["multipole"] == 0: self.bias.update({"cr1": 0., "cr2": 0.})

    def __set_eft_parameters_list(self):
        #MM: Need to inclcude the bispectrum here, please be very carefull with the notation!!!

        if self.c["eft_basis"] in ["eftoflss", "westcoast"]:
            self.gauss_eft_parameters_list = ['cct']
            if self.c["multipole"] >= 2: self.gauss_eft_parameters_list.extend(['cr1', 'cr2'])
        elif self.c["eft_basis"] == "eastcoast":
            self.gauss_eft_parameters_list = ['c0']
            if self.c["multipole"] >= 2: self.gauss_eft_parameters_list.extend(['c2', 'c4'])
        if self.c["with_stoch"]: self.gauss_eft_parameters_list.extend(['ce0', 'ce1', 'ce2'])
        if self.c["with_nnlo_counterterm"]:
            if self.c["eft_basis"] in ["eftoflss", "westcoast"]: self.gauss_eft_parameters_list.extend(['cr4', 'cr6'])
            elif self.c["eft_basis"] == "eastcoast": self.gauss_eft_parameters_list.append('ct')
        self.eft_parameters_list = deepcopy(self.gauss_eft_parameters_list)
        if "b" in self.c["output"]:
            if self.c["eft_basis"] in ["eftoflss", "westcoast"]: self.gauss_eft_parameters_list.append('b3')
            elif self.c["eft_basis"] == "eastcoast": self.gauss_eft_parameters_list.append('bGamma3')
            if self.c["eft_basis"] == "eftoflss": self.eft_parameters_list.extend(['b1', 'b2', 'b3', 'b4'])
            elif self.c["eft_basis"] == "westcoast": self.eft_parameters_list.extend(['b1', 'c2', 'b3', 'c4'])
            elif self.c["eft_basis"] == "eastcoast": self.eft_parameters_list.extend(['b1', 'bt2', 'bG2', 'bGamma3'])
        if self.c["with_tidal_alignments"]: self.eft_parameters_list.append('bq')
        #MM
        if self.c['mg_model'] == 'bootstrap': self.eft_parameters_list.extend(['epsD', 'epsf', 'epsag', 'epsdg', 'epsdga'])

    def __read_config(self, config_dict):

        # Checking if the inputs are consistent with the options
        for config_key in config_dict:
            is_config = False
            for (name, config) in zip(self.c_catalog, self.c_catalog.values()):
                if config_key == name:
                    config.check(config_key, config_dict[config_key])
                    is_config = True
            ### Keep this warning for typos in options that are then unread...
            if not is_config:
                raise Exception("%s is not an available configuration option. Please check correlator.info() for help. " % config_key)

        # Setting unspecified configs to default value
        for (name, config) in zip(self.c_catalog, self.c_catalog.values()):
            if config.value is None: config.value = config.default

        # Translating the catalog to a dict
        self.c = translate_catalog_to_dict(self.c_catalog)

        self.c["accboost"] = float(self.c["accboost"])

    def __is_config_conflict(self):

        if "Cf" in self.c["output"]: self.c.update({"with_cf": True, "with_survey_mask": False, "with_stoch": False})
        else: self.c["with_cf"] = False

        if "bm" in self.c["output"]: self.c["halohalo"] = False
        else: self.c["halohalo"] = True

        if self.c["with_quintessence"]: self.c["with_exact_time"] = True

        self.c["with_common_nonequal_time"] = False # this is to pass for the common Class to setup the numbers of loops (22 and 13 gathered by default)
        if self.c["with_nonequal_time"]:
            self.c.update({"with_bias": False, "with_time": False, "with_common_nonequal_time": True}) # with_common_nonequal_time is to pass for the common Class to setup the numbers of loops (22 and 13 seperated since they have different time dependence)
            if self.c["z1"] is None or self.c["z2"] is None: print("Please specify \'z1\' and \'z2\' for nonequaltime correlator. ")

        if self.c["with_ap"] and (self.c["H_fid"] is None or self.c["D_fid"] is None):
                raise Exception("You asked to apply the AP effect. Please specify \'H_fid\' and \'D_fid\'. ")

        if self.c["with_survey_mask"] and (self.c["survey_mask_arr_p"] is None or self.c["survey_mask_mat_kp"] is None): raise Exception("Survey mask: on. Please specify \'survey_mask_arr_p\' and \'survey_mask_mat_kp\'. ")
        if self.c["with_binning"] and self.c["binsize"] is None: raise Exception("Binning: on. Please provide \'binsize\'.")
        if self.c["with_redshift_bin"]:
            self.c.update({"with_bias": False, "with_time": False, "with_cf": True}) # even for the Pk, we first do the line-of-sight integral in configuration space, then Fourier transform the integrated Cf to get the integrated Pk
            if self.c["redshift_bin_zz"] is None or self.c["redshift_bin_nz"] is None: raise Exception("You asked to account for the galaxy counts distribution over a redshift bins. Please provide a distribution \'redshift_bin_nz\' and corresponding \'redshift_bin_zz\'. ")
        if self.c["with_wedge"] and self.c["wedge_mat_wl"] is None: raise Exception("Please specify \'wedge_mat_wl\'.")

        #MM: just setting it to be sure
        if self.c["mg_model"] == 'bootstrap':
            self.c["with_exact_time"] = True

    def set_cosmo(self, cosmo_dict, module='class', engine=None):

        cosmo = {}

        if module == 'class':
            #MM: notice that inside the likelihood code, when correlator.compute is called, it is always set
            # module = 'class' and this is automatically passed to set_cosmo
            # This is important to us because we will always enter this module!!!

            log10kmax = 0
            if self.c["with_nnlo_counterterm"]: log10kmax = 1 # slower, but required for the wiggle-no-wiggle split scheme

            if not engine:
                from classy import Class
                cosmo_dict_local = cosmo_dict.copy()
                if self.c["with_bias"] and "bias" in cosmo_dict: del cosmo_dict_local["bias"] # remove to not pass it to classy that otherwise complains
                if not self.c["with_time"] and "A" in cosmo_dict: del cosmo_dict_local["A"] # same as above
                if self.c["with_redshift_bin"]: zmax = max(self.c["redshift_bin_zz"])
                else: zmax = self.c["z"]

                if self.c["expansion_model"] == 'w0wa':
                    cosmo_dict_local["Omega_Lambda"] = 0.#1 - (cosmo_dict_local["omega_b"] + cosmo_dict_local["omega_cdm"])/(cosmo_dict_local["h"]**2)

                # MP #
                #if self.c["mg_model"]=="bootstrap":
                #    del cosmo_dict_local["mg_parameters"]
                ######

            
                M = Class()
                M.set(cosmo_dict_local)
                M.set({'output': 'mPk', 'P_k_max_h/Mpc': 10.**log10kmax, 'z_max_pk': zmax })
                M.compute()
            else: M = engine

            cosmo["kk"] = np.logspace(-5, log10kmax, 200)  # k in h/Mpc
            cosmo["pk_lin"] = np.array([M.pk_lin(k*M.h(), self.c["z"])*M.h()**3 for k in cosmo["kk"]]) # P(k) in (Mpc/h)**3

            if self.c["multipole"] > 0: cosmo["f"] = M.scale_independent_growth_factor_f(self.c["z"])
            if not self.c["with_time"]: cosmo["D"] = M.scale_independent_growth_factor(self.c["z"])
            if self.c["with_nonequal_time"]:
                cosmo["D1"] = M.scale_independent_growth_factor(self.c["z1"])
                cosmo["D2"] = M.scale_independent_growth_factor(self.c["z2"])
                cosmo["f1"] = M.scale_independent_growth_factor_f(self.c["z1"])
                cosmo["f2"] = M.scale_independent_growth_factor_f(self.c["z2"])
            if self.c["with_exact_time"] or self.c["with_quintessence"]:
                cosmo["z"] = self.c["z"]
                cosmo["Omega0_m"] = M.Omega0_m()
                try: cosmo["w0_fld"] = cosmo_dict["w0_fld"]#, cosmo["wa_fld"] = cosmo_dict["wa_fld"]
                except: pass
                if self.c["mg_model"] == "EFTofDE":
                    try: self.c["expansion_model"] = M.pars["expansion_model"]
                    except: raise('Asked for EFTofDE, but no expansion_model provided in the log.param')
                    try: self.c["gravity_model"] = M.pars["gravity_model"]
                    except: raise('Asked for EFTofDE, but no gravity_model provided in the log.param')
                if (self.c["expansion_model"] == 'w0wa' and self.c["mg_model"] != "EFTofDE"):
                    try:
                        cosmo["w0_fld"] = M.pars['w0_fld']
                        cosmo["wa_fld"] = M.pars['wa_fld']
                    except: raise('No w0-wa selected inside the likelihood')
                elif self.c["mg_model"] == "EFTofDE":
                    # parameters_smg__1 ---> alpha_B0
                    try: cosmo["alpha_B0"] = float(M.pars['parameters_smg'].split(', ')[0])
                    except: pass
                    # parameters_smg__2 ---> alpha_M0
                    try: cosmo["alpha_M0"] = float(M.pars['parameters_smg'].split(', ')[1])
                    except: pass
                    # parameters_smg__3 ---> alpha_T0
                    try: cosmo["alpha_T0"] = float(M.pars['parameters_smg'].split(', ')[2])
                    except: pass
                    # parameters_smg__4 ---> eta  (time-dep a^eta)
                    try: cosmo["eta"] = float(M.pars['parameters_smg'].split(', ')[3])
                    except: pass
                    if self.c['expansion_model'] == 'w0wa':
                        try:
                            cosmo["w0_fld"] = float(M.pars['expansion_smg'].split(', ')[1])
                            cosmo["wa_fld"] = float(M.pars['expansion_smg'].split(', ')[2])
                        except: raise('You selected w0-wa as background for the EFTofDE model but w0 and wa are not specified in the cosmo dictionary. Check expansion_smg!')
            
            if self.c["with_ap"]:
                cosmo["H"], cosmo["DA"] = M.Hubble(self.c["z"]) / M.Hubble(0.), M.angular_distance(self.c["z"]) * M.Hubble(0.)

            if self.c["with_redshift_bin"]:
                def comoving_distance(z): return M.angular_distance(z) * (1+z) * M.h()
                cosmo["Dz"] = np.array([M.scale_independent_growth_factor(z) for z in self.c["redshift_bin_zz"]])
                cosmo["fz"] = np.array([M.scale_independent_growth_factor_f(z) for z in self.c["redshift_bin_zz"]])
                cosmo["rz"] = np.array([comoving_distance(z) for z in self.c["redshift_bin_zz"]])

            if self.c["mg_model"] == 'nDGP':
                zm = 5 # z in matter domination
                def scale_factor(z): return 1/(1.+z)
                Omega0_m = cosmo["Omega0_m"]
                Om_rc = self.c["Omega_rc"]
                self.GF = GreenFunction(Omega0_m, background='lcdm', model = self.c['model'], Omega_rc = Om_rc)
                Dp = self.GF.D(scale_factor(z))/self.GF.D(scale_factor(zm))
                #Dm = self.GF.Dminus(scale_factor(zfid))/self.GF.Dminus(scale_factor(zm)) #needed?
                D_class = M.scale_independent_growth_factor(self.c["z"]) / M.scale_independent_growth_factor(zm)
                D_lcdm = self.GF.D_LCDM(scale_factor(self.c["z"]))/self.GF.D(scale_factor(zm))
                cosmo["pk_lin"] *= Dp**2 / D_lcdm**2 #GF.D(scale_factor(zfid))**2/M.scale_independent_growth_factor(zfid)**2 #
                cosmo["D"] = self.GF.D(scale_factor(self.c["z"]))/self.GF.D(scale_factor(0))
                #if self.c["multipole"]!=0: cosmo["f"] = GF.fplus(scale_factor(zfid))
                cosmo["f"] = self.GF.fplus(scale_factor(self.c["z"]))

            if self.c["expansion_model"] == 'w0wa':
                #no need to rescale, I can take it directly from Class
                cosmo["f"] = M.scale_independent_growth_factor_f(self.c["z"])#self.GF.fplus(scale_factor(self.c["z"]))
                
            if self.c["mg_model"] == 'EFTofDE':
                #starting depp inside matter domination
                zm = 5.
                def scale_factor(zz): return 1./(1.+zz)
                Omega0_m = cosmo["Omega0_m"]
                alphaB0  = cosmo["alpha_B0"]
                alphaM0  = cosmo["alpha_M0"]
                alphaT0  = cosmo["alpha_T0"]
                eta   = cosmo["eta"]
                back  = self.c["expansion_model"]
                mod   = self.c["mg_model"]
                timed = self.c["gravity_model"]
                if self.c["expansion_model"] == 'w0wa': 
                    w0 = cosmo["w0_fld"]
                    wa = cosmo["wa_fld"]
                else:
                    w0 = -1.
                    wa = 0.
                self.GF = GreenFunction(Omega0_m,w = w0, wa = wa,
                                        alphaM=alphaM0, alphaT=alphaT0,alphaB=alphaB0, eta = eta,
                                        background = back, model = mod, timedep = timed)
                cosmo["D"] = self.GF.D(scale_factor(self.c["z"]))/self.GF.D(scale_factor(0))
                cosmo["f"] = self.GF.fplus(scale_factor(self.c["z"]))
                
            if self.c["mg_model"] == "quintessence":
                # starting deep inside matter domination and evolving to the total adiabatic linear power spectrum.
                # This does not work in the general case, e.g. with massive neutrinos (okish for minimal mass though)
                # This does not work for 'with_redshift_bin': True. # eventually to code up
                zm = 5. # z in matter domination
                def scale_factor(z): return 1/(1.+z)
                cosmo["Omega0_m"] = M.Omega0_m()
                Omega0_m = cosmo["Omega0_m"]
                w = cosmo["w0_fld"]
                self.GF = GreenFunction(Omega0_m, w=w, background='w0wa', model = 'quintessence')
                Dq = self.GF.D(scale_factor(self.c["z"])) / self.GF.D(scale_factor(zm))
                Dm = M.scale_independent_growth_factor(self.c["z"]) / M.scale_independent_growth_factor(zm)
                cosmo["pk_lin"] *= Dq**2 / Dm**2 * ( 1 + (1+w)/(1.-3*w) * (1-Omega0_m)/Omega0_m * (1+zm)**(3*w) )**2 # 1611.07966 eq. (4.15)
                cosmo["f"] = self.GF.fplus(1/(1.+self.c["z"]))

            # wiggle-no-wiggle split # algo: 1003.3999; details: 2004.10607
            def get_smooth_wiggle_resc(kk, pk, alpha_rs=1.): # k [h/Mpc], pk [(Mpc/h)**3]
                kp = np.linspace(1.e-7, 7, 2**16)   # 1/Mpc
                ilogpk = interp1d(np.log(kk * M.h()), np.log(pk / M.h()**3), fill_value="extrapolate") # Mpc**3
                lnkpk = np.log(kp) + ilogpk(np.log(kp))
                harmonics = dst(lnkpk, type=2, norm='ortho')
                odd, even = harmonics[::2], harmonics[1::2]
                nn = np.arange(0, odd.shape[0], 1)
                nobao = np.delete(nn, np.arange(120, 240,1))
                smooth_odd = interp1d(nn, odd, kind='cubic')(nobao)
                smooth_even = interp1d(nn, even, kind='cubic')(nobao)
                smooth_odd = interp1d(nobao, smooth_odd, kind='cubic')(nn)
                smooth_even = interp1d(nobao, smooth_even, kind='cubic')(nn)
                smooth_harmonics =  np.array([[o, e] for (o, e) in zip(smooth_odd, smooth_even)]).reshape(-1)
                smooth_lnkpk = dst(smooth_harmonics, type=3, norm='ortho')
                smooth_pk = np.exp(smooth_lnkpk) / kp
                wiggle_pk = np.exp(ilogpk(np.log(kp))) - smooth_pk
                spk = interp1d(kp, smooth_pk, bounds_error=False)(kk * M.h()) * M.h()**3 # (Mpc/h)**3
                wpk_resc = interp1d(kp, wiggle_pk, bounds_error=False)(alpha_rs * kk * M.h()) * M.h()**3 # (Mpc/h)**3 # wiggle rescaling
                kmask = np.where(kk < 1.02)[0]
                return kk[kmask], spk[kmask], pk[kmask] #spk[kmask]+wpk_resc[kmask]

            if self.c["with_nnlo_counterterm"] or self.c["with_bisp"]:
                cosmo["kk"], cosmo["Psmooth"], cosmo["pk_lin"] = get_smooth_wiggle_resc(cosmo["kk"], cosmo["pk_lin"])
            
            if self.c["with_bisp"]: 
                from scipy.special import spherical_jn
                kx, Pnwx = cosmo["k11"], cosmo["P11"]
                lbao = 110. # M.rs_drag() * M.h()
                ir_fudge_factor = 1. 
                cosmo["Sigma2"] = ir_fudge_factor * np.trapz((1. - spherical_jn(0, kx * lbao) + 2. * spherical_jn(2, kx * lbao)) * Pnwx, kx) / (6.*np.pi**2) # 2 (A0 + A2) [1810.11855, eqs. (A.6) + (B.9), but to match 1804.05080, eq. (7.5)]
                cosmo["deltaSigma2"] = ir_fudge_factor * np.trapz(spherical_jn(2, kx * lbao) * Pnwx, kx) / (2.*np.pi**2)

            # np.savetxt("pk_lin_smooth_wiggle.dat", np.vstack([cosmo["k11"], cosmo["Pnw"], cosmo["Pw"], cosmo["P11"]]).T,
            #     header='k, pk_smooth, pk_wiggle, pk_tot; sigma2 = %.3e, deltaSigma2 = %.3e' % (cosmo["Sigma2"], cosmo["deltaSigma2"])
            #     )


            # MP #
            #if self.c["mg_model"]=="bootstrap":
            #    cosmo["mg_parameters"] = cosmo_dict["mg_parameters"]
            ######

            return cosmo

class BiasCorrelator(Correlator):
    '''
    Class to load pre-computed correlator
    '''
    def __init__(self, config_dict=None, load_engines=False):
        Correlator.__init__(self, config_dict, load_engines=load_engines)

def translate_catalog_to_dict(catalog):
    newdict = dict.fromkeys(catalog)
    for key, option in zip(catalog, catalog.values()):
        newdict[key] = option.value
    return newdict

def typename(onetype):
    if isinstance(onetype, tuple): return [t.__name__ for t in onetype]
    else: return [onetype.__name__]

class Option(object):

    def __init__(self, config_name, config_type, config_list=None, description='', default=None, verbose=False):

        self.verbose = verbose
        self.name = config_name
        self.type = config_type
        self.list = config_list
        self.description = description
        self.default = default
        self.value = None

    def check(self, config_key, config_value):
        is_config = False
        if self.verbose: print("\'%s\': \'%s\'" % (config_key, config_value))
        if isinstance(config_value, self.type):
            if self.list is None: is_config = True
            elif isinstance(config_value, str):
                if any(config_value in o for o in self.list): is_config = True
            elif isinstance(config_value, (int, float, bool)):
                if any(config_value == o for o in self.list): is_config = True
        if is_config:
            self.value = config_value
        else: self.error()
        return is_config

    def error(self):
        if self.list is None:
            try: raise Exception("Input error in \'%s\'; input configs: %s. Check Correlator.info() in any doubt." % (self.name, typename(self.type)))
            except Exception as e: print(e)
        else:
            try: raise Exception("Input error in \'%s\'; input configs: %s. Check Correlator.info() in any doubt." % (self.name, self.list))
            except Exception as e: print(e)
