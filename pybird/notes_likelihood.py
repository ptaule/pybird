########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################

# THIS LIKELIHOOD FILE IS FOR CODING PURPOSES ONLY!!!
# I REMOVED THE ONE-LOOP BISPECTRUM AND THE FNL PARTS TO COMPARE WITH PIERRE'S LIKELIHOOD ON GH
# I ALSO REMOVED ALL THE PARTS THAT I CAN **EASILY** RECOGNIZE IN THE OTHER LIKELIHOOD

########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
import yaml
import numpy as np
from copy import deepcopy
from astropy.io import fits
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
from scipy.special import spherical_jn
from scipy.linalg import block_diag
from scipy.fftpack import dst
from scipy import stats
def pvalue(minchi2, dof): return 1. - stats.chi2.cdf(minchi2, dof)
# import sys
# sys.path.append('/Users/guido/github/pybirdbisp/pybird_fusion/')
import pybird as pb

# MP #
# Skip MontePython bullshittery
# class Likelihood_PBsky(Likelihood):
class Likelihood_PBsky(Likelihood):
    """Likelihood for the power spectrum + bispectrum calculation, for several skycuts with correlation among EFT parameters."""
    def __init__(self, path, data, command_line):
        Likelihood.__init__(self, path, data, command_line)
        self.config = yaml.full_load(open(os.path.join(self.data_directory, self.config_file), 'r'))
        self.conflist = [{} for i in range(self.config["multisky"])]
        self.k, self.triangles = [], []
        self.xmask, self.pkmask, self.bkmask = [], [], []
        self.ydata, self.pkdata, self.bkdata = [], [], []
        self.invcov = []
        self.yerr, self.pkerr, self.bkerr = [], [], []
        self.bg_centers = []
        #MM: all these options were present in my local likelihood code, need this to check step by step
        #with_AP: already in likelihood.py (with_ap)
        #with_bao: already in likelihood.py (with_bao_rec)
        #with_redshift_bin: in io_pb.py (with_redshift_bin)
        #with_survey_mask: in io_pb.py (with_survey_mask)
        #with_window: removed
        #with_binning: in io_pb.py (with_binning)
        #with_fibercol: correlator.py
        #with_exact_time: I put it back
        #with_quintessence: I put it back
        #with_tidal_alignments: correlator.py
        #with_nnlo_counterterm: io_pb.py
        #with_nnlo_higher_derivative: removed
        #with_cf_sys: removed
        #xmaxspacing: removed
        #with_prior_nongauss: removed
        #get_chi2_from_marg: implemented with smarter procedure
        #get_synth: implemented with smarter procedure. NB-->here we need to implement the BISPECTRUM!!!!
        #with_marg_gauss_eft_parameters_centers: removed
        #with_marg_gauss_eft_parameters_tiny_sigma: removed
        #with_loop_from_ext_plin_matching: removed
        #with_bbn: removed and smartly implemented
        #with_perturbativity_prior: removed; not sure what it was doing
        #with_bin_compression: removed
        #with_quadratic_biases_relation_to_linear_bias: removed; would be nice to add
        #with_invcov: removed, probably it was nedeed for large non-diagonal covariances
        #fullcorr_ngc-sgc: removed, but probably included smartly

        #BISPECTURM STUFF
        #with_bisp: not included
        #bisp_resum: not included
        #with_bisp_window: not inclcuded
        #with_bin_bisp: not included
        #with_bisp_nnlo: not included
        #tree_level: not included (this would be important for the priors)
        #with_triangle_selection: not included
        #with_bisp_rebin: not included
        #with_common_shot_noise_parameter_in_pk_and_bk
        #with_bisp_normalization


        options = []
        
        # These are used only in the likelihood, don't need to split in conflist
        for keys in options:
            if not keys in self.config: self.config[keys] = False
            print (keys, ':', self.config[keys])

        options_true_by_default = ["with_resum", "marg", "with_hartlap"]

        for keys in options_true_by_default:
            if not keys in self.config: self.config[keys] = True
            print (keys, ':', self.config[keys])
            
        # loading EFT parameters to vary and priors
        self.eft_parameters_list = [param for param in self.config["eft_prior"]]
        self.marg_gauss_eft_parameters_list = [param for param, prior in self.config["eft_prior"].items() if prior["type"] == 'marg_gauss']
        self.marg_gauss_eft_parameters_prior_mean = np.array([self.config["eft_prior"][param]["mean"] for param in self.marg_gauss_eft_parameters_list])
        self.marg_gauss_eft_parameters_prior_sigma = np.array([self.config["eft_prior"][param]["range"] for param in self.marg_gauss_eft_parameters_list]) 
        # self.marg_gauss_eft_parameters_prior_matrix = np.array([np.diagflat(1. / sigma**2) for sigma in self.marg_gauss_eft_parameters_prior_sigma])
        self.nonmarg_gauss_eft_parameters_list = [param for param, prior in self.config["eft_prior"].items() if prior["type"] == 'gauss']
        self.nonmarg_gauss_eft_parameters_prior_mean = np.array([self.config["eft_prior"][param]["mean"] for param in self.nonmarg_gauss_eft_parameters_list])
        self.nonmarg_gauss_eft_parameters_prior_sigma = np.array([self.config["eft_prior"][param]["range"] for param in self.nonmarg_gauss_eft_parameters_list])
        self.nonmarg_lognormal_eft_parameters_list = [param for param, prior in self.config["eft_prior"].items() if prior["type"] == 'lognormal']
        self.nonmarg_lognormal_eft_parameters_prior_mean = np.array([self.config["eft_prior"][param]["mean"] for param in self.nonmarg_lognormal_eft_parameters_list])
        self.nonmarg_lognormal_eft_parameters_prior_sigma = np.array([self.config["eft_prior"][param]["range"] for param in self.nonmarg_lognormal_eft_parameters_list])
        self.nonmarg_gauss_single_parameters_list = [param for param, prior in self.config["eft_prior"].items() if prior["type"] == 'single_gauss']
        self.nonmarg_gauss_single_parameters_prior_mean = np.array([self.config["eft_prior"][param]["mean"] for param in self.nonmarg_gauss_single_parameters_list]).reshape(-1)
        self.nonmarg_gauss_single_parameters_prior_sigma = np.array([self.config["eft_prior"][param]["range"] for param in self.nonmarg_gauss_single_parameters_list]).reshape(-1)

        ### priors
        if not "epsilon12" in self.config: self.config["epsilon12"] = 0.10
        if not "epsilon13" in self.config: self.config["epsilon13"] = 0.20
        if self.config["multisky"] > 1: 
            print("Correlation CMASS NGC-SGC = ", 1.-self.config["epsilon12"])
            print("Correlation NGC CMASS-LOWZ = ", 1.-self.config["epsilon13"])

        # Off-diagonal correlations between skycuts. Here assume 4 skycuts
        # Parametrization of correlations is 1 - \epsilon^2/2, where \epsilon is the standard deviation on the difference!
        def build_corr(lenbias=1): 
            corr12 = np.diag(np.array(lenbias * [1 - 0.5 * self.config['epsilon12']**2]))
            corr13 = np.diag(np.array(lenbias * [1 - 0.5 * self.config['epsilon13']**2]))
            corr23 = corr12 * corr13
            corr14, corr24, corr34 = corr23, corr13, corr12
            corr = np.block([   [np.eye(lenbias), corr12, corr13, corr14], 
                                [corr12, np.eye(lenbias), corr23, corr24],
                                [corr13, corr23, np.eye(lenbias), corr34], 
                                [corr14, corr24, corr34, np.eye(lenbias)]    ])
            return corr[:lenbias * self.config["multisky"],  :lenbias * self.config["multisky"]]
        self.Nmarg = len(self.marg_gauss_eft_parameters_list) # priors on marginalized Gaussian EFT parameters
        if self.Nmarg > 1: 
            self.priormat = 1/np.concatenate(self.marg_gauss_eft_parameters_prior_sigma.T[:self.config["multisky"]]**2) * np.linalg.inv(build_corr(self.Nmarg)) # prior inverse covariance matrix for marginalization

        # inverse correlation matrix for non-marg EFT parameters : b1, c2, c4
        self.inv_corr_mat = np.linalg.inv(build_corr(1)) 

        # Option conflicts. Let's just resolve them here
        dummy = self.config["multisky"]*[0]

        if "bisp_quad" not in self.config: self.config["bisp_quad"] = 0
        if self.config["bisp_quad"] == 0: self.config["kmin_bisp_quad"], self.config["kmax_bisp_quad"] = dummy, dummy

        # Load data and set up the different configs
        for i in range(self.config["multisky"]):                

            # MP #
            options_for_pybird_in_config = ["with_bisp_stoch", "with_bias", "mg_model",
                                            "output", "multipole", "with_bisp", "wedge",
                                            "with_stoch", "km",
                                            "kr", "with_resum",
                                            "bisp_quad", "eft_basis",
                                            "with_exact_time"]
            ######
            for option in options_for_pybird_in_config:
                self.conflist[i][option] = self.config[option]

            self.conflist[i]["z"] = self.config["z"][i]
            self.conflist[i]["nd"] = self.config["nd"]#[i]

            # Load data and covariance, and mask PS and bispectrum.
            # Notice that ydata has the 2 BAO numbers at the end, correspondingly invcov's last 2 rows and columns regard BAO.
            # So shape of ydata (and of invcov) is len(kmask) + len(bmask) + 2 if there are BAO
            # k, triangles, xmask_for_pybird, kmask, bmask, ydata, pkdata[kmask], bkdata[bmask], yerr, pkerr, bkerr, invcov
            ki, trianglesi, xmaski, pkmaski, bkmaski, ydatai, pkdatai, bkdatai, yerri, pkerri, bkerri, invcovi = self.__load_data(
            self.config["multipole"], self.config["wedge"], self.config["with_bisp"], self.data_directory, self.config["spectrum_file"][i], self.config["bispectrum_file"][i], self.config["covmat_file"][i],
            kmin=self.config["xmin"][i], kmax=self.config["xmax"][i], kmin_bisp=self.config["kmin_bisp"][i], kmax_bisp=self.config["kmax_bisp"][i], 
            bisp_quad=self.config["bisp_quad"], kmin_bisp_quad=self.config["kmin_bisp_quad"][i], kmax_bisp_quad=self.config["kmax_bisp_quad"][i],
            isky=i)
            # print("ki: ", ki)
            self.k.append(ki)
            self.triangles.append(trianglesi)
            self.xmask.append(xmaski)
            self.pkmask.append(pkmaski)
            self.bkmask.append(bkmaski)
            self.ydata.append(ydatai)
            self.pkdata.append(pkdatai)
            self.bkdata.append(bkdatai)
            self.yerr.append(yerri)
            self.pkerr.append(pkerri)
            self.bkerr.append(bkerri)
            self.invcov.append(invcovi)

            self.conflist[i]["xdata"] = ki
            self.conflist[i]["kmax"] = self.config["xmax"][0]# + 0.2 # MP: why the 0.2???

            if self.config["with_bisp"]: 
                self.conflist[i]["triangle_data"] = trianglesi
                self.conflist[i]["matrix_path"] = os.path.join(self.data_directory, self.config["matrix_path"][i])
                if self.config["bisp_quad"] != 0:
                    if "matrix_path_quad" in self.config: self.conflist[i]["matrix_path_quad"] = os.path.join(self.data_directory, self.config["matrix_path_quad"][i])
                    else: self.conflist[i]["matrix_path_quad"] = None # tree-level mode???
                
            


        self.allsky_invcov, self.allsky_ydata = block_diag(*self.invcov), np.concatenate(self.ydata) # for multiskies

        self.cosmovec = []

        # setting classy for pybird
        log10kmax_classy = 0
        self.need_cosmo_arguments(data, {'output': 'mPk', 'z_max_pk': max(self.config["z"]), 'P_k_max_h/Mpc': 10.**log10kmax_classy})
        self.kin = np.logspace(-5, log10kmax_classy, 1000)
        
        self.first_evaluation = True
        self.indent = True # for code readability

    def loglkl(self, cosmo, data):
        # IF FIRST EVALUATION... ALREADY DONE IN THE NEW io_pb.py code.
        if data.need_cosmo_update:
            for i in range(self.config['multisky']):
                #MM: the __set_cosmo func is not needed anymore
                #MM: but need to understand how to handle bootstrap parameters, not sure yet....
                thiscosmo = self.__set_cosmo(cosmo, data, isky=i)
                self.cosmovec.append(thiscosmo)
                self.correlators[i].compute(thiscosmo)

        free_eft_parameters_list = self.use_nuisance
        free_eft_parameters_list = np.array(free_eft_parameters_list).reshape(self.config["multisky"], -1)
        bdict = [] # list of dictionaries of free EFT parameters per sky
        for free_eft_parameters_list_per_sky in free_eft_parameters_list:   
            bdict.append( {p: data.mcmc_parameters[k]['current'] * data.mcmc_parameters[k]['scale'] 
                            for k in free_eft_parameters_list_per_sky for p in self.eft_parameters_list if k.split('_', 1)[0] == p} )
        for i in range(self.config["multisky"]): 
            bdict[i].update({p: 0. for p in self.eft_parameters_list if p not in bdict[i]})

        if self.config["marg"]: # MP: entra qui # 
            if self.indent: # for code readability
                if self.indent:                         ### pieces for data likelihood
                    Tng_k, Tg_ik = [], []
                    for i in range(self.config["multisky"]):
                        Tng_k.append( self.correlators[i].get(bdict[i], pk=True, bk=self.config["with_bisp"])[self.xmask[i]] )
                        Tg_ik.append( self.correlators[i].getmarg(bdict[i], self.marg_gauss_eft_parameters_list)[:, self.xmask[i]] )
                    Png_k, Pg_ik, invcov_kk = [np.concatenate(Tng_k)-self.allsky_ydata], [block_diag(*Tg_ik)], [self.allsky_invcov]
                
            chi2, bg = self.__get_chi2_marg(Png_k, Pg_ik, invcov_kk, self.priormat, data)

        prior = 0.
        #MM: ALL THE PRIOR PROCEDURE WAS IMPLEMENTED MORE SMARTLY 

        lkl = - 0.5 * chi2 + prior

        return lkl

    def __get_chi2_marg(self, Png, Pg, invcov, priormat, montepython_data): 

        def get_Fs(Tng, Tg, invcov):
            F2 = np.einsum('ak,bp,kp->ab', Tg, Tg, invcov, optimize=self.optipath_F2) # + priormat
            F1 = np.einsum('ak,p,kp->a', Tg, Tng, invcov, optimize=self.optipath_F1)
            F0 = self.__get_chi2_non_marg(Tng, invcov) 
            return F2, F1, F0

        for i, (png_i, pg_i, invcov_i) in enumerate(zip(Png, Pg, invcov)):
            F2_i, F1_i, F0_i = get_Fs(png_i, pg_i, invcov_i)
            if i == 0: F2 = F2_i; F1 = F1_i; F0 = F0_i
            else: F2 += F2_i; F1 += F1_i; F0 += F0_i

        F1 -= self.F1_bg_centers # np.einsum('a,ab->b', marg_gauss_eft_parameters_prior_mean, priormat, optimize=self.optipath_F1)
        F2 += priormat 
        invF2 = np.linalg.inv(F2)

        chi2 = F0 - np.einsum('a,b,ab->', F1, F1, invF2, optimize=self.optipath_chi2) + np.linalg.slogdet(F2)[1] 
        chi2 += self.chi2_bg_centers #np.einsum('a,b,ab->', marg_gauss_eft_parameters_prior_mean, marg_gauss_eft_parameters_prior_mean, priormat, optimize=self.optipath_chi2)
        bg = - np.einsum('a,ab->b', F1, invF2, optimize=self.optipath_bg) 
        
        # MP TOBEDELETED (see what happens if I remove the effect of marginalization)
        # return F0, bg
        return chi2, bg

    def __get_chi2_non_marg(self, P, invcov, ydata=None):
        """Standard non-marginalized chi2"""
        tmd = P
        if ydata is not None: tmd -= ydata
        chi2 = np.einsum('k,p,kp->', tmd, tmd, invcov)#, optimize=self.optipath_chi2) # 
        return chi2
    

    # MM: I REMOVED load_data FOR THE MOMENT, JUST FROM MY MENTAL SANITY
    # MM: OF COURSE WE WILL NEED TO INCLUDE THE BISPECTRUM
    
