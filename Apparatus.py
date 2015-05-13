import numpy as np


class Apparatus(object):

    """
    Container for pertinent information about homodyne measurement 
    setups. Includes detunings delta, couplings chi and cavity decay
    rates kappa. Also contains qubit decay rates gamma_1, gamma_phi and
    purcell. Homodyne efficiency eta and phase phi show up too.  
    """

    def __init__(self, omega, delta, chi, kappa, gamma_1, gamma_phi,
                 purcell, eta, phi):
        arg_dict = {'delta': delta, 'chi': chi, 'kappa': kappa,
                    'omega': omega, 'gamma_1': gamma_1, 
                    'gamma_phi': gamma_phi, 'purcell': purcell}
        
        for key in arg_dict.keys():
            try:
                arg_dict[key] = np.array(arg_dict[key], dtype=np.complex128)
            except Exception, e:
                pre_str = "Error while casting "+\
                            "{} to array:\n".format(key)
                e.message = pre_str + e.message
                raise e
        
        if arg_dict['delta'].shape != arg_dict['kappa'].shape:
            raise ValueError("Number of cavity modes ill-defined. "
                "delta indicates {} , ".format(
                    arg_dict['delta'].shape[0])+
                "kappa indicates {}.".format(
                    arg_dict['kappa'].shape[0]))

        if arg_dict['chi'].shape[0] != arg_dict['kappa'].shape[0]:
            raise ValueError("Number of cavity modes ill-defined. "
                "chi indicates {} , ".format(
                    arg_dict['chi'].shape[0])+
                "kappa indicates {}.".format(
                    arg_dict['kappa'].shape[0]))

        if arg_dict['chi'].shape[1] != arg_dict['omega'].shape[0]:
            raise ValueError("Number of qubits ill-defined. "
                "chi indicates {} , ".format(
                    arg_dict['chi'].shape[1])+
                "omega indicates {}.".format(
                    arg_dict['omega'].shape[0]))

        #If these tests pass, we can set delta, chi, kappa, omega, nq,
        # and nm:
        self.delta = arg_dict['delta']
        self.chi = arg_dict['chi']
        self.kappa = arg_dict['kappa']
        self.omega = arg_dict['omega']
        self.nq = arg_dict['chi'].shape[1]
        self.nm = arg_dict['kappa'].shape[0]

        #remaining length checks
        _ = [arg_dict.pop(key) for key in ['delta', 'chi', 'kappa']]
        for key in arg_dict.keys():
            test_len = arg_dict[key].shape[0]
            if test_len != self.nq:
                raise ValueError("Argument {} must have nq ({})".format(
                    key, self.nq)+
                    " entries, has {}.".format(test_len))

        self.gamma_1 = arg_dict['gamma_1'] 
        self.gamma_phi = arg_dict['gamma_phi']
        self.purcell = arg_dict['purcell']
        
        scalar_dict = {'phi': phi, 'eta': eta}
        for key in scalar_dict.keys():
            test_type = type(scalar_dict[key])
            if not np.issubdtype(test_type, np.float):
                raise TypeError("Argument {} must be float".format(key)+
                    ", got {}".format(test_type))

        self.eta = eta
        self.phi = phi
