import os
import numpy as np
import pandas as pd
import ase.io as aio
from scipy.integrate import simpson
import scipy.interpolate as si

from ase import Atoms
    
class AtomicEnergy():
    """
    base class to calculate atomic energies
    """
    def __init__(self):
        self.atoms = {} # every item stores information about a single atom as dictionary
        self.atomic_energies = dict()
        self.atomic_energies_scaled = dict()
        
    def calculate_atomic_energies(self, energy_contr, method, add_lam0):       
        self.atomic_energies[energy_contr] = []
        for atom_key in self.atoms.keys():            
            atomic_energy = self.calculate_atomic_energy(atom_key , energy_contr, method, add_lam0)
            self.atomic_energies[energy_contr].append(atomic_energy)
        self.atomic_energies[energy_contr] = np.array(self.atomic_energies[energy_contr])
        
    def rescale_atomic_energies(self, energy_contr):
        self.atomic_energies_scaled[energy_contr] = self.atomic_energies[energy_contr]/self.atomic_energies[energy_contr].sum()*self.mol_energy[energy_contr]
       
    
    def calculate_atomic_energy(self, atom_key, energy_contr, method, add_lam0):
        x = self.lambda_values
        if f'd_{energy_contr}' in self.atoms[atom_key].keys():
            y = self.atoms[atom_key][f'd_{energy_contr}']
        else:
            self.calculate_atomic_derivative(atom_key, energy_contr, add_lam0)
            y = self.atoms[atom_key][f'd_{energy_contr}']
        if method == 'simpson':
            self.atoms[atom_key][f'ae_{energy_contr}'] = simpson(x=x, y=y)
        elif method == 'cubic_splines':
            spline = si.CubicSpline(x, y)
            self.atoms[atom_key][f'ae_{energy_contr}'] = spline.integrate(0,1)
        elif method == 'trapz':
            self.atoms[atom_key][f'ae_{energy_contr}'] = np.trapz(y=y,x=x)
        else:
            raise ValueError('Integration method not implemented')
            
        return(self.atoms[atom_key][f'ae_{energy_contr}'])
    
    def calculate_nuclear_repulsion_noisy(self, method, add_lam0):
        """
        ion_self and ion_esr can have huge spikes for certain lambda values leading to large 
        integration errors. Since these terms are almost linear we approximate the integrals
        by only considering the term at lambda = 1, ion_pseudo is calculated normally
        """
        self.calculate_atomic_energies('ion_pseudo', method, add_lam0)
        self.calculate_all_derivatives('ion_self', add_lam0)
        self.calculate_all_derivatives('ion_esr', add_lam0)
        atomic_nuclear_repulsion = []
        for i, atom in enumerate(self.atoms.keys()):
            atomic_nuclear_repulsion.append(self.atomic_energies['ion_pseudo'][i] - self.atoms[atom]['d_ion_self'][-1]/2+self.atoms[atom]['d_ion_esr'][-1]/2)
        self.atomic_energies['nuclear'] = np.array(atomic_nuclear_repulsion)
            
    def calculate_total_energy_comfort(self, method, add_lam0):
        """
        calculate total energy using calculate_nuclear_repulsion_noisy for the nuclear repulsion
        """
        self.calculate_atomic_energies('electronic', method, add_lam0)
        self.calculate_nuclear_repulsion(method, add_lam0)
        for atom in self.atoms.keys():
            self.atoms[atom]['ae_total'] = self.atoms[atom]['ae_electronic'] + self.atoms[atom]['ae_nuclear']

    def calculate_all_derivatives(self, energy_contr, add_lam0=True):
        for atom in self.atoms.keys():
            self.calculate_atomic_derivative(atom, energy_contr, add_lam0)
    
    def calculate_atomic_derivative(self, atom_key, energy_contr, add_lam0 = True):
        self.atoms[atom_key][f'd_{energy_contr}'] = []
        # add zeroth order derivative = 0.0
        if add_lam0:
            self.atoms[atom_key][f'd_{energy_contr}'].append(0.0)
        
        for logpath0, logpath1 in zip(self.atoms[atom_key]['log0'], self.atoms[atom_key]['log1']):
            self.atoms[atom_key][f'd_{energy_contr}'].append(self.derivative_from_logfiles(logpath0, logpath1, self.delta_lambda, energy_contr))
        self.atoms[atom_key][f'd_{energy_contr}'] = np.array(self.atoms[atom_key][f'd_{energy_contr}'])
            

    def derivative_from_logfiles(self, logpath0, logpath1, delta_lambda, energy_contr):
        """
        parse two logfiles for energy contribution and calculate derivative via finite differences
        """
        try:
            e1 = self.get_energy_contribution_alias(logpath1, energy_contr)
            e0 = self.get_energy_contribution_alias(logpath0, energy_contr)
            derivative = (e1-e0)/delta_lambda
            return(derivative)
        except FileNotFoundError:
            
            if 'lam_0.0' in logpath1:
                print('Lambda = 0.0, setting derivative to 0')
                return(0.0)
            else:
                print('Warning logfile not found, returning nan')
                return(np.nan)
    
    
    def save_data(self, save_path):
        pass

    
    def get_energy_contribution(self, logfile, name):
        energy = 0
        for line in logfile:
            if name in line:
                energy = float(line.split()[-2]) # return last value (should be after calc is converged)
        return(energy)


    def get_energy_contribution_alias(self, logfile_path, energy_contr):
        """
        replace alias of energy contribution with name given in CPMD-logfile
        """
        energy = None
        with open(logfile_path, 'r') as f:
            logfile = f.readlines()
                # get energy
        if energy_contr == 'total':
            energy = self.get_energy_contribution(logfile, 'TOTAL ENERGY =')
        elif energy_contr == 'ion_pseudo':
            energy = self.get_energy_contribution(logfile, '(PSEUDO CHARGE I-I) ENERGY =')
        elif energy_contr == 'ion_self':
            energy = self.get_energy_contribution(logfile, 'ESELF =')
        elif energy_contr == 'ion_esr':
            energy = self.get_energy_contribution(logfile, 'ESR =')
        elif energy_contr == 'kinetic':
            energy = self.get_energy_contribution(logfile, 'KINETIC ENERGY =')
        elif energy_contr == 'xc':
            energy = self.get_energy_contribution(logfile, 'EXCHANGE-CORRELATION ENERGY =') 
        elif energy_contr == 'local_pp':
            energy = self.get_energy_contribution(logfile, 'LOCAL PSEUDOPOTENTIAL ENERGY')
        elif energy_contr == 'nonlocal_pp':
            energy = self.get_energy_contribution(logfile, 'N-L PSEUDOPOTENTIAL ENERGY =')
        elif energy_contr == 'electrostatic':
            energy = self.get_energy_contribution(logfile, 'ELECTROSTATIC ENERGY =')
        elif energy_contr == 'electronic':
            e_el_parts = []
            for e in ['TOTAL ENERGY =','(PSEUDO CHARGE I-I) ENERGY =', 'ESR =','ESELF =']:
                e_el_parts.append(self.get_energy_contribution(logfile, e))
            e_nuc = e_el_parts[1]+e_el_parts[2]-e_el_parts[3]
            e_el = e_el_parts[0]-e_nuc
            energy = e_el
        elif energy_contr == 'potential':
            e_pot_parts = []
            for e in ['TOTAL ENERGY =','(PSEUDO CHARGE I-I) ENERGY =', 'ESR =','ESELF =','KINETIC ENERGY =']:
                e_pot_parts.append(self.get_energy_contribution(logfile, e))
            epot = e_pot_parts[0] - (e_pot_parts[1]+e_pot_parts[2]-e_pot_parts[3]) - e_pot_parts[4]
            energy = epot
        elif energy_contr == 'nuclear':
            e_nuc_parts = []
            for e in ['(PSEUDO CHARGE I-I) ENERGY =', 'ESR =','ESELF =']:
                e_nuc_parts.append(self.get_energy_contribution(logfile, e))
            e_nuc = e_nuc_parts[0]+e_nuc_parts[1]-e_nuc_parts[2]
            energy = e_nuc
        elif energy_contr == 'pp':
            energy = self.get_energy_contribution(logfile, 'LOCAL PSEUDOPOTENTIAL ENERGY') + self.get_energy_contribution(logfile, 'N-L PSEUDOPOTENTIAL ENERGY =')

        elif energy_contr == 'coulomb':
            # electrostatics + pseudopotential
            energy = self.get_energy_contribution(logfile, 'ELECTROSTATIC ENERGY =')
            energy += self.get_energy_contribution(logfile, 'LOCAL PSEUDOPOTENTIAL ENERGY')
            energy += self.get_energy_contribution(logfile, 'N-L PSEUDOPOTENTIAL ENERGY =')

            assert energy != None, f'Could not extract {energy_contr} from {logfile_path}'
        return(energy)

    def get_valence_charge(self, nuclear_charge):
        """
        return valence charge
        """
        if nuclear_charge <=2:
            valence_charge = nuclear_charge
        elif nuclear_charge >= 3 and nuclear_charge <= 10:
            valence_charge = nuclear_charge - 2
        elif nuclear_charge >= 11 and nuclear_charge <= 18:
            valence_charge = nuclear_charge - 10
        elif nuclear_charge >= 30 and nuclear_charge <= 36:
            valence_charge = nuclear_charge - 28
        else:
            raise ValueError('Cannot calculate number of valence electrons!')
        return(valence_charge)
    
    def parse_xyz_for_CPMD_input(self, path2xyz):
        # get structure information from xyz file
        molecule = aio.read(path2xyz)

        atom_symbolsEl = []
        atom_symbolsIdx = []
        for i, el in enumerate(molecule.get_chemical_symbols()):
            atom_symbolsEl.append(el)
            atom_symbolsIdx.append(el + str(i+1))
        atom_symbols = {'el':atom_symbolsEl, 'elIdx':atom_symbolsIdx}

        nuc_charges = molecule.get_atomic_numbers()
        valence_charges = []
        for n in nuc_charges:
            valence_charges.append(self.get_valence_charge(n))

        positions = molecule.get_positions()
        return(atom_symbols, nuc_charges, positions, valence_charges)
    
class AtomicEnergyPerturbation(AtomicEnergy):
    """in contrast to AtomicEnergy the first energy value is read from the logfile not the final value"""
    def __init__(self):
        pass
        
    def get_energy_contribution(self, logfile, name):
        """
        reads energy contribution from logfile, takes the first value (before SCF cycle)
        as energy in perturbation ansatz
        """
        energy = 0
        for line in logfile:
            if name in line:
                energy = float(line.split()[-2]) # return first value (perturbation)
                break
        return(energy)
                
# class PertFrag2(PertFrag):
#     def rescale_atomic_energies(self, energy_contr):
#         self.atomic_energies_scaled = dict()
#         self.atomic_energies_scaled[energy_contr] = self.atomic_energies[energy_contr]/self.atomic_energies[energy_contr].sum()*self.mol_energy[energy_contr]


#     def get_logfile_paths(self):
#         """
#         assume that files are stored under
#         f'/data/sahre/projects/finite_differences/QM9/pert_logfiles/{name}/pert_{name}_{lvalstr}_heavy_{idx}_bw.log'
#         and
#         f'/data/sahre/projects/finite_differences/QM9/pert_logfiles/{name}/pert_{name}_{lvalstr}_heavy_{idx}_fw.log'
#         """
        
#         for atom in self.atoms.keys():
#             self.atoms[atom]['log0'] = []
#             self.atoms[atom]['log1'] = []
            
#             for lvalstr in self.lambda_strings:
#                 idx = self.atoms[atom]['idx']
#                 name = self.compound_name
#                 self.atoms[atom]['log0'].append(f'/data/sahre/projects/finite_differences/QM9/pert_logfiles/{name}/pert_{name}_{lvalstr}_groups_heavy_{idx}_bw.log')
#                 self.atoms[atom]['log1'].append(f'/data/sahre/projects/finite_differences/QM9/pert_logfiles/{name}/pert_{name}_{lvalstr}_groups_heavy_{idx}_fw.log')


                
# class PertSmallMol(AtomicEnergyPerturbation):
#     """
#     Calculate atomic energies for small molecules with perturbation ansatz
#     """
    
#     def __init__(self, compound_name, xyz_path, lam_vals, delta_lambda):
#         self.compound_name = compound_name
#         self.lambda_values = np.round(lam_vals, 2)
#         self.delta_lambda = delta_lambda
#         # initialize atoms
#         atom_symbols, nuc_charges, positions, valence_charges = self.parse_xyz_for_CPMD_input(xyz_path)
#         # store information about every atom
#         self.atoms = dict()
#         for a_id in atom_symbols['elIdx']:
#             self.atoms[a_id] = {'idx':a_id}
#         # get logfile paths
#         self.get_logfile_paths()
#         # save atomic energies in here
#         self.atomic_energies = dict()

        
#     def get_logfile_paths(self):
#         """
#         assume that files are stored under
#         f'/data/sahre/projects/finite_differences/small_molecules/{self.compound_name}/lam_{lam_val}/{idx}/bw/run_pert.log'
#         and
#         f'/data/sahre/projects/finite_differences/small_molecules/{self.compound_name}/lam_{lam_val}/{idx}/fw/run_pert.log'
#         """
#         for atom in self.atoms.keys():
#             self.atoms[atom]['log0'] = []
#             self.atoms[atom]['log1'] = []

#             for lam_val in self.lambda_values:
#                 idx = self.atoms[atom]['idx']
#                 name = self.compound_name
#                 self.atoms[atom]['log0'].append(f'/data/sahre/projects/finite_differences/small_molecules/{name}/lam_{lam_val}/{idx}/bw/run_pert.log')
#                 self.atoms[atom]['log1'].append(f'/data/sahre/projects/finite_differences/small_molecules/{name}/lam_{lam_val}/{idx}/fw/run_pert.log')
                
#     def calculate_nuclear_repulsion(self, method, add_lam0):
#         self.calculate_atomic_energies('ion_pseudo', method, add_lam0)
#         self.calculate_all_derivatives('ion_self', add_lam0)
#         self.calculate_all_derivatives('ion_esr', add_lam0)
        
#         for atom in self.atoms.keys():
#             self.atoms[atom]['ae_nuclear'] = self.atoms[atom]['ae_ion_pseudo'] - self.atoms[atom]['d_ion_self'][-1]/2+self.atoms[atom]['d_ion_esr'][-1]/2
            
#     def calculate_total_energy_comfort(self, method, add_lam0):
#         self.calculate_atomic_energies('electronic', method, add_lam0)
#         self.calculate_nuclear_repulsion(method, add_lam0)
#         for atom in self.atoms.keys():
#             self.atoms[atom]['ae_total'] = self.atoms[atom]['ae_electronic'] + self.atoms[atom]['ae_nuclear']
    
#     def save_results(self, method, path=None):
#         if path == None:
#             path = f'/data/sahre/projects/finite_differences/small_molecules/{self.compound_name}/atomic_energies_pert_{method}.csv'
#         labels = []
#         energies = []
#         energies_el = []
#         energies_nuclear = []
#         for atom in self.atoms:
#             labels.append(self.atoms[atom]['idx'])
#             energies.append(self.atoms[atom]['ae_total'])
#             energies_el.append(self.atoms[atom]['ae_electronic'])
#             energies_nuclear.append(self.atoms[atom]['ae_nuclear'])
#         data = {'atom label':labels, 'E_el':energies_el, 'E_nuc':energies_nuclear,'E_tot':energies}
#         df = pd.DataFrame(data)
#         df.to_csv(path, index=False)
        
# class SmallMol(PertSmallMol):
#     def get_logfile_paths(self):
#         """
#         assume that files are stored under
#         f'/data/sahre/projects/finite_differences/small_molecules/{self.compound_name}/lam_{lam_val}/{idx}/bw/run.log'
#         and
#         f'/data/sahre/projects/finite_differences/small_molecules/{self.compound_name}/lam_{lam_val}/{idx}/fw/run.log'
#         """
#         for atom in self.atoms.keys():
#             self.atoms[atom]['log0'] = []
#             self.atoms[atom]['log1'] = []

#             for lam_val in self.lambda_values:
#                 idx = self.atoms[atom]['idx']
#                 name = self.compound_name
#                 self.atoms[atom]['log0'].append(f'/data/sahre/projects/finite_differences/small_molecules/{name}/lam_{lam_val}/{idx}/bw/run.log')
#                 self.atoms[atom]['log1'].append(f'/data/sahre/projects/finite_differences/small_molecules/{name}/lam_{lam_val}/{idx}/fw/run.log')
                

#     def save_results(self, method, path=None):
#         if path == None:
#             path = f'/data/sahre/projects/finite_differences/small_molecules/{self.compound_name}/atomic_energies_{method}.csv'
#         labels = []
#         energies = []
#         energies_el = []
#         energies_nuclear = []
#         for atom in self.atoms:
#             labels.append(self.atoms[atom]['idx'])
#             energies.append(self.atoms[atom]['ae_total'])
#             energies_el.append(self.atoms[atom]['ae_electronic'])
#             energies_nuclear.append(self.atoms[atom]['ae_nuclear'])
#         data = {'atom label':labels, 'E_el':energies_el, 'E_nuc':energies_nuclear,'E_tot':energies}
#         df = pd.DataFrame(data)
#         df.to_csv(path, index=False)
        
#     def get_energy_contribution(self, logfile, name):
#         energy = 0
#         for line in logfile:
#             if name in line:
#                 energy = float(line.split()[-2]) # return last value (should be after calc is converged)
#         return(energy)