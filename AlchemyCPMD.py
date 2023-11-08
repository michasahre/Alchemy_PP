from decimal import Decimal, ROUND_HALF_UP
import numpy as np
import ase.io as aio
import os
import shutil

class SetupCPMD(object):
    def __init__(self, parameter_dict):
        # user specified input
        self.parameter_dict = parameter_dict
        # generate remaining input automatically
        self.update_parameters()
        
        # Input file generator
        self.Input = InputCPMD(self.parameter_dict)
        # pp-file generator
        self.PPs = PP_Generator(self.parameter_dict)

    def setup_cpmd_calculation(self, run_inp = None, run_dir = None):
        if run_dir:
            self.parameter_dict['run_dir'] = run_dir
        if run_inp == None:
            run_inp = 'run.inp'
        assert 'run_dir' in self.parameter_dict.keys(), 'run_dir not specified'
        
        # create run_dir
        os.makedirs(self.parameter_dict['run_dir'], exist_ok = True)
        
        # write input file
        self.Input.write_input(os.path.join(self.parameter_dict['run_dir'], run_inp))
        # self.PPs
        self.PPs.write_pp_files(self.parameter_dict['run_dir'])
        # copy restart file if available
        if 'restart_src' in self.parameter_dict.keys() and 'restart_dest' in self.parameter_dict.keys():
            rsrc = self.parameter_dict['restart_src']
            assert os.path.isfile(self.parameter_dict['restart_src']), f'Restart file {rsrc} does not exist.'
            shutil.copyfile(self.parameter_dict['restart_src'], self.parameter_dict['restart_dest'])

    def update_parameters(self):
        """
        define parameters to generate input file and pp-files for CPMD calculation for arbitrary lambda
        values for each atom
        """
        # read positions and nuclear charges from xyz
        atom_symbols, nuc_charges, positions, valence_charges = self.parse_xyz_for_CPMD_input(self.parameter_dict['xyz'])
        self.parameter_dict['atom_symbols_idx'] = atom_symbols['elIdx']
        self.parameter_dict['atom_symbols'] = atom_symbols['el']
        self.parameter_dict['valence_charges'] = np.array(valence_charges)

        # shift molecule to center of box if centering enabled
        if 'center' in self.parameter_dict.keys() and self.parameter_dict['center']:
            coords_final = self.shift2center(positions, np.array([self.parameter_dict['cell absolute'], self.parameter_dict['cell absolute'], self.parameter_dict['cell absolute']])/2)
        else:
            coords_final = positions
        self.parameter_dict['coords'] = coords_final
        
        # assign the lambda values for all atoms 
        # if lval not in parameter dict 'lambda_values' must be manually added to parameter dict for all atoms
        if 'lval' in self.parameter_dict.keys() and type(self.parameter_dict['lval']) == float:
            self.parameter_dict['lambda_values'] = np.full(len(self.parameter_dict['atom_symbols_idx']) , fill_value=self.parameter_dict['lval'])
        else:
            assert 'lambda_values' in self.parameter_dict.keys(), 'Lambda values not specified in parameter dict'

        # add the delta lambda for finite difference calculations
        if 'dlam_fd' in self.parameter_dict.keys():
            if type(self.parameter_dict['atom_fd']) == int or type(self.parameter_dict['atom_fd']) == np.int64:
                self.parameter_dict['lambda_values'][self.parameter_dict['atom_fd']] = self.parameter_dict['lambda_values'][self.parameter_dict['atom_fd']] + self.parameter_dict['dlam_fd']
            else:
                for aid, dlam in zip(self.parameter_dict['atom_fd'], self.parameter_dict['dlam_fd']):
                    self.parameter_dict['lambda_values'][aid] = self.parameter_dict['lambda_values'][aid] + dlam

        # add correct number of electrons (=charge) such that system stays isoelectronic to target molecule
        charge = self.calculate_charge(self.parameter_dict['lambda_values'], self.parameter_dict['valence_charges'])
        self.parameter_dict['charge'] = charge
    
    
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
    
    def shift2center(self, coordinates_initial, centroid_final):
        """
        shifts set of coordinates so that centroid is at centroid_final
        """
        centroid_initial = np.mean(coordinates_initial, axis=0)
        shift = centroid_final - centroid_initial
        return(coordinates_initial+shift)
    
    def calculate_charge(self, lambda_values, valence_charges):
        """
        the charge that must be added to conserve the number of electrons for rescaled pp's
        lambda_values: lambda value for each nucleus
        valence_charges: valence electrons of each nucleus
        """
        elec_pp = float(Decimal((lambda_values*valence_charges).sum()).quantize(0, ROUND_HALF_UP)) # electrons from the PP's after rescaling by lambda
        charge = elec_pp-valence_charges.sum() # electrons that must be added to conserve total number of electrons

        if elec_pp != float(Decimal((lambda_values*valence_charges).sum()).quantize(0, ROUND_HALF_UP)):
            print("Warning, something doesnt add up")
        return(charge)
    
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

class InputCPMD(object):
    """
    writes an input file using a template file for the input and parameters specified in a dictionary
    """
    
    def __init__(self, parameter_dict):
        self.parameter_dict = parameter_dict
    
    def write_input(self, input_file_path):
        """
        writes input file for molecule with specified parameters boxisze L, charge, number of gpts for mesh
        """
        param_section = self.write_params()
        atom_section = self.write_atom_section()
        with open(input_file_path, 'w') as f:
            f.writelines(param_section+['\n']+atom_section)
            
    def write_params(self):
        """
        add correct parameters for boxsize L, charge and gpts to template
        """
        with open(self.parameter_dict['template_path'], 'r') as f:
            template_params = f.readlines()

        for i, line in enumerate(template_params):
            if 'CELL ABSOLUTE' in line:
                cell = self.parameter_dict['cell absolute']
                template_params[i+1] = f'        {cell} {cell} {cell} 0.0 0.0 0.0\n'
            elif 'CHARGE' in line:
                charge = self.parameter_dict['charge']
                template_params[i+1] = f'        {charge}\n'
            elif 'MESH' in line and 'mesh' in self.parameter_dict.keys():
                mesh = self.parameter_dict['mesh']
                template_params[i+1] = f'    {mesh} {mesh} {mesh}\n'
            elif 'FUNCTIONAL' in line and 'functional' in self.parameter_dict.keys():
                functional = self.parameter_dict['functional']
                template_params[i] = f'  FUNCTIONAL {functional}\n'
        return(template_params)
    
    def write_atom(self, atomsym, coordinates, pp_type):
        """
        prepare the input for one atom:
        the name of the pp is 'element_name' + idx of atom in Compound object + 'SG_LDA'
        the coordinates are read from Compund as well (must be shifted to center before)
        """
        line1 = f'*{atomsym}{pp_type} FRAC\n'
        line2 = ' LMAX=S\n'
        line3 = ' 1\n'
        line4 = ' ' + str(coordinates[0]) + ' ' + str(coordinates[1]) + ' ' + str(coordinates[2]) + '\n'
        return( [line1, line2, line3, line4] )

    def write_atom_section(self):
        """
        atomsymbols: list of element names
        coordinates: list of coordinates
        concantenates inputs for individual atoms to one list where each element is one line of the input file
        """
        atom_section = ['&ATOMS\n']
        for atsym, c in zip(self.parameter_dict['atom_symbols_idx'], self.parameter_dict['coords']):
            atom = self.write_atom(atsym, c, self.parameter_dict['pp_type'])
            atom_section.extend(atom)
        atom_section.append('&END')
        return(atom_section)

class PP_Generator(object):
    def __init__(self, parameter_dict):
        self.parameter_dict = parameter_dict
        
    def write_pp_files(self, work_dir):
        elements = self.parameter_dict['atom_symbols']
        elements_indexed = self.parameter_dict['atom_symbols_idx']
        lam_vals = self.parameter_dict['lambda_values']
        pp_dir = self.parameter_dict['pp_dir']
        pp_type = self.parameter_dict['pp_type']
        for el, elIdx, lam_val in zip(elements, elements_indexed, lam_vals):
            pp_file = self.generate_pp_file(lam_val, el, pp_dir, pp_type)
            path_file = os.path.join(work_dir, elIdx + f'{pp_type}')
            with open(path_file, 'w') as f:
                f.writelines(pp_file)
                
    def generate_pp_file(self, lam_val, element, pp_dir, pp_type):
        """
        rescales ZV, local and non local pp-parameters
        """
        name_pp = element + pp_type
        f_pp = os.path.join(pp_dir, name_pp)
        assert os.path.isfile(f_pp), f"PP-file {f_pp} does not exist. PP-files can be downloaded from (select the \"CPMD\"-format) https://htmlpreview.github.io/?https://github.com/cp2k/cp2k-data/blob/master/potentials/Goedecker/index.html"
        new_pp_file = []
        for line in open(f_pp).readlines():
            if 'ZV' in line:
                new_pp_file.append(self.scale_ZV(line, lam_val))
                continue
            if '#C' in line:
                new_pp_file.append(self.scale_coeffs(line, lam_val))
                continue
            if 'H(s)' in line:
                new_pp_file.append(self.scale_hij(line, lam_val))
                continue
            if 'H(p)' in line:
                new_pp_file.append(self.scale_hij(line, lam_val))
                continue
            if 'H(d)' in line:
                new_pp_file.append(self.scale_hij(line, lam_val))
                continue

            new_pp_file.append(line)
        new_pp_file[len(new_pp_file)-1] = new_pp_file[len(new_pp_file)-1].rstrip('\n')
        return(new_pp_file)
    
    def scale_ZV(self, zv_line, lamb):
        """
        rescale nuclear valence charge ZV
        """
        zv=float(zv_line.strip('ZV ='))
        new_zv = zv*lamb
        new_zv_line = '  ZV = {}\n'.format(new_zv)
        return(new_zv_line)

    def scale_coeffs(self, coeffs_line, lam_val):
        """
        rescaling of coefficients in local part of pp
        """

        # extract coefficients and scale by lambda
        parts = np.array([float(c) for c in coeffs_line.split('#C')[0].strip().split()])
        coeffs_rescaled = lam_val*parts[1:]
        num_coeffs = int(parts[0])

        # write line with rescaled parameters
        new_line = '%4d ' %num_coeffs # number of coeffients
        for cs in coeffs_rescaled: # rescaled coefficients
            new_line += ' %20.15f' %cs

        new_line += '   #C ' # coeffcient numbering, e.g. #C C1 C2...
        for i in range(1, num_coeffs+1):
            new_line += f' C{i}'
        new_line += '\n'
        return(new_line)

    def scale_hij(self, coeff_line, lam_val):
        """
        rescaling of h_ij in non local part of PP
        """
        coeff_line_splitted = coeff_line.split()
        # second value is number of projector rows n, we can calculate number of projector values from it as n*(n+1)
        projector_rows = int(coeff_line_splitted[1])
        num_projectors = int(projector_rows*(projector_rows+1)/2)

        # rescale projectors by lambda
        rescaled_projectors = []
        for i in range(2, 2+num_projectors):
            rescaled_hij = float(coeff_line_splitted[i])*lam_val
            coeff_line_splitted[i] = str(rescaled_hij)

        # write line with rescaled parameters
        new_line = 7*' '+coeff_line_splitted[0]+2*' '+coeff_line_splitted[1]+2*' '+' '.join(coeff_line_splitted[2:])+'\n'
        return(new_line)