from collections import OrderedDict
from copy import deepcopy
from numbers import Real, Integral
from pathlib import Path
import warnings
from xml.etree import ElementTree as ET

import numpy as np

import openmc
import openmc.data
import openmc.checkvalue as cv
from openmc._xml import clean_indentation
from .mixin import IDManagerMixin

# cvmt FETs 
from openmc.zernike import * 


# Units for density supported by OpenMC
DENSITY_UNITS = ['g/cm3', 'g/cc', 'kg/m3', 'atom/b-cm', 'atom/cm3', 'sum',
                 'macro']


class Material(IDManagerMixin):
    """A material composed of a collection of nuclides/elements.

    To create a material, one should create an instance of this class, add
    nuclides or elements with :meth:`Material.add_nuclide` or
    `Material.add_element`, respectively, and set the total material density
    with `Material.set_density()`. The material can then be assigned to a cell
    using the :attr:`Cell.fill` attribute.

    Parameters
    ----------
    material_id : int, optional
        Unique identifier for the material. If not specified, an identifier will
        automatically be assigned.
    name : str, optional
        Name of the material. If not specified, the name will be the empty
        string.
    temperature : float, optional
        Temperature of the material in Kelvin. If not specified, the material
        inherits the default temperature applied to the model.

    Attributes
    ----------
    id : int
        Unique identifier for the material
    temperature : float
        Temperature of the material in Kelvin.
    density : float
        Density of the material (units defined separately)
    density_units : str
        Units used for `density`. Can be one of 'g/cm3', 'g/cc', 'kg/m3',
        'atom/b-cm', 'atom/cm3', 'sum', or 'macro'.  The 'macro' unit only
        applies in the case of a multi-group calculation.
    depletable : bool
        Indicate whether the material is depletable.
    nuclides : list of tuple
        List in which each item is a 3-tuple consisting of a nuclide string, the
        percent density, and the percent type ('ao' or 'wo').
    isotropic : list of str
        Nuclides for which elastic scattering should be treated as though it
        were isotropic in the laboratory system.
    average_molar_mass : float
        The average molar mass of nuclides in the material in units of grams per
        mol.  For example, UO2 with 3 nuclides will have an average molar mass
        of 270 / 3 = 90 g / mol.
    volume : float
        Volume of the material in cm^3. This can either be set manually or
        calculated in a stochastic volume calculation and added via the
        :meth:`Material.add_volume_information` method.
    paths : list of str
        The paths traversed through the CSG tree to reach each material
        instance. This property is initialized by calling the
        :meth:`Geometry.determine_paths` method.
    num_instances : int
        The number of instances of this material throughout the geometry.
    fissionable_mass : float
        Mass of fissionable nuclides in the material in [g]. Requires that the
        :attr:`volume` attribute is set.

    """

    next_id = 1
    used_ids = set()

    def __init__(self, material_id=None, name='', temperature=None):
        # Initialize class attributes
        self.id = material_id
        self.name = name
        self.temperature = temperature
        self._density = None
        self._density_units = 'sum'
        self._depletable = False
        self._paths = None
        self._num_instances = None
        self._volume = None
        self._atoms = {}
        self._isotropic = []
        # FETs
        self._zernike_order = None
        self._zernike_type = None
        self._legendre_order = None

        # A list of tuples (nuclide, percent, percent type, poly_zernike) # FETs
        self._nuclides = []

        # The single instance of Macroscopic data present in this material
        # (only one is allowed, hence this is different than _nuclides, etc)
        self._macroscopic = None

        # If specified, a list of table names
        self._sab = []

        # If true, the material will be initialized as distributed
        self._convert_to_distrib_comps = False

        # If specified, this file will be used instead of composition values
        self._distrib_otf_file = None

    def __repr__(self):
        string = 'Material\n'
        string += '{: <16}=\t{}\n'.format('\tID', self._id)
        string += '{: <16}=\t{}\n'.format('\tName', self._name)
        string += '{: <16}=\t{}\n'.format('\tTemperature', self._temperature)
        
        #FETs 
        if self._zernike_order is not None:
            string += '{: <16}=\t{}\n'.format('\tZernike_order', self._zernike_order)
        if self._zernike_type is not None:
            string += '{: <16}=\t{}\n'.format('\tZernike_type', self._zernike_type)
        if self._legendre_order is not None:
            string += '{: <16}=\t{}\n'.format('\tLegendre_order', self._legendre_order)
            
        string += '{: <16}=\t{}'.format('\tDensity', self._density)
        string += ' [{}]\n'.format(self._density_units)

        string += '{: <16}\n'.format('\tS(a,b) Tables')

        for sab in self._sab:
            string += '{: <16}=\t{}\n'.format('\tS(a,b)', sab)

        string += '{: <16}\n'.format('\tNuclides')
        
        for nuclide, percent, percent_type, *tmp_args in self._nuclides: # FETs
            string += '{: <16}'.format('\t{}'.format(nuclide))
            string += '=\t{: <12} [{}]\n'.format(percent, percent_type) 
            if len(tmp_args) > 0: # cvmt FETs zernike, zernikeld, legendre
                string += '\t{: <14} =\t [{}]\n'.format('poly_coeffs', \
                          ' '.join([str(c) for c in tmp_args[0]]))               
                string += '\t{: <14} =\t [{}]\n'.format('poly_type', tmp_args[1])       
            if len(tmp_args) > 2: # cvmt FETs axial coeffts
                string += '\t{: <14} =\t [{}]\n'.format('axial_coeffs', \
                          ' '.join([str(c) for c in tmp_args[2]]))

        if self._macroscopic is not None:
            string += '{: <16}\n'.format('\tMacroscopic Data')
            string += '{: <16}'.format('\t{}'.format(self._macroscopic))

        return string

    @property
    def name(self):
        return self._name

    @property
    def temperature(self):
        return self._temperature

    @property
    def density(self):
        return self._density

    @property
    def density_units(self):
        return self._density_units

    @property
    def depletable(self):
        return self._depletable

    @property
    def paths(self):
        if self._paths is None:
            raise ValueError('Material instance paths have not been determined. '
                             'Call the Geometry.determine_paths() method.')
        return self._paths

    @property
    def num_instances(self):
        if self._num_instances is None:
            raise ValueError(
                'Number of material instances have not been determined. Call '
                'the Geometry.determine_paths() method.')
        return self._num_instances

    @property
    def nuclides(self):
        return self._nuclides

    @property
    def isotropic(self):
        return self._isotropic

    @property
    def convert_to_distrib_comps(self):
        return self._convert_to_distrib_comps

    @property
    def distrib_otf_file(self):
        return self._distrib_otf_file

    @property
    def average_molar_mass(self):

        # Get a list of all the nuclides, with elements expanded
        nuclide_densities = self.get_nuclide_densities()

        # Using the sum of specified atomic or weight amounts as a basis, sum
        # the mass and moles of the material
        mass = 0.
        moles = 0.
        for nuc, vals in nuclide_densities.items():
            if vals[2] == 'ao':
                mass += vals[1] * openmc.data.atomic_mass(nuc)
                moles += vals[1]
            else:
                moles += vals[1] / openmc.data.atomic_mass(nuc)
                mass += vals[1]

        # Compute and return the molar mass
        return mass / moles

    @property
    def volume(self):
        return self._volume

    @name.setter
    def name(self, name):
        if name is not None:
            cv.check_type('name for Material ID="{}"'.format(self._id),
                          name, str)
            self._name = name
        else:
            self._name = ''

    @temperature.setter
    def temperature(self, temperature):
        cv.check_type('Temperature for Material ID="{}"'.format(self._id),
                      temperature, (Real, type(None)))
        self._temperature = temperature

    @depletable.setter
    def depletable(self, depletable):
        cv.check_type('Depletable flag for Material ID="{}"'.format(self.id),
                      depletable, bool)
        self._depletable = depletable

    @volume.setter
    def volume(self, volume):
        if volume is not None:
            cv.check_type('material volume', volume, Real)
        self._volume = volume

    @isotropic.setter
    def isotropic(self, isotropic):
        cv.check_iterable_type('Isotropic scattering nuclides', isotropic,
                               str)
        self._isotropic = list(isotropic)

    @property
    def fissionable_mass(self):
        if self.volume is None:
            raise ValueError("Volume must be set in order to determine mass.")
        density = 0.0
        for nuc, atoms_per_cc, *temp_args in self.get_nuclide_atom_densities().values():
            Z = openmc.data.zam(nuc)[0]
            if Z >= 90:
                density += 1e24 * atoms_per_cc * openmc.data.atomic_mass(nuc) \
                           / openmc.data.AVOGADRO
        return density*self.volume
    # FETs 
    @property
    def zernike_order(self):
        return self._zernike_order
    
    @property
    def zernike_type(self):
        return self._zernike_type
    
    @property    
    def legendre_order(self):
        return self._legendre_order
    
    @zernike_order.setter
    def zernike_order(self, order):
        self._zernike_order = order
    
    @zernike_type.setter
    def zernike_type(self, type):
        self._zernike_type = type
    
    @legendre_order.setter
    def legendre_order(self, order):
        self._legendre_order = order    
    
    @classmethod
    def from_hdf5(cls, group):
        """Create material from HDF5 group

        Parameters
        ----------
        group : h5py.Group
            Group in HDF5 file

        Returns
        -------
        openmc.Material
            Material instance

        """
        mat_id = int(group.name.split('/')[-1].lstrip('material '))

        name = group['name'][()].decode() if 'name' in group else ''
        density = group['atom_density'][()]
        if 'nuclide_densities' in group:
            nuc_densities = group['nuclide_densities'][()]

        # Create the Material
        material = cls(mat_id, name)
        material.depletable = bool(group.attrs['depletable'])
        if 'volume' in group.attrs:
            material.volume = group.attrs['volume']
        if "temperature" in group.attrs:
            material.temperature = group.attrs["temperature"]

        # Read the names of the S(a,b) tables for this Material and add them
        if 'sab_names' in group:
            sab_tables = group['sab_names'][()]
            for sab_table in sab_tables:
                name = sab_table.decode()
                material.add_s_alpha_beta(name)

        # Set the Material's density to atom/b-cm as used by OpenMC
        material.set_density(density=density, units='atom/b-cm')

        if 'nuclides' in group:
            nuclides = group['nuclides'][()]
            # Add all nuclides to the Material
            for fullname, density in zip(nuclides, nuc_densities):
                name = fullname.decode().strip()
                material.add_nuclide(name, percent=density, percent_type='ao')
        if 'macroscopics' in group:
            macroscopics = group['macroscopics'][()]
            # Add all macroscopics to the Material
            for fullname in macroscopics:
                name = fullname.decode().strip()
                material.add_macroscopic(name)

        return material

    def add_volume_information(self, volume_calc):
        """Add volume information to a material.

        Parameters
        ----------
        volume_calc : openmc.VolumeCalculation
            Results from a stochastic volume calculation

        """
        if volume_calc.domain_type == 'material':
            if self.id in volume_calc.volumes:
                self._volume = volume_calc.volumes[self.id].n
                self._atoms = volume_calc.atoms[self.id]
            else:
                raise ValueError('No volume information found for this material.')
        else:
            raise ValueError('No volume information found for this material.')

    def set_density(self, units, density=None):
        """Set the density of the material

        Parameters
        ----------
        units : {'g/cm3', 'g/cc', 'kg/m3', 'atom/b-cm', 'atom/cm3', 'sum', 'macro'}
            Physical units of density.
        density : float, optional
            Value of the density. Must be specified unless units is given as
            'sum'.

        """

        cv.check_value('density units', units, DENSITY_UNITS)
        self._density_units = units

        if units == 'sum':
            if density is not None:
                msg = 'Density "{}" for Material ID="{}" is ignored ' \
                      'because the unit is "sum"'.format(density, self.id)
                warnings.warn(msg)
        else:
            if density is None:
                msg = 'Unable to set the density for Material ID="{}" ' \
                      'because a density value must be given when not using ' \
                      '"sum" unit'.format(self.id)
                raise ValueError(msg)

            cv.check_type('the density for Material ID="{}"'.format(self.id),
                          density, Real)
            self._density = density

    @distrib_otf_file.setter
    def distrib_otf_file(self, filename):
        # TODO: remove this when distributed materials are merged
        warnings.warn('This feature is not yet implemented in a release '
                      'version of openmc')

        if not isinstance(filename, str) and filename is not None:
            msg = 'Unable to add OTF material file to Material ID="{}" with a ' \
                  'non-string name "{}"'.format(self._id, filename)
            raise ValueError(msg)

        self._distrib_otf_file = filename

    @convert_to_distrib_comps.setter
    def convert_to_distrib_comps(self):
        # TODO: remove this when distributed materials are merged
        warnings.warn('This feature is not yet implemented in a release '
                      'version of openmc')

        self._convert_to_distrib_comps = True
    
    # cvmt 
    def add_nuclide(self, nuclide, percent, percent_type='ao', 
                    poly=None, zernike=None, zernike1d=None, 
                    poly_coeffs=None, legendre=None, 
                    legendre_x=None, legendre_y=None, legendre_z=None,
                    legendre_xy=None, legendre_yz=None, legendre_xz=None,):
        """Add a nuclide to the material

        Parameters
        ----------
        nuclide : str
            Nuclide to add, e.g., 'Mo95'
        percent : float
            Atom or weight percent
        percent_type : {'ao', 'wo'}
            'ao' for atom percent and 'wo' for weight percent
        poly: {'poly_type', 'fet_order', 'fet_norm', 'enable'}
            FETs in number density 

        """
        cv.check_type('nuclide', nuclide, str)
        cv.check_type('percent', percent, Real)
        cv.check_value('percent type', percent_type, {'ao', 'wo'})

        if self._macroscopic is not None:
            msg = 'Unable to add a Nuclide to Material ID="{}" as a ' \
                  'macroscopic data-set has already been added'.format(self._id)
            raise ValueError(msg)

        # If nuclide name doesn't look valid, give a warning
        try:
            Z, _, _ = openmc.data.zam(nuclide)
        except ValueError as e:
            warnings.warn(str(e))
        else:
            # For actinides, have the material be depletable by default
            if Z >= 89:
                self.depletable = True

        # cvmt 
        # self._nuclides.append((nuclide, percent, percent_type))
        # FETs 
        if poly is not None:
            if poly['enable'] == True:
                order  = poly['fet_order']
                radius = poly['fet_norm'] 
                if(poly['poly_type']=='zernike'):
                    n      = num_poly(order)
                    coeffs = np.zeros(n)
                    poly_coeffs = np.zeros(n+1)
                    zp = ZernikePolynomial(order, coeffs, radial_norm=radius, sqrt_normed=False)
                elif(poly['poly_type']=='zernike1d'):
                    n      = num_poly1d(order)
                    coeffs = np.zeros(n)
                    poly_coeffs = np.zeros(n+1)
                    zp = Zernike1DPolynomial(order, coeffs, radial_norm=radius)               
                #  
                poly_coeffs[0]  = radius
                poly_coeffs[1:] = zp.coeffs 
                poly_type       = poly['poly_type']
                self._nuclides.append((nuclide, percent, percent_type, poly_coeffs, poly_type))
            else:
                self._nuclides.append((nuclide, percent, percent_type))
        elif zernike is not None:
            poly_type = 'zernike'
            if legendre is not None:
                self._nuclides.append((nuclide, percent, percent_type, zernike, poly_type, legendre))
            else:
                self._nuclides.append((nuclide, percent, percent_type, zernike, poly_type))  
        elif zernike1d is not None:
            poly_type = 'zernike1d'
            if legendre is not None:
                self._nuclides.append((nuclide, percent, percent_type, zernike1d, poly_type, legendre))
            else:
                self._nuclides.append((nuclide, percent, percent_type, zernike1d, poly_type))
        elif legendre_x is not None:
            poly_type = 'legendre-x'
            if legendre is not None:
                self._nuclides.append((nuclide, percent, percent_type, legendre_x, poly_type, legendre))
            else:
                self._nuclides.append((nuclide, percent, percent_type, legendre_x, poly_type))
        elif legendre_y is not None:
            poly_type = 'legendre-y'
            if legendre is not None:
                self._nuclides.append((nuclide, percent, percent_type, legendre_y, poly_type, legendre))
            else:
                self._nuclides.append((nuclide, percent, percent_type, legendre_y, poly_type))
        elif legendre_z is not None:
            poly_type = 'legendre-z'
            if legendre is not None:
                self._nuclides.append((nuclide, percent, percent_type, legendre_z, poly_type, legendre))
            else:
                self._nuclides.append((nuclide, percent, percent_type, legendre_z, poly_type))
        elif legendre_xy is not None:
            poly_type = 'legendre-xy'
            if legendre is not None:
                self._nuclides.append((nuclide, percent, percent_type, legendre_xy, poly_type, legendre))
            else:
                self._nuclides.append((nuclide, percent, percent_type, legendre_xy, poly_type))
        elif legendre_yz is not None:
            poly_type = 'legendre-yz'
            if legendre is not None:
                self._nuclides.append((nuclide, percent, percent_type, legendre_yz, poly_type, legendre))
            else:
                self._nuclides.append((nuclide, percent, percent_type, legendre_yz, poly_type))
        elif legendre_xz is not None:
            poly_type = 'legendre-xz'
            if legendre is not None:
                self._nuclides.append((nuclide, percent, percent_type, legendre_xz, poly_type, legendre))
            else:
                self._nuclides.append((nuclide, percent, percent_type, legendre_xz, poly_type))
        else:
            self._nuclides.append((nuclide, percent, percent_type))
        
        # FETs
        if poly_coeffs is not None:
            self._nuclides.append((nuclide, percent, percent_type, poly_coeffs))
        
        
    # FETs    
    def add_nuclide_fet(self, nuclide, val, percent_type='ao', fet_deplete=None):
        """
        Add a nuclide with FETs coeffs
        """
        if fet_deplete is not None:
            offset = fet_deplete['offset'][int(self.id)]
            if (fet_deplete['name']=='zernike' or \
                fet_deplete['name']=='zernike1d'):
                newval = [offset[0], offset[1], fet_deplete['radius']] + [i for i in val] # offsets 
            elif (fet_deplete['name']=='legendre-x' or \
                  fet_deplete['name']=='legendre-y' or \
                  fet_deplete['name']=='legendre-z' or \
                  fet_deplete['name']=='legendre'):
                newval = [offset[0], offset[1]] + [i for i in val] # offsets  
            self._nuclides.append((nuclide, val[0], percent_type, newval, fet_deplete['name']))
        
        
    def update_nuclide(self, nuclide, val, fet_deplete=None):
        """Update a nuclide with FETs coeffs
        """
        cv.check_type('nuclide', nuclide, str)
        #FETs 
        mp = 1
        if fet_deplete is not None:
            order = fet_deplete['order']
            radius = fet_deplete['radius']
            poly_type = fet_deplete['name']
            offset = fet_deplete['offset'][int(self.id)]
            if fet_deplete['name']== 'zernike':
                mp = (order+1)*(order+2)//2
            elif fet_deplete['name']=='zernike1d':
                mp = order//2 + 1
            elif (fet_deplete['name']=='legendre-x' or \
                  fet_deplete['name']=='legendre-y' or \
                  fet_deplete['name']=='legendre-z' or \
                  fet_deplete['name']=='legendre'):
                mp = order + 1 
            #
        for nuc in self._nuclides:
            if nuclide == nuc[0]:
                percent_type = nuc[2]
                self._nuclides.remove(nuc)
                if fet_deplete is not None:
                    percent = val[0]
                    if (fet_deplete['name']=='zernike' or fet_deplete['name']=='zernike1d'):
                        fet = [offset[0], offset[1], radius] + [i for i in val] # offests
                    elif (fet_deplete['name']=='legendre-x' or \
                          fet_deplete['name']=='legendre-y' or \
                          fet_deplete['name']=='legendre-z' or \
                          fet_deplete['name']=='legendre'):
                        fet = [offset[0], offset[1]] + [i for i in val]
                    self._nuclides.append((nuclide, percent, percent_type, fet, poly_type))
                else:
                    percent = val
                    self._nuclides.append((nuclide, percent, percent_type))
    
    
    def find_nuclide(self, nuclide):
        """ Find the location of the nuclide in material
            If none return -1
        """        
        cv.check_type('nuclide', nuclide, str)
        for loc_nuc in range(len(self._nuclides)):
            nuc = self._nuclides[loc_nuc]
            if nuclide == nuc[0]:
                return loc_nuc
        return -1 
        
    
    def remove_nuclide(self, nuclide):
        """Remove a nuclide from the material

        Parameters
        ----------
        nuclide : str
            Nuclide to remove

        """
        cv.check_type('nuclide', nuclide, str)

        # If the Material contains the Nuclide, delete it
        for nuc in self._nuclides:
            if nuclide == nuc[0]:
                self._nuclides.remove(nuc)
                break

    def add_macroscopic(self, macroscopic):
        """Add a macroscopic to the material.  This will also set the
        density of the material to 1.0, unless it has been otherwise set,
        as a default for Macroscopic cross sections.

        Parameters
        ----------
        macroscopic : str
            Macroscopic to add

        """

        # Ensure no nuclides, elements, or sab are added since these would be
        # incompatible with macroscopics
        if self._nuclides or self._sab:
            msg = 'Unable to add a Macroscopic data set to Material ID="{}" ' \
                  'with a macroscopic value "{}" as an incompatible data ' \
                  'member (i.e., nuclide or S(a,b) table) ' \
                  'has already been added'.format(self._id, macroscopic)
            raise ValueError(msg)

        if not isinstance(macroscopic, str):
            msg = 'Unable to add a Macroscopic to Material ID="{}" with a ' \
                  'non-string value "{}"'.format(self._id, macroscopic)
            raise ValueError(msg)

        if self._macroscopic is None:
            self._macroscopic = macroscopic
        else:
            msg = 'Unable to add a Macroscopic to Material ID="{}". ' \
                  'Only one Macroscopic allowed per ' \
                  'Material.'.format(self._id)
            raise ValueError(msg)

        # Generally speaking, the density for a macroscopic object will
        # be 1.0. Therefore, lets set density to 1.0 so that the user
        # doesnt need to set it unless its needed.
        # Of course, if the user has already set a value of density,
        # then we will not override it.
        if self._density is None:
            self.set_density('macro', 1.0)

    def remove_macroscopic(self, macroscopic):
        """Remove a macroscopic from the material

        Parameters
        ----------
        macroscopic : str
            Macroscopic to remove

        """

        if not isinstance(macroscopic, str):
            msg = 'Unable to remove a Macroscopic "{}" in Material ID="{}" ' \
                  'since it is not a string'.format(self._id, macroscopic)
            raise ValueError(msg)

        # If the Material contains the Macroscopic, delete it
        if macroscopic == self._macroscopic:
            self._macroscopic = None

    def add_element(self, element, percent, percent_type='ao', enrichment=None):
        """Add a natural element to the material

        Parameters
        ----------
        element : str
            Element to add, e.g., 'Zr'
        percent : float
            Atom or weight percent
        percent_type : {'ao', 'wo'}, optional
            'ao' for atom percent and 'wo' for weight percent. Defaults to atom
            percent.
        enrichment : float, optional
            Enrichment for U235 in weight percent. For example, input 4.95 for
            4.95 weight percent enriched U. Default is None
            (natural composition).

        """
        cv.check_type('nuclide', element, str)
        cv.check_type('percent', percent, Real)
        cv.check_value('percent type', percent_type, {'ao', 'wo'})

        if self._macroscopic is not None:
            msg = 'Unable to add an Element to Material ID="{}" as a ' \
                  'macroscopic data-set has already been added'.format(self._id)
            raise ValueError(msg)

        if enrichment is not None:
            if not isinstance(enrichment, Real):
                msg = 'Unable to add an Element to Material ID="{}" with a ' \
                      'non-floating point enrichment value "{}"'\
                      .format(self._id, enrichment)
                raise ValueError(msg)

            elif element != 'U':
                msg = 'Unable to use enrichment for element {} which is not ' \
                      'uranium for Material ID="{}"'.format(element, self._id)
                raise ValueError(msg)

            # Check that the enrichment is in the valid range
            cv.check_less_than('enrichment', enrichment, 100./1.008)
            cv.check_greater_than('enrichment', enrichment, 0., equality=True)

            if enrichment > 5.0:
                msg = 'A uranium enrichment of {} was given for Material ID='\
                      '"{}". OpenMC assumes the U234/U235 mass ratio is '\
                      'constant at 0.008, which is only valid at low ' \
                      'enrichments. Consider setting the isotopic ' \
                      'composition manually for enrichments over 5%.'.\
                      format(enrichment, self._id)
                warnings.warn(msg)

        # Make sure element name is just that
        if not element.isalpha():
            raise ValueError("Element name should be given by the "
                             "element's symbol, e.g., 'Zr'")

        # Add naturally-occuring isotopes
        element = openmc.Element(element)
        for nuclide in element.expand(percent, percent_type, enrichment):
            self.add_nuclide(*nuclide)

    def add_s_alpha_beta(self, name, fraction=1.0):
        r"""Add an :math:`S(\alpha,\beta)` table to the material

        Parameters
        ----------
        name : str
            Name of the :math:`S(\alpha,\beta)` table
        fraction : float
            The fraction of relevant nuclei that are affected by the
            :math:`S(\alpha,\beta)` table.  For example, if the material is a
            block of carbon that is 60% graphite and 40% amorphous then add a
            graphite :math:`S(\alpha,\beta)` table with fraction=0.6.

        """

        if self._macroscopic is not None:
            msg = 'Unable to add an S(a,b) table to Material ID="{}" as a ' \
                  'macroscopic data-set has already been added'.format(self._id)
            raise ValueError(msg)

        if not isinstance(name, str):
            msg = 'Unable to add an S(a,b) table to Material ID="{}" with a ' \
                        'non-string table name "{}"'.format(self._id, name)
            raise ValueError(msg)

        cv.check_type('S(a,b) fraction', fraction, Real)
        cv.check_greater_than('S(a,b) fraction', fraction, 0.0, True)
        cv.check_less_than('S(a,b) fraction', fraction, 1.0, True)

        new_name = openmc.data.get_thermal_name(name)
        if new_name != name:
            msg = 'OpenMC S(a,b) tables follow the GND naming convention. ' \
                  'Table "{}" is being renamed as "{}".'.format(name, new_name)
            warnings.warn(msg)

        self._sab.append((new_name, fraction))

    def make_isotropic_in_lab(self):
        self.isotropic = [x[0] for x in self._nuclides]

    def get_nuclides(self):
        """Returns all nuclides in the material

        Returns
        -------
        nuclides : list of str
            List of nuclide names

        """
        return [x[0] for x in self._nuclides]

    def get_nuclide_densities(self):
        """Returns all nuclides in the material and their densities

        Returns
        -------
        nuclides : dict
            Dictionary whose keys are nuclide names and values are 3-tuples of
            (nuclide, density percent, density percent type)

        """

        nuclides = OrderedDict()
        
        #cvmt FETs
        for nuclide, density, density_type, *tmp_args in self._nuclides:
            if len(tmp_args) > 0:
                tmp_list = [nuclide, density, density_type] + [i for i in tmp_args]
                nuclides[nuclide] = tuple(tmp_list)
            else:
                nuclides[nuclide] = (nuclide, density, density_type)

        return nuclides

    def get_nuclide_atom_densities(self):
        """Returns all nuclides in the material and their atomic densities in
        units of atom/b-cm

        Returns
        -------
        nuclides : dict
            Dictionary whose keys are nuclide names and values are tuples of
            (nuclide, density in atom/b-cm)

        """

        # Expand elements in to nuclides
        nuclides = self.get_nuclide_densities()

        sum_density = False
        if self.density_units == 'sum':
            sum_density = True
            density = 0.
        elif self.density_units == 'macro':
            density = self.density
        elif self.density_units == 'g/cc' or self.density_units == 'g/cm3':
            density = -self.density
        elif self.density_units == 'kg/m3':
            density = -0.001 * self.density
        elif self.density_units == 'atom/b-cm':
            density = self.density
        elif self.density_units == 'atom/cm3' or self.density_units == 'atom/cc':
            density = 1.E-24 * self.density

        # For ease of processing split out nuc, nuc_density,
        # and nuc_density_type in to separate arrays
        nucs = []
        nuc_densities = []
        nuc_density_types = []
        #FETs
        nuc_dummy = []

        for nuclide in nuclides.items():
            nuc, nuc_density, nuc_density_type, *temp_args = nuclide[1] # FETs 
            nucs.append(nuc)
            nuc_densities.append(nuc_density)
            nuc_density_types.append(nuc_density_type)
            # FETs 
            if len(temp_args) > 0: 
                #print(temp_args)
                nuc_dummy.append(temp_args)

        nucs = np.array(nucs)
        nuc_densities = np.array(nuc_densities)
        nuc_density_types = np.array(nuc_density_types)

        if sum_density:
            density = np.sum(nuc_densities)

        percent_in_atom = np.all(nuc_density_types == 'ao')
        density_in_atom = density > 0.
        sum_percent = 0.

        # Convert the weight amounts to atomic amounts
        if not percent_in_atom:
            for n, nuc in enumerate(nucs):
                nuc_densities[n] *= self.average_molar_mass / \
                                    openmc.data.atomic_mass(nuc)

        # Now that we have the atomic amounts, lets finish calculating densities
        sum_percent = np.sum(nuc_densities)
        nuc_densities = nuc_densities / sum_percent

        # Convert the mass density to an atom density
        if not density_in_atom:
            density = -density / self.average_molar_mass * 1.E-24 \
                      * openmc.data.AVOGADRO

        nuc_densities = density * nuc_densities

        nuclides = OrderedDict()
        # FETs 
        for n, nuc in enumerate(nucs):
            if len(nuc_dummy) > 0:
                fet_type = nuc_dummy[n][1]
                n_skip = 3 # skip two values of offsets and radius
                if (fet_type=='legendre-x' or 
                    fet_type=='legendre-y' or
                    fet_type=='legendre-z' or
                    fet_type=='legendre'):
                    n_skip = 2  #skip two values of offsets in legendre 
                #print(nuc_dummy[n][0][:])
                nuc_dummy[n][0][n_skip] = nuc_densities[n] # 
                nuclides[nuc] = (nuc, nuc_densities[n], nuc_dummy[n])
            else:
                nuclides[nuc] = (nuc, nuc_densities[n])
            #print(nuclides[nuc]) # testing for FETs 
        return nuclides

    def get_mass_density(self, nuclide=None):
        """Return mass density of one or all nuclides

        Parameters
        ----------
        nuclides : str, optional
            Nuclide for which density is desired. If not specified, the density
            for the entire material is given.

        Returns
        -------
        float
            Density of the nuclide/material in [g/cm^3]

        """
        mass_density = 0.0
        for nuc, atoms_per_cc in self.get_nuclide_atom_densities().values():
            if nuclide is None or nuclide == nuc:
                density_i = 1e24 * atoms_per_cc * openmc.data.atomic_mass(nuc) \
                            / openmc.data.AVOGADRO
                mass_density += density_i
        return mass_density

    def get_mass(self, nuclide=None):
        """Return mass of one or all nuclides.

        Note that this method requires that the :attr:`Material.volume` has
        already been set.

        Parameters
        ----------
        nuclides : str, optional
            Nuclide for which mass is desired. If not specified, the density
            for the entire material is given.

        Returns
        -------
        float
            Mass of the nuclide/material in [g]

        """
        if self.volume is None:
            raise ValueError("Volume must be set in order to determine mass.")
        return self.volume*self.get_mass_density(nuclide)

    def clone(self, memo=None):
        """Create a copy of this material with a new unique ID.

        Parameters
        ----------
        memo : dict or None
            A nested dictionary of previously cloned objects. This parameter
            is used internally and should not be specified by the user.

        Returns
        -------
        clone : openmc.Material
            The clone of this material

        """

        if memo is None:
            memo = {}

        # If no nemoize'd clone exists, instantiate one
        if self not in memo:
            # Temporarily remove paths -- this is done so that when the clone is
            # made, it doesn't create a copy of the paths (which are specific to
            # an instance)
            paths = self._paths
            self._paths = None

            clone = deepcopy(self)
            clone.id = None
            clone._num_instances = None

            # Restore paths on original instance
            self._paths = paths

            # Memoize the clone
            memo[self] = clone

        return memo[self]

    def _get_nuclide_xml(self, nuclide, distrib=False):
        xml_element = ET.Element("nuclide")
        xml_element.set("name", nuclide[0])

        if not distrib:
            if nuclide[2] == 'ao':
                xml_element.set("ao", str(nuclide[1]))
            else:
                xml_element.set("wo", str(nuclide[1]))
            #FETs CVMT 
            if len(nuclide) > 3:
                #print(nuclide[3])
                tmp_list = nuclide[3]
                xml_element.set("poly_coeffs", ' '.join([str(c) for c in tmp_list])) 
                xml_element.set("poly_type", nuclide[4])
            if len(nuclide) > 5:
                xml_element.set("axial_coeffs", ' '.join([str(c) for c in nuclide[5]]))             
            

        return xml_element

    def _get_macroscopic_xml(self, macroscopic):
        xml_element = ET.Element("macroscopic")
        xml_element.set("name", macroscopic)

        return xml_element

    def _get_nuclides_xml(self, nuclides, distrib=False):
        xml_elements = []
        for nuclide in nuclides:
            xml_elements.append(self._get_nuclide_xml(nuclide, distrib))
        return xml_elements

    def to_xml_element(self, cross_sections=None):
        """Return XML representation of the material

        Parameters
        ----------
        cross_sections : str
            Path to an XML cross sections listing file

        Returns
        -------
        element : xml.etree.ElementTree.Element
            XML element containing material data

        """

        # Create Material XML element
        element = ET.Element("material")
        element.set("id", str(self._id))

        if len(self._name) > 0:
            element.set("name", str(self._name))

        if self._depletable:
            element.set("depletable", "true")

        if self._volume:
            element.set("volume", str(self._volume))

        # Create temperature XML subelement
        if self.temperature is not None:
            element.set("temperature", str(self.temperature))

        # Create density XML subelement
        if self._density is not None or self._density_units == 'sum':
            subelement = ET.SubElement(element, "density")
            if self._density_units != 'sum':
                subelement.set("value", str(self._density))
            subelement.set("units", self._density_units)
        else:
            raise ValueError('Density has not been set for material {}!'
                             .format(self.id))

        if not self._convert_to_distrib_comps:
            if self._macroscopic is None:
                # Create nuclide XML subelements
                subelements = self._get_nuclides_xml(self._nuclides)
                for subelement in subelements:
                    element.append(subelement)
            else:
                # Create macroscopic XML subelements
                subelement = self._get_macroscopic_xml(self._macroscopic)
                element.append(subelement)

        else:
            subelement = ET.SubElement(element, "compositions")

            comps = []
            allnucs = self._nuclides
            dist_per_type = allnucs[0][2]
            for nuc in allnucs:
                if nuc[2] != dist_per_type:
                    msg = 'All nuclides and elements in a distributed ' \
                          'material must have the same type, either ao or wo'
                    raise ValueError(msg)
                comps.append(nuc[1])

            if self._distrib_otf_file is None:
                # Create values and units subelements
                subsubelement = ET.SubElement(subelement, "values")
                subsubelement.text = ' '.join([str(c) for c in comps])
                subsubelement = ET.SubElement(subelement, "units")
                subsubelement.text = dist_per_type
            else:
                # Specify the materials file
                subsubelement = ET.SubElement(subelement, "otf_file_path")
                subsubelement.text = self._distrib_otf_file

            if self._macroscopic is None:
                # Create nuclide XML subelements
                subelements = self._get_nuclides_xml(self._nuclides,
                                                     distrib=True)
                for subelement_nuc in subelements:
                    subelement.append(subelement_nuc)
            else:
                # Create macroscopic XML subelements
                subsubelement = self._get_macroscopic_xml(self._macroscopic)
                subelement.append(subsubelement)

        if self._sab:
            for sab in self._sab:
                subelement = ET.SubElement(element, "sab")
                subelement.set("name", sab[0])
                if sab[1] != 1.0:
                    subelement.set("fraction", str(sab[1]))

        if self._isotropic:
            subelement = ET.SubElement(element, "isotropic")
            subelement.text = ' '.join(self._isotropic)

        return element

    @classmethod
    def from_xml_element(cls, elem):
        """Generate material from an XML element

        Parameters
        ----------
        elem : xml.etree.ElementTree.Element
            XML element

        Returns
        -------
        openmc.Material
            Material generated from XML element

        """
        from collections import OrderedDict
        
        mat_id = int(elem.get('id'))
        mat = cls(mat_id)
        mat.name = elem.get('name')

        if "temperature" in elem.attrib:
            mat.temperature = float(elem.get("temperature"))

        if 'volume' in elem.attrib:
            mat.volume = float(elem.get('volume'))
        mat.depletable = bool(elem.get('depletable'))

        # Get each nuclide
        for nuclide in elem.findall('nuclide'):
            name = nuclide.attrib['name']
            if 'ao' in nuclide.attrib:
                mat.add_nuclide(name, float(nuclide.attrib['ao']))
            elif 'wo' in nuclide.attrib:
                mat.add_nuclide(name, float(nuclide.attrib['wo']), 'wo')
            # FETs 
            if 'poly_coeffs' in nuclide.attrib:
                coeffs = nuclide.attrib['poly_coeffs']
                coeffs = coeffs.split(' ')
                fet_type = nuclide.attrib['poly_type']
                radius = coeffs[2] # skip offset values in zernike or zernike1d
                m = len(coeffs) - 1
                order = 0
                offset = OrderedDict()
                offset[mat_id] = (coeffs[0], coeffs[1]) # offset values 
                if fet_type == 'zernike':
                    delta = 8 * m + 1
                    order = int(np.sqrt(delta) - 3)//2
                elif fet_type == 'zernike1d':
                    order = (m - 1)*2
                elif fet_type == 'legendre': # to do FETs 
                    order = (m - 1) 
                fet_deplete = {'name' : fet_type,
                               'order': order,
                               'radius': radius,
                               'offset': offset}
                if (fet_type=='legendre-x' or \
                    fet_type=='legendre-y' or \
                    fet_type=='legendre-z'):
                    mat.update_nuclide(name, coeffs[2::], fet_deplete=fet_deplete) 
                elif (fet_type=='zernike' or fet_type=='zernike1d'):
                    mat.update_nuclide(name, coeffs[3::], fet_deplete=fet_deplete) # skip offset values 
            
        # Get each S(a,b) table
        for sab in elem.findall('sab'):
            fraction = float(sab.get('fraction', 1.0))
            mat.add_s_alpha_beta(sab.get('name'), fraction)

        # Get total material density
        density = elem.find('density')
        units = density.get('units')
        if units == 'sum':
            mat.set_density(units)
        else:
            value = float(density.get('value'))
            mat.set_density(units, value)

        # Check for isotropic scattering nuclides
        isotropic = elem.find('isotropic')
        if isotropic is not None:
            mat.isotropic = isotropic.text.split()

        return mat
    
    # works for FETs only 
    def export_to_pdf(self, nuc, fet_deplete=None, multiplier=1.0):
        """Export pdf for given nuclide in material
        """
        import openmc.zernike as zer
        
        n_loc = self.find_nuclide(nuc)
        
        if n_loc==-1:
            print("Could not find nuclide in given material.")
            return
        else:
            nuclide = self._nuclides[n_loc]
            if (len(nuclide)<=3):
                print("Not FETs expression.")
                return 
            coeff_tmp = nuclide[3]
            radius = float(coeff_tmp[2]) # skip offset values 
            coeff = [float(i)*multiplier for i in coeff_tmp[3:]] # skip offset values 
            poly_type = nuclide[-1]
            order = 0 
            if poly_type == 'zernike':
                m = len(coeff)
                delta = np.sqrt(8*m + 1)
                order = (int(delta) - 3)//2
            elif poly_type == 'zernike1d':
                order = int(len(coeff) - 1)*2 
            else:
                print("Unknown type of FETs expression.")
                return 
            filename = 'mat-' + str(self.id) + "-" + nuc + ".pdf"
            #print(order, coeff, radius)
            zer_file = zer.ZernikePolynomial(order, coeff, radius, sqrt_normed=False)
            zer_file.plot_disk(20, 32, filename)


class Materials(cv.CheckedList):
    """Collection of Materials used for an OpenMC simulation.

    This class corresponds directly to the materials.xml input file. It can be
    thought of as a normal Python list where each member is a
    :class:`Material`. It behaves like a list as the following example
    demonstrates:

    >>> fuel = openmc.Material()
    >>> clad = openmc.Material()
    >>> water = openmc.Material()
    >>> m = openmc.Materials([fuel])
    >>> m.append(water)
    >>> m += [clad]

    Parameters
    ----------
    materials : Iterable of openmc.Material
        Materials to add to the collection
    cross_sections : str
        Indicates the path to an XML cross section listing file (usually named
        cross_sections.xml). If it is not set, the
        :envvar:`OPENMC_CROSS_SECTIONS` environment variable will be used for
        continuous-energy calculations and
        :envvar:`OPENMC_MG_CROSS_SECTIONS` will be used for multi-group
        calculations to find the path to the HDF5 cross section file.

    """

    def __init__(self, materials=None):
        super().__init__(Material, 'materials collection')
        self._cross_sections = None

        if materials is not None:
            self += materials

    @property
    def cross_sections(self):
        return self._cross_sections

    @cross_sections.setter
    def cross_sections(self, cross_sections):
        cv.check_type('cross sections', cross_sections, str)
        self._cross_sections = cross_sections

    def append(self, material):
        """Append material to collection

        Parameters
        ----------
        material : openmc.Material
            Material to append

        """
        super().append(material)

    def insert(self, index, material):
        """Insert material before index

        Parameters
        ----------
        index : int
            Index in list
        material : openmc.Material
            Material to insert

        """
        super().insert(index, material)

    def make_isotropic_in_lab(self):
        for material in self:
            material.make_isotropic_in_lab()

    def _create_material_subelements(self, root_element):
        for material in sorted(self, key=lambda x: x.id):
            root_element.append(material.to_xml_element(self.cross_sections))

    def _create_cross_sections_subelement(self, root_element):
        if self._cross_sections is not None:
            element = ET.SubElement(root_element, "cross_sections")
            element.text = str(self._cross_sections)

    def export_to_xml(self, path='materials.xml'):
        """Export material collection to an XML file.

        Parameters
        ----------
        path : str
            Path to file to write. Defaults to 'materials.xml'.

        """

        root_element = ET.Element("materials")
        self._create_cross_sections_subelement(root_element)
        self._create_material_subelements(root_element)

        # Clean the indentation in the file to be user-readable
        clean_indentation(root_element)

        # Check if path is a directory
        p = Path(path)
        if p.is_dir():
            p /= 'materials.xml'

        # Write the XML Tree to the materials.xml file
        tree = ET.ElementTree(root_element)
        tree.write(str(p), xml_declaration=True, encoding='utf-8')

    @classmethod
    def from_xml(cls, path='materials.xml'):
        """Generate materials collection from XML file

        Parameters
        ----------
        path : str, optional
            Path to materials XML file

        Returns
        -------
        openmc.Materials
            Materials collection

        """
        tree = ET.parse(path)
        root = tree.getroot()

        # Generate each material
        materials = cls()
        for material in root.findall('material'):
            materials.append(Material.from_xml_element(material))

        # Check for cross sections settings
        xs = tree.find('cross_sections')
        if xs is not None:
            materials.cross_sections = xs.text

        return materials
    