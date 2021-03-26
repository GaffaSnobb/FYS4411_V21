import os
import numpy as np

class Container:
    def __init__(
        self,
        data,
        method,
        n_particles,
        n_dims,
        n_mc_cycles,
        step_size,
        numerical,
        interaction,
        data_type,
        fname
    ):
        """
        Parameters
        ----------

        data : numpy.ndarray
            Array with data from data file.
        
        method : string
            'brute', 'importance', or 'gradient'.

        n_particles : integer
            Total number of particles.

        n_dims : integer
            Number of spatial dimensions.

        n_mc_cycles : integer
            Number of Monte Carlo cycles.

        step_size : float
            Either importance time step or brute force step size.

        numerical : boolean
            True if numerical differentiation, False if analytical.

        interaction : boolean
            Interaction term on / off.

        data_type : string
            'particles', 'onebody', or 'energies'.

        fname : string
            Original file name.
        """
        self.data = data
        self.method = method
        self.n_particles = n_particles
        self.n_dims = n_dims
        self.n_mc_cycles = n_mc_cycles
        self.step_size = step_size
        self.numerical = numerical
        self.interaction = interaction
        self.data_type = data_type
        self.fname = fname


def read_all_files(filter_method=None, filter_data_type=None):
    """
    Read all text files in generated_data/ and store all relevant data
    in 'Container' objects. See Container docstring for details.

    Parameters
    ----------
    filter_method : NoneType, string
        Filter for only reading certain data files. Valid filters are
        'brute', 'importance', 'gradient'. If None, no filter is
        applied.

    filter_data_type : NoneType, string
        Filter for only reading certain data types. Valid filters are
        'particle', 'onebody', 'energies'. If None, no filter is
        applied.

    Returns
    -------
    data_list : list
        List of Container objects.
    """
    fnames = os.listdir("generated_data/")
    data_list = []

    for i in range(len(fnames)):
        fname = fnames[i].split("_")
        if len(fname) != 10:
            """
            Skip 'README.md' (or other files which do not match the
            naming convention).
            """
            print(f"File {fnames[i]} skipped!")
            continue
        
        data = np.loadtxt(fname = "generated_data/" + fnames[i], skiprows=1)
        method = fname[1]
        data_type = fname[8]
        
        if (filter_method != method) and (filter_method is not None):
            """
            Read only certain data files if filter_method is given as
            input.
            """
            continue
        
        if (filter_data_type != data_type) and (filter_data_type is not None):
            """
            Read only certain data files if filter_data_type is given as
            input.
            """
            continue
        
        n_particles = int(fname[2])
        n_dims = int(fname[3])
        n_mc_cycles = int(fname[4])
        step_size = float(fname[5])
        
        if fname[6] == "numerical":
            numerical = True
        elif fname[6] == "analytical":
            numerical = False
        else:
            msg = f"Expected 'numerical' or 'analytical'. Got '{fname[6]}'."
            raise ValueError(msg)

        if fname[7] == "interaction":
            interaction = True
        elif fname[7] == "nointeraction":
            interaction = False
        else:
            msg = f"Expected 'interaction' or 'nointeraction'. Got '{fname[7]}'."
            raise ValueError(msg)
        
        data_list.append(Container(
            data,
            method,
            n_particles,
            n_dims,
            n_mc_cycles,
            step_size,
            numerical,
            interaction,
            data_type,
            fnames[i]
        ))

    return data_list


def get_number_particles(filename):
    """
    DEPRECATED
    """
    f = open(filename)
    n_particles = (f.readline()).split()[1]
    f.close()
    return float(n_particles)

def read_from_file(filename):
    # n_particles = get_number_particles(filename)
    alpha, var, exp, time = np.loadtxt(fname=filename, skiprows=2, unpack=True)
    return alpha, var, exp, time#, n_particles


def read_energy_from_file_v1(filename, clip = False):
    """
    DEPRECATED
    for files with names: output_energy_*.txt
    """
    n_particles = get_number_particles(filename)
    data = np.loadtxt(filename, skiprows=1)
    alphas = data[0,:]
    energies = data[1:,:]

    if clip:
        first_zero_elm = first_zero(energies[0,:], axis=0)
        if first_zero_elm < 0:
            pass
        else:
            new_alphas = alphas[:first_zero_elm]
            new_energies = energies[:,:first_zero_elm]

            energies = new_energies
            alphas = new_alphas
    return alphas, energies, n_particles


def read_energy_from_file(filename, clip = False):
    """
    for files with names: output_energy_*.txt
    """
    n_particles = get_number_particles(filename)
    data = np.loadtxt(filename, skiprows=1)
    alphas = data[0,:]
    energies = data[1:,:]

    if clip:
        first_zero_elm = first_zero(alphas, axis=0)
        new_alphas = alphas[:first_zero_elm]
        new_energies = energies[:,:first_zero_elm]

        energies = new_energies
        alphas = new_alphas

    return alphas, energies, n_particles



def first_zero(arr, axis, invalid_val=-1):
    mask = arr==0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)


if __name__ == "__main__":
    filename = "generated_data/output_energy_gradient_descent.txt"
    alphas, energies, n_particles = read_energy_from_file(filename, clip=True)

    print(alphas)
